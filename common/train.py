"""Fine-tune one (pretrained) speaker-embedding model with AAM-softmax on the train split (dev speakers never seen).

Usage from an experiment folder (run.py):
    from common import train
    train.run(HERE, dict(model="redimnet_M_multi", epochs=2, ...))

Safe start: the AAM head is initialised with the L2-normalised speaker centroids of the INITIAL model's embeddings,
the backbone gets a lower LR than the head (optionally frozen for the first `freeze_steps` steps), linear warmup,
cosine decay, margin ramp. EMA of the weights is kept and evaluated next to the last checkpoint.
KD hooks (used by distill/): teacher utterance embeddings precomputed ONCE on clean full utterances
(npz with `rels`, `emb`), the student sees short augmented crops; modes cos / proj / rel (per-teacher z-scored
similarity matrices), dkd (Decoupled KD, Zhao 2022: teacher logits = clean-utterance embedding vs
leave-one-out speaker centroids of the teacher; student logits = AAM cosines without margin). `l2sp` penalises the distance to the initial weights (anti-forgetting).
`subcenters` K>1: sub-center AAM (Deng 2020), class cosine = max over K sub-centers (noisy-label robust; not with speed_perturb).
`multi_crop` ({"crops": [s..], "bs": {"s": batch}}, for LMFT): every batch has ONE crop length drawn per row among the lengths
the utterance can fill (MultiCropBatchSampler); cfg["bs"] is then unused for sampling.
"""
import copy
import json
import math
import os
import random
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C
from . import evaluate as E
from . import models as M
from . import sv

DEFAULTS = dict(
    model="redimnet_M_multi", init=None, desc="",
    epochs=2, crop=3.0, bs=32, accum=1, workers=6, utt_cap=None,
    lr_backbone=1e-4, lr_head=1e-3, wd=1e-4, warmup=500, min_lr_ratio=0.05, freeze_steps=0,
    margin=0.2, scale=32.0, margin_ramp_epochs=0.5,
    aug="mid", speed_perturb=False, subcenters=1,
    ema=0.999, kd=[], l2sp=0.0, head_init="centroid", hard_batch=None, multi_crop=None,
    seed=0, mem_frac=0.85, amp=True, smoke=False, final_eval=True, eval_devB=False,
    all_data=False, final_pick="ema",  # all_data: final round on ALL 1900 spk (dev included -> no dev eval)
)


# ----------------------------------------------------------------------------- data
def train_rows(utt_cap=None, seed=0, all_data=False):
    splits = ["split_train.txt"] + (["split_devA.txt", "split_devB.txt"] if all_data else [])
    train = {s for f in splits for s in C.read_lines(os.path.join(C.LISTS, f))}
    rows = [l.split("\t") for l in C.read_lines(os.path.join(C.LISTS, "train_index.tsv"))]
    rows = [(s, r, int(n)) for s, r, n in rows if s in train]
    if utt_cap:
        rng = random.Random(seed)
        by = {}
        for x in rows:
            by.setdefault(x[0], []).append(x)
        rows = [y for s in sorted(by) for y in (rng.sample(by[s], utt_cap) if len(by[s]) > utt_cap else by[s])]
    return rows


class CropSet(torch.utils.data.Dataset):
    def __init__(self, rows, spk2id, crop, codec_set, speed_perturb=False):
        self.rows, self.spk2id, self.n = rows, spk2id, int(crop * C.SR)
        self.codec = codec_set
        self.sp = speed_perturb
        self.nspk = len(spk2id)

    def __len__(self):
        return len(self.rows)

    def _read(self, path, st, n, total):
        if total > n:
            w = sv.load_wav(path, st, st + n)
        else:
            w = sv.load_wav(path)
        if len(w) < n:
            w = np.tile(w, n // max(len(w), 1) + 1)[:n]
        return w[:n]

    def __getitem__(self, i):
        n_crop = self.n
        if isinstance(i, tuple):  # (row index, crop samples) from MultiCropBatchSampler
            assert not self.sp, "multi_crop is incompatible with speed_perturb"
            i, n_crop = i
        spk, rel, total = self.rows[i]
        label = self.spk2id[spk]
        factor = 1.0
        if self.sp:
            k = random.randint(0, 2)  # 0: 1.0, 1: 0.9 (new speaker), 2: 1.1 (new speaker)
            factor = [1.0, 0.9, 1.1][k]
            label += k * self.nspk
        need = int(round(n_crop * factor))
        st = random.randint(0, total - need) if total > need else 0
        w = self._read(os.path.join(C.TRAIN_WAV, rel), st, need, total)
        has = rel in self.codec
        cw = self._read(os.path.join(C.CACHE, "codec_train", rel), st, need, total) if has else np.zeros(need, np.float32)
        if factor != 1.0:
            from scipy.signal import resample_poly

            # sox-like speed change (tempo + pitch): n*factor samples -> n samples
            up, down = (10, 9) if factor == 0.9 else (10, 11)
            w = resample_poly(w, up, down).astype(np.float32)[: self.n]
            cw = resample_poly(cw, up, down).astype(np.float32)[: self.n]
            if len(w) < self.n:
                w = np.pad(w, (0, self.n - len(w)))
                cw = np.pad(cw, (0, self.n - len(cw)))
        return torch.from_numpy(np.ascontiguousarray(w)), torch.from_numpy(np.ascontiguousarray(cw)), has, label, i


# ----------------------------------------------------------------------------- head
class AAM(nn.Module):
    def __init__(self, dim, n_cls, scale=32.0, init=None, K=1):
        super().__init__()
        self.K = K
        if init is None:
            W = torch.randn(n_cls * K, dim) * 0.01
        elif K > 1:  # rows [c0_k0, c0_k1, .., c1_k0, ..] so reshape(B, n_cls, K) groups sub-centers per class
            W = F.normalize(init.repeat_interleave(K, 0) + 0.01 * torch.randn(init.shape[0] * K, dim), dim=1)
        else:
            W = init.clone()
        self.W = nn.Parameter(W)
        self.s = scale
        self.m = 0.0

    def forward(self, e, y):
        cos = F.linear(F.normalize(e.float()), F.normalize(self.W.float()))
        if self.K > 1:
            cos = cos.view(cos.shape[0], -1, self.K).max(2).values
        m = self.m
        sin = (1 - cos.pow(2)).clamp(1e-7, 1).sqrt()
        phi = cos * math.cos(m) - sin * math.sin(m)
        phi = torch.where(cos > math.cos(math.pi - m), phi, cos - math.sin(math.pi - m) * m)
        onehot = F.one_hot(y, cos.shape[1]).bool()
        logits = self.s * torch.where(onehot, phi, cos)
        return F.cross_entropy(logits, y), (cos.argmax(1) == y).float().mean(), cos


# ----------------------------------------------------------------------------- KD
class KD(nn.Module):
    """One teacher: precomputed clean full-utterance embeddings aligned to the training rows."""

    def __init__(self, spec, rows, s_dim, spks=None):
        super().__init__()
        z = np.load(spec["path"])
        pos = {r: i for i, r in enumerate(z["rels"].tolist())}
        T = z["emb"].astype(np.float32)
        missing = [r for _, r, _ in rows if r not in pos]
        assert not missing, f"{len(missing)} train utts missing in {spec['path']}"
        self.register_buffer("T", torch.from_numpy(sv.l2n(T[[pos[r] for _, r, _ in rows]])))
        self.mode, self.w = spec["mode"], float(spec["weight"])
        self.name = spec.get("name", os.path.basename(spec["path"]))
        if self.mode == "proj" or (self.mode == "cos" and self.T.shape[1] != s_dim):
            self.proj = nn.Linear(s_dim, self.T.shape[1], bias=False)
        else:
            self.proj = None
        if self.mode == "dkd":
            # classes must match spk2id in run() (sorted speakers); `spks` given when rows were subsampled (smoke)
            spks = spks or sorted({s for s, _, _ in rows})
            sid = {s: i for i, s in enumerate(spks)}
            y_row = torch.tensor([sid[s] for s, _, _ in rows], dtype=torch.long)
            S = torch.zeros(len(spks), self.T.shape[1]).index_add_(0, y_row, self.T.cpu())
            self.register_buffer("y_row", y_row)
            self.register_buffer("S", S)
            self.register_buffer("cnt", torch.bincount(y_row, minlength=len(spks)))
            self.alpha, self.beta = float(spec.get("alpha", 1.0)), float(spec.get("beta", 8.0))
            self.temp, self.t_scale = float(spec.get("T", 4.0)), float(spec.get("t_scale", 32.0))
            self.proj = None

    @staticmethod
    def _z(m):
        off = ~torch.eye(m.shape[0], dtype=torch.bool, device=m.device)
        v = m[off]
        return (m - v.mean()) / v.std().clamp_min(1e-6), off

    def _dkd(self, idx, cos, y):
        T_ = self.temp
        t, yy = self.T[idx], self.y_row[idx]
        assert torch.equal(yy, y), "dkd: class order differs from spk2id"
        cos_t = t @ F.normalize(self.S).T
        own = F.normalize(self.S[yy] - t)
        own_c = torch.where(self.cnt[yy] > 1, (t * own).sum(1), torch.ones_like(cos_t[:, 0]))
        cos_t = cos_t.scatter(1, yy[:, None], own_c[:, None])
        z_t, z_s = self.t_scale * cos_t / T_, 32.0 * cos.float() / T_
        gt = F.one_hot(yy, z_t.shape[1]).bool()
        pt, ps = F.softmax(z_t, 1), F.softmax(z_s, 1)
        bt = torch.stack([pt[gt], 1 - pt[gt]], 1).clamp_min(1e-8)
        bs = torch.stack([ps[gt], 1 - ps[gt]], 1).clamp_min(1e-8)
        tckd = (bt * (bt.log() - bs.log())).sum(1).mean() * T_ ** 2
        lt = F.log_softmax(z_t - 1000.0 * gt, 1)
        ls = F.log_softmax(z_s - 1000.0 * gt, 1)
        nckd = (lt.exp() * (lt - ls)).sum(1).mean() * T_ ** 2
        return self.w * (self.alpha * tckd + self.beta * nckd)

    def forward(self, e, idx, cos=None, y=None):
        if self.mode == "dkd":
            with torch.autocast("cuda", enabled=False):
                return self._dkd(idx, cos, y)
        t = self.T[idx]
        e = e.float()
        if self.mode in ("cos", "proj"):
            s = self.proj(e) if self.proj is not None else e
            return self.w * (1 - F.cosine_similarity(s, t)).mean()
        # relational: z-scored in-batch cosine matrices (per-teacher scale normalisation)
        en = F.normalize(e)
        ms, off = self._z(en @ en.T)
        mt, _ = self._z(t @ t.T)
        return self.w * F.mse_loss(ms[off], mt[off])


# ----------------------------------------------------------------------------- hard batches
class HardBatchSampler(torch.utils.data.Sampler):
    """A fraction `frac` of the batches is built around a random seed speaker and its nearest speakers
    according to a TEACHER (centroids of its clean full-utterance embeddings), `per_spk` utts each;
    the other batches are plain random batches."""

    def __init__(self, rows, teacher_npz, bs, per_spk=2, k_near=40, frac=0.5, seed=0):
        z = np.load(teacher_npz)
        pos = {r: i for i, r in enumerate(z["rels"].tolist())}
        T = sv.l2n(z["emb"].astype(np.float32))
        self.by = {}
        for i, (s, r, _) in enumerate(rows):
            self.by.setdefault(s, []).append(i)
        self.spks = sorted(self.by)
        cent = sv.l2n(np.stack([T[[pos[rows[i][1]] for i in self.by[s]]].mean(0) for s in self.spks]))
        sim = cent @ cent.T
        np.fill_diagonal(sim, -2)
        self.nn = np.argsort(-sim, 1)[:, :k_near]
        self.bs, self.per, self.frac, self.n = bs, per_spk, frac, len(rows)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n // self.bs

    def __iter__(self):
        for _ in range(len(self)):
            if self.rng.random() < self.frac:
                k = self.bs // self.per
                s0 = self.rng.integers(len(self.spks))
                group = [s0] + list(self.rng.choice(self.nn[s0], k - 1, replace=False))
                idx = []
                for g in group:
                    u = self.by[self.spks[g]]
                    idx += list(self.rng.choice(u, self.per, replace=len(u) < self.per))
                yield [int(i) for i in idx]
            else:
                yield [int(i) for i in self.rng.integers(self.n, size=self.bs)]


class MultiCropBatchSampler(torch.utils.data.Sampler):
    """Large-margin FT with several crop lengths. Each epoch every row gets a random crop length among those it can
    fill (total >= L*SR; a row shorter than all crops gets the smallest one, tiled as before); rows are grouped by length,
    cut into full batches of bs[L] (equal audio per batch), batch order shuffled. Yields lists of (row, crop_samples).
    len() = min #batches over the first 10 epoch plans; every epoch yields exactly that many (extra batches dropped)."""

    def __init__(self, rows, crops, bs, seed=0):
        self.crops = sorted(float(c) for c in crops)
        self.bs = {float(k): int(v) for k, v in bs.items()}
        assert set(self.bs) >= set(self.crops), "bs must cover every crop length"
        self.tot = np.array([r[2] for r in rows])
        self.seed, self.epoch = seed, 0
        self.n_batches = min(len(self.plan(e)) for e in range(10))

    def plan(self, epoch):
        rng = np.random.default_rng([self.seed, epoch])
        need = np.array([int(L * C.SR) for L in self.crops])
        valid = self.tot[:, None] >= need[None, :]
        valid[~valid.any(1), 0] = True  # shorter than every crop -> smallest crop
        pick = np.where(valid, rng.random(valid.shape), -1.0).argmax(1)
        out = []
        for k, L in enumerate(self.crops):
            idx = rng.permutation(np.flatnonzero(pick == k))
            b = self.bs[L]
            out += [[(int(i), int(need[k])) for i in idx[j:j + b]] for j in range(0, len(idx) - b + 1, b)]
        return [out[j] for j in rng.permutation(len(out))]

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        plan = self.plan(self.epoch)
        self.epoch += 1
        for i in range(len(self)):
            yield plan[i % len(plan)]


# ----------------------------------------------------------------------------- helpers
def head_init_centroids(model_name, model, use_frozen, rows, spk2id, log):
    """Normalised speaker centroids of the initial model (cached frozen embeddings when the model is the
    untouched pretrained checkpoint and they exist; otherwise <= 8 utts / speaker are embedded now)."""
    f = os.path.join(C.CACHE, "frozen", f"{model_name}__train.npy")
    n_cls = len(spk2id)
    if os.path.exists(f) and use_frozen:
        idx = [l.split("\t") for l in C.read_lines(os.path.join(C.LISTS, "train_index.tsv"))]
        pos = {r[1]: i for i, r in enumerate(idx)}
        emb = sv.l2n(np.load(f))
        use = [(spk2id[s], pos[r]) for s, r, _ in rows]
        lab = np.array([u[0] for u in use])
        e = emb[[u[1] for u in use]]
    else:
        rng = random.Random(0)
        by = {}
        for s, r, n in rows:
            by.setdefault(s, []).append(r)
        sel = [(s, r) for s in sorted(by) for r in (rng.sample(by[s], 8) if len(by[s]) > 8 else by[s])]
        log(f"head init: embedding {len(sel)} utts with the initial model")
        model.eval()
        e = sv.l2n(sv.extract(model, [os.path.join(C.TRAIN_WAV, r) for _, r in sel], verbose=False))
        lab = np.array([spk2id[s] for s, _ in sel])
    W = np.zeros((n_cls, e.shape[1]), np.float32)
    np.add.at(W, lab, e)
    return torch.from_numpy(sv.l2n(W))


def code_hash():
    import hashlib

    h = hashlib.sha1()
    d = os.path.dirname(os.path.abspath(__file__))
    for f in sorted(os.listdir(d)):
        if f.endswith(".py"):
            h.update(open(os.path.join(d, f), "rb").read())
    return h.hexdigest()[:12]


@torch.no_grad()
def quick_eer(model, log, tag):
    lab, u1, u2, _ = sv.read_trials(os.path.join(C.LISTS, "trials", "devA_random.txt"))
    paths = sorted(set(u1 + u2))
    model.eval()
    emb = sv.extract(model, [C.resolve(p) for p in paths], verbose=False)
    pos = {p: i for i, p in enumerate(paths)}
    s = sv.cosine(emb[[pos[p] for p in u1]], emb[[pos[p] for p in u2]])
    eer = sv.eer_mindcf(s, lab)[0]
    log(f"{tag} devA_random EER(cos) {eer:.3f}")
    return eer


# ----------------------------------------------------------------------------- main
def run(out_dir, cfg):
    cfg = {**DEFAULTS, **cfg}
    os.makedirs(out_dir, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(cfg["mem_frac"])
    C.seed_all(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    log_f = open(os.path.join(out_dir, "train_log.txt"), "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    cfg["code_hash"] = code_hash()
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    log("config " + json.dumps(cfg))

    model = M.load(cfg["model"])
    if cfg["init"]:
        sd = torch.load(C.resolve(cfg["init"]), map_location="cuda", weights_only=False)
        model.load_state_dict(sd, strict=True)
    params = M.count_params(model)
    rows = train_rows(cfg["utt_cap"], cfg["seed"], cfg["all_data"])
    spks = sorted({s for s, _, _ in rows})
    spk2id = {s: i for i, s in enumerate(spks)}
    if cfg["smoke"]:
        rows = rows[:: max(1, len(rows) // 640)]
    codec_set = set()
    cf = os.path.join(C.CACHE, "aug", "codec_train.tsv")
    if os.path.exists(cf):
        codec_set = {l.split("\t")[0] for l in C.read_lines(cf)}
    with torch.no_grad():
        dim = model(torch.zeros(1, C.SR, device="cuda")).shape[-1]
    log(f"model {cfg['model']} params {params / 1e6:.3f}M (<7M: {params < C.PARAM_LIMIT}) emb {dim} | "
        f"train {len(rows)} utts / {len(spks)} spk | codec copies {len(codec_set)}")

    if cfg["head_init"] == "centroid":
        W0 = head_init_centroids(cfg["model"], model, cfg["init"] is None, rows, spk2id, log)
    else:
        W0 = F.normalize(torch.randn(len(spk2id), dim), dim=1)
    if cfg["speed_perturb"]:
        assert not any(k["mode"] == "dkd" for k in cfg["kd"]), "dkd is incompatible with speed_perturb (3x classes)"
        W0 = torch.cat([W0, W0, W0], 0)  # 0.9 / 1.1 copies start at the original centroid
    assert cfg["subcenters"] == 1 or not cfg["speed_perturb"], "subcenters is incompatible with speed_perturb"
    head = AAM(dim, W0.shape[0], cfg["scale"], W0, cfg["subcenters"]).cuda()
    kds = nn.ModuleList([KD(k, rows, dim, spks) for k in cfg["kd"]]).cuda()
    w_init = {n: p.detach().clone() for n, p in model.named_parameters()} if cfg["l2sp"] > 0 else None

    aug = None
    if cfg["aug"]:
        from .augment import Augment3

        aug = Augment3(cfg["aug"])
    ds = CropSet(rows, spk2id, cfg["crop"], codec_set, cfg["speed_perturb"])
    if cfg["multi_crop"]:
        assert not cfg["hard_batch"], "multi_crop is incompatible with hard_batch"
        mc = cfg["multi_crop"]
        sampler = MultiCropBatchSampler(rows, mc["crops"], mc["bs"], cfg["seed"])
        dl = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=cfg["workers"], pin_memory=True,
                                         persistent_workers=True, prefetch_factor=4)
    elif cfg["hard_batch"]:
        hb = cfg["hard_batch"]
        sampler = HardBatchSampler(rows, hb["teacher"], cfg["bs"], hb.get("per_spk", 2), hb.get("k_near", 40),
                                   hb.get("frac", 0.5), cfg["seed"])
        dl = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=cfg["workers"], pin_memory=True,
                                         persistent_workers=True, prefetch_factor=4)
    else:
        dl = torch.utils.data.DataLoader(ds, batch_size=cfg["bs"], shuffle=True, num_workers=cfg["workers"],
                                         drop_last=True, pin_memory=True, persistent_workers=True,
                                         prefetch_factor=4)
    steps_per_epoch = len(dl) // cfg["accum"]
    total = steps_per_epoch * cfg["epochs"]
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": cfg["lr_backbone"],
         "base": cfg["lr_backbone"]},
        {"params": list(head.parameters()) + list(kds.parameters()), "lr": cfg["lr_head"], "base": cfg["lr_head"]},
    ], weight_decay=cfg["wd"])
    ema = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(cfg["ema"]), use_buffers=True)

    if not cfg["smoke"] and not cfg["all_data"]:  # all_data: dev speakers are in training, dev EER is meaningless
        quick_eer(model, log, "epoch 0 (init)")
    step = 0
    t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        acc_l, acc_a, acc_k, nb = 0.0, 0.0, 0.0, 0
        for it, (wav, cw, has, y, idx) in enumerate(dl):
            if it // cfg["accum"] >= steps_per_epoch:
                break
            wav, y, idx = wav.cuda(non_blocking=True), y.cuda(non_blocking=True), idx.cuda(non_blocking=True)
            if aug is not None:
                wav, _ = aug(wav, cw.cuda(non_blocking=True), has)
            frac = step / max(total, 1)
            head.m = cfg["margin"] * min(1.0, step / max(1, cfg["margin_ramp_epochs"] * steps_per_epoch))
            lr_scale = min(1.0, (step + 1) / cfg["warmup"]) * (
                cfg["min_lr_ratio"] + (1 - cfg["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * frac)))
            for g in opt.param_groups:
                g["lr"] = g["base"] * lr_scale
            if step < cfg["freeze_steps"]:
                opt.param_groups[0]["lr"] = 0.0
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["amp"]):
                e = model(wav)
            loss, acc, cos = head(e, y)
            kl = sum(k(e, idx % len(rows), cos, y) for k in kds) if len(kds) else torch.zeros((), device="cuda")
            total_loss = loss + kl
            if w_init is not None:
                total_loss = total_loss + cfg["l2sp"] * sum(
                    (p - w_init[n]).pow(2).sum() for n, p in model.named_parameters())
            (total_loss / cfg["accum"]).backward()
            if (it + 1) % cfg["accum"] == 0:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 5.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                ema.update_parameters(model)
                step += 1
            acc_l += loss.item()
            acc_a += acc.item()
            acc_k += float(kl.detach())
            nb += 1
            if nb % 200 == 0:
                el = time.time() - t0
                log(f"ep {ep} step {step}/{total} aam {acc_l / nb:.3f} acc {acc_a / nb:.3f} kd {acc_k / nb:.3f} "
                    f"lr {opt.param_groups[1]['lr']:.2e} m {head.m:.3f} {el / max(step, 1):.3f}s/step "
                    f"eta {(total - step) * el / max(step, 1) / 60:.0f}min")
            if cfg["smoke"] and nb >= 20:
                break
        log(f"epoch {ep} done: aam {acc_l / max(nb, 1):.3f} acc {acc_a / max(nb, 1):.3f} kd {acc_k / max(nb, 1):.3f}")
        torch.save(model.state_dict(), os.path.join(out_dir, "model_last.pt"))
        if not cfg["smoke"] and not cfg["all_data"]:
            quick_eer(model, log, f"epoch {ep}")
    torch.save(ema.module.state_dict(), os.path.join(out_dir, "model_ema.pt"))
    if cfg["smoke"]:
        log("smoke ok")
        return

    # ------------------------------------------------------------- final evaluation (last + EMA)
    if not cfg["final_eval"]:
        return
    sets = E.DEVA + (E.DEVB if cfg["eval_devB"] else [])
    del opt, dl
    torch.cuda.empty_cache()
    res = {}
    if cfg["all_data"]:  # no held-out dev: the checkpoint variant is fixed in advance, public test = info only
        pick = cfg["final_pick"]
        m = model if pick == "last" else ema.module
        m.eval()
        log(f"all_data run: no dev evaluation; final = {pick} (fixed a priori); public test (info only):")
        res[pick] = E.evaluate(m, os.path.join(out_dir, f"eval_{pick}"), sets=[], pub=True, log=log)
        sets = None
    for tag, m in [("last", model), ("ema", ema.module)] if sets is not None else []:
        m.eval()
        log(f"final eval [{tag}] (dev sets)")
        res[tag] = E.evaluate(m, os.path.join(out_dir, f"eval_{tag}"), sets=sets, pub=False, log=log)
    if sets is not None:
        pick = min(res, key=lambda t: res[t]["devA_random"]["asnorm"]["eer"])
        log(f"selected on devA_random(asnorm): {pick}; public test (info only) for the selected model:")
        res[pick] = E.evaluate(model if pick == "last" else ema.module, os.path.join(out_dir, f"eval_{pick}"),
                               sets=[], pub=True, log=log)
    import shutil

    for f in ["results.json", "results_devB.json"]:
        src = os.path.join(out_dir, f"eval_{pick}", f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, f))
    shutil.copy(os.path.join(out_dir, f"model_{pick}.pt"), os.path.join(out_dir, "model_final.pt"))
    manifest = {"model": cfg["model"], "params": params, "selected": pick, "code_hash": cfg["code_hash"],
                "sha1": {f: C.sha1_file(os.path.join(out_dir, f)) for f in ["model_last.pt", "model_ema.pt",
                                                                            "model_final.pt"]},
                "train_minutes": round((time.time() - t0) / 60, 1)}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return res[pick], params
