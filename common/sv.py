"""Embedding extraction, scoring (cosine / AS-norm) and metrics (EER, minDCF)."""
import os

import numpy as np
import soundfile as sf
import torch

from . import config as C


# ----------------------------------------------------------------------------- audio
def load_wav(path, start=None, stop=None):
    x, sr = sf.read(path, start=start or 0, stop=stop, dtype="float32", always_2d=True)
    assert sr == C.SR, (path, sr)
    return x[:, 0] if x.shape[1] == 1 else x.mean(1)


def num_frames(path):
    return sf.info(path).frames


class _WavSet(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return i, torch.from_numpy(load_wav(self.paths[i]))


def _collate(b):
    return b


def length_batches(lengths, max_utts=64, max_samples=160 * C.SR, max_drop=0.02):
    """Group utterance indices of similar length; each batch is cropped to its shortest member,
    which drops at most `max_drop` of any utterance's samples (long utterances run alone or in small groups)."""
    order = np.argsort(lengths)[::-1]
    batches, cur = [], []
    for i in order:
        if cur:
            longest = lengths[cur[0]]
            ok = (lengths[i] >= (1 - max_drop) * longest and len(cur) < max_utts
                  and (len(cur) + 1) * lengths[i] <= max_samples)
            if not ok:
                batches.append(cur)
                cur = []
        cur.append(int(i))
    if cur:
        batches.append(cur)
    return batches


@torch.no_grad()
def extract(model, paths, lengths=None, num_workers=6, desc="extract", amp=True, min_sec=0.0, verbose=True, max_utts=64):
    """-> (emb (N, D) float32 BEFORE L2-norm, in the order of `paths`).
    Whole-utterance embeddings; utterances shorter than `min_sec` are repeat-padded.
    max_utts=1: every file is embedded alone (no batching with other files, no cropping) -> fully per-file independent."""
    import time

    if lengths is None:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(16) as ex:
            lengths = np.array(list(ex.map(num_frames, paths)))
    lengths = np.asarray(lengths)
    batches = length_batches(lengths, max_utts=max_utts)
    order = [i for b in batches for i in b]
    ds = _WavSet([paths[i] for i in order])
    dl = torch.utils.data.DataLoader(ds, batch_size=None, sampler=None, num_workers=num_workers,
                                     collate_fn=None, prefetch_factor=8 if num_workers else None,
                                     persistent_workers=False)
    it = iter(dl)
    out = [None] * len(paths)
    bench = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # variable shapes: per-shape autotuning would be very slow
    dev = next(model.parameters()).device
    t0 = time.time()
    done = 0
    for bi, b in enumerate(batches):
        wavs = [next(it)[1] for _ in b]
        n = min(len(w) for w in wavs)
        nmin = int(min_sec * C.SR)
        if n < nmin:
            wavs = [w.repeat(nmin // max(len(w), 1) + 1) for w in wavs]
            n = nmin
        x = torch.stack([w[:n] for w in wavs]).to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            e = model(x)
        e = e.float().cpu().numpy()
        for j, i in enumerate(b):
            out[i] = e[j]
        done += len(b)
        if verbose and (bi % 200 == 0 or bi == len(batches) - 1):
            print(f"[{desc}] {done}/{len(paths)} utts, {time.time() - t0:.0f}s", flush=True)
    torch.backends.cudnn.benchmark = bench
    return np.stack(out).astype(np.float32)


class _SegSet(torch.utils.data.Dataset):
    """one item = (file index, (k, crop) tensor of the file's overlapping crops; last crop aligned to the file end)"""

    def __init__(self, paths, crop, hop):
        self.paths, self.crop, self.hop = paths, int(crop * C.SR), int(hop * C.SR)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        w = torch.from_numpy(load_wav(self.paths[i]))
        starts = list(range(0, len(w) - self.crop + 1, self.hop))
        if starts[-1] != len(w) - self.crop:
            starts.append(len(w) - self.crop)
        return i, torch.stack([w[s:s + self.crop] for s in starts])


@torch.no_grad()
def extract_ea(model, paths, crop=3.0, hop=1.5, num_workers=6, desc="ea", amp=True, verbose=True):
    """Embedding averaging -> (N, D) float32: mean of the L2-normalised embeddings of `crop`-s windows (hop `hop`,
    last window aligned to the file end, no padding). Files <= crop s: one whole-file embedding (L2-normalised).
    The mean is NOT re-normalised."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(16) as ex:
        lengths = np.array(list(ex.map(num_frames, paths)))
    out = [None] * len(paths)
    short = [i for i in range(len(paths)) if lengths[i] <= int(crop * C.SR)]
    long_ = [i for i in range(len(paths)) if lengths[i] > int(crop * C.SR)]
    if short:
        e = l2n(extract(model, [paths[i] for i in short], lengths=lengths[short], num_workers=num_workers,
                        desc=desc + "-short", amp=amp, verbose=verbose))
        for j, i in enumerate(short):
            out[i] = e[j]
    dev = next(model.parameters()).device
    t0, done, buf, owner = time.time(), 0, [], []

    def flush():
        x = torch.cat(buf).to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            e = model(x)
        e = l2n(e.float().cpu().numpy())
        o = np.array(owner)
        for i in np.unique(o):
            out[i] = e[o == i].mean(0)
        buf.clear()
        owner.clear()

    if long_:
        dl = torch.utils.data.DataLoader(_SegSet([paths[i] for i in long_], crop, hop), batch_size=None,
                                         num_workers=num_workers, prefetch_factor=8 if num_workers else None)
        bench = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False
        nseg = 0
        for k, (j, segs) in enumerate(dl):
            buf.append(segs)
            owner += [long_[j]] * len(segs)
            nseg += len(segs)
            if len(owner) >= 128:
                flush()
            done += 1
            if verbose and (k % 500 == 0 or k == len(long_) - 1):
                print(f"[{desc}] long {done}/{len(long_)} files, {nseg} crops, {time.time() - t0:.0f}s", flush=True)
        if owner:
            flush()
        torch.backends.cudnn.benchmark = bench
    return np.stack(out).astype(np.float32)


def l2n(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-8)


# ----------------------------------------------------------------------------- metrics
def eer_mindcf(scores, labels, p_target=0.01):
    """labels 1 = target. -> (eer %, minDCF, threshold at EER)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    order = np.argsort(scores)
    s, l = scores[order], labels[order]
    nt, nn_ = l.sum(), (~l).sum()
    # threshold just above s[i]: FRR = targets <= s[i], FAR = nontargets > s[i]
    frr = np.concatenate([[0], np.cumsum(l)]) / max(nt, 1)
    far = 1 - np.concatenate([[0], np.cumsum(~l)]) / max(nn_, 1)
    k = np.argmin(np.abs(frr - far))
    eer = (frr[k] + far[k]) / 2 * 100
    thr = s[max(k - 1, 0)] if k > 0 else s[0] - 1e-6
    dcf = p_target * frr + (1 - p_target) * far
    mindcf = dcf.min() / min(p_target, 1 - p_target)
    return float(eer), float(mindcf), float(thr)


def far_frr_at(scores, labels, thr):
    scores, labels = np.asarray(scores), np.asarray(labels).astype(bool)
    far = float((scores[~labels] > thr).mean() * 100) if (~labels).any() else float("nan")
    frr = float((scores[labels] <= thr).mean() * 100) if labels.any() else float("nan")
    return far, frr


# ----------------------------------------------------------------------------- scoring
def cosine(e1, e2):
    return (l2n(e1) * l2n(e2)).sum(-1)


def asnorm(emb_a, emb_b, cohort, top_n=300, chunk=4096):
    """Adaptive symmetric normalisation. emb_* (N, D) are the trial sides (already L2-normalised or not),
    cohort (K, D). Returns normalised scores (N,)."""
    a, b, c = l2n(emb_a), l2n(emb_b), l2n(cohort)
    raw = (a * b).sum(-1)
    out = np.empty_like(raw)
    ct = torch.from_numpy(c).cuda()
    for s in range(0, len(a), chunk):
        sa = torch.from_numpy(a[s:s + chunk]).cuda() @ ct.T
        sb = torch.from_numpy(b[s:s + chunk]).cuda() @ ct.T
        ta = sa.topk(top_n, dim=1).values
        tb = sb.topk(top_n, dim=1).values
        ma, sda = ta.mean(1), ta.std(1)
        mb, sdb = tb.mean(1), tb.std(1)
        r = torch.from_numpy(raw[s:s + chunk]).cuda()
        out[s:s + chunk] = (0.5 * ((r - ma) / sda + (r - mb) / sdb)).cpu().numpy()
    return out


def snorm(emb_a, emb_b, cohort):
    return asnorm(emb_a, emb_b, cohort, top_n=len(cohort))


# ----------------------------------------------------------------------------- trials
def read_trials(path):
    """lines '<label 0/1> <utt1> <utt2> [group...]' -> labels, u1, u2, groups(list of str)."""
    lab, u1, u2, grp = [], [], [], []
    for l in C.read_lines(path):
        p = l.split()
        lab.append(int(p[0]))
        u1.append(p[1])
        u2.append(p[2])
        grp.append(p[3] if len(p) > 3 else "")
    return np.array(lab), u1, u2, grp
