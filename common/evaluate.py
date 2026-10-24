"""Evaluate one embedding model on the fixed trial lists.

dev-A sets (every decision): devA_random, devA_hard, devA_shortlong, devA_simedge
dev-B sets (final check only, written to results_devB.json and never shown in tables): devB_*
public test (labelled by the organisers; reported, NOT used for selection): pub

Scores: cos (raw cosine) and asnorm (adaptive s-norm, top-300, cohort = centroids of the train-split speakers
computed with the same model from <= 5 utts each). Per-tag EER and FAR/FRR at the pooled-EER threshold.
"""
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from . import config as C
from . import sv

DEVA = ["devA_random", "devA_hard", "devA_shortlong", "devA_simedge"]
DEVB = ["devB_random", "devB_hard", "devB_shortlong", "devB_simedge"]


def cohort_paths(per_spk=5, seed=0):
    p = os.path.join(C.LISTS, "cohort_utts.txt")
    if not os.path.exists(p):
        rng = np.random.default_rng(seed)
        train = set(C.read_lines(os.path.join(C.LISTS, "split_train.txt")))
        by = defaultdict(list)
        for s, r in C.read_train_list():
            if s in train:
                by[s].append(r)
        rows = []
        for s in sorted(by):
            u = sorted(by[s])
            for r in (rng.choice(u, per_spk, replace=False) if len(u) > per_spk else u):
                rows.append(f"{s}\tvlsp/data/{r}")
        C.write_lines(p, rows)
    rows = [l.split("\t") for l in C.read_lines(p)]
    return [r[0] for r in rows], [r[1] for r in rows]


def pub_trials():
    lab, u1, u2 = [], [], []
    for l in C.read_lines(C.PUB_LIST):
        a, b, t = l.split()
        lab.append(int(t == "target"))
        u1.append("public_test/data/" + a)
        u2.append("public_test/data/" + b)
    return np.array(lab), u1, u2, [""] * len(lab)


def centroids(spk, emb):
    spk = np.asarray(spk)
    out = []
    e = sv.l2n(emb)
    for s in sorted(set(spk)):
        out.append(e[spk == s].mean(0))
    return sv.l2n(np.stack(out))


def set_metrics(scores, lab, grp):
    eer, dcf, thr = sv.eer_mindcf(scores, lab)
    res = {"eer": round(eer, 3), "mindcf": round(dcf, 4), "n": int(len(lab))}
    tags = defaultdict(list)
    for i, g in enumerate(grp):
        for t in g.split(","):
            if t:
                tags[t].append(i)
    per = {}
    for t, idx in sorted(tags.items()):
        idx = np.array(idx)
        far, frr = sv.far_frr_at(scores[idx], lab[idx], thr)
        d = {"n": len(idx), "far": round(far, 2), "frr": round(frr, 2)}
        if lab[idx].min() == 0 and lab[idx].max() == 1 and len(idx) >= 200:
            d["eer"] = round(sv.eer_mindcf(scores[idx], lab[idx])[0], 3)
        per[t] = d
    res["groups"] = per
    return res


@torch.no_grad()
def evaluate(model, out_dir, sets=DEVA, pub=True, save_emb=True, top_n=300, log=print):
    t0 = time.time()
    trials = {}
    for s in sets:
        f = os.path.join(C.LISTS, "trials", s + ".txt")
        if os.path.exists(f):
            trials[s] = sv.read_trials(f)
    if pub:
        trials["pub"] = pub_trials()
    coh_spk, coh_paths = cohort_paths()
    paths = sorted({p for t in trials.values() for p in t[1] + t[2]} | set(coh_paths))
    emb = sv.extract(model, [C.resolve(p) for p in paths], desc="eval", verbose=False)
    idx = {p: i for i, p in enumerate(paths)}
    cohort = centroids(coh_spk, emb[[idx[p] for p in coh_paths]])
    results, results_b = {}, {}
    pooled = {"devA": defaultdict(list), "devB": defaultdict(list)}
    for s, (lab, u1, u2, grp) in trials.items():
        e1 = emb[[idx[p] for p in u1]]
        e2 = emb[[idx[p] for p in u2]]
        sc = {"cos": sv.cosine(e1, e2), "asnorm": sv.asnorm(e1, e2, cohort, top_n=top_n)}
        r = {k: set_metrics(v, lab, grp) for k, v in sc.items()}
        for pre in pooled:
            if s.startswith(pre + "_"):
                for k, v in sc.items():
                    pooled[pre][k].append((v, lab))
        (results_b if s.startswith("devB") else results)[s] = r
        if not s.startswith("devB"):
            log(f"  {s:16s} cos {r['cos']['eer']:6.3f}  asnorm {r['asnorm']['eer']:6.3f}  "
                f"minDCF {r['asnorm']['mindcf']:.4f}")
    for pre, d in pooled.items():  # pooled EER over all sets of the split ("EER gop")
        if d and len(d["cos"]) > 1:
            pr = {}
            for k, lst in d.items():
                v = np.concatenate([a for a, _ in lst])
                l = np.concatenate([b for _, b in lst])
                eer, dcf, _ = sv.eer_mindcf(v, l)
                pr[k] = {"eer": round(eer, 3), "mindcf": round(dcf, 4), "n": int(len(l))}
            (results_b if pre == "devB" else results)[f"{pre}_pooled"] = pr
            if pre == "devA":
                log(f"  {pre}_pooled     cos {pr['cos']['eer']:6.3f}  asnorm {pr['asnorm']['eer']:6.3f}")
    os.makedirs(out_dir, exist_ok=True)
    # merge with results of an earlier call on the same folder (e.g. dev-A first, public test later)
    for fname, new_res in [("results.json", results), ("results_devB.json", results_b)]:
        f = os.path.join(out_dir, fname)
        if not new_res:
            continue
        old = json.load(open(f)) if os.path.exists(f) else {}
        old.update(new_res)
        new_res.clear()
        new_res.update(old)
        with open(f, "w") as fh:
            json.dump(new_res, fh, indent=1)
    if save_emb:
        f = os.path.join(out_dir, "eval_emb.npz")
        if os.path.exists(f):  # keep embeddings of earlier calls too
            z = np.load(f)
            keep = [i for i, p in enumerate(z["paths"].tolist()) if p not in idx]
            paths = list(paths) + [z["paths"][i] for i in keep]
            emb = np.concatenate([emb, z["emb"][keep].astype(np.float32)]) if keep else emb
        np.savez_compressed(f, paths=np.array(paths), emb=emb.astype(np.float16))
    log(f"  eval done in {time.time() - t0:.0f}s")
    return results


def append_results_csv(csv_path, name, kind, params, results, desc):
    import csv

    new = not os.path.exists(csv_path)
    g = lambda s, k="asnorm": results.get(s, {}).get(k, {}).get("eer", "")  # noqa: E731
    row = dict(name=name, kind=kind, params_M=round(params / 1e6, 3), under_7M=params < C.PARAM_LIMIT,
               devA_rand_cos=g("devA_random", "cos"), devA_rand=g("devA_random"), devA_hard=g("devA_hard"),
               devA_shortlong=g("devA_shortlong"), devA_simedge=g("devA_simedge"),
               devA_rand_mindcf=results.get("devA_random", {}).get("asnorm", {}).get("mindcf", ""),
               pub_cos=g("pub", "cos"), pub=g("pub"), desc=desc, time=time.strftime("%Y-%m-%d %H:%M"))
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
