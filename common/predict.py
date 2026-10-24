"""Score an official trial list -> submission file "<wav1> <wav2> <score>" (same order as the trial list).

usage (from D:\\project_vlsp):
  python -m common.predict --model redimnet_M_multi --ckpt baseline/ft_redimnet_M_multi/model_final.pt \
      --trials public_test/public-test.txt --wav_dir public_test/data --out submissions/pub_ftM.txt [--score asnorm]
  python -m common.predict --check submissions/pub_ftM.txt --trials public_test/public-test.txt

Each trial is scored independently: whole-utterance embeddings of its two files, cosine, optional AS-norm with a
cohort made ONLY of train-split speaker centroids. No statistic of the test audio is used.
--qmf <coef.json>: quality-measure function fitted on dev-A (distill/be_qmf/qmf_eval.py, entry "both"): logistic
score on [AS-norm, cosine, log min duration, log max duration] of the trial; durations from each file's own header.
"""
import argparse
import json
import os

import numpy as np
import torch

from . import config as C
from . import evaluate as E
from . import models as M
from . import sv


def check(sub, trials):
    t = [l.split()[:2] for l in C.read_lines(trials)]
    s = [l.split() for l in C.read_lines(sub)]
    assert len(s) == len(t), f"line count {len(s)} != {len(t)}"
    bad = [i for i, (a, b) in enumerate(zip(s, t)) if len(a) != 3 or a[:2] != b]
    assert not bad, f"{len(bad)} lines do not match the trial list (first: line {bad[0] + 1})"
    v = np.array([float(a[2]) for a in s])
    assert np.isfinite(v).all(), "non-finite scores"
    print(f"OK {sub}: {len(s)} lines, score range [{v.min():.3f}, {v.max():.3f}]")
    lab = [l.split()[2] for l in C.read_lines(trials) if len(l.split()) > 2]
    if len(lab) == len(v):
        eer = sv.eer_mindcf(v, np.array([x == "target" for x in lab]))[0]
        print(f"(labels present in the trial file) EER {eer:.3f}%")


def qmf_scores(a, pairs, e1, e2, cohort):
    """QMF entry "both" of a.qmf: logistic score on [asnorm, cos, log dmin, log dmax] (coefficients frozen)."""
    q = json.load(open(C.resolve(a.qmf)))["both"]
    assert q["features"] == ["asnorm_top300", "cos", "log_dmin", "log_dmax"], q["features"]
    dur = {u: sv.num_frames(os.path.join(a.wav_dir, u)) / C.SR for u in sorted({u for p in pairs for u in p})}
    v = np.array(list(dur.values()))
    print(f"QMF durations: min {v.min():.2f}s max {v.max():.2f}s, outside [1, 30]s: {int(((v < 1) | (v > 30)).sum())} files")
    # clip to the fitted range (dev-A / public / VieSpeaker are all within [1.0, 30.0] s): no extrapolation, no log(0)
    d1 = np.log(np.clip([dur[p[0]] for p in pairs], 1.0, 30.0))
    d2 = np.log(np.clip([dur[p[1]] for p in pairs], 1.0, 30.0))
    X = np.stack([sv.asnorm(e1, e2, cohort, top_n=300), sv.cosine(e1, e2), np.minimum(d1, d2), np.maximum(d1, d2)], 1)
    X = (X - np.array(q["mu"])) / np.array(q["sd"])
    return X @ np.array(q["coef"]) + q["intercept"]


def from_saved(a):
    if a.model:  # the saved embeddings must come from a < 7M system too
        params = M.count_params(M.build(a.model))
        assert params < C.PARAM_LIMIT, f"{params} params >= 7M"
        print(f"{a.model}: {params / 1e6:.3f}M params (< 7M)")
    z = np.load(C.resolve(a.from_emb))
    pos = {p: i for i, p in enumerate(z["paths"].tolist())}
    emb = z["emb"].astype(np.float32)
    rel = C.relpath(a.wav_dir)
    pairs = [l.split()[:2] for l in C.read_lines(a.trials)]
    e1 = emb[[pos[f"{rel}/{p[0]}"] for p in pairs]]
    e2 = emb[[pos[f"{rel}/{p[1]}"] for p in pairs]]
    coh_spk, coh_paths = E.cohort_paths()
    cohort = E.centroids(coh_spk, emb[[pos[p] for p in coh_paths]])
    if a.qmf:
        scores = qmf_scores(a, pairs, e1, e2, cohort)
    elif a.score == "asnorm":
        scores = sv.asnorm(e1, e2, cohort, top_n=a.top_n)
    else:
        scores = sv.cosine(e1, e2)
    C.write_lines(a.out, [f"{p[0]} {p[1]} {s:.6f}" for p, s in zip(pairs, scores)])
    meta = dict(model=a.model, ckpt=a.ckpt, ckpt_sha1=C.sha1_file(C.resolve(a.ckpt)) if a.ckpt else None,
                from_emb=a.from_emb, score="qmf_both" if a.qmf else a.score, top_n=a.top_n, trials=a.trials,
                qmf=a.qmf, qmf_sha1=C.sha1_file(C.resolve(a.qmf)) if a.qmf else None)
    with open(a.out + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    check(a.out, a.trials)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--trials", default=C.PUB_LIST)
    ap.add_argument("--wav_dir", default=C.PUB_WAV)
    ap.add_argument("--out")
    ap.add_argument("--score", default="asnorm", choices=["cos", "asnorm"])
    ap.add_argument("--top_n", type=int, default=300)
    ap.add_argument("--check", default=None)
    ap.add_argument("--qmf", default=None, help="QMF coefficient json (distill/be_qmf/qmf_coef.json); overrides --score")
    ap.add_argument("--from_emb", default=None,
                    help="reuse embeddings saved by evaluate() (eval_emb.npz of the SAME checkpoint) instead of extracting")
    ap.add_argument("--per_file", action="store_true",
                    help="embed every file alone (batch of 1, no length-bucketing with other test files)")
    a = ap.parse_args()
    for k in ("trials", "wav_dir", "out", "check", "qmf"):  # accept old-layout paths like submissions/x.txt
        if getattr(a, k):
            setattr(a, k, C.resolve(getattr(a, k)))
    if a.check:
        return check(a.check, a.trials)
    if a.from_emb:
        return from_saved(a)
    torch.cuda.set_per_process_memory_fraction(0.8)
    model = M.load(a.model)
    if a.ckpt:
        model.load_state_dict(torch.load(C.resolve(a.ckpt), map_location="cuda", weights_only=False),
                              strict=True)
    model.eval()
    params = M.count_params(model)
    assert params < C.PARAM_LIMIT, f"{params} params >= 7M"
    pairs = [l.split()[:2] for l in C.read_lines(a.trials)]
    utts = sorted({u for p in pairs for u in p})
    mu = 1 if a.per_file else 64
    emb = sv.extract(model, [os.path.join(a.wav_dir, u) for u in utts], desc="test", max_utts=mu)
    pos = {u: i for i, u in enumerate(utts)}
    e1, e2 = emb[[pos[p[0]] for p in pairs]], emb[[pos[p[1]] for p in pairs]]
    if a.score == "asnorm" or a.qmf:
        coh_spk, coh_paths = E.cohort_paths()
        coh = sv.extract(model, [C.resolve(p) for p in coh_paths], desc="cohort", max_utts=mu)
        cohort = E.centroids(coh_spk, coh)
    if a.qmf:
        scores = qmf_scores(a, pairs, e1, e2, cohort)
    elif a.score == "asnorm":
        scores = sv.asnorm(e1, e2, cohort, top_n=a.top_n)
    else:
        scores = sv.cosine(e1, e2)
    C.write_lines(a.out, [f"{p[0]} {p[1]} {s:.6f}" for p, s in zip(pairs, scores)])
    meta = dict(model=a.model, ckpt=a.ckpt, ckpt_sha1=C.sha1_file(C.resolve(a.ckpt)) if a.ckpt else None,
                params=params, per_file=a.per_file, score="qmf_both" if a.qmf else a.score, top_n=a.top_n, trials=a.trials,
                qmf=a.qmf, qmf_sha1=C.sha1_file(C.resolve(a.qmf)) if a.qmf else None)
    with open(a.out + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    check(a.out, a.trials)


if __name__ == "__main__":
    main()
