"""Back-end experiments on saved embeddings (<run>/eval_*/eval_emb.npz or <run>/eval_emb.npz).

- AS-norm sweep: top-N in {100, 200, 300, 400} + full S-norm; cohort = train-speaker centroids.
- QMF: logistic regression on [score, log min/max duration, min/max embedding norm (before L2)], symmetric in the
  two sides; fit on dev-A with speaker-grouped 5-fold CV (CV EER reported), then refit on all dev-A trials.
Everything is selected on dev-A; public-test numbers are information only.
"""
import json
import os
import re

import numpy as np

from . import config as C
from . import evaluate as E
from . import sv


def load_emb(run_dir):
    sel = []
    man = os.path.join(run_dir, "manifest.json")
    if os.path.exists(man):  # the checkpoint selected on dev-A (its folder also holds the public-test embeddings)
        sel = [os.path.join(run_dir, f"eval_{json.load(open(man))['selected']}", "eval_emb.npz")]
    for f in sel + [os.path.join(run_dir, "eval_emb.npz"), os.path.join(run_dir, "eval_ema", "eval_emb.npz"),
                    os.path.join(run_dir, "eval_last", "eval_emb.npz")]:
        if os.path.exists(f):
            z = np.load(f)
            return {p: i for i, p in enumerate(z["paths"].tolist())}, z["emb"].astype(np.float32), f
    raise FileNotFoundError(run_dir)


_DUR = {}


def duration(p):
    if p not in _DUR:
        m = re.search(r"_(\d+)_(\d+)\.wav$", p) if "devcrops" in p else None
        _DUR[p] = int(m.group(2)) / 1000 if m else sv.num_frames(C.resolve(p)) / C.SR
    return _DUR[p]


def qmf_features(s, p1, p2, n1, n2):
    d1 = np.log(np.array([duration(p) for p in p1]))
    d2 = np.log(np.array([duration(p) for p in p2]))
    return np.stack([s, np.minimum(d1, d2), np.maximum(d1, d2), np.minimum(n1, n2), np.maximum(n1, n2)], 1)


def speaker_of(p):
    # dev paths: vlsp/data/<spk>/x.wav, cache/devcrops/<split>/<spk>_x_st_dur.wav, cache/simedge/<split>/k_<spk>_x.wav
    parts = p.split("/")
    if parts[0] == "vlsp":
        return parts[2]
    m = re.search(r"(id\d{5})", parts[-1])
    return m.group(1) if m else parts[-1]


def run(run_dir, out_name="backend.json", sets=("devA_random", "devA_hard", "devA_shortlong", "devA_simedge")):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    idx, emb, src = load_emb(run_dir)
    norms = np.linalg.norm(emb, axis=1)
    coh_spk, coh_paths = E.cohort_paths()
    cohort = E.centroids(coh_spk, emb[[idx[p] for p in coh_paths]])
    trials = {s: sv.read_trials(os.path.join(C.LISTS, "trials", s + ".txt")) for s in sets}
    trials["pub"] = E.pub_trials()
    out = {"source": src, "asnorm": {}, "qmf": {}}
    scores = {}
    for s, (lab, u1, u2, _) in trials.items():
        e1, e2 = emb[[idx[p] for p in u1]], emb[[idx[p] for p in u2]]
        r = {"cos": round(sv.eer_mindcf(sv.cosine(e1, e2), lab)[0], 3)}
        for n in [100, 200, 300, 400]:
            sc = sv.asnorm(e1, e2, cohort, top_n=n)
            r[f"top{n}"] = round(sv.eer_mindcf(sc, lab)[0], 3)
            if n == 300:
                scores[s] = sc
        r["snorm"] = round(sv.eer_mindcf(sv.snorm(e1, e2, cohort), lab)[0], 3)
        out["asnorm"][s] = r
    # QMF on dev-A (pooled over the dev-A sets), speaker-grouped CV
    X, Y, G, S = [], [], [], []
    for s in sets:
        lab, u1, u2, _ = trials[s]
        n1, n2 = norms[[idx[p] for p in u1]], norms[[idx[p] for p in u2]]
        X.append(qmf_features(scores[s], u1, u2, n1, n2))
        Y.append(lab)
        G += [speaker_of(a) for a in u1]
        S += [s] * len(lab)
    X, Y, G, S = np.concatenate(X), np.concatenate(Y), np.array(G), np.array(S)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    cv = np.zeros(len(Y))
    for tr, te in GroupKFold(5).split(Xn, Y, G):
        cv[te] = LogisticRegression(C=1.0, max_iter=1000).fit(Xn[tr], Y[tr]).decision_function(Xn[te])
    for s in sets:
        m = S == s
        out["qmf"][s] = {"asnorm300": out["asnorm"][s]["top300"], "qmf_cv": round(sv.eer_mindcf(cv[m], Y[m])[0], 3)}
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(Xn, Y)
    out["qmf"]["coef"] = dict(zip(["score", "dmin", "dmax", "nmin", "nmax"], np.round(clf.coef_[0], 3).tolist()))
    lab, u1, u2, _ = trials["pub"]
    n1, n2 = norms[[idx[p] for p in u1]], norms[[idx[p] for p in u2]]
    Xp = (qmf_features(scores["pub"], u1, u2, n1, n2) - mu) / sd
    out["qmf"]["pub"] = {"asnorm300": out["asnorm"]["pub"]["top300"],
                         "qmf_fit_devA": round(sv.eer_mindcf(clf.decision_function(Xp), lab)[0], 3)}
    with open(os.path.join(run_dir, out_name), "w") as f:
        json.dump(out, f, indent=1)
    return out
