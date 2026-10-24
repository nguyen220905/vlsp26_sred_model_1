"""Add devA_pooled (and devB_pooled) to results.json of runs evaluated before pooled EER existed,
recomputed from their saved embeddings (same scoring as common/evaluate.py).  usage: python -m common.backfill_pooled <run_dir>..."""
import json
import os
import sys

import numpy as np

from . import backend
from . import config as C
from . import evaluate as E
from . import sv

if __name__ == "__main__":
    for run in sys.argv[1:]:
        idx, emb, _ = backend.load_emb(run)
        coh_spk, coh_paths = E.cohort_paths()
        cohort = E.centroids(coh_spk, emb[[idx[p] for p in coh_paths]])
        f = os.path.join(run, "results.json")
        res = json.load(open(f))
        pr = {}
        for k in ["cos", "asnorm"]:
            vs, ls = [], []
            for s in E.DEVA:
                lab, u1, u2, _ = sv.read_trials(os.path.join(C.LISTS, "trials", s + ".txt"))
                e1, e2 = emb[[idx[p] for p in u1]], emb[[idx[p] for p in u2]]
                vs.append(sv.cosine(e1, e2) if k == "cos" else sv.asnorm(e1, e2, cohort, top_n=300))
                ls.append(lab)
            eer, dcf, _ = sv.eer_mindcf(np.concatenate(vs), np.concatenate(ls))
            pr[k] = {"eer": round(eer, 3), "mindcf": round(dcf, 4), "n": int(sum(len(x) for x in ls))}
        res["devA_pooled"] = pr
        json.dump(res, open(f, "w"), indent=1)
        print(run, pr["cos"]["eer"], pr["asnorm"]["eer"])
