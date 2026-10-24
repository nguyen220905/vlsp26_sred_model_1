"""Compare a reproduced score file with the submitted one, for the case where the sha1 differs (e.g. another GPU,
driver or CUDA/cuDNN version changes the last digits of the floating-point scores).
Checks: same trials in the same order, max / mean |score difference|, Spearman rank correlation, and, if the trial
list has labels (public test), the EER of both files.
usage:  python tools/compare_scores.py outputs/private/results.txt submitted/private_results.zip
        python tools/compare_scores.py outputs/public/results.txt submitted/public_results.zip \
            "data[ban_to_chuc]/public_test/public-test.txt"
"""
import sys
import zipfile

import numpy as np
from scipy.stats import spearmanr


def read(path):
    if path.endswith(".zip"):
        lines = zipfile.ZipFile(path).read("results.txt").decode("utf-8").splitlines()
    else:
        lines = open(path, encoding="utf-8").read().splitlines()
    rows = [l.split() for l in lines if l.strip()]
    return [r[:2] for r in rows], np.array([float(r[2]) for r in rows])


def eer(scores, labels):
    o = np.argsort(scores)
    l = labels[o]
    frr = np.concatenate([[0], np.cumsum(l)]) / l.sum()
    far = 1 - np.concatenate([[0], np.cumsum(~l)]) / (~l).sum()
    k = np.argmin(np.abs(frr - far))
    return (frr[k] + far[k]) / 2 * 100


if __name__ == "__main__":
    pa, sa = read(sys.argv[1])
    pb, sb = read(sys.argv[2])
    assert pa == pb, "the two files do not list the same trials in the same order"
    d = np.abs(sa - sb)
    print(f"trials {len(sa)}  max|diff| {d.max():.2e}  mean|diff| {d.mean():.2e}  "
          f"Spearman {spearmanr(sa, sb)[0]:.6f}  identical {bool((d == 0).all())}")
    if len(sys.argv) > 3:
        lab = np.array([l.split()[2] == "target" for l in open(sys.argv[3], encoding="utf-8") if l.strip()])
        print(f"EER reproduced {eer(sa, lab):.3f}%  submitted {eer(sb, lab):.3f}%")
