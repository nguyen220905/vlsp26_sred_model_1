"""Stage 2 - embed every train utterance (and the public-test utterances) with FROZEN zero-shot models.

Used for: identity check across speaker IDs (s3), hard-trial mining (s4, 3 model families),
zero-shot baselines (baseline/zs_*) and later the cohort / hard-batch sampler.
Outputs cache/frozen/<model>__train.npy (raw, before L2) and <model>__pub.npy, plus the matching .txt lists.

usage: python s2_embed_frozen.py [model ...] [--pub]   (default: M multi, B4 vox2, R34 vox2; train only)
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import config as C  # noqa: E402
from common import models as M  # noqa: E402
from common import sv  # noqa: E402

OUT = os.path.join(C.CACHE, "frozen")


def train_index():
    """rels + lengths (samples) of train.txt, cached."""
    p = os.path.join(C.LISTS, "train_index.tsv")
    if not os.path.exists(p):
        st = os.path.join(C.LISTS, "utt_stats.tsv")
        if os.path.exists(st):
            import pandas as pd

            df = pd.read_csv(st, sep="\t")
            rows = [f"{r.spk}\t{r.rel}\t{int(round(r.dur * C.SR))}" for r in df.itertuples()]
        else:
            from concurrent.futures import ThreadPoolExecutor

            utts = C.read_train_list()
            with ThreadPoolExecutor(16) as ex:
                lens = list(ex.map(lambda r: sv.num_frames(os.path.join(C.TRAIN_WAV, r[1])), utts))
            rows = [f"{s}\t{r}\t{n}" for (s, r), n in zip(utts, lens)]
        C.write_lines(p, rows)
    rows = [l.split("\t") for l in C.read_lines(p)]
    return [r[0] for r in rows], [r[1] for r in rows], np.array([int(r[2]) for r in rows])


def pub_index():
    p = os.path.join(C.LISTS, "pub_index.tsv")
    if not os.path.exists(p):
        from concurrent.futures import ThreadPoolExecutor

        utts = sorted({u for l in C.read_lines(C.PUB_LIST) for u in l.split()[:2]})
        with ThreadPoolExecutor(16) as ex:
            lens = list(ex.map(lambda r: sv.num_frames(os.path.join(C.PUB_WAV, r)), utts))
        C.write_lines(p, [f"{u}\t{n}" for u, n in zip(utts, lens)])
    rows = [l.split("\t") for l in C.read_lines(p)]
    return [r[0] for r in rows], np.array([int(r[1]) for r in rows])


def main(names, with_pub=False):
    os.makedirs(OUT, exist_ok=True)
    spk, rels, lens = train_index()
    sets = [("train", C.TRAIN_WAV, rels, lens)]
    if with_pub:
        prel, plens = pub_index()
        sets.append(("pub", C.PUB_WAV, prel, plens))
    for name in names:
        model = M.load(name)
        for tag, root, rr, ll in sets:
            f = os.path.join(OUT, f"{name}__{tag}.npy")
            if os.path.exists(f):
                continue
            t0 = time.time()
            e = sv.extract(model, [os.path.join(root, r) for r in rr], ll, desc=f"{name}/{tag}")
            np.save(f, e)
            print(f"{name} {tag}: {e.shape} in {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.7)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    # default: the 3 model families used for the identity check and hard-trial mining
    main(args or ["redimnet_M_multi", "redimnet2_b4_vox2", "wespeaker_r34_vox2"], with_pub="--pub" in sys.argv)
