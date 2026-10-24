"""Pre-compute teacher targets ONCE: embeddings of every train.txt utterance, clean and full length.

usage: python make_targets.py <target_name> <model_name> [checkpoint relative to ROOT]
  e.g. python make_targets.py b6_zs redimnet2_b6_multi
       python make_targets.py b6_ft redimnet2_b6_multi distill/teachers/ft_redimnet2_b6_multi/model_final.pt
       python make_targets.py self_M redimnet_M_multi            (the student's own pretrained checkpoint)
-> cache/teachers/<target_name>.npz  (rels, emb before L2-norm, spk)
Frozen zero-shot embeddings already computed by stages/s2_embed_frozen.py are reused when available.
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common import config as C  # noqa: E402
from common import models as M  # noqa: E402
from common import sv  # noqa: E402

if __name__ == "__main__":
    name, model_name = sys.argv[1], sys.argv[2]
    ckpt = sys.argv[3] if len(sys.argv) > 3 else None
    out = os.path.join(C.CACHE, "teachers", f"{name}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = [l.split("\t") for l in C.read_lines(os.path.join(C.LISTS, "train_index.tsv"))]
    frozen = os.path.join(C.CACHE, "frozen", f"{model_name}__train.npy")
    if ckpt is None and os.path.exists(frozen):
        emb = np.load(frozen)
    else:
        torch.cuda.set_per_process_memory_fraction(0.8)
        model = M.load(model_name)
        if ckpt:
            model.load_state_dict(torch.load(C.resolve(ckpt), map_location="cuda", weights_only=False),
                                  strict=True)
        model.eval()
        t0 = time.time()
        emb = sv.extract(model, [os.path.join(C.TRAIN_WAV, r[1]) for r in rows], np.array([int(r[2]) for r in rows]),
                         desc=name)
        print(f"{name}: {emb.shape} in {time.time() - t0:.0f}s")
    np.savez(out, rels=np.array([r[1] for r in rows]), spk=np.array([r[0] for r in rows]), emb=emb.astype(np.float32),
             model=model_name, ckpt=str(ckpt), ckpt_sha1=C.sha1_file(C.resolve(ckpt)) if ckpt else "")
    print("saved", out)
