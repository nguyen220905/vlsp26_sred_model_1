"""Paths and global settings shared by every stage / experiment."""
import os
import random

import numpy as np

ROOT = os.environ.get("VLSP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # package root
DATA_DIR = "data[ban_to_chuc]"                        # organiser data + everything derived from it
BASELINE_DIR = "baseline[du_lieu_ban_to_chuc]"        # zero-shot / fine-tune baselines
DISTILL_DIR = "distill[12,7m_du_lieu_ban_to_chuc]"    # KD from the B6 teacher, final run, submissions
DATA = os.path.join(ROOT, DATA_DIR)
TRAIN_LIST = os.path.join(DATA, "vlsp", "train.txt")      # "<spk>\t<spk>/<utt>.wav"
TRAIN_WAV = os.path.join(DATA, "vlsp", "data")
PUB_LIST = os.path.join(DATA, "public_test", "public-test.txt")  # "<wav1> <wav2> target|nontarget"
PUB_WAV = os.path.join(DATA, "public_test", "data")
LISTS = os.path.join(DATA, "lists")       # generated splits / trial lists (stage outputs)
CACHE = os.path.join(DATA, "cache")       # embeddings, augmentation assets, teacher targets
TP = os.path.join(ROOT, "third_party")    # upstream repos (redimnet, redimnet2, wespeaker)

# Paths stored in lists / caches / configs use the old flat layout ("vlsp/data/..", "cache/..", "final/..");
# resolve() maps their first component to the grouped folders, anything else stays relative to ROOT.
_OLD_TOP = {**{d: f"{DATA_DIR}/{d}" for d in ("vlsp", "public_test", "lists", "cache", "downloads")},
            "baseline": BASELINE_DIR, "distill": DISTILL_DIR,
            "final": f"{DISTILL_DIR}/final", "submissions": f"{DISTILL_DIR}/submissions"}


def resolve(rel):
    """Path relative to ROOT (old or new layout) -> absolute path."""
    if os.path.isabs(rel):
        return rel
    head, _, tail = rel.replace("\\", "/").partition("/")
    if head in _OLD_TOP:
        return os.path.join(ROOT, _OLD_TOP[head], tail)
    return os.path.join(ROOT, rel)


def relpath(path):
    """Absolute path -> old-layout path relative to ROOT (the form stored in lists / embedding files)."""
    rel = os.path.relpath(path, ROOT).replace("\\", "/")
    for old, new in sorted(_OLD_TOP.items(), key=lambda kv: -len(kv[1])):
        if rel == new or rel.startswith(new + "/"):
            return old + rel[len(new):]
    return rel
SR = 16000
PARAM_LIMIT = 7_000_000

os.makedirs(LISTS, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)


def seed_all(seed=0):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_train_list():
    """-> list of (spk, relpath)."""
    out = []
    with open(TRAIN_LIST, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                out.append((p[0], p[1]))
    return out


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for l in lines:
            f.write(l + "\n")


def sha1_file(path, block=1 << 20):
    import hashlib

    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
