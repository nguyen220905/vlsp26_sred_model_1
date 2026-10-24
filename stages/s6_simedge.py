"""Stage 6 - simulated-edge dev trials: <split>_simedge.txt.

3k target + 3k non-target pairs taken from <split>_random.txt; the SECOND side is degraded with assets that
training never sees: eval RIR bank (p=0.5), eval noise bank at 0-15 dB SNR (p=0.7), and one EVAL codec
(AMR-NB 4.75-12.2 kbps @8 kHz, Opus 6-12 kbps, AAC 16-24 kbps, G.722) applied to every degraded file.
Tags: <original tags>,<codec>[,rev][,noise]
"""
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import config as C  # noqa: E402
from common import sv  # noqa: E402

AUG = os.path.join(C.CACHE, "aug")
EVAL_CODECS = [
    ("amrnb", lambda r: ["-c:a", "libopencore_amrnb", "-ar", "8000", "-b:a", str(r.choice([4750, 7400, 12200]))], "amr"),
    ("opus", lambda r: ["-c:a", "libopus", "-b:a", f"{r.choice([6, 8, 12])}k", "-ar", "16000"], "ogg"),
    ("aac", lambda r: ["-c:a", "aac", "-b:a", f"{r.choice([16, 24])}k", "-ar", "16000"], "m4a"),
    ("g722", lambda r: ["-c:a", "g722", "-ar", "16000"], "wav"),
]


def degrade(job):
    src, dst, seed = job
    if os.path.exists(dst):
        return dst, None
    r = np.random.default_rng(seed)
    x = sv.load_wav(C.resolve(src))
    tags = []
    if r.random() < 0.5:
        rirs = np.load(os.path.join(AUG, "rir_eval.npy"), mmap_mode="r")
        h = np.asarray(rirs[r.integers(len(rirs))])
        y = fftconvolve(x, h)[: len(x)]
        x = y / (np.abs(y).max() + 1e-8) * np.abs(x).max()
        tags.append("rev")
    if r.random() < 0.7:
        nb = np.load(os.path.join(AUG, "noise_eval.npy"), mmap_mode="r")
        n = np.asarray(nb[r.integers(len(nb))], np.float32) / 32767
        n = np.tile(n, len(x) // len(n) + 1)[: len(x)]
        snr = r.uniform(0, 15)
        px, pn = (x ** 2).mean() + 1e-10, (n ** 2).mean() + 1e-10
        x = x + n * np.sqrt(px / (pn * 10 ** (snr / 10)))
        tags.append("noise")
    x = x / max(1.0, np.abs(x).max())
    name, opts, ext = EVAL_CODECS[r.integers(len(EVAL_CODECS))]
    tags.insert(0, name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        a, b = os.path.join(td, "in.wav"), os.path.join(td, "enc." + ext)
        sf.write(a, x, C.SR, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a, *opts(r), b], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", b, "-ar", "16000", "-ac", "1",
                        "-c:a", "pcm_s16le", dst], check=True)
    return dst, ",".join(tags)


def main():
    for si, split in enumerate(["devA", "devB"]):
        rng = np.random.default_rng(300 + si)
        lines = C.read_lines(os.path.join(C.LISTS, "trials", f"{split}_random.txt"))
        tg = [l for l in lines if l[0] == "1"]
        nt = [l for l in lines if l[0] == "0"]
        pick = [tg[i] for i in rng.choice(len(tg), 3000, replace=False)] + \
               [nt[i] for i in rng.choice(len(nt), 3000, replace=False)]
        jobs, meta = [], []
        for k, l in enumerate(pick):
            lab, a, b, t = l.split()
            dst = os.path.join(C.CACHE, "simedge", split, f"{k:05d}_" + b.split("/")[-2] + "_" + b.split("/")[-1])
            jobs.append((b, dst, 1000 * si + k))
            meta.append((lab, a, t))
        with ProcessPoolExecutor(12) as ex:
            res = list(ex.map(degrade, jobs, chunksize=16))
        tagf = os.path.join(C.CACHE, "simedge", f"{split}_tags.tsv")
        old = dict(l.split("\t") for l in C.read_lines(tagf)) if os.path.exists(tagf) else {}
        out, tags_rows = [], []
        for (lab, a, t), (dst, tg_) in zip(meta, res):
            tg_ = tg_ or old.get(dst, "")
            tags_rows.append(f"{dst}\t{tg_}")
            rel = C.relpath(dst)
            out.append(f"{lab} {a} {rel} {t},{tg_}")
        C.write_lines(tagf, tags_rows)
        C.write_lines(os.path.join(C.LISTS, "trials", f"{split}_simedge.txt"), out)
        print(split, len(out))


if __name__ == "__main__":
    main()
