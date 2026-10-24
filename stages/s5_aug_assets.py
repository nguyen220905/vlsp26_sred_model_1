"""Stage 5 - augmentation assets built ONLY from the provided training data + code (no MUSAN / external RIRs).

  cache/aug/noise_train.npy, noise_eval.npy   int16 (N, 4 s) clips of non-speech segments mined from
                                              DISJOINT halves of the train-split utterances
  cache/aug/rir_train.npy, rir_eval.npy       float32 (N, 0.8 s) RIRs simulated with pyroomacoustics
                                              (different seeds / room ranges for train and eval)
  cache/codec_train/<rel>                     25% of train-split utterances passed through a TRAINING codec
                                              (mp3, vorbis, A-law/mu-law 8 kHz, G.726, IMA-ADPCM 8 kHz) via ffmpeg
The eval codecs (AMR-NB, Opus, AAC, G.722) are reserved for the simulated-edge dev trials (s6) only.
"""
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import config as C  # noqa: E402
from common import sv  # noqa: E402

AUG = os.path.join(C.CACHE, "aug")
CODEC_DIR = os.path.join(C.CACHE, "codec_train")
CLIP = 4 * C.SR
TRAIN_CODECS = [
    ("mp3_16k", ["-c:a", "libmp3lame", "-b:a", "16k", "-ar", "16000"], "mp3"),
    ("mp3_24k", ["-c:a", "libmp3lame", "-b:a", "24k", "-ar", "16000"], "mp3"),
    ("vorbis_q0", ["-c:a", "libvorbis", "-q:a", "0", "-ar", "16000"], "ogg"),
    ("alaw_8k", ["-c:a", "pcm_alaw", "-ar", "8000"], "wav"),
    ("mulaw_8k", ["-c:a", "pcm_mulaw", "-ar", "8000"], "wav"),
    ("g726_24k", ["-c:a", "g726", "-b:a", "24k", "-ar", "8000"], "wav"),
    ("adpcm_8k", ["-c:a", "adpcm_ima_wav", "-ar", "8000"], "wav"),
]


def mine_noise(rel):
    """Return the concatenated low-energy (non-speech) segments of one utterance (float32)."""
    x = sv.load_wav(os.path.join(C.TRAIN_WAV, rel))
    fl, fs = 400, 160
    if len(x) < 2 * C.SR:
        return np.zeros(0, np.float32)
    n = 1 + (len(x) - fl) // fs
    e = 10 * np.log10((x[np.arange(fl)[None] + fs * np.arange(n)[:, None]] ** 2).mean(1) + 1e-10)
    thr = min(np.percentile(e, 10) + 3, np.percentile(e, 95) - 25)
    quiet = e < thr
    segs, i = [], 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if j - i >= 25:  # >= 0.25 s
                seg = x[i * fs + fl // 2:(j - 1) * fs + fl // 2]
                if np.sqrt((seg ** 2).mean()) > 1e-4:  # not digital silence
                    segs.append(seg)
            i = j
        else:
            i += 1
    return np.concatenate(segs) if segs else np.zeros(0, np.float32)


def noise_bank(rels, n_clips):
    with ProcessPoolExecutor(12) as ex:
        parts = [p for p in ex.map(mine_noise, rels, chunksize=32) if len(p)]
    clips, buf = [], np.zeros(0, np.float32)
    for p in parts:
        buf = np.concatenate([buf, p])
        while len(buf) >= CLIP and len(clips) < n_clips:
            c = buf[:CLIP]
            c = c / (np.abs(c).max() + 1e-6) * 0.5
            clips.append((c * 32767).astype(np.int16))
            buf = buf[CLIP:]
        if len(clips) >= n_clips:
            break
    return np.stack(clips)


def rir_bank(n, seed, rt60_range, dim_range):
    import pyroomacoustics as pra

    rng = np.random.default_rng(seed)
    out = []
    L = int(0.8 * C.SR)
    while len(out) < n:
        dims = rng.uniform(*dim_range, size=3)
        dims[2] = rng.uniform(2.4, 4.0)
        rt60 = rng.uniform(*rt60_range)
        try:
            e_abs, max_order = pra.inverse_sabine(rt60, dims)
        except ValueError:
            continue
        room = pra.ShoeBox(dims, fs=C.SR, materials=pra.Material(e_abs), max_order=min(max_order, 25))
        src = rng.uniform(0.5, dims - 0.5)
        mic = rng.uniform(0.5, dims - 0.5)
        room.add_source(src)
        room.add_microphone(mic)
        room.compute_rir()
        h = np.asarray(room.rir[0][0], np.float32)
        h = h[np.argmax(np.abs(h)) - 40:] if np.argmax(np.abs(h)) > 40 else h
        h = h[:L] if len(h) >= L else np.pad(h, (0, L - len(h)))
        out.append(h / (np.abs(h).max() + 1e-8))
    return np.stack(out)


def encode(args):
    rel, codec = args
    name, opts, ext = next(c for c in TRAIN_CODECS if c[0] == codec)
    src = os.path.join(C.TRAIN_WAV, rel)
    dst = os.path.join(CODEC_DIR, rel)
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst[:-4] + "." + name + "." + ext
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, *opts, tmp], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-ar", "16000", "-ac", "1",
                    "-c:a", "pcm_s16le", dst], check=True)
    os.remove(tmp)


def main():
    os.makedirs(AUG, exist_ok=True)
    st = pd.read_csv(os.path.join(C.LISTS, "utt_stats.tsv"), sep="\t")
    train_spk = set(C.read_lines(os.path.join(C.LISTS, "split_train.txt")))
    rels = st[st.spk.isin(train_spk)].rel.tolist()
    rng = np.random.default_rng(7)
    rng.shuffle(rels)
    half = len(rels) // 2
    for tag, rr, n in [("train", rels[:half], 3000), ("eval", rels[half:], 600)]:
        f = os.path.join(AUG, f"noise_{tag}.npy")
        if not os.path.exists(f):
            nb = noise_bank(rr, n)
            np.save(f, nb)
            print("noise", tag, nb.shape, flush=True)
    for tag, n, seed, rt, dim in [("train", 1000, 1, (0.15, 0.9), (3, 10)), ("eval", 200, 2, (0.2, 1.2), (2.5, 14))]:
        f = os.path.join(AUG, f"rir_{tag}.npy")
        if not os.path.exists(f):
            rb = rir_bank(n, seed, rt, dim)
            np.save(f, rb)
            print("rir", tag, rb.shape, flush=True)
    # offline training codecs on 25% of the train-split utterances
    sel = rels[: len(rels) // 4]
    jobs = [(r, TRAIN_CODECS[i % len(TRAIN_CODECS)][0]) for i, r in enumerate(sel)]
    with ProcessPoolExecutor(12) as ex:
        list(ex.map(encode, jobs, chunksize=16))
    C.write_lines(os.path.join(AUG, "codec_train.tsv"), [f"{r}\t{c}" for r, c in jobs])
    print("codec copies:", len(jobs))


if __name__ == "__main__":
    main()
