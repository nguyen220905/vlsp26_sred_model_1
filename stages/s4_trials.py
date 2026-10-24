"""Stage 4 - fixed dev trial lists (dev-A for every decision, dev-B for the final check only).

Per split:
  <split>_random.txt     10k target + 10k non-target, speakers sampled uniformly (not by #utts)
  <split>_hard.txt       ~3k + 3k mined with the AVERAGE score of 3 frozen models of different families
                         (ReDimNet-M multi, ReDimNet2-B4 vox2, WeSpeaker R34): hardest-but-not-extreme band
                         (targets: 2-25th pct of target scores; non-targets: top 0.1-3% of 1M random pairs)
  <split>_shortlong.txt  3k + 3k: a 1.5-2.5 s crop of one utterance vs. a different full utterance >= 6 s
Line format: "<1|0> <path1> <path2> <tags>", paths relative to ROOT; tags = duration class (ss/sl/ll),
quality class (hq/xq/lq: low quality = SNR < 15 dB or 99%-energy bandwidth < 1.6 kHz).
Dev utterances flagged in label_fixes.tsv (exclude_from_dev_trials) are not used; at most 40 utts / speaker.
Also writes lists/trials/hard_pairs_to_listen.tsv (random sample of 40 hard pairs to check by ear).
"""
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import config as C  # noqa: E402
from common import sv  # noqa: E402

FROZEN = os.path.join(C.CACHE, "frozen")
MODELS = ["redimnet_M_multi", "redimnet2_b4_vox2", "wespeaker_r34_vox2"]
OUT = os.path.join(C.LISTS, "trials")
CROPS = os.path.join(C.CACHE, "devcrops")


def main():
    os.makedirs(OUT, exist_ok=True)
    st = pd.read_csv(os.path.join(C.LISTS, "utt_stats.tsv"), sep="\t")
    rows = [l.split("\t") for l in C.read_lines(os.path.join(C.LISTS, "train_index.tsv"))]
    rel2i = {r[1]: i for i, r in enumerate(rows)}
    embs = [sv.l2n(np.load(os.path.join(FROZEN, f"{m}__train.npy"))) for m in MODELS]
    fixes = pd.read_csv(os.path.join(C.LISTS, "label_fixes.tsv"), sep="\t")
    excl = set(fixes[fixes.kind == "exclude_from_dev_trials"].item)
    st["lowq"] = (st.snr < 15) | (st.bw < 1600)
    info = st.set_index("rel")
    listen = []

    def score(a, b):
        ia = np.array([rel2i[x] for x in a])
        ib = np.array([rel2i[x] for x in b])
        return np.mean([(e[ia] * e[ib]).sum(1) for e in embs], 0)

    def tag(a, b, da=None, db=None):
        da = info.dur[a] if da is None else da
        db = info.dur[b] if db is None else db
        short = int(da < 4) + int(db < 4)
        d = ["ll", "sl", "ss"][short]
        q = int(info.lowq[a]) + int(info.lowq[b])
        return f"{d},{['hq', 'xq', 'lq'][q]}"

    for si, split in enumerate(["devA", "devB"]):
        rng = np.random.default_rng(100 + si)
        spks = C.read_lines(os.path.join(C.LISTS, f"split_{split}.txt"))
        utt = {}
        for s in spks:
            u = [r for r in st[st.spk == s].rel.tolist() if r not in excl]
            if len(u) > 40:
                u = list(rng.choice(u, 40, replace=False))
            if len(u) >= 2:
                utt[s] = u
        spks = sorted(utt)
        P = lambda r: "vlsp/data/" + r  # noqa: E731

        # ---- random
        tg, nt = set(), set()
        while len(tg) < 10000:
            s = spks[rng.integers(len(spks))]
            a, b = rng.choice(len(utt[s]), 2, replace=False)
            tg.add(tuple(sorted((utt[s][a], utt[s][b]))))
            if len(tg) >= sum(len(u) * (len(u) - 1) // 2 for u in utt.values()):
                break
        while len(nt) < 10000:
            s, t = rng.choice(len(spks), 2, replace=False)
            a = utt[spks[s]][rng.integers(len(utt[spks[s]]))]
            b = utt[spks[t]][rng.integers(len(utt[spks[t]]))]
            nt.add(tuple(sorted((a, b))))
        lines = [f"1 {P(a)} {P(b)} {tag(a, b)}" for a, b in sorted(tg)] + \
                [f"0 {P(a)} {P(b)} {tag(a, b)}" for a, b in sorted(nt)]
        C.write_lines(os.path.join(OUT, f"{split}_random.txt"), lines)

        # ---- hard (3-model average score)
        allt = []
        for s in spks:
            u = utt[s]
            pairs = [(u[i], u[j]) for i in range(len(u)) for j in range(i + 1, len(u))]
            if len(pairs) > 300:
                pairs = [pairs[k] for k in rng.choice(len(pairs), 300, replace=False)]
            allt += pairs
        allt = np.array(allt)
        ts = score(allt[:, 0], allt[:, 1])
        lo, hi = np.percentile(ts, [2, 25])
        band = np.where((ts >= lo) & (ts <= hi))[0]
        ht = allt[rng.choice(band, min(3000, len(band)), replace=False)]
        flat = [(s, x) for s in spks for x in utt[s]]
        fi = rng.integers(len(flat), size=(1_000_000, 2))
        fi = fi[np.array([flat[a][0] != flat[b][0] for a, b in fi])]
        na = np.array([flat[a][1] for a in fi[:, 0]])
        nb = np.array([flat[b][1] for b in fi[:, 1]])
        ns = score(na, nb)
        lo, hi = np.percentile(ns, [97, 99.9])
        band = np.where((ns >= lo) & (ns <= hi))[0]
        pick = rng.choice(band, min(3000, len(band)), replace=False)
        hn = np.stack([na[pick], nb[pick]], 1)
        lines = [f"1 {P(a)} {P(b)} {tag(a, b)}" for a, b in ht] + [f"0 {P(a)} {P(b)} {tag(a, b)}" for a, b in hn]
        C.write_lines(os.path.join(OUT, f"{split}_hard.txt"), lines)
        if split == "devA":
            for k in rng.choice(len(lines), 40, replace=False):
                listen.append(lines[k])

        # ---- short-long (materialised crops)
        cdir = os.path.join(CROPS, split)
        os.makedirs(cdir, exist_ok=True)
        longs = {s: [x for x in utt[s] if info.dur[x] >= 6] for s in spks}
        crop_src = {s: [x for x in utt[s] if info.dur[x] >= 4] for s in spks}
        ok_t = [s for s in spks if crop_src[s] and any(len(set(longs[s]) - {c}) for c in crop_src[s])]
        ok_l = [s for s in spks if longs[s]]
        lines = []
        for label, n in [(1, 3000), (0, 3000)]:
            k = 0
            while k < n:
                s = ok_t[rng.integers(len(ok_t))] if label else spks[rng.integers(len(spks))]
                if not crop_src[s]:
                    continue
                a = crop_src[s][rng.integers(len(crop_src[s]))]
                if label:
                    cand = [x for x in longs[s] if x != a]
                else:
                    t = ok_l[rng.integers(len(ok_l))]
                    if t == s:
                        continue
                    cand = longs[t]
                if not cand:
                    continue
                b = cand[rng.integers(len(cand))]
                cd = float(rng.uniform(1.5, 2.5))
                stt = float(rng.uniform(0, info.dur[a] - cd))
                name = f"{a.replace('/', '_')[:-4]}_{int(stt * 1000)}_{int(cd * 1000)}.wav"
                fp = os.path.join(cdir, name)
                if not os.path.exists(fp):
                    x = sv.load_wav(os.path.join(C.TRAIN_WAV, a), int(stt * C.SR), int((stt + cd) * C.SR))
                    sf.write(fp, x, C.SR, subtype="PCM_16")
                lines.append(f"{label} cache/devcrops/{split}/{name} {P(b)} {tag(a, b, da=cd)}")
                k += 1
        C.write_lines(os.path.join(OUT, f"{split}_shortlong.txt"), sorted(set(lines)))
        print(split, {f: len(C.read_lines(os.path.join(OUT, f"{split}_{f}.txt")))
                      for f in ["random", "hard", "shortlong"]})
    C.write_lines(os.path.join(OUT, "hard_pairs_to_listen.tsv"), listen)


if __name__ == "__main__":
    main()
