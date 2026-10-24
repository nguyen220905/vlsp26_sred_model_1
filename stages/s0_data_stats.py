"""Stage 0 - dataset audit on the provided training data only.

Per utterance: sample rate, channels, duration, sha1 (exact duplicates), peak, clipping ratio,
a crude SNR estimate (speech-frame vs. quietest-frame energy) and the 99%-energy bandwidth.
Outputs: lists/utt_stats.tsv, lists/data_report.md
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import config as C  # noqa: E402


def utt_stats(rel):
    path = os.path.join(C.TRAIN_WAV, rel)
    info = sf.info(path)
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x[:, 0]
    dur = len(x) / sr
    peak = float(np.abs(x).max()) if len(x) else 0.0
    clip = float((np.abs(x) > 0.999).mean()) if len(x) else 0.0
    # frame energies (25 ms / 10 ms)
    fl, fs = int(0.025 * sr), int(0.010 * sr)
    if len(x) > fl:
        n = 1 + (len(x) - fl) // fs
        idx = np.arange(fl)[None, :] + fs * np.arange(n)[:, None]
        e = 10 * np.log10((x[idx] ** 2).mean(1) + 1e-10)
        snr = float(np.percentile(e, 95) - np.percentile(e, 10))
        speech_ratio = float((e > np.percentile(e, 95) - 30).mean())
    else:
        snr, speech_ratio = 0.0, 0.0
    # 99% energy bandwidth from the average 512-point power spectrum (first 30 s)
    y = x[: sr * 30]
    if len(y) >= 512:
        n = 1 + (len(y) - 512) // 256
        fr = y[np.arange(512)[None, :] + 256 * np.arange(n)[:, None]] * np.hanning(512)
        spec = (np.abs(np.fft.rfft(fr, axis=1)) ** 2).mean(0)
    else:
        spec = np.zeros(257)
    if spec.sum() > 0:
        c = np.cumsum(spec) / spec.sum()
        bw = float(np.searchsorted(c, 0.99) / len(spec) * sr / 2)
    else:
        bw = 0.0
    return dict(rel=rel, sr=info.samplerate, ch=info.channels, subtype=info.subtype, dur=round(dur, 3),
                peak=round(peak, 4), clip=round(clip, 5), snr=round(snr, 1), speech=round(speech_ratio, 3),
                bw=round(bw), sha1=C.sha1_file(path))


def main():
    utts = C.read_train_list()
    rels = [r for _, r in utts]
    spk_of = {r: s for s, r in utts}
    with ProcessPoolExecutor(max_workers=12) as ex:
        rows = list(ex.map(utt_stats, rels, chunksize=64))
    cols = ["rel", "spk", "sr", "ch", "subtype", "dur", "peak", "clip", "snr", "speech", "bw", "sha1"]
    lines = ["\t".join(cols)]
    for r in rows:
        r["spk"] = spk_of[r["rel"]]
        lines.append("\t".join(str(r[c]) for c in cols))
    C.write_lines(os.path.join(C.LISTS, "utt_stats.tsv"), lines)

    import pandas as pd

    df = pd.DataFrame(rows)
    per_spk = df.groupby("spk").size()
    dup = df[df.duplicated("sha1", keep=False)].sort_values("sha1")
    dup_cross = dup.groupby("sha1")["spk"].nunique()
    rep = ["# Data report (train.txt)", ""]
    rep.append(f"- utterances: {len(df)}, speakers: {df.spk.nunique()}, total hours: {df.dur.sum() / 3600:.1f}")
    rep.append(f"- sample rates: {df.sr.value_counts().to_dict()}, channels: {df.ch.value_counts().to_dict()}, "
               f"subtypes: {df.subtype.value_counts().to_dict()}")
    q = df.dur.quantile([0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1]).round(2).to_dict()
    rep.append(f"- duration quantiles (s): {q}")
    for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 10), (10, 20), (20, 1e9)]:
        rep.append(f"  - {lo}-{hi if hi < 1e9 else 'inf'} s: {((df.dur >= lo) & (df.dur < hi)).mean() * 100:.1f}%")
    q = per_spk.quantile([0, .1, .25, .5, .75, .9, .99, 1]).to_dict()
    rep.append(f"- utts/speaker quantiles: {q}")
    rep.append(f"- exact duplicate files (sha1): {len(dup)} files in {dup.sha1.nunique()} groups, "
               f"{int((dup_cross > 1).sum())} groups span >1 speaker")
    rep.append(f"- clipping (>0.1% samples at full scale): {(df['clip'] > 0.001).mean() * 100:.2f}% of utts")
    rep.append(f"- crude SNR quantiles (dB): {df.snr.quantile([.05, .25, .5, .75, .95]).round(1).to_dict()}")
    rep.append(f"- 99%-energy bandwidth quantiles (Hz): {df.bw.quantile([.05, .25, .5, .75, .95]).round(0).to_dict()}")
    rep.append(f"- narrow-band (bw < 3.5 kHz): {(df.bw < 3500).mean() * 100:.1f}% of utts, "
               f"{(df.groupby('spk').bw.median() < 3500).mean() * 100:.1f}% of speakers (median)")
    rep.append("")
    rep.append("Duplicate groups spanning several speakers (first 30):")
    for h, g in list(dup.groupby("sha1"))[:2000]:
        if g.spk.nunique() > 1:
            rep.append("  - " + ", ".join(g.rel.tolist()))
            if sum(1 for l in rep if l.startswith("  - ")) >= 30:
                break
    C.write_lines(os.path.join(C.LISTS, "data_report.md"), rep)
    print("\n".join(rep[:25]))


if __name__ == "__main__":
    main()
