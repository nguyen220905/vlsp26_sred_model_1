"""Stage 3 - identity check + speaker-disjoint dev split (dev-A / dev-B / train).

Evidence for "two IDs are the same person" uses the AVERAGE of 3 frozen models from different families
(ReDimNet-M multi, ReDimNet2-B4 vox2, WeSpeaker ResNet34): the cross-ID utterance->centroid similarity must be
as high as the within-ID one. Nothing is relabelled or merged: twins are only kept in the same split
(always train), so no dev identity leaks into training / cohorts. Exact duplicate files are linked too.

Also flags single utterances that sit closer to another speaker than to their own (possible label errors);
they are written to lists/suspect_utts.tsv to be checked by ear, and kept out of DEV trials only.

Outputs (lists/): identity_links.tsv, suspect_utts.tsv, split_train.txt, split_devA.txt, split_devB.txt,
                  label_fixes.tsv (every exclusion decision, with its reason), split_report.md
"""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import config as C  # noqa: E402
from common import sv  # noqa: E402

FROZEN = os.path.join(C.CACHE, "frozen")
MODELS = ["redimnet_M_multi", "redimnet2_b4_vox2", "wespeaker_r34_vox2"]
N_DEV_A, N_DEV_B, MIN_DEV_UTTS = 100, 100, 4
SEED = 2026


def load():
    rows = [l.split("\t") for l in C.read_lines(os.path.join(C.LISTS, "train_index.tsv"))]
    spk = np.array([r[0] for r in rows])
    rels = [r[1] for r in rows]
    embs = [sv.l2n(np.load(os.path.join(FROZEN, f"{m}__train.npy"))) for m in MODELS]
    return spk, rels, embs


def centroids(spk, emb, spks):
    idx = {s: i for i, s in enumerate(spks)}
    cs = np.zeros((len(spks), emb.shape[1]), np.float32)
    np.add.at(cs, np.array([idx[s] for s in spk]), emb)
    return cs  # un-normalised sums


def main():
    import pandas as pd

    spk, rels, embs = load()
    spks = sorted(set(spk))
    sidx = {s: i for i, s in enumerate(spks)}
    si = np.array([sidx[s] for s in spk])
    n_utt = np.bincount(si, minlength=len(spks))

    # utterance -> centroid similarity, averaged over the 3 models (leave-one-out for the own speaker)
    sim_utt_cent = np.zeros((len(spk), len(spks)), np.float32)
    own_loo = np.zeros(len(spk), np.float32)
    for e in embs:
        sums = centroids(spk, e, spks)
        cn = sv.l2n(sums)
        sim_utt_cent += e @ cn.T / len(embs)
        loo = sums[si] - e
        loo_n = sv.l2n(loo)
        own = (e * loo_n).sum(1)
        own[n_utt[si] < 2] = np.nan
        own_loo += own / len(embs)

    # --- speaker-level evidence: mean over utts of s of sim(utt, centroid t)  vs  within (LOO) of s
    df = pd.DataFrame({"spk": spk, "si": si, "own": own_loo})
    within = df.groupby("si").own.mean().reindex(range(len(spks))).values  # nan for 1-utt speakers
    cross = np.zeros((len(spks), len(spks)), np.float32)
    np.add.at(cross, si, sim_utt_cent)
    cross /= n_utt[:, None]
    np.fill_diagonal(cross, -1)
    sym = (cross + cross.T) / 2
    # reference: within-speaker LOO similarity distribution, and between-speaker distribution
    w_ok = within[~np.isnan(within)]
    off = sym[np.triu_indices(len(spks), 1)]
    rep = ["# Split report", ""]
    rep.append(f"- within-ID utt->own-centroid (LOO) sim, 3-model avg: p5={np.percentile(w_ok, 5):.3f} "
               f"p50={np.median(w_ok):.3f}")
    rep.append(f"- between-ID mean utt->centroid sim: p50={np.median(off):.3f} p99.9={np.percentile(off, 99.9):.3f} "
               f"p99.99={np.percentile(off, 99.99):.3f} max={off.max():.3f}")

    # twin rule: cross similarity reaches 92% of the lower within-similarity AND lies above the 99.99th
    # percentile of between-ID similarities (1-utt speakers: use the p50 within)
    w_fill = np.where(np.isnan(within), np.median(w_ok), within)
    iu = np.triu_indices(len(spks), 1)
    ratio = sym[iu] / np.minimum(w_fill[iu[0]], w_fill[iu[1]])
    floor = np.percentile(off, 99.99)
    cand = np.where((ratio > 0.92) & (sym[iu] > floor))[0]
    links = []
    for k in cand:
        a, b = iu[0][k], iu[1][k]
        links.append((spks[a], spks[b], float(sym[a, b]), float(w_fill[a]), float(w_fill[b]), float(ratio[k]),
                      "embedding"))
    # exact duplicate files linking two IDs
    st = pd.read_csv(os.path.join(C.LISTS, "utt_stats.tsv"), sep="\t")
    for h, g in st[st.duplicated("sha1", keep=False)].groupby("sha1"):
        ss = sorted(set(g.spk))
        for i in range(len(ss)):
            for j in range(i + 1, len(ss)):
                links.append((ss[i], ss[j], 1.0, np.nan, np.nan, np.nan, "identical_file:" + ",".join(g.rel)))
    links.sort(key=lambda r: -r[2])
    C.write_lines(os.path.join(C.LISTS, "identity_links.tsv"),
                  ["spk_a\tspk_b\tcross_sim\twithin_a\twithin_b\tratio\tevidence"] +
                  ["\t".join(str(round(x, 4)) if isinstance(x, float) else str(x) for x in r) for r in links])
    rep.append(f"- identity links (ratio > 0.92 and cross > {floor:.3f}, or identical file): {len(links)} pairs -> listen to "
               f"lists/identity_links.tsv; IDs are NOT merged")
    for r in links[:15]:
        rep.append(f"  - {r[0]} ~ {r[1]} cross={r[2]:.3f} within=({r[3]:.3f},{r[4]:.3f}) ratio={r[5]:.2f} {r[6][:60]}")

    # identity groups (union-find)
    parent = {s: s for s in spks}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, *_ in links:
        parent[find(a)] = find(b)
    groups = defaultdict(list)
    for s in spks:
        groups[find(s)].append(s)
    multi = {s for g in groups.values() if len(g) > 1 for s in g}
    sizes = sorted((len(g) for g in groups.values() if len(g) > 1), reverse=True)
    rep.append(f"- multi-ID group sizes: {sizes[:20]}")
    rep.append(f"- {len(multi)} IDs are in multi-ID identity groups ({sum(len(g) > 1 for g in groups.values())} "
               f"groups) -> kept in TRAIN")

    # --- suspect utterances: closer to another speaker's centroid than to own (LOO) by a margin
    other = sim_utt_cent.copy()
    other[np.arange(len(spk)), si] = -1
    best_other = other.max(1)
    best_other_spk = np.array(spks)[other.argmax(1)]
    margin = best_other - own_loo
    sus = np.where((n_utt[si] >= 3) & (margin > 0.10))[0]
    C.write_lines(os.path.join(C.LISTS, "suspect_utts.tsv"),
                  ["rel\tspk\town_loo\tbest_other\tbest_other_spk"] +
                  [f"{rels[i]}\t{spk[i]}\t{own_loo[i]:.3f}\t{best_other[i]:.3f}\t{best_other_spk[i]}"
                   for i in sus[np.argsort(-margin[sus])]])
    rep.append(f"- suspect utterances (closer to another ID by > 0.10): {len(sus)} -> lists/suspect_utts.tsv "
               f"(kept in train; excluded from dev trials only)")

    # --- dev split: singleton identities with >= MIN_DEV_UTTS, stratified by #utts and bandwidth
    med_bw = st.groupby("spk").bw.median()
    cands = [s for s in spks if s not in multi and n_utt[sidx[s]] >= MIN_DEV_UTTS]
    rng = np.random.default_rng(SEED)
    ubin = lambda n: 0 if n <= 6 else 1 if n <= 15 else 2 if n <= 50 else 3  # noqa: E731
    bbin = lambda b: int(b >= med_bw.median())  # noqa: E731
    strata = defaultdict(list)
    for s in cands:
        strata[(ubin(n_utt[sidx[s]]), bbin(med_bw[s]))].append(s)
    all_cand = len(cands)
    dev_a, dev_b = [], []
    for key in sorted(strata):
        lst = strata[key]
        rng.shuffle(lst)
        k = len(lst) / all_cand
        na, nb = int(round(k * N_DEV_A)), int(round(k * N_DEV_B))
        dev_a += lst[:na]
        dev_b += lst[na:na + nb]
    dev_set = set(dev_a) | set(dev_b)
    train = [s for s in spks if s not in dev_set]
    for name, ss in [("split_devA.txt", dev_a), ("split_devB.txt", dev_b), ("split_train.txt", train)]:
        C.write_lines(os.path.join(C.LISTS, name), sorted(ss))
    fixes = ["kind\titem\treason"]
    fixes += [f"keep_in_train\t{s}\tmulti-ID identity group" for s in sorted(multi)]
    fixes += [f"exclude_from_dev_trials\t{rels[i]}\tcloser to {best_other_spk[i]} (margin {margin[i]:.3f})"
              for i in sus if spk[i] in dev_set]
    C.write_lines(os.path.join(C.LISTS, "label_fixes.tsv"), fixes)
    for name, ss in [("dev-A", dev_a), ("dev-B", dev_b), ("train", train)]:
        n = sum(n_utt[sidx[s]] for s in ss)
        rep.append(f"- {name}: {len(ss)} speakers, {n} utts")
    C.write_lines(os.path.join(C.LISTS, "split_report.md"), rep)
    print("\n".join(rep))


if __name__ == "__main__":
    main()
