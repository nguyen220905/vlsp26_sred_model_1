"""Build <folder>/RESULTS.md from every <folder>/*/results.json (+ config.json / train_log.txt).

usage: python -m common.summarize baseline   (or distill)
Ranked by dev-A random (AS-norm). Public-test EER is shown for information only (never used for selection).
"""
import json
import os
import re
import sys

from . import config as C

SETS = [("devA_pooled", "cos", "devA-pooled cos"), ("devA_pooled", "asnorm", "**devA-pooled**"),
        ("devA_random", "asnorm", "devA-rand"),
        ("devA_hard", "asnorm", "devA-hard"), ("devA_shortlong", "asnorm", "devA-short/long"),
        ("devA_simedge", "asnorm", "devA-simedge"), ("pub", "cos", "pub cos (info)"),
        ("pub", "asnorm", "pub (info)")]


MINERS = ["redimnet_M_multi", "redimnet2_b4_vox2", "wespeaker_r34_vox2"]  # frozen models that mined devA_hard


def main(folder):
    root = C.resolve(folder)
    rows = []
    dirs = []
    for d in sorted(os.listdir(root)):
        if os.path.isdir(os.path.join(root, d)) and not os.path.exists(os.path.join(root, d, "run.py")):
            dirs += [f"{d}/{x}" for x in sorted(os.listdir(os.path.join(root, d)))]  # e.g. distill/teachers/*
        else:
            dirs.append(d)
    for d in dirs:
        f = os.path.join(root, d, "results.json")
        if not os.path.exists(f):
            continue
        res = json.load(open(f))
        params, kind, desc = "", "", ""
        cfgf = os.path.join(root, d, "config.json")
        man = os.path.join(root, d, "manifest.json")
        if os.path.exists(man):
            params = json.load(open(man))["params"] / 1e6
        if os.path.exists(cfgf):
            desc = json.load(open(cfgf)).get("desc", "")
        elif os.path.exists(os.path.join(root, d, "run.py")):
            desc = open(os.path.join(root, d, "run.py"), encoding="utf-8").read().split('"""')[1].splitlines()[0]
        log = os.path.join(root, d, "log.txt")
        if not params and os.path.exists(log):
            m = re.search(r"([\d.]+)M params", open(log, encoding="utf-8", errors="ignore").read())
            params = float(m.group(1)) if m else ""
        kind = d.split("/")[0] if "/" in d else d.split("_")[0]
        vals = [res.get(s, {}).get(k, {}).get("eer") for s, k, _ in SETS]
        if d.startswith("zs_") and any(m in d for m in MINERS):
            d = d + " †"
        dcf = res.get("devA_random", {}).get("asnorm", {}).get("mindcf")
        curve = ""
        tl = os.path.join(root, d, "train_log.txt")
        if os.path.exists(tl):
            eers = re.findall(r"devA_random EER\(cos\) ([\d.]+)", open(tl, encoding="utf-8").read())
            curve = " -> ".join(eers)
        rows.append((d, kind, params, vals, dcf, desc, curve))
    rows.sort(key=lambda r: (r[3][1] is None, r[3][1] or 0))  # dev-A pooled, AS-norm
    out = [f"# {folder} results", "",
           "EER % (lower is better). Ranked by **dev-A pooled (random+hard+short/long+simedge), AS-norm** "
           "(selection metric, 'EER gop'). "
           "dev-B is final-check only (results_devB.json, not shown). "
           "`pub` = organiser public test with labels — information only, never used for selection. "
           "† = frozen model used to mine the hard trials (its devA-hard is pessimistic).", "",
           "| # | folder | kind | params (M) | <7M | " + " | ".join(s[2] for s in SETS) + " | minDCF | description |",
           "|---|---|---|---|---|" + "---|" * len(SETS) + "---|---|"]
    for i, (d, kind, p, vals, dcf, desc, _) in enumerate(rows, 1):
        ok = "yes" if p != "" and p < 7 else ("**no**" if p != "" else "")
        out.append(f"| {i} | `{d.replace(' †', '')}`{' †' if d.endswith('†') else ''} | {kind} | {p if p == '' else f'{p:.2f}'} | {ok} | " +
                   " | ".join("" if v is None else f"{v:.2f}" for v in vals) +
                   f" | {'' if dcf is None else f'{dcf:.3f}'} | {desc} |")
    curves = [(d, c) for d, _, _, _, _, _, c in rows if c]
    if curves:
        out += ["", "Training curves (devA_random EER, raw cosine; epoch 0 = initial model):", ""]
        out += [f"- `{d}`: {c}" for d, c in curves]
    C.write_lines(os.path.join(root, "RESULTS.md"), out)
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "baseline")
