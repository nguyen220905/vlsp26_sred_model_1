"""FINAL round: ReDimNet-M multi (4.81M) + projection KD from the fine-tuned ReDimNet2-B6 teacher, trained on ALL
1900 speakers (train + dev-A + dev-B). Recipe chosen on dev-A (kd_M_b6ft_proj, 2 seeds) ; the epoch count comes
from kd_M_b6ft_proj_4ep vs the 2-epoch runs (env FINAL_EPOCHS, set by run_final.sh). Checkpoint = EMA (fixed a
priori: EMA won on dev-A in every ReDimNet-M run). No dev evaluation is possible; public test is information only.
Run from this folder:  ..\..\..\.venv\Scripts\python.exe run.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from common import config as C  # noqa: E402
from common import runners  # noqa: E402

EPOCHS = int(os.environ.get("FINAL_EPOCHS", "2"))
CFG = {
    "model": "redimnet_M_multi",
    "epochs": EPOCHS,
    "crop": 3.0,
    "bs": 16,
    "accum": 2,
    "lr_backbone": 0.0001,
    "lr_head": 0.001,
    "aug": "mid",
    "kd": [{"name": "b6ft", "path": os.path.join(C.CACHE, "teachers", "b6_ft.npz"), "mode": "proj", "weight": 2.0}],
    "all_data": True,
    "final_pick": "ema",
    "desc": f"FINAL M + proj KD (B6 FT), ALL 1900 spk, {EPOCHS} ep, EMA",
}

if __name__ == "__main__":  # guard required: DataLoader workers re-import this file on Windows
    runners.finetune(HERE, CFG, kind="final", csv_rel="../results.csv")
