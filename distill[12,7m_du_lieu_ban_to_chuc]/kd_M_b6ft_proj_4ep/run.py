"""[4-epoch twin of kd_M_b6ft_proj] Distillation experiment (student ReDimNet-M multi, same budget as baseline/ft_redimnet_M_multi): AAM + KD through a learned linear projection (training only) to the fine-tuned B6 teacher.
Run from this folder:  ..\..\.venv\Scripts\python.exe run.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from common import config as C  # noqa: E402
from common import runners  # noqa: E402

CFG = {
    "model": "redimnet_M_multi",
    "epochs": 4,
    "crop": 3.0,
    "bs": 16,
    "accum": 2,
    "lr_backbone": 0.0001,
    "lr_head": 0.001,
    "aug": "mid",
    "kd": [
        {
            "name": "b6ft",
            "path": "cache/teachers/b6_ft.npz",
            "mode": "proj",
            "weight": 2.0
        }
    ],
    "desc": "projection KD from fine-tuned B6, 4 epochs (decides the epoch count of the final all-data run)"
}
for k in CFG.get("kd", []):  # teacher targets live in cache/teachers (distill/make_targets.py)
    k["path"] = C.resolve(k["path"])
if CFG.get("hard_batch"):
    CFG["hard_batch"]["teacher"] = C.resolve(CFG["hard_batch"]["teacher"])

if __name__ == "__main__":  # guard required: DataLoader workers re-import this file on Windows
    runners.finetune(HERE, CFG, kind="distill", csv_rel="../results.csv")
