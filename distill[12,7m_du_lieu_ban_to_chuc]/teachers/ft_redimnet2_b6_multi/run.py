"""Teacher 1: ReDimNet2-B6 multilingual (12.3M, official list) fine-tuned on the train split. Teacher only (> 7M): its clean full-utterance embeddings become KD targets (make_targets.py).
Run from this folder:  ..\..\..\.venv\Scripts\python.exe run.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from common import runners  # noqa: E402

CFG = {
    "model": "redimnet2_b6_multi",
    "epochs": 2,
    "crop": 3.0,
    "bs": 16,
    "accum": 2,
    "lr_backbone": 0.0001,
    "lr_head": 0.001,
    "aug": "mid",
    "desc": "TEACHER ReDimNet2-B6 multi FT (2 ep, 3 s, aug mid)"
}

if __name__ == "__main__":  # guard required: DataLoader workers re-import this file on Windows
    runners.finetune(HERE, CFG, kind="teacher", csv_rel="../../results.csv")
