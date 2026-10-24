"""Thin entry points used by the experiment folders' run.py files."""
import os

import torch

from . import evaluate as E
from . import models as M


def zero_shot(here, name, desc, csv_rel="../results.csv", eval_devB=False):
    torch.cuda.set_per_process_memory_fraction(0.8)
    model = M.load(name)
    params = M.count_params(model)
    print(f"{name}: {params / 1e6:.3f}M params")
    res = E.evaluate(model, here, sets=E.DEVA + (E.DEVB if eval_devB else []))
    E.append_results_csv(os.path.join(here, csv_rel), os.path.basename(here), "zero-shot", params, res, desc)
    return res


def finetune(here, cfg, kind="fine-tuned", csv_rel="../results.csv"):
    """Runs train.run; on CUDA OOM retries with half the batch and twice the grad accumulation
    (same effective batch)."""
    import gc

    from . import train

    cfg = dict(cfg)
    while True:
        try:
            out = train.run(here, cfg)
            break
        except torch.OutOfMemoryError:
            if cfg.get("bs", 32) <= 8:
                raise
            gc.collect()
            torch.cuda.empty_cache()
            cfg["bs"] = cfg.get("bs", 32) // 2
            cfg["accum"] = cfg.get("accum", 1) * 2
            print(f"OOM -> retry with bs {cfg['bs']} x accum {cfg['accum']}", flush=True)
    if out:
        res, params = out
        E.append_results_csv(os.path.join(here, csv_rel), os.path.basename(here), kind, params, res,
                             cfg.get("desc", ""))
