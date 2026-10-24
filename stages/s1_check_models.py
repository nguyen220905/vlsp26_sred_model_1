"""Stage 1 - check every allowed checkpoint before using it.

strict=True loading, eval(), dither 0 / no spec-aug at eval, determinism, batched == single,
and the inference parameter count against the 7M limit. Writes lists/model_check.json.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import config as C  # noqa: E402
from common import models as M  # noqa: E402


def main():
    out = {}
    for name, (_, _, role, desc) in M.REGISTRY.items():
        model = M.load(name)
        rep = M.frontend_report(name, model)
        rep.update(role=role, desc=desc, under_7M=rep["params"] < C.PARAM_LIMIT)
        rep["ckpt_sha1"] = None
        hub = os.path.join(torch.hub.get_dir(), "checkpoints")
        kw = M.REGISTRY[name][1]
        if "model_name" in kw:
            f = os.path.join(hub, f"{kw['model_name']}-{kw['dataset']}-{kw['train_type']}.pt")
            if os.path.exists(f):
                rep["ckpt_sha1"] = C.sha1_file(f)
        print(json.dumps(rep, default=str))
        out[name] = rep
        del model
        torch.cuda.empty_cache()
    with open(os.path.join(C.LISTS, "model_check.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
