"""Wrappers for the checkpoints on the OFFICIAL list (BTC email 2026-10-03).

Every model is a torch.nn.Module  wav (B, T) float32 in [-1, 1] @16 kHz  ->  embedding (B, D),
with the original frontend of its upstream code (dither 0, no spec-augment in eval), loaded with
strict=True. `load(name)` returns the module in eval mode on cuda.

Students (< 7M params at inference):  redimnet_S_multi, redimnet_M_multi, redimnet2_b4_vox2, wespeaker_r34_vox2
Teachers (training only):             redimnet2_b6_multi, (wavlm_large / xlsr_300m: frozen SSL, see distill/)
"""
import importlib
import os
import sys
import types

import torch
import torch.nn as nn

from . import config as C

REGISTRY = {
    # name: (family, kwargs, role, description)
    "redimnet_S_multi": ("redimnet_v1", dict(model_name="S", train_type="ft_mix", dataset="vb2+vox2+cnc"),
                         "student", "ReDimNet-S S-vb2+vox2+cnc-ft_mix"),
    "redimnet_M_multi": ("redimnet_v1", dict(model_name="M", train_type="ft_mix", dataset="vb2+vox2+cnc"),
                         "student", "ReDimNet-M M-vb2+vox2+cnc-ft_mix"),
    "redimnet2_b4_vox2": ("redimnet2", dict(model_name="b4", train_type="lm", dataset="vox2"),
                          "student", "ReDimNet2-B4 b4-vox2-lm"),
    "wespeaker_r34_vox2": ("wespeaker", dict(hf_repo="Wespeaker/wespeaker-voxceleb-resnet34-LM", ckpt="avg_model"),
                           "student", "WeSpeaker ResNet34-LM VoxCeleb2"),
    "redimnet2_b6_multi": ("redimnet2", dict(model_name="b6", train_type="lm", dataset="vb2+vox2+cnc2_v0"),
                           "teacher", "ReDimNet2-B6 b6-vb2+vox2+cnc2_v0-lm (12.3M, teacher only)"),
}


def _add_path(p):
    if p not in sys.path:
        sys.path.insert(0, p)


def count_params(model):
    """All parameters used at inference (classifier heads are never part of these wrappers)."""
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------- ReDimNet v1 / v2
def _redimnet_v1(model_name, train_type, dataset):
    _add_path(os.path.join(C.TP, "redimnet"))
    from redimnet.model import ReDimNetWrap

    url = f"https://github.com/IDRnD/ReDimNet/releases/download/latest/{model_name}-{dataset}-{train_type}.pt"
    full = torch.hub.load_state_dict_from_url(url, map_location="cpu", weights_only=False)
    model = ReDimNetWrap(**full["model_config"])
    model.load_state_dict(full["state_dict"], strict=True)
    return model


def _redimnet2(model_name, train_type, dataset):
    _add_path(os.path.join(C.TP, "redimnet2"))
    from redimnet2 import ReDimNet2Wrap

    url = f"https://github.com/PalabraAI/redimnet2/releases/download/v1.0.0/{model_name}-{dataset}-{train_type}.pt"
    full = torch.hub.load_state_dict_from_url(url, map_location="cpu", weights_only=False)
    model = ReDimNet2Wrap(**full["model_config"])
    model.load_state_dict(full["state_dict"], strict=True)
    return model


# ----------------------------------------------------------------------------- WeSpeaker
class KaldiFbank(nn.Module):
    """torchaudio kaldi fbank exactly as WeSpeaker inference: wav*32768, 80 mel, 25/10 ms, hamming,
    dither 0, energy_floor 0, then per-utterance CMN. Parameter-free."""

    def __init__(self, num_mel_bins=80, frame_length=25, frame_shift=10):
        super().__init__()
        self.kw = dict(num_mel_bins=num_mel_bins, frame_length=frame_length, frame_shift=frame_shift,
                       dither=0.0, energy_floor=0.0, window_type="hamming", sample_frequency=C.SR)

    def forward(self, wav):
        import torchaudio.compliance.kaldi as kaldi

        with torch.no_grad(), torch.autocast(wav.device.type, enabled=False):
            x = wav.float() * (1 << 15)
            feats = [kaldi.fbank(r.unsqueeze(0), **self.kw) for r in x]
            feats = torch.stack(feats)
            feats = feats - feats.mean(1, keepdim=True)
        return feats


class WeSpeakerWrap(nn.Module):
    def __init__(self, backbone, fbank_args):
        super().__init__()
        self.fbank = KaldiFbank(fbank_args["num_mel_bins"], fbank_args["frame_length"], fbank_args["frame_shift"])
        self.backbone = backbone

    def forward(self, wav):
        out = self.backbone(self.fbank(wav))
        return out[-1] if isinstance(out, tuple) else out


def _wespeaker(hf_repo, ckpt):
    import yaml
    from huggingface_hub import hf_hub_download

    # stub packages so wespeaker/__init__.py (CLI deps) is not executed
    root = os.path.join(C.TP, "wespeaker", "wespeaker")
    for pkg, path in [("wespeaker", root), ("wespeaker.models", os.path.join(root, "models"))]:
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [path]
            sys.modules[pkg] = m
    cfg = yaml.safe_load(open(hf_hub_download(hf_repo, "config.yaml")))
    sd = torch.load(hf_hub_download(hf_repo, ckpt), map_location="cpu", weights_only=False)
    mod = "resnet" if cfg["model"].startswith("ResNet") else "ecapa_tdnn"
    backbone = getattr(importlib.import_module(f"wespeaker.models.{mod}"), cfg["model"])(**cfg["model_args"])
    # the checkpoint also holds the training classifier ("projection.*"); it is not part of the embedder
    head = [k for k in sd if k.startswith("projection.")]
    sd = {k: v for k, v in sd.items() if not k.startswith("projection.")}
    backbone.load_state_dict(sd, strict=True)
    model = WeSpeakerWrap(backbone, cfg["dataset_args"]["fbank_args"])
    model.dropped_keys = head
    return model


# ----------------------------------------------------------------------------- public API
def build(name):
    if name.startswith("ssl:"):  # frozen SSL teacher + MHFA head, e.g. "ssl:wavlm_large" (teacher only)
        from .ssl_teacher import SSLMHFA

        return SSLMHFA(name.split(":", 1)[1])
    family, kw, _, _ = REGISTRY[name]
    return {"redimnet_v1": _redimnet_v1, "redimnet2": _redimnet2, "wespeaker": _wespeaker}[family](**kw)


def load(name, device="cuda"):
    model = build(name).to(device).eval()
    return model


def frontend_report(name, model):
    """Facts checked before fine-tuning: eval mode, no dither, no spec-aug at eval, deterministic output."""
    rep = {"name": name, "params": count_params(model), "training": model.training}
    spec = getattr(model, "spec", None)
    if spec is not None:
        rep["spec_aug_module"] = type(getattr(spec, "specaug", nn.Identity())).__name__
    if isinstance(model, WeSpeakerWrap):
        rep["fbank"] = model.fbank.kw
        rep["dropped_keys"] = model.dropped_keys
    dev = next(model.parameters()).device
    g = torch.Generator().manual_seed(0)
    wav = (torch.randn(2, 3 * C.SR, generator=g) * 0.05).to(dev)
    with torch.no_grad():
        e1 = model(wav)
        e2 = model(wav)
        e_single = model(wav[:1])
    rep["emb_dim"] = int(e1.shape[-1])
    rep["deterministic"] = bool(torch.allclose(e1, e2, atol=1e-5))
    rep["batch_eq_single"] = float((e1[:1] - e_single).abs().max())
    return rep
