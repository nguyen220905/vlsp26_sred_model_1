"""Frozen SSL teacher (WavLM-Large / XLS-R 300M from the official list) + a small trainable MHFA back-end.

The SSL encoder is NEVER fine-tuned (no gradients, eval mode); only the layer weights and the multi-head
factorized attentive pooling (MHFA, Peng et al. 2022) are trained with AAM on the train split.
Used only as an extra teacher for distillation (never at inference: > 7M params).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

HF = {"wavlm_large": "microsoft/wavlm-large", "xlsr_300m": "facebook/wav2vec2-xls-r-300m"}


class SSLMHFA(nn.Module):
    def __init__(self, which="wavlm_large", heads=64, compression=128, emb_dim=256):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        cfg = AutoConfig.from_pretrained(HF[which])
        self.ssl = AutoModel.from_pretrained(HF[which])
        self.ssl.eval()
        for p in self.ssl.parameters():
            p.requires_grad_(False)
        L, D = cfg.num_hidden_layers + 1, cfg.hidden_size
        self.w_k = nn.Parameter(torch.zeros(L))
        self.w_v = nn.Parameter(torch.zeros(L))
        self.cmp_k = nn.Linear(D, compression)
        self.cmp_v = nn.Linear(D, compression)
        self.att = nn.Linear(compression, heads)
        self.out = nn.Linear(heads * compression, emb_dim)
        self.norm_wav = True  # both checkpoints were pre-trained on zero-mean / unit-variance waveforms

    def train(self, mode=True):
        super().train(mode)
        self.ssl.eval()  # keep dropout / layerdrop off in the frozen encoder
        return self

    def forward(self, wav):
        x = wav.float()
        if self.norm_wav:
            x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-5)
        with torch.no_grad():
            hs = self.ssl(x, output_hidden_states=True).hidden_states
            hs = torch.stack(hs, 0)  # (L, B, T, D)
        hs = hs.to(self.cmp_k.weight.dtype if not torch.is_autocast_enabled() else torch.bfloat16)
        k = torch.einsum("l,lbtd->btd", F.softmax(self.w_k, 0).to(hs.dtype), hs)
        v = torch.einsum("l,lbtd->btd", F.softmax(self.w_v, 0).to(hs.dtype), hs)
        k, v = self.cmp_k(k), self.cmp_v(v)
        a = F.softmax(self.att(k).float(), dim=1).to(v.dtype)  # (B, T, H) attention over time
        pooled = torch.einsum("bth,btc->bhc", a, v).reshape(x.shape[0], -1)
        return self.out(pooled)

    def trainable_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("ssl.")]
