"""3-branch GPU augmentation built only from the provided data + code (assets: stages/s5_aug_assets.py).

Each sample of a batch goes to one branch:
  0 clean        : untouched
  1 acoustic     : simulated reverb (pyroomacoustics RIR bank, p_rir) + additive noise at SNR in snr_range;
                   noise = non-speech segments mined from train (60%), synthetic coloured noise / hum (20%),
                   babble = other utterances of the same batch (20%)
  2 device       : ONE device transform: offline training-codec copy (mp3/vorbis/G.711/G.726/ADPCM) when the
                   utterance has one, else one of 8 kHz band-limit, mu-law 8 kHz, random EQ / low-pass,
                   hard or soft clipping, gain change (+ clipping)
`level` sets the difficulty: "mid" (default) or "hard" (lower SNR, more reverb, stronger device effects).
Level "mid_slr28" = "mid" + a branch with REAL RIRs / point-source noise from OpenSLR 28 (stages/s5b_slr28_assets.py,
train halves only): ext_rir of the reverb events and ext_noise of the noise events. OpenSLR 28 is an augmentation-only
dataset (allowed by BTC, 13/10); its files are never used as speaker labels. Older levels do not load it.
"""
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C

LEVELS = {
    "mid": dict(p=(1 / 3, 1 / 3, 1 / 3), snr=(5.0, 20.0), p_rir=0.5, clip=(0.2, 0.6), lp=(2500, 7000)),
    "hard": dict(p=(0.25, 0.375, 0.375), snr=(0.0, 15.0), p_rir=0.7, clip=(0.08, 0.4), lp=(1800, 5000)),
    "light": dict(p=(0.5, 0.25, 0.25), snr=(10.0, 25.0), p_rir=0.3, clip=(0.3, 0.8), lp=(3000, 7600)),
}
LEVELS["mid_slr28"] = dict(LEVELS["mid"], ext_rir=0.5, ext_noise=0.2)


def _rms(x):
    return x.pow(2).mean(-1, keepdim=True).clamp_min(1e-10).sqrt()


class Augment3:
    def __init__(self, level="mid", device="cuda"):
        self.cfg = LEVELS[level]
        aug = os.path.join(C.CACHE, "aug")
        nb = np.load(os.path.join(aug, "noise_train.npy"))
        self.noise = torch.from_numpy(nb.astype(np.float32) / 32767).to(device)
        self.rir = torch.from_numpy(np.load(os.path.join(aug, "rir_train.npy"))).to(device)
        self.device = device
        self.ext_rir = self.cfg.get("ext_rir", 0.0)
        self.ext_noise = self.cfg.get("ext_noise", 0.0)
        self.rir_ext = self.noise_ext = None
        if self.ext_rir > 0:
            self.rir_ext = torch.from_numpy(np.load(os.path.join(aug, "rir_slr28_train.npy"))).to(device)
        if self.ext_noise > 0:
            nx = np.load(os.path.join(aug, "noise_slr28_train.npy"))
            self.noise_ext = torch.from_numpy(nx.astype(np.float32) / 32767).to(device)

    # ------------------------------------------------------------------ helpers
    def _reverb(self, x):
        B, T = x.shape
        h = self.rir[torch.randint(len(self.rir), (B,), device=x.device)]
        if self.rir_ext is not None:
            he = self.rir_ext[torch.randint(len(self.rir_ext), (B,), device=x.device)]
            L = max(h.shape[1], he.shape[1])
            h, he = F.pad(h, (0, L - h.shape[1])), F.pad(he, (0, L - he.shape[1]))
            h = torch.where((torch.rand(B, 1, device=x.device) < self.ext_rir), he, h)
        n = T + h.shape[1]
        nfft = 1 << (n - 1).bit_length()
        y = torch.fft.irfft(torch.fft.rfft(x, nfft) * torch.fft.rfft(h, nfft), nfft)[:, :T]
        return y * (_rms(x) / _rms(y))

    def _colored(self, B, T, dev):
        w = torch.randn(B, T, device=dev)
        spec = torch.fft.rfft(w)
        f = torch.linspace(1, spec.shape[1], spec.shape[1], device=dev)
        beta = torch.rand(B, 1, device=dev) * 2  # 0 white .. 1 pink .. 2 brown
        y = torch.fft.irfft(spec / f.pow(beta / 2), T)
        hum = torch.rand(B, 1, device=dev) < 0.2  # mains hum + harmonics
        t = torch.arange(T, device=dev) / C.SR
        f0 = torch.where(torch.rand(B, 1, device=dev) < 0.5, 50.0, 60.0)
        h = sum(torch.sin(2 * math.pi * k * f0 * t) / k for k in range(1, 6))
        return torch.where(hum, h, y)

    def _noise(self, x):
        B, T = x.shape
        dev = x.device
        src = torch.rand(B, device=dev)
        bank = self.noise[torch.randint(len(self.noise), (B,), device=dev)]
        if bank.shape[1] < T:
            bank = bank.repeat(1, T // bank.shape[1] + 1)
        st = torch.randint(0, bank.shape[1] - T + 1, (1,)).item()
        bank = bank[:, st:st + T]
        col = self._colored(B, T, dev)
        bab = sum(torch.roll(x, k, 0) for k in range(1, 4))  # 3 other utterances of the batch
        if self.noise_ext is not None:  # slr28 point-source noise takes src in [0, ext_noise) from the mined share [0, 0.6)
            be = self.noise_ext[torch.randint(len(self.noise_ext), (B,), device=dev)]
            if be.shape[1] < T:
                be = be.repeat(1, T // be.shape[1] + 1)
            st = torch.randint(0, be.shape[1] - T + 1, (1,)).item()
            bank = torch.where((src < self.ext_noise)[:, None], be[:, st:st + T], bank)
        n = torch.where((src < 0.6)[:, None], bank, torch.where((src < 0.8)[:, None], col, bab))
        lo, hi = self.cfg["snr"]
        snr = torch.empty(B, 1, device=dev).uniform_(lo, hi)
        return x + n * (_rms(x) / _rms(n)) / (10 ** (snr / 20))

    def _eq(self, x):
        B, T = x.shape
        dev = x.device
        X = torch.fft.rfft(x)
        f = torch.linspace(0, C.SR / 2, X.shape[1], device=dev)
        lo, hi = self.cfg["lp"]
        fc = torch.empty(B, 1, device=dev).uniform_(lo, hi)
        order = torch.randint(2, 8, (B, 1), device=dev).float()
        lp = 1 / torch.sqrt(1 + (f / fc).pow(2 * order))
        hc = torch.empty(B, 1, device=dev).uniform_(50, 400)
        hp = 1 / torch.sqrt(1 + (hc / f.clamp_min(1)).pow(4))
        # random smooth tilt / resonance (+-6 dB)
        k = torch.randn(B, 4, device=dev) * 3
        basis = torch.stack([torch.cos(math.pi * j * f / (C.SR / 2)) for j in range(1, 5)], 0)
        tilt = 10 ** ((k @ basis) / 20)
        return torch.fft.irfft(X * lp * hp * tilt, T)

    def _band8k(self, x, mulaw=False):
        import torchaudio.functional as AF

        y = AF.resample(x, C.SR, 8000)
        if mulaw:
            y = y / y.abs().amax(-1, keepdim=True).clamp_min(1e-6) * 0.9
            y = AF.mu_law_decoding(AF.mu_law_encoding(y, 256), 256)
        return AF.resample(y, 8000, C.SR)[:, : x.shape[1]]

    def _clip(self, x):
        B = x.shape[0]
        dev = x.device
        peak = x.abs().amax(-1, keepdim=True).clamp_min(1e-6)
        lo, hi = self.cfg["clip"]
        c = peak * torch.empty(B, 1, device=dev).uniform_(lo, hi)
        hard = torch.rand(B, 1, device=dev) < 0.5
        g = torch.empty(B, 1, device=dev).uniform_(2, 8)
        soft = torch.tanh(g * x / peak) / math.tanh(1) * c
        return torch.where(hard, x.clamp(-c, c), soft)

    def _gain(self, x):
        B = x.shape[0]
        g = 10 ** (torch.empty(B, 1, device=x.device).uniform_(-20, 12) / 20)
        return (x * g).clamp(-1, 1)

    # ------------------------------------------------------------------ main
    @torch.no_grad()
    def __call__(self, x, codec=None, has_codec=None):
        """x (B, T) clean; codec (B, T) training-codec copy (or None); has_codec (B,) bool."""
        B, T = x.shape
        dev = x.device
        p = torch.tensor(self.cfg["p"], device=dev)
        br = torch.multinomial(p, B, replacement=True)
        out = x.clone()
        m1 = br == 1
        if m1.any():
            y = x[m1]
            rv = torch.rand(len(y), 1, device=dev) < self.cfg["p_rir"]
            y = torch.where(rv, self._reverb(y), y)
            out[m1] = self._noise(y)
        m2 = br == 2
        if m2.any():
            y = x[m2]
            op = torch.randint(0, 6, (len(y),), device=dev)
            if codec is not None and has_codec is not None:
                use_c = has_codec.to(dev)[m2] & (torch.rand(len(y), device=dev) < 0.6)
                op = torch.where(use_c, torch.full_like(op, 6), op)
            res = y.clone()
            for k, fn in enumerate([lambda z: self._band8k(z), lambda z: self._band8k(z, True), self._eq,
                                    self._eq, self._clip, self._gain]):
                mk = op == k
                if mk.any():
                    res[mk] = fn(y[mk])
            mk = op == 6
            if mk.any():
                res[mk] = codec.to(dev)[m2][mk]
            out[m2] = res
        peak = out.abs().amax(-1, keepdim=True)
        return torch.where(peak > 1, out / peak, out), br
