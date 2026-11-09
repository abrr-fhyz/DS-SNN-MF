import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from torch import Tensor

from config import cfg

# ──────────────────────────────────────────────────────────────
# 3-D spiking building blocks
# ──────────────────────────────────────────────────────────────

def _group_norm(ch: int) -> nn.GroupNorm:
    g = min(8, ch)
    while ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)

def spike_fn(u: Tensor, threshold: Tensor, alpha: float) -> Tensor:
    return ATanSurrogate.apply(u - threshold, alpha)

class ATanSurrogate(torch.autograd.Function):
    """
    Surrogate gradient — arctan (smoother than sigmoid-deriv, standard in modern deep-SNN work, e.g. Fang et al. 2021):
    forward:  S = 1[x ≥ 0]              (x := u − u_thr)
    backward: ∂S/∂x ≈ α / (2·(1 + (π/2·α·x)²))
    """
    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        a = ctx.alpha
        grad = a / (2.0 * (1.0 + (math.pi / 2.0 * a * x) ** 2))
        return grad_output * grad, None

class PLIFNeuron(nn.Module):
    """
    Parametric LIF neuron (PLIF): per-channel learnable decay, learnable positive threshold, soft reset.
    u[t] = σ(w_c) · u[t-1] + I[t]         (decay in (0,1), learned)
    S[t] = 1[u[t] ≥ softplus(θ)]
    u[t] ← u[t] · (1 − S[t])              (soft reset)
    """
    def __init__(self, num_channels: int, tau: float = cfg.LIF_TAU, threshold: float = cfg.LIF_THRESHOLD, alpha: float = cfg.SURROGATE_ALPHA):
        super().__init__()
        init_decay = min(max(1.0 - 1.0 / tau, 1e-3), 1.0 - 1e-3)
        init_logit = math.log(init_decay / (1.0 - init_decay))
        self.decay_logit = nn.Parameter(
            torch.full((num_channels,), init_logit))
        init_thr = 0.5
        init_thr_raw = math.log(math.expm1(init_thr))
        self.thr_raw = nn.Parameter(torch.tensor(init_thr_raw))
        self.alpha = alpha

    @property
    def threshold(self) -> Tensor:
        return F.softplus(self.thr_raw)

    def forward(self, x: Tensor, mem: Optional[Tensor] = None
                ) -> Tuple[Tensor, Tensor]:
        if mem is None:
            mem = torch.zeros_like(x)
        shape = [1, -1] + [1] * (x.dim() - 2)
        decay = torch.sigmoid(self.decay_logit).view(*shape)
        mem = decay * mem + x
        spike = spike_fn(mem, self.threshold, self.alpha)
        mem = mem * (1.0 - spike)
        return spike, mem

class SpikeDown3D(nn.Module):
    """Stride-2 spiking downsample:  Conv → GN → PLIF."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel,
                              stride=2, padding=kernel // 2, bias=False)
        self.norm = _group_norm(out_ch)
        self.lif  = PLIFNeuron(out_ch)

    def forward(self, x: Tensor, mem: Optional[Tensor] = None):
        return self.lif(self.norm(self.conv(x)), mem)

class SEWResBlock3D(nn.Module):
    """Spike-Element-Wise residual block (Fang et al., NeurIPS 2021).

        s1 = LIF(GN(conv1(x)))
        s2 = LIF(GN(conv2(s1)))
        y  = s2 + x                    (SEW-ADD)
    """
    def __init__(self, ch: int, kernel: int = 3):
        super().__init__()
        p = kernel // 2
        self.conv1 = nn.Conv3d(ch, ch, kernel, padding=p, bias=False)
        self.norm1 = _group_norm(ch)
        self.lif1  = PLIFNeuron(ch)
        self.conv2 = nn.Conv3d(ch, ch, kernel, padding=p, bias=False)
        self.norm2 = _group_norm(ch)
        self.lif2  = PLIFNeuron(ch)

    def forward(self, x: Tensor,
                mems: Optional[List[Optional[Tensor]]] = None):
        if mems is None:
            mems = [None, None]
        s1, mems[0] = self.lif1(self.norm1(self.conv1(x)),  mems[0])
        s2, mems[1] = self.lif2(self.norm2(self.conv2(s1)), mems[1])
        return s2 + x, mems                              # SEW-ADD

# ──────────────────────────────────────────────────────────────
# Spiking 3-D encoder
# ──────────────────────────────────────────────────────────────

class SNN3DEncoder(nn.Module):
    """Compact 3-D spiking encoder for one structural-MRI view.
      1. PLIF neurons
      2. Arctan surrogate
      3. Spiking stem
      4. SEW-ADD residual blocks
      5. Learnable temporal aggregation
    Input:   (B, in_ch, D, H, W)
    Output:  (B, out_dim)  — non-negative rate-coded feature vector.
    """
    def __init__(self, in_ch: int = 1, channels: Tuple[int, ...] = cfg.ENCODER_CHANNELS, T: int = cfg.T_STEPS, out_dim: int = cfg.VIEW_FEAT_DIM, dropout: float = cfg.ENCODER_DROPOUT, input_coding: str = cfg.SNN_INPUT_CODING, blocks_per_stage: int = cfg.SNN_BLOCKS_PER_STAGE):
        super().__init__()
        assert len(channels) >= 1, "channels must be non-empty"
        self.T = T
        self.input_coding = input_coding

        # ---- Spiking input stem (stride 1, learns spike encoding) ----
        stem_ch = channels[0]
        self.stem_conv = nn.Conv3d(in_ch, stem_ch, 3, padding=1, bias=False)
        self.stem_norm = _group_norm(stem_ch)
        self.stem_lif  = PLIFNeuron(stem_ch)

        # ---- Downsample + residual stages ----
        self.downs = nn.ModuleList()
        self.stages = nn.ModuleList()
        prev = stem_ch
        for ch in channels:
            self.downs.append(SpikeDown3D(prev, ch))
            self.stages.append(
                nn.ModuleList([SEWResBlock3D(ch)
                               for _ in range(blocks_per_stage)]))
            prev = ch

        # ---- Head ----
        self.pool    = nn.AdaptiveAvgPool3d(1)
        self.feat_bn = nn.BatchNorm1d(prev)
        self.dropout = nn.Dropout(dropout)
        self.proj    = nn.Linear(prev, out_dim)

        # Learnable temporal weights (softmax over T)
        self.time_logits = nn.Parameter(torch.zeros(T))

        self._init_conv_weights()

    def _init_conv_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def _encode_step(self, mri: Tensor) -> Tensor:
        if self.input_coding == 'bernoulli':
            return (torch.rand_like(mri) < mri.clamp(0, 1)).float()
        return mri

    def forward(self, mri: Tensor) -> Tensor:
        stem_mem: Optional[Tensor] = None
        down_mems: List[Optional[Tensor]] = [None] * len(self.downs)
        stage_mems: List[List[List[Optional[Tensor]]]] = [
            [[None, None] for _ in stage] for stage in self.stages
        ]

        step_feats: List[Tensor] = []
        fire_sum = 0.0
        fire_n = 0
        for _ in range(self.T):
            h = self._encode_step(mri)
            h, stem_mem = self.stem_lif(
                self.stem_norm(self.stem_conv(h)), stem_mem)
            fire_sum += float(h.mean().detach()); fire_n += 1

            for i, (down, stage) in enumerate(zip(self.downs, self.stages)):
                h, down_mems[i] = down(h, down_mems[i])
                for j, block in enumerate(stage):
                    h, stage_mems[i][j] = block(h, stage_mems[i][j])
            step_feats.append(self.pool(h).flatten(1))

        self.last_stem_fire_rate = fire_sum / max(fire_n, 1)

        feats = torch.stack(step_feats, dim=0)
        w = torch.softmax(self.time_logits, dim=0).view(-1, 1, 1)
        f = (feats * w).sum(dim=0)
        f = self.feat_bn(f)
        return self.proj(self.dropout(f))

# ──────────────────────────────────────────────────────────────
# Clinical tabular encoder
# ──────────────────────────────────────────────────────────────
class ClinicalMLP(nn.Module):
    """MLP encoder for the clinical/demographic features."""
    def __init__(self, n_features: int, out_dim: int = cfg.CLINICAL_FEAT_DIM, dropout: float = cfg.ENCODER_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LayerNorm(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.LayerNorm(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)