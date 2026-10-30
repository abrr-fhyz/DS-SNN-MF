"""Dempster-Shafer algebra over the frame Θ = {CN, MCI, AD}.

Subsets are encoded as bitmasks (CN=1, MCI=2, AD=4). A BBA is a (B, 8) tensor indexed by
bitmask: 0=∅ 1={CN} 2={MCI} 3={CN,MCI} 4={AD} 5={CN,AD} 6={MCI,AD} 7=Θ, with column 0 always 0.

CN, MCI, AD lie on one severity axis, so on the ordinal frame {CN,AD} is implausible: the
``ordinal`` rule never assigns mass to it. It stays in the table so ``yager`` / ``pcr5`` work
unchanged.
"""
from typing import Dict, NamedTuple, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import cfg

FULL_FRAME = 0b111
SINGLETONS = [0b001, 0b010, 0b100]
POS_CN, POS_MCI, POS_CNMCI, POS_AD, POS_CNAD, POS_MCIAD, POS_THETA = range(7)  # index in m[:, 1:]


class _Tables(NamedTuple):
    intersection: Tensor   # (7, 7, 8) one-hot of A∩B over the 8 bitmasks
    conflict: Tensor       # (7, 7) 1 where A∩B = ∅
    plausibility: Tensor   # (8, 3) m @ . = Pl of each singleton
    pignistic: Tensor      # (8, 3) m @ . = BetP


_TABLES: Dict[torch.device, _Tables] = {}


def _tables(device: torch.device) -> _Tables:
    if device not in _TABLES:
        idx = torch.arange(1, 8, device=device)
        inter = idx[:, None] & idx[None, :]
        subsets = torch.arange(8, device=device)
        member = torch.stack([(subsets & s) != 0 for s in SINGLETONS], dim=1).float()   # (8, 3)
        _TABLES[device] = _Tables(
            intersection=F.one_hot(inter, 8).float(),
            conflict=(inter == 0).float(),
            plausibility=member,
            pignistic=member / member.sum(dim=1, keepdim=True).clamp(min=1.0),
        )
    return _TABLES[device]


# ──────────────────────────────────────────────────────────────
# Combination rules
# ──────────────────────────────────────────────────────────────

def _conjunctive(m1: Tensor, m2: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Unnormalised conjunctive pooling. Returns (pairwise products (B,7,7), mass (B,8) with
    column 0 zeroed, conflict K)."""
    tables = _tables(m1.device)
    prod = m1[:, 1:].unsqueeze(2) * m2[:, 1:].unsqueeze(1)
    pooled = torch.einsum('bij,ijc->bc', prod, tables.intersection.to(prod.dtype))
    K = pooled[:, 0]
    mass = torch.cat([torch.zeros_like(K).unsqueeze(1), pooled[:, 1:]], dim=1)
    return prod, mass, K


def dempster_combine(m1: Tensor, m2: Tensor, use_yager: bool = False) -> Tuple[Tensor, Tensor]:
    """Dempster's rule (renormalise by 1−K) or Yager's (K moved onto Θ). Returns (m, K)."""
    _, mass, K = _conjunctive(m1, m2)
    if use_yager:
        mass[:, FULL_FRAME] = mass[:, FULL_FRAME] + K
        return mass, K
    return mass / (1.0 - K).clamp(min=1e-8).unsqueeze(1), K


def conflict_breakdown(m1: Tensor, m2: Tensor) -> Dict[str, Tensor]:
    """Conflict mass K split by class pair (singleton disagreements) plus everything else."""
    prod, _, K = _conjunctive(m1, m2)
    cn_mci = prod[:, POS_CN, POS_MCI] + prod[:, POS_MCI, POS_CN]
    mci_ad = prod[:, POS_MCI, POS_AD] + prod[:, POS_AD, POS_MCI]
    cn_ad = prod[:, POS_CN, POS_AD] + prod[:, POS_AD, POS_CN]
    other = (K - cn_mci - mci_ad - cn_ad).clamp(min=0.0)
    return {'cn_mci': cn_mci, 'mci_ad': mci_ad, 'cn_ad': cn_ad, 'other': other, 'total': K}


def ordinal_yager_combine(m1: Tensor, m2: Tensor) -> Tuple[Tensor, Tensor]:
    """Yager variant for the ordinal frame: CN↔MCI conflict goes to {CN,MCI}, MCI↔AD conflict to
    {MCI,AD}; CN↔AD and conflicts involving compound sets go to Θ."""
    _, mass, K = _conjunctive(m1, m2)
    br = conflict_breakdown(m1, m2)
    mass[:, POS_CNMCI + 1] = mass[:, POS_CNMCI + 1] + br['cn_mci']
    mass[:, POS_MCIAD + 1] = mass[:, POS_MCIAD + 1] + br['mci_ad']
    mass[:, FULL_FRAME] = mass[:, FULL_FRAME] + br['cn_ad'] + br['other']
    return mass, K


def pcr5_combine(m1: Tensor, m2: Tensor) -> Tuple[Tensor, Tensor]:
    """Proportional Conflict Redistribution 5 (Smarandache & Dezert). Each conflicting pair
    (A, B) returns its product mass to A and B proportionally to m1(A)² m2(B)/(m1(A)+m2(B)) and
    m2(B)² m1(A)/(m1(A)+m2(B)); no renormalisation is needed."""
    conflict = _tables(m1.device).conflict.unsqueeze(0)
    a, b = m1[:, 1:].unsqueeze(2), m2[:, 1:].unsqueeze(1)
    _, mass, K = _conjunctive(m1, m2)
    denom = (a + b).clamp(min=1e-8)
    to_a = (conflict * a.pow(2) * b / denom).sum(dim=2)
    to_b = (conflict * b.pow(2) * a / denom).sum(dim=1)
    mass[:, 1:] = mass[:, 1:] + to_a + to_b
    return mass, K


def discount_bba(m: Tensor, reliability) -> Tensor:
    """Shafer discounting before combination: m'(A) = r·m(A) for A ≠ Θ, remainder moved to Θ.
    r=1 leaves the BBA unchanged, r=0 yields the vacuous BBA. ``reliability`` is a float or a (B,) tensor."""
    if isinstance(reliability, Tensor):
        r = reliability.clamp(0.0, 1.0).to(dtype=m.dtype, device=m.device).unsqueeze(1)
    else:
        r = float(min(max(reliability, 0.0), 1.0))
    out = m.clone()
    scaled = m[:, 1:FULL_FRAME] * r
    out[:, 1:FULL_FRAME] = scaled
    out[:, FULL_FRAME] = 1.0 - scaled.sum(dim=1)
    return out


# ──────────────────────────────────────────────────────────────
# Decision quantities
# ──────────────────────────────────────────────────────────────

def compute_belief_plausibility(m: Tensor) -> Tuple[Tensor, Tensor]:
    """Belief and plausibility of each singleton class."""
    return m[:, SINGLETONS], m @ _tables(m.device).plausibility.to(m.dtype)


def pignistic_probability(m: Tensor) -> Tensor:
    """Smets' pignistic transform: BBA → class probabilities."""
    return m @ _tables(m.device).pignistic.to(m.dtype)


def vacuous_bba(batch: int, device: torch.device) -> Tensor:
    m = torch.zeros(batch, 8, device=device)
    m[:, FULL_FRAME] = 1.0
    return m


# ──────────────────────────────────────────────────────────────
# Modules
# ──────────────────────────────────────────────────────────────

class DSEvidenceLayer(nn.Module):
    """Feature vector → subjective-logic BBA.

    e_c = softplus(head(feat)) (capped, scaled); S = Σe_c + |Θ|; m({c}) = e_c/S; m(Θ) = |Θ|/S.
    """

    def __init__(self, feat_dim: int, n_classes: int = cfg.NUM_CLASSES,
                 evidence_cap: float = cfg.EVIDENCE_CAP, evidence_scale: float = 1.0):
        super().__init__()
        self.n_classes = n_classes
        self.evidence_cap = evidence_cap
        self.evidence_scale = float(evidence_scale)
        hidden = max(feat_dim // 2, n_classes)
        self.net = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(),
                                 nn.Dropout(cfg.ENCODER_DROPOUT), nn.Linear(hidden, n_classes))
        self.register_buffer('singleton_indices', torch.tensor(SINGLETONS, dtype=torch.long))

    def forward(self, feat: Tensor) -> Tensor:
        evidence = F.softplus(self.net(feat)) * self.evidence_scale
        if self.evidence_cap and self.evidence_cap > 0:
            evidence = evidence.clamp(max=self.evidence_cap)
        strength = evidence.sum(dim=1, keepdim=True) + self.n_classes
        m = torch.zeros(feat.shape[0], 8, device=feat.device, dtype=feat.dtype)
        m[:, self.singleton_indices] = evidence / strength
        m[:, FULL_FRAME] = (self.n_classes / strength).squeeze(1)
        return m


class ConflictDetectionModule(nn.Module):
    """Gated fusion of two BBAs: Dempster at low conflict, ``rule`` at high conflict, blended by
    sigmoid(γ(K − κ))."""

    def __init__(self, kappa: float = cfg.CONFLICT_KAPPA, sharpness: float = cfg.CONFLICT_GAMMA,
                 rule: str = cfg.CONFLICT_RULE):
        super().__init__()
        assert rule in ('yager', 'ordinal', 'pcr5'), f"Unknown CONFLICT_RULE: {rule!r}"
        self.kappa, self.sharpness, self.rule = kappa, sharpness, rule

    def forward(self, m1: Tensor, m2: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns (fused BBA, K, K > report threshold, soft gate weight)."""
        m_dempster, K = dempster_combine(m1, m2)
        if self.rule == 'pcr5':
            m_alt, _ = pcr5_combine(m1, m2)
        elif self.rule == 'yager':
            m_alt, _ = dempster_combine(m1, m2, use_yager=True)
        else:
            m_alt, _ = ordinal_yager_combine(m1, m2)
        gate = torch.sigmoid(self.sharpness * (K - self.kappa))
        m_out = gate.unsqueeze(1) * m_alt + (1.0 - gate.unsqueeze(1)) * m_dempster
        return m_out, K, K > cfg.CONFLICT_REPORT_KAPPA, gate


class TemporalAccumulator:
    """Sequential Dempster accumulation of per-visit BBAs from a vacuous prior."""

    def __init__(self, batch_size: int, device: torch.device):
        self.m_acc = vacuous_bba(batch_size, device)

    def accumulate(self, m_fused: Tensor, mask: Tensor = None, use_yager: bool = False) -> Tensor:
        """Fold ``m_fused`` into the running BBA; rows where ``mask`` is False are left unchanged."""
        m_new, _ = dempster_combine(self.m_acc, m_fused, use_yager=use_yager)
        if mask is not None:
            m_new = torch.where(mask.unsqueeze(1), m_new, self.m_acc)
        self.m_acc = m_new.detach()
        return self.m_acc
