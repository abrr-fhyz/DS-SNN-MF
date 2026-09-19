# Dempster-Shafer utilities over Θ = {CN=0, MCI=1, AD=2}
# Power set indices: 0=∅ 1={CN} 2={MCI} 3={CN,MCI} 4={AD} 5={CN,AD} 6={MCI,AD} 7=Θ
# Positions inside the 7-wide mass tensors (m[:,1:]) are idx-1:
#   0=CN 1=MCI 2={CN,MCI} 3=AD 4={CN,AD} 5={MCI,AD} 6=Θ
# NOTE: on the ordinal frame (MD §B, "exploit ordinal structure"): CN, MCI, AD are adjacent stages along one severity axis, so {CN,AD} (idx 5 / pos 4) is clinically implausible as a resting place for conflict and is *never
# populated* by ordinal_yager_combine below — it stays structurally at 0 mass whenever that rule is used. It is kept in the power-set table purely so the classical 'yager'/'pcr5' rules stay usable unchanged for backward-compatible / ablation reruns.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from torch import Tensor
from config import Config
from SNNModule import SNN3DEncoder, ClinicalMLP

# ──────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ──────────────────────────────────────────────────────────────

cfg = Config()
EMPTY_SET  = 0b000
FULL_FRAME = 0b111
SINGLETONS = [0b001, 0b010, 0b100]   # {CN}, {MCI}, {AD}
POS_CN, POS_MCI, POS_CNMCI, POS_AD, POS_CNAD, POS_MCIAD, POS_THETA = 0, 1, 2, 3, 4, 5, 6
_COMBINATION_TABLE_CACHE: Dict[torch.device, Tuple[Tensor, Tensor]] = {}
IGNORANCE_TARGET = {
    0: cfg.IGNORANCE_TARGET_CN,
    1: cfg.IGNORANCE_TARGET_MCI,
    2: cfg.IGNORANCE_TARGET_AD,
}

# ──────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────

def _build_combination_tables(device: torch.device) -> Tuple[Tensor, Tensor]:
    """Static lookup tables for the 7×7 non-empty-subset intersection matrix."""
    if device in _COMBINATION_TABLE_CACHE:
        return _COMBINATION_TABLE_CACHE[device]
    idx = torch.arange(1, 8, device=device)
    inter_table   = idx.unsqueeze(1) & idx.unsqueeze(0)
    conflict_mask = (inter_table == 0).float()
    _COMBINATION_TABLE_CACHE[device] = (inter_table, conflict_mask)
    return inter_table, conflict_mask

def dempster_combine(m1: Tensor, m2: Tensor, use_yager: bool = False) -> Tuple[Tensor, Tensor]:
    """Combine two BBAs via Dempster or Yager. Returns (m_combined, K)"""
    inter_table, conflict_mask = _build_combination_tables(m1.device)
    prod = m1[:, 1:].unsqueeze(2) * m2[:, 1:].unsqueeze(1)
    K    = (prod * conflict_mask.unsqueeze(0)).sum(dim=(1, 2))

    m_combined = torch.zeros(m1.shape[0], 8, device=m1.device, dtype=m1.dtype)
    for i in range(1, 8):
        mask_i = (inter_table == i).float()
        m_combined[:, i] = (prod * mask_i.unsqueeze(0)).sum(dim=(1, 2))

    if use_yager:
        m_combined = m_combined.clone()
        m_combined[:, FULL_FRAME] = m_combined[:, FULL_FRAME] + K
    else:
        denom = (1.0 - K).clamp(min=1e-8).unsqueeze(1)
        m_combined = m_combined / denom
    return m_combined, K

def conflict_breakdown(m1: Tensor, m2: Tensor) -> Dict[str, Tensor]:
    """Per-class-pair decomposition of conflict mass K between two BBAs."""
    prod = m1[:, 1:].unsqueeze(2) * m2[:, 1:].unsqueeze(1)
    _, conflict_mask = _build_combination_tables(m1.device)
    k_cn_mci = prod[:, POS_CN, POS_MCI] + prod[:, POS_MCI, POS_CN]
    k_mci_ad = prod[:, POS_MCI, POS_AD] + prod[:, POS_AD, POS_MCI]
    k_cn_ad  = prod[:, POS_CN, POS_AD]  + prod[:, POS_AD, POS_CN]
    K_total  = (prod * conflict_mask.unsqueeze(0)).sum(dim=(1, 2))
    k_other  = (K_total - k_cn_mci - k_mci_ad - k_cn_ad).clamp(min=0.0)
    return {'cn_mci': k_cn_mci, 'mci_ad': k_mci_ad, 'cn_ad': k_cn_ad,
            'other': k_other, 'total': K_total}

def ordinal_yager_combine(m1: Tensor, m2: Tensor) -> Tuple[Tensor, Tensor]:
    """Conflict from an adjacent-stage singleton disagreement is redirected to the corresponding two-element focal set ({CN,MCI} or {MCI,AD}) instead of straight to Θ.
    Conflict from a CN↔AD disagreement (clinically implausible) and any conflict involving an already-compound focal element still redirects to Θ, matching classical Yager."""
    inter_table, conflict_mask = _build_combination_tables(m1.device)
    prod = m1[:, 1:].unsqueeze(2) * m2[:, 1:].unsqueeze(1)
    K = (prod * conflict_mask.unsqueeze(0)).sum(dim=(1, 2))

    m_out = torch.zeros(m1.shape[0], 8, device=m1.device, dtype=m1.dtype)
    for i in range(1, 8):
        mask_i = (inter_table == i).float()
        m_out[:, i] = (prod * mask_i.unsqueeze(0)).sum(dim=(1, 2))

    br = conflict_breakdown(m1, m2)
    m_out[:, POS_CNMCI + 1] = m_out[:, POS_CNMCI + 1] + br['cn_mci']
    m_out[:, POS_MCIAD + 1] = m_out[:, POS_MCIAD + 1] + br['mci_ad']
    m_out[:, FULL_FRAME]    = m_out[:, FULL_FRAME] + br['cn_ad'] + br['other']
    return m_out, K

def pcr5_combine(m1: Tensor, m2: Tensor) -> Tuple[Tensor, Tensor]:
    """Proportional Conflict Redistribution (Smarandache & Dezert).
    Non-conflicting focal elements combine as in the classical conjunctive rule:
        Δm(A) += m1(A)^2 * m2(B) / (m1(A) + m2(B))
        Δm(B) += m2(B)^2 * m1(A) / (m1(A) + m2(B))
    No renormalisation step is needed afterwards."""
    inter_table, conflict_mask = _build_combination_tables(m1.device)
    m1s, m2s = m1[:, 1:], m2[:, 1:]             
    prod = m1s.unsqueeze(2) * m2s.unsqueeze(1)

    m_out = torch.zeros(m1.shape[0], 8, device=m1.device, dtype=m1.dtype)
    for i in range(1, 8):
        mask_i = (inter_table == i).float()
        m_out[:, i] = (prod * mask_i.unsqueeze(0)).sum(dim=(1, 2))
    K = (prod * conflict_mask.unsqueeze(0)).sum(dim=(1, 2))

    denom = (m1s.unsqueeze(2) + m2s.unsqueeze(1)).clamp(min=1e-8)
    contrib_to_i = conflict_mask.unsqueeze(0) * (m1s.unsqueeze(2) ** 2) * m2s.unsqueeze(1) / denom
    contrib_to_j = conflict_mask.unsqueeze(0) * (m2s.unsqueeze(1) ** 2) * m1s.unsqueeze(2) / denom
    m_out[:, 1:] = m_out[:, 1:] + contrib_to_i.sum(dim=2) + contrib_to_j.sum(dim=1)
    return m_out, K

### NEEDS CHECKING
def discount_bba(m: Tensor, reliability) -> Tensor:
    """Shafer's classical discounting operator, applied BEFORE combination:

        m'(A) = r · m(A)   for A ≠ Θ
        m'(Θ) = 1 − r·(1 − m(Θ))  =  (1-r) + r·m(Θ)

    `reliability` may be a python float (applied to the whole batch, e.g. a
    fixed per-branch estimate from out-of-fold macro-F1) or a (B,) tensor for
    per-sample/per-fold reliability.

    NOTE: the MD write-up phrases this as "m'(A) = (1−r)·m(A)". Taken
    literally that inverts the intent — a *more* reliable branch (higher r,
    e.g. higher macro-F1) would then get *more* discounted. This
    implementation uses the standard Shafer convention (r=1 ⇒ no discount,
    r=0 ⇒ fully discounted to Θ) so that "estimate r from macro-F1" behaves
    the way the action item describes."""
    if isinstance(reliability, Tensor):
        r = reliability.clamp(0.0, 1.0).to(dtype=m.dtype, device=m.device).unsqueeze(1)
    else:
        r = float(max(0.0, min(1.0, reliability)))
    m_disc = m.clone()
    non_theta = m[:, 1:FULL_FRAME]                     
    m_disc[:, 1:FULL_FRAME] = non_theta * r
    m_disc[:, FULL_FRAME]   = 1.0 - (non_theta * r).sum(dim=1)
    return m_disc

def compute_belief_plausibility(m: Tensor) -> Tuple[Tensor, Tensor]:
    """Belief and plausibility for each singleton class."""
    bel = m[:, SINGLETONS]
    pl_parts = []
    for singleton in SINGLETONS:
        intersecting = [s for s in range(1, 8) if (s & singleton) != 0]
        pl_parts.append(m[:, intersecting].sum(dim=1, keepdim=True))
    pl = torch.cat(pl_parts, dim=1)
    return bel, pl

def pignistic_probability(m: Tensor) -> Tensor:
    """Smets' pignistic transform: BBA → probability distribution over classes."""
    BetP_CN  = m[:, 1] + m[:, 3] / 2 + m[:, 5] / 2 + m[:, 7] / 3
    BetP_MCI = m[:, 2] + m[:, 3] / 2 + m[:, 6] / 2 + m[:, 7] / 3
    BetP_AD  = m[:, 4] + m[:, 5] / 2 + m[:, 6] / 2 + m[:, 7] / 3
    return torch.stack([BetP_CN, BetP_MCI, BetP_AD], dim=1)

def evidential_utility_loss(bba_singletons: Tensor, labels: Tensor, n_classes: int = cfg.NUM_CLASSES) -> Tensor:
    """Reward mass on the true singleton; penalise mass on wrong singletons. Does not penalise Θ mass, so ambiguous samples may route to ignorance."""
    B = labels.shape[0]
    loss = torch.zeros(B, device=labels.device)
    for c in range(n_classes):
        mask = (labels == c)
        if mask.sum() == 0:
            continue
        m_correct = bba_singletons[mask, c]
        m_wrong   = bba_singletons[mask].clone()
        m_wrong[:, c] = 0.0
        loss[mask] = (1.0 - m_correct).pow(2) + m_wrong.pow(2).sum(dim=1)
    return loss.mean()

def conflict_regularisation_loss(K: Tensor, kappa: float = cfg.CONFLICT_KAPPA) -> Tensor:
    """Hinge on conflict above κ: L_K = mean(max(0, K − κ)²)."""
    return F.relu(K - kappa).pow(2).mean()

def ignorance_anchoring_loss(bba_ignorance: Tensor, labels: Tensor, targets: Dict[int, float] = IGNORANCE_TARGET) -> Tensor:
    """Pull each class's Θ-mass toward its target corridor."""
    B = labels.shape[0]
    loss = torch.zeros(B, device=labels.device)
    for c, tau in targets.items():
        mask = (labels == c)
        if mask.sum() == 0:
            continue
        loss[mask] = (bba_ignorance[mask] - tau).pow(2)
    return loss.mean()

def focal_ce_from_probs(probs: Tensor, labels: Tensor, weight: Tensor, gamma: float, label_smoothing: float) -> Tensor:
    """Class-weighted focal CE on a probability simplex (e.g. pignistic BetP)."""
    log_p = torch.log(probs.clamp(min=1e-8))
    nll = F.nll_loss(log_p, labels, weight=weight, reduction='none')
    if label_smoothing > 0:
        smooth = -(log_p.mean(dim=1)) * weight[labels]
        nll = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    if gamma and gamma > 0:
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
        nll = (1.0 - pt).pow(gamma) * nll
    return nll.mean()

def focal_ce_from_logits(logits: Tensor, labels: Tensor, weight: Tensor, gamma: float, label_smoothing: float) -> Tensor:
    """Class-weighted focal CE on logits (auxiliary head)."""
    log_p = F.log_softmax(logits, dim=1)
    probs = log_p.exp()
    nll = F.nll_loss(log_p, labels, weight=weight, reduction='none')
    if label_smoothing > 0:
        n_cls = logits.shape[1]
        with torch.no_grad():
            true_dist = torch.zeros_like(log_p)
            true_dist.fill_(label_smoothing / (n_cls - 1))
            true_dist.scatter_(1, labels.unsqueeze(1), 1.0 - label_smoothing)
        nll = -(true_dist * log_p).sum(dim=1) * weight[labels]
    if gamma and gamma > 0:
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
        nll = (1.0 - pt).pow(gamma) * nll
    return nll.mean()

def extract_view(mri: Tensor, box: Tuple[float, ...], out_shape: Tuple[int, int, int]) -> Tensor:
    """Crop a fractional box (z0,z1,y0,y1,x0,x1) from (B,C,D,H,W) and resize."""
    _, _, D, H, W = mri.shape
    z0, z1, y0, y1, x0, x1 = box
    d0, d1 = int(z0 * D), max(int(z1 * D), int(z0 * D) + 1)
    h0, h1 = int(y0 * H), max(int(y1 * H), int(y0 * H) + 1)
    w0, w1 = int(x0 * W), max(int(x1 * W), int(x0 * W) + 1)
    crop = mri[:, :, d0:d1, h0:h1, w0:w1]
    return F.interpolate(crop, size=out_shape, mode='trilinear', align_corners=False)

# ──────────────────────────────────────────────────────────────
# CLASSES
# ──────────────────────────────────────────────────────────────

class DSEvidenceLayer(nn.Module):
    def __init__(self, feat_dim: int, n_classes: int = cfg.NUM_CLASSES, evidence_cap: float = cfg.EVIDENCE_CAP, evidence_scale: float = 1.0):
        super().__init__()
        self.n_classes      = n_classes
        self.evidence_cap   = evidence_cap
        self.evidence_scale = float(evidence_scale)
        hidden = max(feat_dim // 2, n_classes)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(cfg.ENCODER_DROPOUT),
            nn.Linear(hidden, n_classes),
        )
        self.register_buffer('singleton_indices', torch.tensor([1, 2, 4], dtype=torch.long))

    def forward(self, feat: Tensor) -> Tensor:
        """Evidential (subjective-logic) BBA:

            e_c   = softplus(head(feat))                 (non-negative evidence)
            S     = Σ_c e_c + |Θ|                          (Dirichlet strength)
            m({c})= e_c / S,     m(Θ) = |Θ| / S
        """
        B = feat.shape[0]
        evidence = F.softplus(self.net(feat)) * self.evidence_scale   # (B, C)
        if self.evidence_cap and self.evidence_cap > 0:
            evidence = evidence.clamp(max=self.evidence_cap)
        S = evidence.sum(dim=1, keepdim=True) + self.n_classes       # (B, 1)
        m_singletons = evidence / S                                  # (B, C)
        m_theta      = self.n_classes / S                            # (B, 1)

        zeros = torch.zeros(B, 1, device=feat.device, dtype=feat.dtype)
        return torch.cat([
            zeros,                  # ∅
            m_singletons[:, 0:1],   # {CN}
            m_singletons[:, 1:2],   # {MCI}
            zeros,                  # {CN,MCI}
            m_singletons[:, 2:3],   # {AD}
            zeros,                  # {CN,AD}
            zeros,                  # {MCI,AD}
            m_theta,                # Θ
        ], dim=1)

class ConflictDetectionModule(nn.Module):
    def __init__(self, kappa: float = cfg.CONFLICT_KAPPA, sharpness: float = float(getattr(cfg, 'CONFLICT_GAMMA', 20.0)), rule: str = str(getattr(cfg, 'CONFLICT_RULE', 'ordinal'))):
        super().__init__()
        self.kappa = kappa
        self.sharpness = sharpness
        assert rule in ('yager', 'ordinal', 'pcr5'), f"Unknown CONFLICT_RULE: {rule!r}"
        self.rule = rule

    def forward(self, m1: Tensor, m2: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        K_pre = self._compute_conflict(m1, m2)
        m_dempster, _ = dempster_combine(m1, m2, use_yager=False)
        if self.rule == 'pcr5':
            m_alt, _ = pcr5_combine(m1, m2)
        elif self.rule == 'yager':
            m_alt, _ = dempster_combine(m1, m2, use_yager=True)
        else:
            m_alt, _ = ordinal_yager_combine(m1, m2)
        gate  = torch.sigmoid(self.sharpness * (K_pre - self.kappa))
        m_out = gate.unsqueeze(1) * m_alt + (1.0 - gate.unsqueeze(1)) * m_dempster
        report_kappa = float(getattr(cfg, 'CONFLICT_REPORT_KAPPA', self.kappa))
        return m_out, K_pre, (K_pre > report_kappa), gate

    @staticmethod
    def _compute_conflict(m1: Tensor, m2: Tensor) -> Tensor:
        _, conflict_mask = _build_combination_tables(m1.device)
        prod = m1[:, 1:].unsqueeze(2) * m2[:, 1:].unsqueeze(1)
        return (prod * conflict_mask.unsqueeze(0)).sum(dim=(1, 2))

class TemporalAccumulator:
    def __init__(self, batch_size: int, device: torch.device):
        self.device = device
        self.m_acc = torch.zeros(batch_size, 8, device=device)
        self.m_acc[:, FULL_FRAME] = 1.0
        self.visit_count = 0

    def accumulate(self, m_fused: Tensor) -> Tensor:
        m_new, _ = dempster_combine(self.m_acc, m_fused, use_yager=True)
        self.m_acc = m_new.detach()
        self.visit_count += 1
        return self.m_acc

class DSSNNLoss(nn.Module):
    """L = L_CE(betp) + λ_aux·L_aux + λ_ds·L_DS + λ_k·L_K + λ_ign·L_ign.

    CE terms are class-weighted focal losses (γ from cfg.FOCAL_GAMMA). DS and
    ignorance terms warm up so early noisy BBAs don't dominate.
    """
    def __init__(self, class_weights: Optional[Tensor] = None):
        super().__init__()
        # λ_ign / λ_aux are fusion-context terms. The Θ corridor in IGNORANCE_TARGET exists so a
        # branch keeps mass in reserve for the *other* source to act on downstream; a standalone
        # ablation has no other source, so enforcing the fused target pins a strong branch
        # (clinical) below what its data supports and smears BetP toward 1/3-1/3-1/3. Decoupled
        # here the same way the evidence cap is decoupled below. Defaults preserve old behaviour
        # for λ_aux and drop the ignorance tax for λ_ign.
        self.ablation   = getattr(cfg, 'ABLATION', 'full')
        self.standalone = (self.ablation != 'full')
        self.lambda_aux = (float(getattr(cfg, 'LAMBDA_AUX_STANDALONE', cfg.LAMBDA_AUX))
                           if self.standalone else cfg.LAMBDA_AUX)
        self.lambda_ds  = cfg.LAMBDA_DS
        self.lambda_k   = cfg.LAMBDA_K
        self.lambda_ign = (float(getattr(cfg, 'LAMBDA_IGN_STANDALONE', 0.0))
                           if self.standalone else cfg.LAMBDA_IGN)
        self.kappa      = cfg.CONFLICT_KAPPA
        self.label_smoothing = cfg.LABEL_SMOOTHING
        self.focal_gamma     = cfg.FOCAL_GAMMA
        self.warmup_epochs   = max(cfg.NUM_EPOCHS // 10, 3)
        self.current_epoch   = 0

        if class_weights is None:
            class_weights = torch.ones(cfg.NUM_CLASSES)
        self.register_buffer('class_weights', class_weights.float())

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def forward(self, out: Dict[str, Tensor], labels: Tensor) -> Dict[str, Tensor]:
        m_fused        = out['m_fused']
        bba_singletons = m_fused[:, SINGLETONS]
        bba_ignorance  = m_fused[:, FULL_FRAME]

        L_ce  = focal_ce_from_probs(out['betp'], labels, self.class_weights,
                                    self.focal_gamma, self.label_smoothing)
        L_aux = focal_ce_from_logits(out['aux_logits'], labels, self.class_weights,
                                     self.focal_gamma, self.label_smoothing)
        L_ds  = evidential_utility_loss(bba_singletons, labels)
        L_k   = conflict_regularisation_loss(out['K'], kappa=self.kappa)
        L_ign = ignorance_anchoring_loss(bba_ignorance, labels)

        warmup = min(1.0, (self.current_epoch + 1) / max(1, self.warmup_epochs))
        L_total = (L_ce
                   + self.lambda_aux * L_aux
                   + self.lambda_ds  * warmup * L_ds
                   + self.lambda_k   * L_k
                   + self.lambda_ign * warmup * L_ign)

        return {'total': L_total, 'ce': L_ce, 'aux': L_aux,
                'ds': L_ds, 'conflict': L_k, 'ign': L_ign}

# ──────────────────────────────────────────────────────────────
# MAIN MODULE
# ──────────────────────────────────────────────────────────────

class DSSNMMF(nn.Module):
    """Dempster-Shafer fusion of multiple evidence sources:
    cfg.ABLATION selects the active sources: 'full' | 'mri' | 'clinical'
    """
    def __init__(self, n_clinical_features: int = len(cfg.CLINICAL_COLS), n_classes: int = cfg.NUM_CLASSES):
        super().__init__()
        self.view_names   = list(cfg.MRI_VIEWS.keys())
        self.views        = dict(cfg.MRI_VIEWS)
        self.out_shape    = cfg.VIEW_SHAPE
        self.ablation     = getattr(cfg, 'ABLATION', 'full')
        self.use_mri      = self.ablation in ('full', 'mri')
        self.use_clinical = (cfg.USE_CLINICAL and self.ablation in ('full', 'clinical'))
        self.aux_clinical_only = bool(getattr(cfg, 'AUX_CLINICAL_ONLY', True)) and self.use_clinical

        if self.use_mri:
            in_ch = int(cfg.mri_in_channels()) if hasattr(cfg, 'mri_in_channels') else 1
            if cfg.SHARE_MRI_ENCODER:
                self.shared_encoder = SNN3DEncoder(in_ch=in_ch, out_dim=cfg.VIEW_FEAT_DIM)
                self.encoders = None
            else:
                self.shared_encoder = None
                self.encoders = nn.ModuleDict({
                    name: SNN3DEncoder(in_ch=in_ch, out_dim=cfg.VIEW_FEAT_DIM)
                    for name in self.view_names
                })
            mri_cap = float(getattr(cfg, 'MRI_EVIDENCE_CAP', cfg.EVIDENCE_CAP))
            if self.ablation == 'mri':
                mri_cap = float(getattr(cfg, 'MRI_EVIDENCE_CAP_STANDALONE', mri_cap))
            self.ds_layers = nn.ModuleDict({
                name: DSEvidenceLayer(
                    feat_dim=cfg.VIEW_FEAT_DIM,
                    n_classes=n_classes,
                    evidence_cap=mri_cap,
                    evidence_scale=float(getattr(cfg, 'MRI_EVIDENCE_SCALE', 1.0)),
                )
                for name in self.view_names
            })

        if self.use_clinical:
            self.clinical_encoder = ClinicalMLP(n_features=n_clinical_features)
            clin_cap = float(cfg.EVIDENCE_CAP)
            if self.ablation == 'clinical':
                clin_cap = float(getattr(cfg, 'CLINICAL_EVIDENCE_CAP_STANDALONE', clin_cap))
            self.ds_clinical      = DSEvidenceLayer(
                feat_dim=cfg.CLINICAL_FEAT_DIM,
                n_classes=n_classes,
                evidence_cap=clin_cap,
                evidence_scale=1.0,
            )

        if self.aux_clinical_only:
            aux_in = cfg.CLINICAL_FEAT_DIM
        elif self.use_mri and self.use_clinical:
            aux_in = len(self.view_names) * cfg.VIEW_FEAT_DIM + cfg.CLINICAL_FEAT_DIM
        elif self.use_mri:
            aux_in = len(self.view_names) * cfg.VIEW_FEAT_DIM
        else:
            aux_in = cfg.CLINICAL_FEAT_DIM

        self.cdm = ConflictDetectionModule(kappa=cfg.CONFLICT_KAPPA)
        self.aux_head = nn.Sequential(
            nn.Dropout(cfg.ENCODER_DROPOUT),
            nn.Linear(max(aux_in, 1), n_classes),
        )

        # Pre-fusion discounting (MD §B bullet 1): {view_name or 'clinical': r}
        self.reliability_cfg = dict(getattr(cfg, 'BRANCH_RELIABILITY', {}) or {})

        # K-aware decision blend (MD §C bullet 2) + ablation-purity fix.
        self.ablation_pure_decision = bool(getattr(cfg, 'ABLATION_PURE_DECISION', True))
        self.decision_blend_base = min(max(float(getattr(cfg, 'DECISION_BLEND', 0.0)), 0.0), 1.0)
        self.decision_blend_max  = min(max(float(getattr(cfg, 'DECISION_BLEND_MAX',
                                                           self.decision_blend_base)), 0.0), 1.0)
        self.decision_blend_kappa = float(getattr(cfg, 'DECISION_BLEND_KAPPA', cfg.CONFLICT_KAPPA))
        self.decision_blend_sharpness = float(getattr(
            cfg, 'DECISION_BLEND_SHARPNESS', float(getattr(cfg, 'CONFLICT_GAMMA', 20.0))))

    def _encode_view(self, name: str, view: Tensor) -> Tensor:
        if self.shared_encoder is not None:
            return self.shared_encoder(view)
        return self.encoders[name](view)

    def _fold(self, bbas: List[Tensor]):
        """Sequentially fuse a list of ≥1 same-tier BBAs with the gated CDM."""
        m_combined = bbas[0]
        B = m_combined.shape[0]
        K_max = torch.zeros(B, device=m_combined.device)
        K_sum = torch.zeros(B, device=m_combined.device)
        flag  = torch.zeros(B, dtype=torch.bool, device=m_combined.device)
        gate_max = torch.zeros(B, device=m_combined.device)
        for m_i in bbas[1:]:
            m_combined, K_new, flag_new, gate_new = self.cdm(m_combined, m_i)
            K_sum = K_sum + K_new
            K_max = torch.maximum(K_max, K_new)
            gate_max = torch.maximum(gate_max, gate_new)
            flag = flag | flag_new
        return m_combined, K_max, K_sum, flag, gate_max

    def forward(self, mri: Optional[Tensor] = None, clinical: Optional[Tensor] = None) -> Dict[str, Tensor]:
        mri_bbas, clin_bbas = [], []
        mri_feats, clin_feats = [], []

        if self.use_mri:
            assert mri is not None and mri.numel() > 0, "MRI volume is required."
            for name in self.view_names:
                view = extract_view(mri, self.views[name], self.out_shape)
                feat = self._encode_view(name, view)
                mri_feats.append(feat)
                mri_bbas.append(self.ds_layers[name](feat))

        if self.use_clinical:
            assert clinical is not None and clinical.numel() > 0, "Clinical features required."
            cfeat = self.clinical_encoder(clinical)
            clin_feats.append(cfeat)
            clin_bbas.append(self.ds_clinical(cfeat))

        # Pre-fusion discounting by reliability (MD §B bullet 1), before any combination.
        # Only meaningful when ≥2 BBAs are actually combined: discounting a lone source is a pure
        # affine flattening of BetP (BetP' = r·BetP + (1-r)/3), so it cannot change the argmax but
        # does inflate Θ, which then distorts L_ce / L_ign. Third fusion-context leak, same family
        # as λ_ign and the evidence cap.
        if self.reliability_cfg and (len(mri_bbas) + len(clin_bbas)) > 1:
            if self.use_mri:
                mri_bbas = [
                    discount_bba(bba, self.reliability_cfg[name]) if name in self.reliability_cfg else bba
                    for name, bba in zip(self.view_names, mri_bbas)
                ]
            if self.use_clinical and clin_bbas and 'clinical' in self.reliability_cfg:
                clin_bbas = [discount_bba(clin_bbas[0], self.reliability_cfg['clinical'])]

        assert mri_bbas or clin_bbas, "No evidence sources active — check cfg.ABLATION / USE_CLINICAL."

        # Stage 1: consolidate same-modality MRI-ROI branches.
        mri_consolidated, K_mri_max, K_mri_sum, flag_mri, gate_mri_max = (
            self._fold(mri_bbas) if mri_bbas else (None, None, None, None, None)
        )
        clin_bba = clin_bbas[0] if clin_bbas else None

        # Stage 2: consolidated MRI BBA × clinical BBA.
        K_breakdown = None
        if mri_consolidated is not None and clin_bba is not None:
            m_combined, K_new, flag_new, gate_new = self.cdm(mri_consolidated, clin_bba)
            K_breakdown = conflict_breakdown(mri_consolidated, clin_bba)
            K_max = torch.maximum(K_mri_max, K_new)
            K_sum = K_mri_sum + K_new
            flag  = flag_mri | flag_new
            gate_max = torch.maximum(gate_mri_max, gate_new)
        elif mri_consolidated is not None:
            m_combined, K_max, K_sum, flag, gate_max = mri_consolidated, K_mri_max, K_mri_sum, flag_mri, gate_mri_max
        else:
            m_combined = clin_bba
            B = m_combined.shape[0]
            K_max = torch.zeros(B, device=m_combined.device)
            K_sum = torch.zeros(B, device=m_combined.device)
            flag  = torch.zeros(B, dtype=torch.bool, device=m_combined.device)
            gate_max = torch.zeros(B, device=m_combined.device)

        bel, pl = compute_belief_plausibility(m_combined)
        betp = pignistic_probability(m_combined)

        # Aux input: clinical-only when configured (full model); else all feats.
        if self.aux_clinical_only:
            aux_in_feats = clin_feats
        else:
            aux_in_feats = mri_feats + clin_feats
        assert aux_in_feats, "No features for aux head."
        aux_logits = self.aux_head(torch.cat(aux_in_feats, dim=1))
        aux_p = F.softmax(aux_logits, dim=1)

        # Decision blend: K-aware (MD §C bullet 2) + ablation-purity.
        if self.ablation != 'full' and self.ablation_pure_decision:
            blend = torch.zeros(m_combined.shape[0], 1, device=m_combined.device, dtype=m_combined.dtype)
        else:
            blend_gate = torch.sigmoid(self.decision_blend_sharpness * (K_max - self.decision_blend_kappa))
            blend = self.decision_blend_base + (self.decision_blend_max - self.decision_blend_base) * blend_gate
            blend = blend.clamp(0.0, 1.0).unsqueeze(1)
        decision = (1.0 - blend) * betp + blend * aux_p

        return {
            'm_fused':       m_combined,
            'bel':           bel,
            'pl':            pl,
            'betp':          betp,
            'decision':      decision,
            'pred':          decision.argmax(dim=1),
            'K':             K_max,          # meanK in logs = mean max-pairwise K
            'K_sum':         K_sum,
            'K_breakdown':   K_breakdown,    # per-class-pair K at the MRI×clinical stage, or None
            'conflict_flag': flag,           # K > CONFLICT_REPORT_KAPPA
            'conflict_gate': gate_max,       # soft high-conflict-rule weight in (0,1)
            'decision_blend': blend.squeeze(1),  # actual per-sample blend used, for logging/audit
            'aux_logits':    aux_logits,
        }