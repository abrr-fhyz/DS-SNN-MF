import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from torch import Tensor

from config import Config
from SNNModule import SNN3DEncoder, ClinicalMLP

cfg = Config()

# ──────────────────────────────────────────────────────────────
# Dempster-Shafer utilities over Θ = {CN=0, MCI=1, AD=2}
# Power set indices: 0=∅ 1={CN} 2={MCI} 3={CN,MCI} 4={AD} 5={CN,AD} 6={MCI,AD} 7=Θ
# ──────────────────────────────────────────────────────────────
EMPTY_SET  = 0b000
FULL_FRAME = 0b111
SINGLETONS = [0b001, 0b010, 0b100]   # {CN}, {MCI}, {AD} → indices 1, 2, 4

_COMBINATION_TABLE_CACHE: Dict[torch.device, Tuple[Tensor, Tensor]] = {}


def _build_combination_tables(device: torch.device) -> Tuple[Tensor, Tensor]:
    """Static lookup tables for the 7×7 non-empty-subset intersection matrix."""
    if device in _COMBINATION_TABLE_CACHE:
        return _COMBINATION_TABLE_CACHE[device]
    idx = torch.arange(1, 8, device=device)
    inter_table   = idx.unsqueeze(1) & idx.unsqueeze(0)     # (7, 7)
    conflict_mask = (inter_table == 0).float()               # disjoint pairs
    _COMBINATION_TABLE_CACHE[device] = (inter_table, conflict_mask)
    return inter_table, conflict_mask


def dempster_combine(m1: Tensor, m2: Tensor, use_yager: bool = False) -> Tuple[Tensor, Tensor]:
    """Combine two BBAs via Dempster's rule (normalised) or Yager's rule
    (conflict mass reassigned to Θ). Returns (m_combined, conflict K)."""
    inter_table, conflict_mask = _build_combination_tables(m1.device)
    prod = m1[:, 1:].unsqueeze(2) * m2[:, 1:].unsqueeze(1)   # (B, 7, 7)
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


# ──────────────────────────────────────────────────────────────
# DS evidence layer — feature vector → BBA via evidential (subjective-logic) head
# ──────────────────────────────────────────────────────────────
class DSEvidenceLayer(nn.Module):
    def __init__(self, feat_dim: int, n_classes: int = cfg.NUM_CLASSES,
                 evidence_cap: float = cfg.EVIDENCE_CAP,
                 evidence_scale: float = 1.0):
        super().__init__()
        self.n_classes      = n_classes
        self.evidence_cap   = evidence_cap
        self.evidence_scale = float(evidence_scale)

        # Small evidence head: feat → per-class non-negative evidence.
        hidden = max(feat_dim // 2, n_classes)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(cfg.ENCODER_DROPOUT),
            nn.Linear(hidden, n_classes),
        )
        self.register_buffer('singleton_indices', torch.tensor([1, 2, 4], dtype=torch.long))

    def forward(self, feat: Tensor) -> Tensor:
        """Evidential (subjective-logic) BBA — a valid Dempster-Shafer mass
        function with support only on singletons and Θ:

            e_c   = softplus(head(feat))                 (non-negative evidence)
            S     = Σ_c e_c + |Θ|                          (Dirichlet strength)
            m({c})= e_c / S,     m(Θ) = |Θ| / S

        Unlike the prototype-RBF layer, softplus(linear) can drive evidence
        large, so masses concentrate on the correct singleton (low Θ) when the
        source is informative and stay vacuous (high Θ) when it is not — which
        is exactly what makes cross-source conflict meaningful.
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


# ──────────────────────────────────────────────────────────────
# Conflict detection — soft Dempster↔Yager gate
# ──────────────────────────────────────────────────────────────
class ConflictDetectionModule(nn.Module):
    def __init__(self, kappa: float = cfg.CONFLICT_KAPPA, sharpness: float = 20.0):
        super().__init__()
        self.kappa = kappa
        self.sharpness = sharpness

    def forward(self, m1: Tensor, m2: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        K_pre = self._compute_conflict(m1, m2)
        m_dempster, _ = dempster_combine(m1, m2, use_yager=False)
        m_yager,    _ = dempster_combine(m1, m2, use_yager=True)
        gate  = torch.sigmoid(self.sharpness * (K_pre - self.kappa))
        m_out = gate.unsqueeze(1) * m_yager + (1.0 - gate.unsqueeze(1)) * m_dempster
        # Soft gate ∈ (0,1) is the true conflict utilization; hard flag uses a
        # higher reporting threshold so "conflict_rate" isn't ~100% of noise.
        report_kappa = float(getattr(cfg, 'CONFLICT_REPORT_KAPPA', self.kappa))
        return m_out, K_pre, (K_pre > report_kappa), gate

    @staticmethod
    def _compute_conflict(m1: Tensor, m2: Tensor) -> Tensor:
        _, conflict_mask = _build_combination_tables(m1.device)
        prod = m1[:, 1:].unsqueeze(2) * m2[:, 1:].unsqueeze(1)
        return (prod * conflict_mask.unsqueeze(0)).sum(dim=(1, 2))


# ──────────────────────────────────────────────────────────────
# Temporal accumulator (longitudinal DS combination across visits)
# ──────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────
# View extraction (unimodal MRI evidence sources)
# ──────────────────────────────────────────────────────────────
def extract_view(mri: Tensor, box: Tuple[float, ...],
                 out_shape: Tuple[int, int, int]) -> Tensor:
    """Crop a fractional box (z0,z1,y0,y1,x0,x1) from (B,C,D,H,W) and resize."""
    _, _, D, H, W = mri.shape
    z0, z1, y0, y1, x0, x1 = box
    d0, d1 = int(z0 * D), max(int(z1 * D), int(z0 * D) + 1)
    h0, h1 = int(y0 * H), max(int(y1 * H), int(y0 * H) + 1)
    w0, w1 = int(x0 * W), max(int(x1 * W), int(x0 * W) + 1)
    crop = mri[:, :, d0:d1, h0:h1, w0:w1]
    return F.interpolate(crop, size=out_shape, mode='trilinear', align_corners=False)


# ──────────────────────────────────────────────────────────────
# DS-SNN model: multi-view MRI (SNN) + clinical, fused with DS
# ──────────────────────────────────────────────────────────────
class DSSNMMF(nn.Module):
    """Dempster-Shafer fusion of multiple evidence sources:
      • shared (or per-view) 3D SNN encoder over MRI anatomical crops, and
      • one MLP over clinical/demographic features (if enabled).

    ``cfg.ABLATION`` selects the active sources:
      'full' | 'mri' | 'clinical'

    When ``cfg.AUX_CLINICAL_ONLY`` is True and clinical is active, the aux
    decision head sees *clinical features only*. MRI still contributes via
    DS BBAs / BetP, but cannot pollute the aux classifier through concat.
    """
    def __init__(self, n_clinical_features: int = len(cfg.CLINICAL_COLS),
                 n_classes: int = cfg.NUM_CLASSES):
        super().__init__()
        self.view_names   = list(cfg.MRI_VIEWS.keys())
        self.views        = dict(cfg.MRI_VIEWS)
        self.out_shape    = cfg.VIEW_SHAPE
        self.ablation     = getattr(cfg, 'ABLATION', 'full')
        self.use_mri      = self.ablation in ('full', 'mri')
        self.use_clinical = (cfg.USE_CLINICAL and self.ablation in ('full', 'clinical'))
        # Clinical-only aux when clinical is present; MRI-only ablation still
        # needs an MRI aux head (there is nothing else to decide from).
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
            self.ds_layers = nn.ModuleDict({
                name: DSEvidenceLayer(
                    feat_dim=cfg.VIEW_FEAT_DIM,
                    n_classes=n_classes,
                    evidence_cap=float(getattr(cfg, 'MRI_EVIDENCE_CAP', cfg.EVIDENCE_CAP)),
                    evidence_scale=float(getattr(cfg, 'MRI_EVIDENCE_SCALE', 1.0)),
                )
                for name in self.view_names
            })

        if self.use_clinical:
            self.clinical_encoder = ClinicalMLP(n_features=n_clinical_features)
            self.ds_clinical      = DSEvidenceLayer(
                feat_dim=cfg.CLINICAL_FEAT_DIM,
                n_classes=n_classes,
                evidence_cap=cfg.EVIDENCE_CAP,
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

    def _encode_view(self, name: str, view: Tensor) -> Tensor:
        if self.shared_encoder is not None:
            return self.shared_encoder(view)
        return self.encoders[name](view)

    def forward(self, mri: Optional[Tensor] = None,
                clinical: Optional[Tensor] = None) -> Dict[str, Tensor]:
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

        # Clinical-first: start from the stronger tabular BBA, then update with MRI.
        if getattr(cfg, 'CLINICAL_FIRST', False) and clin_bbas:
            bbas = clin_bbas + mri_bbas
        else:
            bbas = mri_bbas + clin_bbas
        assert bbas, "No evidence sources active — check cfg.ABLATION / USE_CLINICAL."

        m_combined = None
        # Track max pairwise K (interpretable vs κ) and sum (legacy total conflict).
        K_max = None
        K_sum = None
        flag = None
        gate_max = None
        for m_i in bbas:
            if m_combined is None:
                m_combined = m_i
                K_max = torch.zeros(m_i.shape[0], device=m_i.device)
                K_sum = torch.zeros(m_i.shape[0], device=m_i.device)
                flag  = torch.zeros(m_i.shape[0], dtype=torch.bool, device=m_i.device)
                gate_max = torch.zeros(m_i.shape[0], device=m_i.device)
            else:
                m_combined, K_new, flag_new, gate_new = self.cdm(m_combined, m_i)
                K_sum = K_sum + K_new
                K_max = torch.maximum(K_max, K_new)
                gate_max = torch.maximum(gate_max, gate_new)
                flag  = flag | flag_new

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
        blend = float(getattr(cfg, 'DECISION_BLEND', 0.0))
        blend = min(max(blend, 0.0), 1.0)
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
            'conflict_flag': flag,           # K > CONFLICT_REPORT_KAPPA
            'conflict_gate': gate_max,       # soft Yager weight in (0,1)
            'aux_logits':    aux_logits,
        }


# ──────────────────────────────────────────────────────────────
# Loss terms
# ──────────────────────────────────────────────────────────────
IGNORANCE_TARGET = {
    0: cfg.IGNORANCE_TARGET_CN,
    1: cfg.IGNORANCE_TARGET_MCI,
    2: cfg.IGNORANCE_TARGET_AD,
}


def evidential_utility_loss(bba_singletons: Tensor, labels: Tensor,
                            n_classes: int = cfg.NUM_CLASSES) -> Tensor:
    """Reward mass on the true singleton; penalise mass on wrong singletons.
    Does not penalise Θ mass, so ambiguous samples may route to ignorance."""
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


def ignorance_anchoring_loss(bba_ignorance: Tensor, labels: Tensor,
                             targets: Dict[int, float] = IGNORANCE_TARGET) -> Tensor:
    """Pull each class's Θ-mass toward its target corridor."""
    B = labels.shape[0]
    loss = torch.zeros(B, device=labels.device)
    for c, tau in targets.items():
        mask = (labels == c)
        if mask.sum() == 0:
            continue
        loss[mask] = (bba_ignorance[mask] - tau).pow(2)
    return loss.mean()


def focal_ce_from_probs(probs: Tensor, labels: Tensor, weight: Tensor,
                        gamma: float, label_smoothing: float) -> Tensor:
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


def focal_ce_from_logits(logits: Tensor, labels: Tensor, weight: Tensor,
                         gamma: float, label_smoothing: float) -> Tensor:
    """Class-weighted focal CE on logits (auxiliary head)."""
    log_p = F.log_softmax(logits, dim=1)
    probs = log_p.exp()
    nll = F.nll_loss(log_p, labels, weight=weight, reduction='none')
    if label_smoothing > 0:
        # Match PyTorch label_smoothing mixture on the log-softmax.
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


class DSSNNLoss(nn.Module):
    """L = L_CE(betp) + λ_aux·L_aux + λ_ds·L_DS + λ_k·L_K + λ_ign·L_ign.

    CE terms are class-weighted focal losses (γ from cfg.FOCAL_GAMMA). DS and
    ignorance terms warm up so early noisy BBAs don't dominate.
    """
    def __init__(self, class_weights: Optional[Tensor] = None):
        super().__init__()
        self.lambda_aux = cfg.LAMBDA_AUX
        self.lambda_ds  = cfg.LAMBDA_DS
        self.lambda_k   = cfg.LAMBDA_K
        self.lambda_ign = cfg.LAMBDA_IGN
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
