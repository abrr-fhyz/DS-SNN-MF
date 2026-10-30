"""DS-SNN-MF: spiking MRI encoders and a clinical MLP, fused with Dempster-Shafer theory.

    MRI views ─ SNN ─ evidential head ─┐ (BBAs folded with the gated rule)
                                       ├─ gated combine ─ fused BBA ─ BetP ─┐
    clinical ─ MLP ─ evidential head ──┘                                    ├─ decision
                       clinical-only aux head (softmax) ── K-aware blend ──┘

``cfg.ABLATION`` selects the active sources: 'full' | 'mri' | 'clinical'.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import cfg
from DSModule import (ConflictDetectionModule, DSEvidenceLayer, compute_belief_plausibility,
                      conflict_breakdown, discount_bba, pignistic_probability)
from SNNModule import ClinicalMLP, SNN3DEncoder


def extract_view(mri: Tensor, box: Tuple[float, ...], out_shape: Tuple[int, int, int]) -> Tensor:
    """Crop the fractional box (z0, z1, y0, y1, x0, x1) from (B, C, D, H, W) and resize."""
    sizes = mri.shape[2:]
    slices = []
    for axis, size in enumerate(sizes):
        lo, hi = box[2 * axis], box[2 * axis + 1]
        start = int(lo * size)
        slices.append(slice(start, max(int(hi * size), start + 1)))
    crop = mri[:, :, slices[0], slices[1], slices[2]]
    return F.interpolate(crop, size=out_shape, mode='trilinear', align_corners=False)


class DSSNMMF(nn.Module):
    def __init__(self, n_clinical_features: Optional[int] = None, n_classes: int = cfg.NUM_CLASSES):
        super().__init__()
        n_clinical = n_clinical_features or len(cfg.CLINICAL_COLS)
        self.view_names = list(cfg.MRI_VIEWS)
        self.views = dict(cfg.MRI_VIEWS)
        self.out_shape = cfg.VIEW_SHAPE
        self.ablation = cfg.ABLATION
        self.use_mri = self.ablation in ('full', 'mri')
        self.use_clinical = cfg.USE_CLINICAL and self.ablation in ('full', 'clinical')
        self.aux_clinical_only = cfg.AUX_CLINICAL_ONLY and self.use_clinical
        self.reliability = dict(cfg.BRANCH_RELIABILITY)

        if self.use_mri:
            self._build_mri_branch(n_classes)
        if self.use_clinical:
            self.clinical_encoder = ClinicalMLP(n_features=n_clinical)
            cap = cfg.CLINICAL_EVIDENCE_CAP_STANDALONE if self.ablation == 'clinical' else cfg.EVIDENCE_CAP
            self.ds_clinical = DSEvidenceLayer(cfg.CLINICAL_FEAT_DIM, n_classes, evidence_cap=cap)

        if self.aux_clinical_only or not self.use_mri:
            aux_in = cfg.CLINICAL_FEAT_DIM
        else:
            aux_in = len(self.view_names) * cfg.VIEW_FEAT_DIM + (cfg.CLINICAL_FEAT_DIM if self.use_clinical else 0)
        self.aux_head = nn.Sequential(nn.Dropout(cfg.ENCODER_DROPOUT), nn.Linear(aux_in, n_classes))
        self.cdm = ConflictDetectionModule()

        self.pure_decision = self.ablation != 'full' and cfg.ABLATION_PURE_DECISION
        self.blend_base = min(max(cfg.DECISION_BLEND, 0.0), 1.0)
        self.blend_max = min(max(cfg.DECISION_BLEND_MAX, 0.0), 1.0)

    def _build_mri_branch(self, n_classes: int) -> None:
        in_ch = cfg.mri_in_channels()
        if cfg.SHARE_MRI_ENCODER:
            self.shared_encoder = SNN3DEncoder(in_ch=in_ch, out_dim=cfg.VIEW_FEAT_DIM)
            self.encoders = None
        else:
            self.shared_encoder = None
            self.encoders = nn.ModuleDict({n: SNN3DEncoder(in_ch=in_ch, out_dim=cfg.VIEW_FEAT_DIM)
                                           for n in self.view_names})
        cap = cfg.MRI_EVIDENCE_CAP_STANDALONE if self.ablation == 'mri' else cfg.MRI_EVIDENCE_CAP
        self.ds_layers = nn.ModuleDict({
            n: DSEvidenceLayer(cfg.VIEW_FEAT_DIM, n_classes, evidence_cap=cap,
                               evidence_scale=cfg.MRI_EVIDENCE_SCALE)
            for n in self.view_names})

    # ── fusion ────────────────────────────────────────────────
    def _fold(self, bbas: List[Tensor]):
        """Sequentially fuse same-modality BBAs. Returns (m, K_max, K_sum, flag, gate_max)."""
        m = bbas[0]
        zeros = torch.zeros(m.shape[0], device=m.device)
        K_max, K_sum, gate_max = zeros, zeros, zeros
        flag = torch.zeros(m.shape[0], dtype=torch.bool, device=m.device)
        for other in bbas[1:]:
            m, K, flag_new, gate = self.cdm(m, other)
            K_sum, K_max = K_sum + K, torch.maximum(K_max, K)
            gate_max, flag = torch.maximum(gate_max, gate), flag | flag_new
        return m, K_max, K_sum, flag, gate_max

    def _fuse(self, mri_bbas: List[Tensor], clin_bbas: List[Tensor]) -> dict:
        """MRI views are consolidated first (positively correlated), then fused with clinical."""
        breakdown = None
        if mri_bbas:
            m, K_max, K_sum, flag, gate = self._fold(mri_bbas)
        if clin_bbas and mri_bbas:
            m_new, K, flag_new, gate_new = self.cdm(m, clin_bbas[0])
            breakdown = conflict_breakdown(m, clin_bbas[0])
            m, K_max, K_sum = m_new, torch.maximum(K_max, K), K_sum + K
            flag, gate = flag | flag_new, torch.maximum(gate, gate_new)
        elif clin_bbas:
            m = clin_bbas[0]
            zeros = torch.zeros(m.shape[0], device=m.device)
            K_max, K_sum, gate = zeros, zeros, zeros
            flag = torch.zeros(m.shape[0], dtype=torch.bool, device=m.device)
        return {'m': m, 'K_max': K_max, 'K_sum': K_sum, 'flag': flag, 'gate': gate, 'breakdown': breakdown}

    def _discount(self, mri_bbas: List[Tensor], clin_bbas: List[Tensor]):
        """Shafer discounting by branch reliability; only meaningful when ≥2 sources are combined
        (discounting a lone source cannot change its argmax but inflates Θ)."""
        if not self.reliability or len(mri_bbas) + len(clin_bbas) < 2:
            return mri_bbas, clin_bbas
        mri_bbas = [discount_bba(b, self.reliability[n]) if n in self.reliability else b
                    for n, b in zip(self.view_names, mri_bbas)]
        if clin_bbas and 'clinical' in self.reliability:
            clin_bbas = [discount_bba(clin_bbas[0], self.reliability['clinical'])]
        return mri_bbas, clin_bbas

    def _blend(self, K_max: Tensor) -> Tensor:
        """Weight of the aux head in the decision, rising with conflict."""
        if self.pure_decision:
            return torch.zeros_like(K_max).unsqueeze(1)
        gate = torch.sigmoid(cfg.DECISION_BLEND_SHARPNESS * (K_max - cfg.DECISION_BLEND_KAPPA))
        return (self.blend_base + (self.blend_max - self.blend_base) * gate).clamp(0.0, 1.0).unsqueeze(1)

    # ── forward ───────────────────────────────────────────────
    def forward(self, mri: Optional[Tensor] = None, clinical: Optional[Tensor] = None) -> Dict[str, Tensor]:
        mri_feats, mri_bbas, clin_feats, clin_bbas = [], [], [], []
        if self.use_mri:
            assert mri is not None and mri.numel() > 0, "MRI volume is required."
            for name in self.view_names:
                view = extract_view(mri, self.views[name], self.out_shape)
                feat = self.shared_encoder(view) if self.shared_encoder is not None else self.encoders[name](view)
                mri_feats.append(feat)
                mri_bbas.append(self.ds_layers[name](feat))
        if self.use_clinical:
            assert clinical is not None and clinical.numel() > 0, "Clinical features required."
            feat = self.clinical_encoder(clinical)
            clin_feats.append(feat)
            clin_bbas.append(self.ds_clinical(feat))
        assert mri_bbas or clin_bbas, "No evidence sources active — check cfg.ABLATION / USE_CLINICAL."

        fused = self._fuse(*self._discount(mri_bbas, clin_bbas))
        m = fused['m']
        bel, pl = compute_belief_plausibility(m)
        betp = pignistic_probability(m)

        aux_feats = clin_feats if self.aux_clinical_only else mri_feats + clin_feats
        aux_logits = self.aux_head(torch.cat(aux_feats, dim=1))
        blend = self._blend(fused['K_max'])
        decision = (1.0 - blend) * betp + blend * F.softmax(aux_logits, dim=1)

        return {
            'm_fused': m, 'bel': bel, 'pl': pl, 'betp': betp,
            'decision': decision, 'pred': decision.argmax(dim=1),
            'uncertainty': (pl - bel).mean(dim=1),
            'K': fused['K_max'], 'K_sum': fused['K_sum'], 'K_breakdown': fused['breakdown'],
            'conflict_flag': fused['flag'], 'conflict_gate': fused['gate'],
            'decision_blend': blend.squeeze(1), 'aux_logits': aux_logits,
        }
