"""DS-SNN-MF: spiking MRI/PET/CT encoders and a clinical MLP, fused with Dempster-Shafer theory.

    MRI (MTL crop) ─ SNN ─ evidential head ─┐
    PET (whole)    ─ SNN ─ evidential head ─┤ gated sequential combine ─ fused BBA ─ BetP ─┐
    CT  (whole)    ─ SNN ─ evidential head ─┤                                              ├─ decision
    clinical       ─ MLP ─ evidential head ─┘           clinical-only aux (softmax) ── K-aware blend ─┘

Missing PET/CT skip that encoder and contribute a vacuous BBA (Dempster identity).
``cfg.ABLATION`` is ``full`` or a plus-separated subset of ``mri``, ``pet``, ``ct``, ``clinical``.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import cfg
from MODULE.DSModule import (ConflictDetectionModule, DSEvidenceLayer, compute_belief_plausibility,
                      conflict_breakdown, discount_bba, pignistic_probability, vacuous_bba)
from MODULE.SNNModule import ClinicalMLP, SNN3DEncoder


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


def _as_mask(flag: Optional[Tensor], batch: int, device: torch.device) -> Tensor:
    if flag is None:
        return torch.ones(batch, dtype=torch.bool, device=device)
    return flag.reshape(-1).to(device=device, dtype=torch.bool)


class DSSNMMF(nn.Module):
    def __init__(self, n_clinical_features: Optional[int] = None, n_classes: int = cfg.NUM_CLASSES):
        super().__init__()
        n_clinical = n_clinical_features or len(cfg.CLINICAL_COLS)
        self.view_names = list(cfg.MRI_VIEWS)
        self.views = dict(cfg.MRI_VIEWS)
        self.out_shape = cfg.VIEW_SHAPE
        self.ablation = cfg.ABLATION
        self.sources = cfg.active_sources()
        self.use_mri = 'mri' in self.sources
        self.use_pet = 'pet' in self.sources
        self.use_ct = 'ct' in self.sources
        self.use_clinical = 'clinical' in self.sources
        self.aux_clinical_only = cfg.AUX_CLINICAL_ONLY and self.use_clinical
        self.reliability = dict(cfg.BRANCH_RELIABILITY)

        if self.use_mri:
            self._build_mri_branch(n_classes)
        if self.use_pet:
            self.pet_encoder = SNN3DEncoder(in_ch=1, out_dim=cfg.VIEW_FEAT_DIM)
            self.ds_pet = DSEvidenceLayer(cfg.VIEW_FEAT_DIM, n_classes,
                                          evidence_cap=self._imaging_cap('pet'),
                                          evidence_scale=cfg.MRI_EVIDENCE_SCALE)
        if self.use_ct:
            self.ct_encoder = SNN3DEncoder(in_ch=1, out_dim=cfg.VIEW_FEAT_DIM)
            self.ds_ct = DSEvidenceLayer(cfg.VIEW_FEAT_DIM, n_classes,
                                         evidence_cap=self._imaging_cap('ct'),
                                         evidence_scale=cfg.MRI_EVIDENCE_SCALE)
        if self.use_clinical:
            self.clinical_encoder = ClinicalMLP(n_features=n_clinical)
            cap = cfg.CLINICAL_EVIDENCE_CAP_STANDALONE if self.sources == {'clinical'} else cfg.EVIDENCE_CAP
            self.ds_clinical = DSEvidenceLayer(cfg.CLINICAL_FEAT_DIM, n_classes, evidence_cap=cap)

        if self.aux_clinical_only:
            aux_in = cfg.CLINICAL_FEAT_DIM
        else:
            n_img = (len(self.view_names) if self.use_mri else 0) + int(self.use_pet) + int(self.use_ct)
            aux_in = n_img * cfg.VIEW_FEAT_DIM + (cfg.CLINICAL_FEAT_DIM if self.use_clinical else 0)
        self.aux_head = nn.Sequential(nn.Dropout(cfg.ENCODER_DROPOUT), nn.Linear(aux_in, n_classes))
        self.cdm = ConflictDetectionModule()

        self.pure_decision = self.ablation != 'full' and cfg.ABLATION_PURE_DECISION
        self.blend_base = min(max(cfg.DECISION_BLEND, 0.0), 1.0)
        self.blend_max = min(max(cfg.DECISION_BLEND_MAX, 0.0), 1.0)

    def _imaging_cap(self, source: str) -> float:
        return cfg.MRI_EVIDENCE_CAP_STANDALONE if self.sources == {source} else cfg.MRI_EVIDENCE_CAP

    def _build_mri_branch(self, n_classes: int) -> None:
        in_ch = cfg.mri_in_channels()
        if cfg.SHARE_MRI_ENCODER:
            self.shared_encoder = SNN3DEncoder(in_ch=in_ch, out_dim=cfg.VIEW_FEAT_DIM)
            self.encoders = None
        else:
            self.shared_encoder = None
            self.encoders = nn.ModuleDict({n: SNN3DEncoder(in_ch=in_ch, out_dim=cfg.VIEW_FEAT_DIM)
                                           for n in self.view_names})
        cap = self._imaging_cap('mri')
        self.ds_layers = nn.ModuleDict({
            n: DSEvidenceLayer(cfg.VIEW_FEAT_DIM, n_classes, evidence_cap=cap,
                               evidence_scale=cfg.MRI_EVIDENCE_SCALE)
            for n in self.view_names})

    def _fold(self, bbas: List[Tensor]):
        """Sequentially fuse BBAs. Returns (m, K_max, K_sum, flag, gate_max, breakdown)."""
        m = bbas[0]
        zeros = torch.zeros(m.shape[0], device=m.device)
        K_max, K_sum, gate_max = zeros, zeros, zeros
        flag = torch.zeros(m.shape[0], dtype=torch.bool, device=m.device)
        breakdown = None
        for other in bbas[1:]:
            breakdown = conflict_breakdown(m, other)
            m, K, flag_new, gate = self.cdm(m, other)
            K_sum, K_max = K_sum + K, torch.maximum(K_max, K)
            gate_max, flag = torch.maximum(gate_max, gate), flag | flag_new
        return m, K_max, K_sum, flag, gate_max, breakdown

    def _discount(self, named: List[Tuple[str, Tensor]], n_present: Tensor) -> List[Tuple[str, Tensor]]:
        """Shafer discounting by branch reliability, only on samples with ≥2 present sources."""
        if not self.reliability or not named:
            return named
        should = (n_present >= 2).to(dtype=named[0][1].dtype)
        out = []
        for name, bba in named:
            alpha = self.reliability.get(name)
            if alpha is None or alpha >= 1.0:
                out.append((name, bba))
                continue
            r = torch.where(should >= 1, torch.full_like(should, float(alpha)), torch.ones_like(should))
            out.append((name, discount_bba(bba, r)))
        return out

    def _blend(self, K_max: Tensor) -> Tensor:
        """Weight of the aux head in the decision, rising with conflict."""
        if self.pure_decision:
            return torch.zeros_like(K_max).unsqueeze(1)
        gate = torch.sigmoid(cfg.DECISION_BLEND_SHARPNESS * (K_max - cfg.DECISION_BLEND_KAPPA))
        return (self.blend_base + (self.blend_max - self.blend_base) * gate).clamp(0.0, 1.0).unsqueeze(1)

    def _masked_bba(self, volume: Tensor, encoder: nn.Module, ds_layer: nn.Module,
                    present: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        """Encode samples that have the volume; vacuous BBA elsewhere (avoids dummy zeros in BN)."""
        batch, device = volume.shape[0], volume.device
        empty = vacuous_bba(batch, device)
        mask = _as_mask(present, batch, device)
        feats = volume.new_zeros(batch, cfg.VIEW_FEAT_DIM)
        if not bool(mask.any()):
            return empty, feats
        idx = mask.nonzero(as_tuple=False).view(-1)
        view = F.interpolate(volume.index_select(0, idx), size=self.out_shape,
                             mode='trilinear', align_corners=False)
        feat = encoder(view)
        bba = ds_layer(feat)
        out = empty.clone()
        out[idx] = bba
        feats[idx] = feat
        return out, feats

    def forward(self, mri: Optional[Tensor] = None, clinical: Optional[Tensor] = None,
                pet: Optional[Tensor] = None, ct: Optional[Tensor] = None,
                has_pet: Optional[Tensor] = None, has_ct: Optional[Tensor] = None) -> Dict[str, Tensor]:
        named: List[Tuple[str, Tensor]] = []
        mri_feats, pet_feats, ct_feats, clin_feats = [], [], [], []
        batch = None
        device = None

        if self.use_mri:
            assert mri is not None and mri.numel() > 0, "MRI volume is required."
            batch, device = mri.shape[0], mri.device
            for name in self.view_names:
                view = extract_view(mri, self.views[name], self.out_shape)
                feat = self.shared_encoder(view) if self.shared_encoder is not None else self.encoders[name](view)
                mri_feats.append(feat)
                named.append((name, self.ds_layers[name](feat)))
        if self.use_pet:
            assert pet is not None and pet.numel() > 0, "PET volume is required."
            batch, device = pet.shape[0], pet.device
            bba, feat = self._masked_bba(pet, self.pet_encoder, self.ds_pet, has_pet)
            pet_feats.append(feat)
            named.append(('pet', bba))
        if self.use_ct:
            assert ct is not None and ct.numel() > 0, "CT volume is required."
            batch, device = ct.shape[0], ct.device
            bba, feat = self._masked_bba(ct, self.ct_encoder, self.ds_ct, has_ct)
            ct_feats.append(feat)
            named.append(('ct', bba))
        if self.use_clinical:
            assert clinical is not None and clinical.numel() > 0, "Clinical features required."
            batch, device = clinical.shape[0], clinical.device
            feat = self.clinical_encoder(clinical)
            clin_feats.append(feat)
            named.append(('clinical', self.ds_clinical(feat)))
        assert named, "No evidence sources active — check cfg.ABLATION / USE_CLINICAL."

        n_present = torch.zeros(batch, device=device)
        if self.use_mri:
            n_present = n_present + 1
        if self.use_pet:
            n_present = n_present + _as_mask(has_pet, batch, device).to(n_present.dtype)
        if self.use_ct:
            n_present = n_present + _as_mask(has_ct, batch, device).to(n_present.dtype)
        if self.use_clinical:
            n_present = n_present + 1

        discounted = self._discount(named, n_present)
        view_set = set(self.view_names)
        view_bbas = [b for n, b in discounted if n in view_set]
        rest = [b for n, b in discounted if n not in view_set]
        if view_bbas:
            folded, kmax, ksum, flag, gate, breakdown = self._fold(view_bbas)
            chain = [folded] + rest
            if rest:
                m, kmax2, ksum2, flag2, gate2, breakdown = self._fold(chain)
                kmax, ksum = torch.maximum(kmax, kmax2), ksum + ksum2
                flag, gate = flag | flag2, torch.maximum(gate, gate2)
            else:
                m = folded
        else:
            m, kmax, ksum, flag, gate, breakdown = self._fold(rest)

        bel, pl = compute_belief_plausibility(m)
        betp = pignistic_probability(m)

        aux_feats = clin_feats if self.aux_clinical_only else mri_feats + pet_feats + ct_feats + clin_feats
        aux_logits = self.aux_head(torch.cat(aux_feats, dim=1))
        blend = self._blend(kmax)
        decision = (1.0 - blend) * betp + blend * F.softmax(aux_logits, dim=1)

        return {
            'm_fused': m, 'bel': bel, 'pl': pl, 'betp': betp,
            'decision': decision, 'pred': decision.argmax(dim=1),
            'uncertainty': (pl - bel).mean(dim=1),
            'K': kmax, 'K_sum': ksum, 'K_breakdown': breakdown,
            'conflict_flag': flag, 'conflict_gate': gate,
            'decision_blend': blend.squeeze(1), 'aux_logits': aux_logits,
        }
