"""Losses for the DS-SNN-MF model.

L = L_focal(decision) + λ_aux·L_aux + λ_ds·L_DS + λ_k·L_K + λ_ign·L_ign

The DS and ignorance terms warm up linearly so early, noisy BBAs do not dominate.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import cfg
from DSModule import FULL_FRAME, SINGLETONS

_IGNORANCE_TARGETS = (cfg.IGNORANCE_TARGET_CN, cfg.IGNORANCE_TARGET_MCI, cfg.IGNORANCE_TARGET_AD)


def evidential_utility_loss(bba_singletons: Tensor, labels: Tensor) -> Tensor:
    """Reward mass on the true singleton, penalise mass on wrong ones; Θ mass is free."""
    true_mass = bba_singletons.gather(1, labels.unsqueeze(1)).squeeze(1)
    wrong_sq = bba_singletons.pow(2).sum(dim=1) - true_mass.pow(2)
    return ((1.0 - true_mass).pow(2) + wrong_sq).mean()


def conflict_regularisation_loss(K: Tensor, kappa: float = cfg.CONFLICT_KAPPA) -> Tensor:
    """Hinge on conflict above κ: mean(max(0, K − κ)²)."""
    return F.relu(K - kappa).pow(2).mean()


def ignorance_anchoring_loss(bba_ignorance: Tensor, labels: Tensor) -> Tensor:
    """Pull each class's Θ mass toward its target."""
    targets = torch.tensor(_IGNORANCE_TARGETS, device=labels.device, dtype=bba_ignorance.dtype)
    return (bba_ignorance - targets[labels]).pow(2).mean()


def focal_ce_from_probs(probs: Tensor, labels: Tensor, weight: Tensor, gamma: float,
                        label_smoothing: float) -> Tensor:
    """Class-weighted focal CE on a probability simplex (e.g. BetP or the blended decision)."""
    log_p = torch.log(probs.clamp(min=1e-8))
    nll = F.nll_loss(log_p, labels, weight=weight, reduction='none')
    if label_smoothing > 0:
        smooth = -log_p.mean(dim=1) * weight[labels]
        nll = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    if gamma and gamma > 0:
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
        nll = (1.0 - pt).pow(gamma) * nll
    return nll.mean()


def focal_ce_from_logits(logits: Tensor, labels: Tensor, weight: Tensor, gamma: float,
                         label_smoothing: float) -> Tensor:
    """Class-weighted focal CE with label smoothing on logits (auxiliary head)."""
    log_p = F.log_softmax(logits, dim=1)
    nll = F.nll_loss(log_p, labels, weight=weight, reduction='none')
    if label_smoothing > 0:
        with torch.no_grad():
            target = torch.full_like(log_p, label_smoothing / (logits.shape[1] - 1))
            target.scatter_(1, labels.unsqueeze(1), 1.0 - label_smoothing)
        nll = -(target * log_p).sum(dim=1) * weight[labels]
    if gamma and gamma > 0:
        pt = log_p.exp().gather(1, labels.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
        nll = (1.0 - pt).pow(gamma) * nll
    return nll.mean()


class DSSNNLoss(nn.Module):
    """Weighted sum of the five terms. Standalone ablations (anything but 'full') use the
    ``*_STANDALONE`` weights: the Θ corridor only makes sense when another source can act on it."""

    def __init__(self, class_weights: Optional[Tensor] = None):
        super().__init__()
        standalone = cfg.ABLATION != 'full'
        self.lambda_aux = cfg.LAMBDA_AUX_STANDALONE if standalone else cfg.LAMBDA_AUX
        self.lambda_ign = cfg.LAMBDA_IGN_STANDALONE if standalone else cfg.LAMBDA_IGN
        self.lambda_ds, self.lambda_k = cfg.LAMBDA_DS, cfg.LAMBDA_K
        self.warmup_epochs = max(cfg.NUM_EPOCHS // 10, 3)
        self.current_epoch = 0
        if class_weights is None:
            class_weights = torch.ones(cfg.NUM_CLASSES)
        self.register_buffer('class_weights', class_weights.float())

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def forward(self, out: Dict[str, Tensor], labels: Tensor) -> Dict[str, Tensor]:
        m_fused = out['m_fused']
        w, gamma, smooth = self.class_weights, cfg.FOCAL_GAMMA, cfg.LABEL_SMOOTHING
        L_ce = focal_ce_from_probs(out['decision'], labels, w, gamma, smooth)
        L_aux = focal_ce_from_logits(out['aux_logits'], labels, w, gamma, smooth)
        L_ds = evidential_utility_loss(m_fused[:, SINGLETONS], labels)
        L_k = conflict_regularisation_loss(out['K'])
        L_ign = ignorance_anchoring_loss(m_fused[:, FULL_FRAME], labels)

        warmup = min(1.0, (self.current_epoch + 1) / max(1, self.warmup_epochs))
        total = (L_ce + self.lambda_aux * L_aux + self.lambda_ds * warmup * L_ds
                 + self.lambda_k * L_k + self.lambda_ign * warmup * L_ign)
        return {'total': total, 'ce': L_ce, 'aux': L_aux, 'ds': L_ds, 'conflict': L_k, 'ign': L_ign}
