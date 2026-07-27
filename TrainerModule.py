import time
import logging
import numpy as np
import torch
from typing import Dict, List, Optional
from collections import Counter
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from config import Config
from DSModule import (
    DSSNMMF, DSSNNLoss, TemporalAccumulator,
    pignistic_probability, compute_belief_plausibility, dempster_combine,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
log = logging.getLogger("DS-SNN")
cfg = Config()

CLASS_NAMES = cfg.CLASSES


class Trainer:
    def __init__(self, model: DSSNMMF, device: torch.device = cfg.DEVICE,
                 class_weights: Optional[Tensor] = None,
                 lr: Optional[float] = None,
                 param_groups: Optional[List[dict]] = None):
        self.model  = model.to(device)
        self.device = device
        self.loss_fn = DSSNNLoss(
            class_weights=class_weights.to(device) if class_weights is not None else None)
        if param_groups is not None:
            self.optimizer = AdamW(param_groups, weight_decay=cfg.WEIGHT_DECAY)
            lrs = [g.get('lr', cfg.LEARNING_RATE) for g in param_groups]
            log.info(f"  AdamW param groups LRs: {[round(x, 6) for x in lrs]}")
        else:
            lr = cfg.LEARNING_RATE if lr is None else lr
            self.optimizer = AdamW(model.parameters(), lr=lr,
                                   weight_decay=cfg.WEIGHT_DECAY)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.NUM_EPOCHS,
                                           eta_min=cfg.MIN_LR)
        self.best_val_f1  = 0.0
        self.best_val_acc = 0.0
        self.best_score   = -1.0

    # ── one epoch ─────────────────────────────────────────────
    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.loss_fn.set_epoch(epoch - 1)

        accum = max(int(cfg.GRAD_ACCUM_STEPS), 1)
        self.optimizer.zero_grad()

        total_loss, correct, n = 0.0, 0, 0
        comp = {'ce': 0.0, 'aux': 0.0, 'ds': 0.0, 'conflict': 0.0, 'ign': 0.0}
        all_preds, all_labels, n_batches = [], [], 0

        for step, batch in enumerate(loader):
            mri      = batch['mri'].to(self.device)
            clinical = batch['clinical'].to(self.device)
            labels   = batch['label'].to(self.device)

            out    = self.model(mri=mri, clinical=clinical)
            losses = self.loss_fn(out, labels)
            (losses['total'] / accum).backward()

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            bs = labels.shape[0]
            total_loss += losses['total'].item() * bs
            for k in comp:
                comp[k] += losses[k].item() * bs
            correct   += (out['pred'] == labels).sum().item()
            n         += bs
            n_batches  = step + 1
            all_preds.extend(out['pred'].detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

        if n_batches % accum != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.scheduler.step()

        log.info("  Loss | " + "  ".join(f"{k.upper()}: {comp[k]/n:.4f}" for k in comp))
        return {'loss': total_loss / n, 'acc': correct / n,
                'f1': f1_score(all_labels, all_preds, average='macro', zero_division=0)}

    # ── evaluation ────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, verbose: bool = False) -> Dict:
        self.model.eval()
        correct, n, conflict_count, k_sum, gate_sum = 0, 0, 0, 0.0, 0.0
        all_preds, all_labels = [], []
        all_bel, all_pl = [], []

        for batch in loader:
            mri      = batch['mri'].to(self.device)
            clinical = batch['clinical'].to(self.device)
            labels   = batch['label'].to(self.device)
            out = self.model(mri=mri, clinical=clinical)

            correct        += (out['pred'] == labels).sum().item()
            n              += labels.shape[0]
            conflict_count += out['conflict_flag'].sum().item()
            k_sum          += out['K'].sum().item()
            gate_sum       += out['conflict_gate'].sum().item()
            all_bel.append(out['bel'].cpu())
            all_pl.append(out['pl'].cpu())
            all_preds.extend(out['pred'].cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        bel = torch.cat(all_bel); pl = torch.cat(all_pl)
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(cfg.NUM_CLASSES)))
        results = {
            'acc':          correct / n,
            'macro_f1':     f1_score(all_labels, all_preds, average='macro', zero_division=0),
            'per_class_f1': f1_score(all_labels, all_preds, average=None,
                                     labels=list(range(cfg.NUM_CLASSES)), zero_division=0).tolist(),
            # High-conflict fraction (K > CONFLICT_REPORT_KAPPA), not the soft-gate center.
            'conflict_rate': conflict_count / n,
            'mean_K':        k_sum / n,
            'mean_gate':     gate_sum / n,   # avg soft Yager weight ∈ (0,1)
            'avg_uncertainty': (pl - bel).mean().item(),
            'confusion': cm,
        }
        if verbose:
            self._log_confusion(cm, all_labels, all_preds)
        return results

    @staticmethod
    def _log_confusion(cm, y_true, y_pred):
        log.info("Confusion (rows=true, cols=pred): " + " ".join(f"{c:>5}" for c in CLASS_NAMES))
        for i, row in enumerate(cm):
            log.info(f"  {CLASS_NAMES[i]:<4} " + " ".join(f"{v:>5d}" for v in row))
        pred_counts = Counter(y_pred)
        if sum(1 for v in pred_counts.values() if v > 0) == 1:
            sole = list(pred_counts.keys())[0]
            log.warning(f"  ⚠ MAJORITY-CLASS COLLAPSE: predicts only '{CLASS_NAMES[sole]}'.")

    # ── full loop ─────────────────────────────────────────────
    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            save_path: str = "ds_snn_mf_best.pt", verbose_every: int = 5) -> List[Dict]:
        history = []
        warmup = int(getattr(cfg, 'CHECKPOINT_WARMUP', 1))
        require_all = bool(getattr(cfg, 'CKPT_REQUIRE_ALL_CLASSES', True))
        max_gap = float(getattr(cfg, 'CKPT_MAX_VAL_TRAIN_GAP', 0.15))
        train_w = float(getattr(cfg, 'CKPT_TRAIN_F1_WEIGHT', 0.25))
        train_w = min(max(train_w, 0.0), 0.5)
        log.info(f"Training on {self.device} | {cfg.NUM_EPOCHS} epochs "
                 f"(warmup={warmup}, min_epochs={cfg.MIN_EPOCHS}, patience={cfg.PATIENCE})")
        log.info(f"  Checkpoint rules: require_all_classes={require_all}  "
                 f"max_val−train_F1_gap={max_gap}  train_F1_weight={train_w}")
        epochs_since_improve = 0

        # Selection score mixes val macro-F1 / bal-acc with train F1 so a lucky
        # tiny-val spike cannot beat a weaker-but-consistent train state.
        best_score = -1.0

        for epoch in range(1, cfg.NUM_EPOCHS + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader, epoch)
            verbose = (epoch == 1) or (epoch % verbose_every == 0) or (epoch == cfg.NUM_EPOCHS)
            val_metrics = self.evaluate(val_loader, verbose=verbose)

            bal_acc = float(np.mean(np.diag(val_metrics['confusion']) /
                                    np.clip(val_metrics['confusion'].sum(axis=1), 1, None)))
            pred_cols = int((val_metrics['confusion'].sum(axis=0) > 0).sum())
            train_f1 = float(train_metrics['f1'])
            val_f1 = float(val_metrics['macro_f1'])
            gap = val_f1 - train_f1

            # Base score: (1-w)/2 * (val_f1 + bal) + w * train_f1
            val_part = 0.5 * val_f1 + 0.5 * bal_acc
            score = (1.0 - train_w) * val_part + train_w * train_f1

            # Soft penalty if a class is never predicted (kept for MRI-only /
            # tiny-val edge cases when hard reject is off).
            if pred_cols < cfg.NUM_CLASSES:
                score -= 0.10 * (cfg.NUM_CLASSES - pred_cols)

            # Ignore lucky ep-1 spikes on ~18-subject val until warmup ends.
            eligible = epoch >= warmup
            reject_reason = None
            if eligible:
                # Val must contain every class in its labels for the hard gate
                # to be meaningful (otherwise AD may be absent from val).
                val_label_cols = int((val_metrics['confusion'].sum(axis=1) > 0).sum())
                if require_all and val_label_cols >= cfg.NUM_CLASSES and pred_cols < cfg.NUM_CLASSES:
                    reject_reason = (f"missing class on val "
                                    f"(pred_cols={pred_cols}/{cfg.NUM_CLASSES})")
                elif gap > max_gap:
                    reject_reason = (f"val−train F1 gap {gap:.3f} > {max_gap:.3f} "
                                    f"(train_f1={train_f1:.3f}, val_f1={val_f1:.3f})")

            if eligible and reject_reason is None and score > best_score + 1e-4:
                best_score = score
                self.best_score   = score
                self.best_val_f1  = val_metrics['macro_f1']
                self.best_val_acc = val_metrics['acc']
                torch.save(self.model.state_dict(), save_path)
                epochs_since_improve = 0
                log.info(f"  ✓ New best (score={score:.4f}, val_acc={val_metrics['acc']:.4f}, "
                         f"macro_f1={val_f1:.4f}, bal_acc={bal_acc:.4f}, "
                         f"train_f1={train_f1:.4f})")
            elif eligible and reject_reason is not None:
                epochs_since_improve += 1
                if epoch % verbose_every == 0 or epoch == warmup:
                    log.info(f"  ✗ Skip checkpoint: {reject_reason}")
            elif eligible:
                epochs_since_improve += 1
            elif epoch == warmup - 1 or (epoch < warmup and epoch % verbose_every == 0):
                log.info(f"  … warmup {epoch}/{warmup - 1} (not checkpointing yet; "
                         f"val_acc={val_metrics['acc']:.4f}, macro_f1={val_f1:.4f})")

            history.append({**train_metrics,
                            **{f'val_{k}': v for k, v in val_metrics.items() if k != 'confusion'}})
            log.info(
                f"Epoch {epoch:3d}/{cfg.NUM_EPOCHS} | Loss {train_metrics['loss']:.4f} | "
                f"Train Acc {train_metrics['acc']:.4f} F1 {train_metrics['f1']:.4f} | "
                f"Val Acc {val_metrics['acc']:.4f} F1 {val_metrics['macro_f1']:.4f} | "
                f"Per-class F1 {[f'{f:.2f}' for f in val_metrics['per_class_f1']]} | "
                f"meanK {val_metrics['mean_K']:.3f} (hi-rate {val_metrics['conflict_rate']:.2f}, "
                f"gate {val_metrics['mean_gate']:.2f}) | "
                f"t={time.time()-t0:.1f}s")

            if epoch >= cfg.MIN_EPOCHS and epochs_since_improve >= cfg.PATIENCE:
                log.info(f"Early stopping (best score={best_score:.4f}, "
                         f"macro_f1={self.best_val_f1:.4f}).")
                break

        # Safety: if warmup never produced a save (pathological), save last state.
        if best_score < 0:
            torch.save(self.model.state_dict(), save_path)
            log.warning("No checkpoint saved during warmup/selection — wrote final weights.")
        self.best_score = best_score
        return history


# ──────────────────────────────────────────────────────────────
# Longitudinal evaluation
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_longitudinal(model: DSSNMMF, loader: DataLoader,
                          device: torch.device = cfg.DEVICE) -> Dict:
    """Score each subject against the final-visit label.

    Primary metric = **last real visit only** (same decision rule as
    cross-sectional). Equal Dempster accumulation across all visits is kept
    as a secondary diagnostic only — it mixes earlier healthier visits into
    the final diagnosis and systematically biases predictions."""
    model.eval()
    last_correct = demp_correct = n = 0
    last_preds, demp_preds, all_labels = [], [], []

    for batch in loader:
        visits = batch['visits']
        labels = batch['label'].to(device)
        B = labels.shape[0]
        acc = TemporalAccumulator(batch_size=B, device=device)
        # Track the most recent real visit's decision per subject.
        last_decision = torch.full((B, cfg.NUM_CLASSES), 1.0 / cfg.NUM_CLASSES,
                                   device=device)

        for visit in visits:
            mri       = visit['mri'].to(device)
            clinical  = visit['clinical'].to(device)
            real_mask = torch.tensor(visit['has_visit'], dtype=torch.bool, device=device)
            if not real_mask.any():
                continue

            out = model(mri=mri, clinical=clinical)
            # Last-visit decision (aux/DS blend — same as cross-sectional pred).
            last_decision[real_mask] = out['decision'][real_mask].detach()

            # Secondary: equal Dempster accumulation of fused BBAs.
            m_new_real, _ = dempster_combine(acc.m_acc[real_mask],
                                             out['m_fused'][real_mask], use_yager=False)
            new_m_acc = acc.m_acc.clone()
            new_m_acc[real_mask] = m_new_real
            acc.m_acc = new_m_acc.detach()

        pred_last = last_decision.argmax(dim=1)
        pred_demp = pignistic_probability(acc.m_acc).argmax(dim=1)
        last_correct += (pred_last == labels).sum().item()
        demp_correct += (pred_demp == labels).sum().item()
        n            += B
        last_preds.extend(pred_last.cpu().tolist())
        demp_preds.extend(pred_demp.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    log.info("Longitudinal (LAST-visit) confusion:\n" + str(
        confusion_matrix(all_labels, last_preds, labels=list(range(cfg.NUM_CLASSES)))))
    log.info("\n" + classification_report(all_labels, last_preds,
                                          labels=list(range(cfg.NUM_CLASSES)),
                                          target_names=CLASS_NAMES, zero_division=0, digits=3))
    log.info("Longitudinal (equal Dempster accum) confusion:\n" + str(
        confusion_matrix(all_labels, demp_preds, labels=list(range(cfg.NUM_CLASSES)))))
    return {
        'longitudinal_acc': last_correct / max(n, 1),          # primary
        'longitudinal_acc_dempster': demp_correct / max(n, 1), # diagnostic
    }
