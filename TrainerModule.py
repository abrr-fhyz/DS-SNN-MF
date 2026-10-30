"""Training loop, checkpoint selection and longitudinal evaluation."""
import logging
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix as sk_confusion
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from config import cfg
from DSModule import TemporalAccumulator, pignistic_probability
from FusionModule import DSSNMMF
from LossModule import DSSNNLoss
from MetricsModule import Predictions, compute_metrics, macro_f1

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
log = logging.getLogger("DS-SNN")
_LOSS_TERMS = ('ce', 'aux', 'ds', 'conflict', 'ign')


def build_param_groups(model: DSSNMMF) -> List[dict]:
    """Clinical branch + aux head train at CLINICAL_LR, the MRI encoder / heads at LEARNING_RATE.
    Single-source models use one group."""
    if not (model.use_mri and model.use_clinical):
        lr = cfg.CLINICAL_LR if model.use_clinical else cfg.LEARNING_RATE
        return [{'params': list(model.parameters()), 'lr': lr}]
    hot = (list(model.clinical_encoder.parameters()) + list(model.ds_clinical.parameters())
           + list(model.aux_head.parameters()))
    hot_ids = {id(p) for p in hot}
    cold = [p for p in model.parameters() if id(p) not in hot_ids]
    groups = [{'params': hot, 'lr': cfg.CLINICAL_LR}]
    return groups + [{'params': cold, 'lr': cfg.LEARNING_RATE}] if cold else groups


class Trainer:
    def __init__(self, model: DSSNMMF, device: torch.device = cfg.DEVICE,
                 class_weights: Optional[Tensor] = None):
        self.model, self.device = model.to(device), device
        self.loss_fn = DSSNNLoss(class_weights.to(device) if class_weights is not None else None).to(device)
        groups = build_param_groups(model)
        self.optimizer = AdamW(groups, weight_decay=cfg.WEIGHT_DECAY)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.NUM_EPOCHS, eta_min=cfg.MIN_LR)
        self.best_val_f1 = self.best_val_acc = 0.0
        self.best_score = -1.0
        log.info(f"  AdamW LRs: {[round(g['lr'], 6) for g in groups]}")

    def load_checkpoint(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    # ── training ──────────────────────────────────────────────
    def _optimizer_step(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.loss_fn.set_epoch(epoch - 1)
        accum = max(int(cfg.GRAD_ACCUM_STEPS), 1)
        self.optimizer.zero_grad()

        total, n, steps = 0.0, 0, 0
        terms = dict.fromkeys(_LOSS_TERMS, 0.0)
        preds, labels_all = [], []
        for step, batch in enumerate(loader):
            labels = batch['label'].to(self.device)
            out = self.model(mri=batch['mri'].to(self.device), clinical=batch['clinical'].to(self.device))
            losses = self.loss_fn(out, labels)
            (losses['total'] / accum).backward()
            if (step + 1) % accum == 0:
                self._optimizer_step()

            bs = labels.shape[0]
            total += losses['total'].item() * bs
            for k in terms:
                terms[k] += losses[k].item() * bs
            n, steps = n + bs, step + 1
            preds.extend(out['pred'].detach().cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

        if steps % accum != 0:
            self._optimizer_step()
        self.scheduler.step()
        log.info("  Loss | " + "  ".join(f"{k.upper()}: {v / n:.4f}" for k, v in terms.items()))
        labels_np, preds_np = np.array(labels_all), np.array(preds)
        return {'loss': total / n, 'acc': float((labels_np == preds_np).mean()),
                'f1': macro_f1(labels_np, preds_np)}

    # ── evaluation ────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Predictions:
        self.model.eval()
        keys = ('labels', 'probs', 'uncertainty', 'K', 'gate', 'blend', 'conflict_flag')
        cols: Dict[str, list] = {k: [] for k in keys}
        for batch in loader:
            out = self.model(mri=batch['mri'].to(self.device), clinical=batch['clinical'].to(self.device))
            for key, value in zip(keys, (batch['label'], out['decision'], out['uncertainty'], out['K'],
                                         out['conflict_gate'], out['decision_blend'], out['conflict_flag'])):
                cols[key].append(value.detach().cpu().numpy())
        return Predictions(**{k: np.concatenate(v) for k, v in cols.items()})

    def evaluate(self, loader: DataLoader, verbose: bool = False) -> Dict:
        metrics = compute_metrics(self.predict(loader))
        if verbose:
            self._log_confusion(metrics['confusion'])
        return metrics

    @staticmethod
    def _log_confusion(cm: np.ndarray) -> None:
        names = cfg.CLASSES
        log.info("Confusion (rows=true, cols=pred): " + " ".join(f"{c:>5}" for c in names))
        for name, row in zip(names, cm):
            log.info(f"  {name:<4} " + " ".join(f"{v:>5d}" for v in row))
        predicted = np.nonzero(cm.sum(axis=0))[0]
        if len(predicted) == 1:
            log.warning(f"  MAJORITY-CLASS COLLAPSE: predicts only '{names[predicted[0]]}'.")

    # ── checkpoint selection ──────────────────────────────────
    @staticmethod
    def _score(train_f1: float, val: Dict) -> float:
        """Validation score: mix of macro-F1 and balanced accuracy, plus a small train-F1 term,
        penalised for every class the model never predicts."""
        w = min(max(cfg.CKPT_TRAIN_F1_WEIGHT, 0.0), 0.5)
        score = (1.0 - w) * (0.5 * val['macro_f1'] + 0.5 * val['bal_acc']) + w * train_f1
        predicted = int((val['confusion'].sum(axis=0) > 0).sum())
        return score - 0.10 * (cfg.NUM_CLASSES - predicted)

    @staticmethod
    def _reject_reason(train_f1: float, val: Dict) -> Optional[str]:
        cm = val['confusion']
        pred_cols, label_cols = int((cm.sum(axis=0) > 0).sum()), int((cm.sum(axis=1) > 0).sum())
        if cfg.CKPT_REQUIRE_ALL_CLASSES and label_cols >= cfg.NUM_CLASSES and pred_cols < cfg.NUM_CLASSES:
            return f"missing class on val (pred_cols={pred_cols}/{cfg.NUM_CLASSES})"
        gap = val['macro_f1'] - train_f1
        if gap > cfg.CKPT_MAX_VAL_TRAIN_GAP:
            return f"val−train F1 gap {gap:.3f} > {cfg.CKPT_MAX_VAL_TRAIN_GAP:.3f}"
        return None

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, save_path: str = "ds_snn_mf_best.pt",
            verbose_every: int = 5) -> List[Dict]:
        history: List[Dict] = []
        warmup, stale = cfg.CHECKPOINT_WARMUP, 0
        log.info(f"Training on {self.device} | {cfg.NUM_EPOCHS} epochs (checkpoint warmup={warmup}, "
                 f"min_epochs={cfg.MIN_EPOCHS}, patience={cfg.PATIENCE})")

        for epoch in range(1, cfg.NUM_EPOCHS + 1):
            t0 = time.time()
            train = self.train_epoch(train_loader, epoch)
            verbose = epoch == 1 or epoch % verbose_every == 0 or epoch == cfg.NUM_EPOCHS
            val = self.evaluate(val_loader, verbose=verbose)
            score = self._score(train['f1'], val)

            if epoch >= warmup:
                reason = self._reject_reason(train['f1'], val)
                if reason is None and score > self.best_score + 1e-4:
                    self.best_score, self.best_val_f1, self.best_val_acc = score, val['macro_f1'], val['acc']
                    torch.save(self.model.state_dict(), save_path)
                    stale = 0
                    log.info(f"  ✓ New best (score={score:.4f}, val_acc={val['acc']:.4f}, "
                             f"macro_f1={val['macro_f1']:.4f}, bal_acc={val['bal_acc']:.4f})")
                else:
                    stale += 1
                    if reason and epoch % verbose_every == 0:
                        log.info(f"  ✗ Skip checkpoint: {reason}")

            history.append({**train, **{f'val_{k}': v for k, v in val.items() if k != 'confusion'}})
            log.info(f"Epoch {epoch:3d}/{cfg.NUM_EPOCHS} | Loss {train['loss']:.4f} | "
                     f"Train Acc {train['acc']:.4f} F1 {train['f1']:.4f} | "
                     f"Val Acc {val['acc']:.4f} F1 {val['macro_f1']:.4f} | "
                     f"Per-class F1 {[f'{f:.2f}' for f in val['per_class_f1']]} | "
                     f"meanK {val['mean_K']:.3f} (hi-rate {val['conflict_rate']:.2f}, "
                     f"gate {val['mean_gate']:.2f}, blend {val['mean_decision_blend']:.2f}) | "
                     f"t={time.time() - t0:.1f}s")

            if epoch >= cfg.MIN_EPOCHS and stale >= cfg.PATIENCE:
                log.info(f"Early stopping (best score={self.best_score:.4f}, macro_f1={self.best_val_f1:.4f}).")
                break

        if self.best_score < 0:  # no epoch passed the checkpoint rules
            torch.save(self.model.state_dict(), save_path)
            log.warning("No checkpoint saved during selection — wrote final weights.")
        return history


# ──────────────────────────────────────────────────────────────
# Longitudinal evaluation
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_longitudinal(model: torch.nn.Module, loader: DataLoader, device: torch.device = cfg.DEVICE) -> Dict:
    """Score each subject against the final-visit label.

    Primary: the last real visit's decision (same rule as cross-sectional evaluation).
    Diagnostic: equal Dempster accumulation of all visits' fused BBAs, which mixes earlier
    healthier visits into the final diagnosis and is biased toward them."""
    model.eval()
    k = cfg.NUM_CLASSES
    last_preds, demp_preds, labels_all = [], [], []
    for batch in loader:
        labels = batch['label'].to(device)
        acc = TemporalAccumulator(batch_size=labels.shape[0], device=device)
        last_decision = torch.full((labels.shape[0], k), 1.0 / k, device=device)
        for visit in batch['visits']:
            real = torch.tensor(visit['has_visit'], dtype=torch.bool, device=device)
            if not real.any():
                continue
            out = model(mri=visit['mri'].to(device), clinical=visit['clinical'].to(device))
            last_decision[real] = out['decision'][real]
            acc.accumulate(out['m_fused'], mask=real)
        last_preds.extend(last_decision.argmax(dim=1).cpu().tolist())
        demp_preds.extend(pignistic_probability(acc.m_acc).argmax(dim=1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

    labels_np = np.array(labels_all)
    names = list(range(k))
    log.info("Longitudinal (LAST-visit) confusion:\n" + str(sk_confusion(labels_all, last_preds, labels=names)))
    log.info("\n" + classification_report(labels_all, last_preds, labels=names, target_names=cfg.CLASSES,
                                          zero_division=0, digits=3))
    log.info("Longitudinal (equal Dempster accum) confusion:\n"
             + str(sk_confusion(labels_all, demp_preds, labels=names)))
    return {'longitudinal_acc': float((labels_np == np.array(last_preds)).mean()),
            'longitudinal_acc_dempster': float((labels_np == np.array(demp_preds)).mean())}
