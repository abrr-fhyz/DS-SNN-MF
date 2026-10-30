"""Subject-level (repeated) cross-validation experiments.

For every fold: normalise on train subjects, train with validation early stopping, test the
selected checkpoint once. The RF / LR baselines run on the *same* folds, so the report includes
paired comparisons and bootstrap confidence intervals on pooled out-of-fold predictions.

Each run writes ``<OUT_DIR>/<dataset>_<ablation>_<timestamp>/``:
    config.json            every setting used
    fold<R>_<F>_*.{pt,json} checkpoint and metrics per fold
    oof_predictions.csv    one row per (repeat, subject): label, prediction, probabilities
    summary.json           per-fold results, summary statistics, baseline comparison
    run.log                the full log
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import cfg
from DataModule import (CrossSectionalWrapper, Sequences, SubjectDataset, load_dataset, longitudinal_collate_fn,
                        make_holdout_split, make_stratified_folds, set_seed, split_sequences, subject_labels)
from FusionModule import DSSNMMF
from MetricsModule import Predictions, compute_metrics, macro_f1
from OtherModules.baselines import run_baselines
from TrainerModule import Trainer, evaluate_longitudinal

log = logging.getLogger("DS-SNN")
_SUMMARY_KEYS = [
    ('test_acc', 'Test accuracy'), ('test_macro_f1', 'Test macro-F1'), ('test_bal_acc', 'Test balanced accuracy'),
    ('test_auroc', 'Test macro AUROC'), ('test_brier', 'Test Brier'), ('test_ece', 'Test ECE'),
    ('test_sel_acc_unc', 'Selective acc (Pl−Bel)'), ('test_sel_acc_conf', 'Selective acc (max prob)'),
    ('test_mean_K', 'Test meanK'), ('test_conflict_rate', 'Test high-conflict rate'),
    ('test_mean_gate', 'Test mean soft-gate'), ('test_mean_decision_blend', 'Test mean decision blend'),
    ('longitudinal_acc', 'Longitudinal (last-visit)'), ('longitudinal_acc_dempster', 'Longitudinal (Dempster)'),
    ('val_acc', 'Val accuracy (selected)'), ('val_macro_f1', 'Val macro-F1 (selected)'),
]


# ──────────────────────────────────────────────────────────────
# One train/val/test split
# ──────────────────────────────────────────────────────────────

def _inverse_frequency(counts: List[int]) -> torch.Tensor:
    total = sum(counts)
    return torch.tensor([total / (cfg.NUM_CLASSES * max(c, 1)) for c in counts], dtype=torch.float32)


def build_loaders(data: Dict[str, Sequences]):
    """Returns (train, val, test, test-longitudinal loaders, loss class weights, train class counts)."""
    train_ds = CrossSectionalWrapper(data['train'], visits=cfg.spec.train_visits, augment=True)
    counts = train_ds.label_counts()
    inv_freq = _inverse_frequency(counts)
    log.info(f"  Train samples: {len(train_ds)} ({cfg.spec.train_visits} visits) | class counts {counts} "
             f"→ sampler weights {[round(w, 3) for w in inv_freq.tolist()]}")

    common = dict(batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS, pin_memory=torch.cuda.is_available())
    sampler = WeightedRandomSampler([inv_freq[y].item() for y in train_ds.labels()],
                                    num_samples=len(train_ds), replacement=True)
    lone_tail = len(train_ds) > cfg.BATCH_SIZE and len(train_ds) % cfg.BATCH_SIZE == 1  # BatchNorm needs >1 sample
    loaders = (
        DataLoader(train_ds, sampler=sampler, drop_last=lone_tail, **common),
        DataLoader(CrossSectionalWrapper(data['val']), shuffle=False, **common),
        DataLoader(CrossSectionalWrapper(data['test']), shuffle=False, **common),
        DataLoader(SubjectDataset(data['test']), shuffle=False, collate_fn=longitudinal_collate_fn,
                   batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS),
    )
    if cfg.USE_LOSS_CLASS_WEIGHTS:
        weights = inv_freq.clone()
        weights[1] *= cfg.MCI_LOSS_BOOST
    else:
        weights = torch.ones(cfg.NUM_CLASSES)
    return (*loaders, weights, counts)


def run_split(data: Dict[str, Sequences], device: torch.device, ckpt_path: str) -> Tuple[dict, Predictions]:
    """Train on ``data['train']``, select on val, evaluate once on test."""
    train_loader, val_loader, test_loader, long_loader, weights, counts = build_loaders(data)
    trainer = Trainer(DSSNMMF().to(device), device=device, class_weights=weights)
    trainer.fit(train_loader, val_loader, save_path=ckpt_path)
    trainer.load_checkpoint(ckpt_path)

    preds = trainer.predict(test_loader)
    metrics = compute_metrics(preds)
    trainer._log_confusion(metrics['confusion'])
    longitudinal = evaluate_longitudinal(trainer.eval_model, long_loader, device=device)

    result = {'train_counts': counts, 'val_acc': trainer.best_val_acc, 'val_macro_f1': trainer.best_val_f1,
              'val_score': trainer.best_score, 'confusion': metrics['confusion'].tolist(), **longitudinal}
    result.update({f'test_{k}': v for k, v in metrics.items() if k != 'confusion'})
    log.info(f"\n  Test | acc={result['test_acc']:.4f} macroF1={result['test_macro_f1']:.4f} "
             f"bal_acc={result['test_bal_acc']:.4f} auroc={result['test_auroc']:.3f} ece={result['test_ece']:.3f} "
             f"per_class_f1={[f'{f:.2f}' for f in result['test_per_class_f1']]} meanK={result['test_mean_K']:.3f} "
             f"hi-rate={result['test_conflict_rate']:.2f} gate={result['test_mean_gate']:.2f} "
             f"blend={result['test_mean_decision_blend']:.2f}")
    log.info(f"  Longitudinal last-visit {longitudinal['longitudinal_acc']:.4f} | "
             f"equal-Dempster {longitudinal['longitudinal_acc_dempster']:.4f}")
    return result, preds


# ──────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────

def _mean_std(values) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {'mean': float(np.nanmean(arr)), 'std': float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0,
            'values': arr.tolist()}


def summarise(results: List[dict]) -> dict:
    summary = {key: _mean_std([r[key] for r in results]) for key, _ in _SUMMARY_KEYS}
    per_class = np.array([r['test_per_class_f1'] for r in results])
    summary['test_per_class_f1'] = {name: _mean_std(per_class[:, i]) for i, name in enumerate(cfg.CLASSES)}
    return summary


def bootstrap_comparison(labels: np.ndarray, oof: Dict[str, np.ndarray], reference: str,
                         rounds: int, seed: int) -> Dict[str, dict]:
    """Bootstrap subjects; each draw averages macro-F1 over repeats.

    ``oof[name]`` is (repeats, N) predictions (-1 = not tested). Returns, per method, the
    pooled macro-F1 with a 95% CI and, for methods other than ``reference``, the paired
    difference ``reference − method`` with the share of draws where it is ≤ 0.
    """
    covered = np.where((oof[reference] >= 0).all(axis=0))[0]
    draws = np.random.default_rng(seed).integers(0, len(covered), size=(rounds, len(covered)))

    def pooled(preds: np.ndarray, idx: np.ndarray) -> float:
        return float(np.mean([macro_f1(labels[idx], p[idx]) for p in preds]))

    boot = {name: np.array([pooled(p, covered[d]) for d in draws]) for name, p in oof.items()}
    out = {}
    for name, p in oof.items():
        lo, hi = np.percentile(boot[name], [2.5, 97.5])
        out[name] = {'macro_f1': pooled(p, covered), 'ci95': [float(lo), float(hi)]}
        if name != reference:
            diff = boot[reference] - boot[name]
            dlo, dhi = np.percentile(diff, [2.5, 97.5])
            out[name].update(delta_vs_reference=pooled(oof[reference], covered) - out[name]['macro_f1'],
                             delta_ci95=[float(dlo), float(dhi)], frac_delta_le_zero=float((diff <= 0).mean()))
    return out


# ──────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────

def _fmt(values) -> str:
    return str([f'{v:.3f}' for v in values])


def log_summary(summary: dict, n_units: int, baselines: dict, pooled: Optional[dict]) -> None:
    bar = "=" * 76
    log.info(f"\n{bar}\nSUBJECT-LEVEL CV SUMMARY — {n_units} fold(s)  (mean ± std)\n{bar}")
    for key, name in _SUMMARY_KEYS:
        s = summary[key]
        log.info(f"  {name:<28} {s['mean']:.4f} ± {s['std']:.4f}   {_fmt(s['values'])}")
    log.info("  Per-class F1 (test):")
    for name, s in summary['test_per_class_f1'].items():
        log.info(f"    {name:<6} {s['mean']:.4f} ± {s['std']:.4f}   {_fmt(s['values'])}")

    log.info("-" * 76 + "\n  Baselines on identical folds (macro-F1 mean ± std; paired fold Δ = model − baseline)")
    model_f1 = np.array(summary['test_macro_f1']['values'])
    for name, folds in baselines.items():
        f1 = np.array([f['macro_f1'] for f in folds])
        log.info(f"    {name:<8} {f1.mean():.4f} ± {(f1.std(ddof=1) if len(f1) > 1 else 0.0):.4f}   "
                 f"Δ={float((model_f1 - f1).mean()):+.4f}   model ≥ baseline in "
                 f"{int((model_f1 >= f1).sum())}/{len(f1)} folds")
    if pooled:
        log.info("-" * 76 + "\n  Pooled out-of-fold macro-F1, bootstrap 95% CI over subjects")
        for name, p in pooled.items():
            extra = (f"   Δ(model−{name})={p['delta_vs_reference']:+.4f} CI [{p['delta_ci95'][0]:+.3f}, "
                     f"{p['delta_ci95'][1]:+.3f}]  P(Δ≤0)={p['frac_delta_le_zero']:.3f}"
                     if 'delta_vs_reference' in p else "")
            log.info(f"    {name:<8} {p['macro_f1']:.4f}  [{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}]{extra}")
    log.info(bar)
    log.info("Report test metrics only; validation numbers are for early stopping. "
             "Tune hyperparameters on validation, never on these test folds.")


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _dump(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


# ──────────────────────────────────────────────────────────────
# Experiment driver
# ──────────────────────────────────────────────────────────────

def _plan_folds(sequences: Sequences) -> List[List[Dict[str, np.ndarray]]]:
    """Per repeat, a list of train/val/test index dicts (a single holdout when USE_CV is off)."""
    if not cfg.USE_CV:
        return [[make_holdout_split(sequences, seed=cfg.SPLIT_SEED)]]
    return [make_stratified_folds(sequences, cfg.N_FOLDS, cfg.SPLIT_SEED + r, cfg.CV_VAL_FRAC)
            for r in range(cfg.REPEATS)]


def run_experiment(device: torch.device, sequences: Optional[Sequences] = None,
                   run_dir: Optional[Path] = None) -> dict:
    sequences = sequences if sequences is not None else load_dataset()
    labels = subject_labels(sequences)
    log.info(f"Dataset {cfg.DATASET}: {len(sequences)} subjects | final-label counts {cfg.CLASSES} = "
             f"{[int((labels == c).sum()) for c in range(cfg.NUM_CLASSES)]}")

    run_dir = Path(run_dir or cfg.OUT_DIR / f"{cfg.DATASET}_{cfg.ABLATION}_{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(run_dir / 'run.log', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(levelname)s | %(message)s'))
    log.addHandler(handler)
    try:
        _dump(run_dir / 'config.json', cfg.snapshot())
        folds_by_repeat = _plan_folds(sequences)
        n_rep, n_subj = len(folds_by_repeat), len(sequences)
        oof = {'model': np.full((n_rep, n_subj), -1, dtype=np.int64)}
        baseline_folds: Dict[str, List[dict]] = {}
        rows, results = [], []

        for r, folds in enumerate(folds_by_repeat):
            baselines = run_baselines(sequences, folds)
            for name, res in baselines.items():
                baseline_folds.setdefault(name, []).extend(res['folds'])
                oof.setdefault(name, np.full((n_rep, n_subj), -1, dtype=np.int64))[r] = res['oof']

            for f, indices in enumerate(folds):
                tag = f"r{r + 1}f{f + 1}"
                log.info(f"\n{'#' * 76}\n# REPEAT {r + 1}/{n_rep}  FOLD {f + 1}/{len(folds)}  "
                         f"train={len(indices['train'])} val={len(indices['val'])} test={len(indices['test'])}\n{'#' * 76}")
                set_seed(cfg.SEED + f + 100 * r)
                result, preds = run_split(split_sequences(sequences, indices), device,
                                          str(run_dir / f"fold_{tag}_best.pt"))
                result.update(repeat=r + 1, fold=f + 1)
                results.append(result)
                _dump(run_dir / f"fold_{tag}_metrics.json", result)

                oof['model'][r, indices['test']] = preds.preds
                for j, subj in enumerate(indices['test']):
                    rows.append({'repeat': r + 1, 'fold': f + 1, 'subject_idx': int(subj),
                                 'subject_id': sequences[subj][-1]['subject_id'], 'label': int(preds.labels[j]),
                                 'pred': int(preds.preds[j]), 'uncertainty': float(preds.uncertainty[j]),
                                 'K': float(preds.K[j]),
                                 **{f'p_{cfg.CLASSES[c]}': float(preds.probs[j, c]) for c in range(cfg.NUM_CLASSES)},
                                 **{f'pred_{n}': int(res['oof'][subj]) for n, res in baselines.items()}})

        pd.DataFrame(rows).to_csv(run_dir / 'oof_predictions.csv', index=False)
        summary = summarise(results)
        pooled = bootstrap_comparison(labels, oof, 'model', cfg.BOOTSTRAP_ROUNDS, cfg.SEED) if cfg.USE_CV else None
        log_summary(summary, len(results), baseline_folds, pooled)
        _dump(run_dir / 'summary.json', {'folds': results, 'summary': summary,
                                         'baselines': baseline_folds, 'pooled_oof': pooled})
        log.info(f"\nWrote checkpoints, metrics and predictions to {run_dir}/")
        return summary
    finally:
        log.removeHandler(handler)
        handler.close()
