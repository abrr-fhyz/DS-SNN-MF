"""Tabular baselines (logistic regression, random forest) on the same subject-level folds.

CLI:   python -m OtherModules.baselines [--dataset oasis2|oasis3]
Import: ``run_baselines(sequences, folds)`` is also used by ``ExperimentModule`` so the neural
model and the baselines are always compared on identical splits.
"""
import argparse
from typing import Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import cfg
from DataModule import Sequences, load_dataset, make_stratified_folds, subject_labels
from MetricsModule import balanced_accuracy, confusion_matrix, per_class_f1


def _matrix(sequences: Sequences, indices: np.ndarray, last_only: bool):
    """Raw clinical rows: the final visit per subject, or every visit flattened."""
    rows = [(v['clinical_raw'], v['label']) for i in indices
            for v in ([sequences[i][-1]] if last_only else sequences[i])]
    return np.array([r for r, _ in rows], dtype=np.float64), np.array([y for _, y in rows], dtype=np.int64)


def _impute(X: np.ndarray, mean: np.ndarray) -> np.ndarray:
    X = X.copy()
    rows, cols = np.where(np.isnan(X))
    X[rows, cols] = mean[cols]
    return X


def _models(seed: int) -> Dict[str, object]:
    return {
        'logreg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight='balanced')),
        'rf': RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=3,
                                     class_weight='balanced_subsample', random_state=seed),
    }


def run_baselines(sequences: Sequences, folds: List[Dict[str, np.ndarray]], last_only: bool = True
                  ) -> Dict[str, Dict]:
    """Fit every baseline per fold on train subjects (imputing with train means).

    Returns ``{name: {'folds': [metrics per fold], 'oof': (N,) predictions per subject}}``;
    ``oof`` is -1 for subjects never tested and only filled when ``last_only``.
    """
    results = {name: {'folds': [], 'oof': np.full(len(sequences), -1, dtype=np.int64)}
               for name in _models(0)}
    for fold_i, idx in enumerate(folds):
        X_train, y_train = _matrix(sequences, idx['train'], last_only)
        X_test, y_test = _matrix(sequences, idx['test'], last_only)
        mean = np.nanmean(X_train, axis=0)
        X_train, X_test = _impute(X_train, mean), _impute(X_test, mean)
        for name, model in _models(cfg.SEED + fold_i).items():
            pred = model.fit(X_train, y_train).predict(X_test)
            cm = confusion_matrix(y_test, pred)
            results[name]['folds'].append({
                'acc': float((pred == y_test).mean()), 'macro_f1': float(per_class_f1(cm).mean()),
                'bal_acc': balanced_accuracy(cm), 'per_class_f1': per_class_f1(cm).tolist()})
            if last_only:
                results[name]['oof'][idx['test']] = pred
    return results


def _print(name: str, folds: List[Dict]) -> None:
    f1 = np.array([f['macro_f1'] for f in folds])
    acc = np.array([f['acc'] for f in folds])
    bal = np.array([f['bal_acc'] for f in folds])
    pc = np.array([f['per_class_f1'] for f in folds])
    print(f"  {name:8s} acc {acc.mean():.3f}±{acc.std(ddof=1):.3f}  macroF1 {f1.mean():.3f}±{f1.std(ddof=1):.3f}"
          f"  bal {bal.mean():.3f}±{bal.std(ddof=1):.3f}")
    print("           per-class F1  " + "  ".join(
        f"{c}={pc[:, i].mean():.3f}±{pc[:, i].std(ddof=1):.3f}" for i, c in enumerate(cfg.CLASSES)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', default=cfg.DATASET)
    cfg.DATASET = parser.parse_args().dataset

    sequences = load_dataset()
    y = subject_labels(sequences)
    print(f"Subjects: {len(sequences)} | final-label counts {cfg.CLASSES} = "
          f"{[int((y == c).sum()) for c in range(cfg.NUM_CLASSES)]}\nFeatures: {list(cfg.CLINICAL_COLS)}")
    folds = make_stratified_folds(sequences, cfg.N_FOLDS, cfg.SPLIT_SEED, cfg.CV_VAL_FRAC)
    for mode, last_only in (('subject-final-visit', True), ('all-visits-flattened', False)):
        print(f"\n{'=' * 72}\nMODE: {mode}\n{'=' * 72}")
        for name, res in run_baselines(sequences, folds, last_only).items():
            _print(name, res['folds'])
    print(f"\nMajority-class baseline (subject final label): {np.bincount(y).max() / len(y):.3f}")


if __name__ == '__main__':
    main()
