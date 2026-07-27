"""Quick tabular baselines on the same subject-level 5-fold splits.

Run:  python baselines.py

This answers: 'how well do Age/EDUC/SES/eTIV/nWBV/ASF/MF separate CN/MCI/AD
without any neural net / DS machinery?' If this is ~0.55–0.70 macro-F1 and
the clinical MLP ablation is ~0.20, the neural training recipe is broken —
not the features.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import Config
from OasisModule import Oasis2Dataset, make_stratified_folds, subject_labels

cfg = Config()


def _matrix_from_indices(sequences, indices, use_last_visit_only: bool = True):
    """Build (X, y). Default: one row per subject = final visit (matches
    subject-level stratification). Set use_last_visit_only=False to flatten
    all visits (matches the neural cross-sectional training view)."""
    X, y = [], []
    for i in indices:
        visits = sequences[i]
        chosen = [visits[-1]] if use_last_visit_only else visits
        for v in chosen:
            X.append(np.asarray(v['clinical_raw'], dtype=np.float64))
            y.append(int(v['label']))
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    # Impute NaN with column means computed on this split's X only would leak
    # val into itself slightly; for a diagnostic baseline, global-train impute
    # is done by the caller.
    return X, y


def _impute(X, mean):
    X = X.copy()
    inds = np.where(np.isnan(X))
    X[inds] = np.take(mean, inds[1])
    return X


def _eval(name, clf, Xtr, ytr, Xte, yte):
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    return {
        'name': name,
        'acc': float(accuracy_score(yte, pred)),
        'macro_f1': float(f1_score(yte, pred, average='macro', zero_division=0)),
        'bal_acc': float(balanced_accuracy_score(yte, pred)),
        'per_class_f1': f1_score(yte, pred, average=None,
                                 labels=list(range(cfg.NUM_CLASSES)),
                                 zero_division=0).tolist(),
    }


def main():
    sequences = Oasis2Dataset.load_all_sequences()
    y_subj = subject_labels(sequences)
    print(f"Subjects: {len(sequences)} | final-label CN/MCI/AD = "
          f"{[int((y_subj == c).sum()) for c in range(cfg.NUM_CLASSES)]}")
    print(f"Features: {cfg.CLINICAL_COLS}")
    folds = make_stratified_folds(sequences, n_folds=cfg.N_FOLDS,
                                  seed=cfg.SPLIT_SEED, val_frac=cfg.CV_VAL_FRAC)

    for mode, last_only in [('subject-final-visit', True),
                            ('all-visits-flattened', False)]:
        print("\n" + "=" * 72)
        print(f"MODE: {mode}")
        print("=" * 72)
        results = {'logreg': [], 'rf': []}
        for fold_i, idx in enumerate(folds):
            # Fit impute mean on TRAIN subjects only (no leakage).
            Xtr_raw, ytr = _matrix_from_indices(sequences, idx['train'], last_only)
            Xte_raw, yte = _matrix_from_indices(sequences, idx['test'], last_only)
            mean = np.nanmean(Xtr_raw, axis=0)
            Xtr = _impute(Xtr_raw, mean)
            Xte = _impute(Xte_raw, mean)

            lr = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, class_weight='balanced'),
            )
            rf = RandomForestClassifier(
                n_estimators=300, max_depth=4, min_samples_leaf=3,
                class_weight='balanced_subsample', random_state=cfg.SEED + fold_i,
            )
            r_lr = _eval('logreg', lr, Xtr, ytr, Xte, yte)
            r_rf = _eval('rf', rf, Xtr, ytr, Xte, yte)
            results['logreg'].append(r_lr)
            results['rf'].append(r_rf)
            print(f"  Fold {fold_i+1}: LR acc={r_lr['acc']:.3f} F1={r_lr['macro_f1']:.3f} "
                  f"pc={[f'{x:.2f}' for x in r_lr['per_class_f1']]} | "
                  f"RF acc={r_rf['acc']:.3f} F1={r_rf['macro_f1']:.3f} "
                  f"pc={[f'{x:.2f}' for x in r_rf['per_class_f1']]}")

        for key in ('logreg', 'rf'):
            acc = np.array([r['acc'] for r in results[key]])
            f1  = np.array([r['macro_f1'] for r in results[key]])
            bal = np.array([r['bal_acc'] for r in results[key]])
            pc  = np.array([r['per_class_f1'] for r in results[key]])
            print(f"  {key:8s}  acc {acc.mean():.3f}±{acc.std(ddof=1):.3f}  "
                  f"macroF1 {f1.mean():.3f}±{f1.std(ddof=1):.3f}  "
                  f"bal {bal.mean():.3f}±{bal.std(ddof=1):.3f}")
            print(f"           per-class F1  " + "  ".join(
                f"{cfg.CLASSES[i]}={pc[:, i].mean():.3f}±{pc[:, i].std(ddof=1):.3f}"
                for i in range(cfg.NUM_CLASSES)))

    maj = max(np.bincount(y_subj)) / len(y_subj)
    print(f"\nMajority-class baseline (subject final label): {maj:.3f}")
    #print("If sklearn >> clinical MLP ablation, the neural loss recipe is the bug.")


if __name__ == '__main__':
    main()
