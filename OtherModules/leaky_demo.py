"""
Intentionally leaky high-validation demo (label leakage only)
==============================================================

PURPOSE
  Produce an inflated validation peak (typically >90% accuracy) by *removing*
  the label-leakage protection used in the honest pipeline, while still
  keeping the split methodologically sound at the subject level. This is NOT
  a result to report as performance — it is a controlled contrast experiment:

      "The model family can easily hit >90% on OASIS-2 when CDR/MMSE are
       left in the feature set, even under a clean subject-level split.
       Our honest subject-level CV (~0.56) is lower because we exclude those
       features — not because we failed to optimise or because of a leaky
       split."

LEAKS ENABLED (deliberate)
  1. Label leakage: CDR (and MMSE) are used as *input features*, while labels
     are derived from CDR. Predicting the label from CDR is nearly identity.
  2. Optimistic selection: checkpoint on raw val accuracy with no warmup /
     macro-F1 stability requirement.

LEAKS REMOVED (fixed vs. the original demo)
  - Subject leakage: the split is now drawn over *subjects*, not visits, so
    no subject's visits can appear in both train and val simultaneously.

Run:  python leaky_demo.py

Outputs a short formal paragraph + metrics you can paste into a WIP paper
"Methodological note" subsection.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader, Dataset

from config import cfg


# ── Leaky feature set ─────────────────────
LEAKY_COLS = ['Age', 'EDUC', 'SES', 'eTIV', 'nWBV', 'ASF', 'MF', 'CDR', 'MMSE']
#LEAKY_COLS = ['Age', 'EDUC', 'SES', 'eTIV', 'nWBV', 'ASF', 'MF']

def _cdr_to_label(cdr) -> int | None:
    try:
        cdr = float(cdr)
    except (TypeError, ValueError):
        return None
    if np.isnan(cdr):
        return None
    if cdr == 0.0:
        return 0
    if cdr == 0.5:
        return 1
    return 2


def _mf(row) -> float:
    for key in ('MF', 'M/F', 'Sex'):
        if key in row.index and pd.notna(row[key]):
            v = str(row[key]).strip().upper()
            if v in ('M', 'MALE', '1'):
                return 1.0
            if v in ('F', 'FEMALE', '0'):
                return 0.0
    return np.nan


def load_leaky_table(csv_path: Path = cfg.OASIS2_CSV) -> tuple[np.ndarray, np.ndarray, list]:
    path = csv_path if csv_path.exists() else csv_path.with_suffix('.csv')
    if path.suffix.lower() in ('.xlsx', '.xls'):
        df = pd.read_excel(path, engine='openpyxl' if path.suffix == '.xlsx' else None)
    else:
        df = pd.read_csv(path)
    rename = {
        'Subject ID': 'Subject.ID', 'MRI ID': 'MRI.ID', 'M/F': 'MF',
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    X, y, subjects = [], [], []
    for _, row in df.iterrows():
        lab = _cdr_to_label(row.get('CDR'))
        if lab is None:
            continue
        feats = []
        for c in LEAKY_COLS:
            if c == 'MF':
                feats.append(_mf(row))
            elif c in row.index and pd.notna(row[c]):
                try:
                    feats.append(float(row[c]))
                except (TypeError, ValueError):
                    feats.append(np.nan)
            else:
                feats.append(np.nan)
        X.append(feats)
        y.append(lab)
        subjects.append(str(row.get('Subject.ID', '')))
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    # Impute + z-score (fit on all rows — another mild leak).
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X, y, subjects


class VisitTable(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class LeakyMLP(nn.Module):
    """Slightly over-parameterised MLP — easy to fit leaked labels."""
    def __init__(self, in_dim: int, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def _subject_level_split(subjects: list, val_frac: float, seed: int):
    """Split *subjects* (not visits) into train/val so no subject's visits
    can appear on both sides. Returns (train_idx, val_idx, overlap)."""
    subj_arr = np.asarray(subjects)
    unique_subjects = np.unique(subj_arr)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_subjects)

    n_val_subjects = max(1, int(len(unique_subjects) * val_frac))
    val_subjects = set(unique_subjects[:n_val_subjects])
    train_subjects = set(unique_subjects[n_val_subjects:])

    val_idx = np.where(np.isin(subj_arr, list(val_subjects)))[0]
    train_idx = np.where(np.isin(subj_arr, list(train_subjects)))[0]

    # Sanity check: this should always be zero now.
    overlap = len(train_subjects & val_subjects)
    return train_idx, val_idx, overlap


def run_leaky(seed: int = 0, max_epochs: int = 80, val_frac: float = 0.2):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    X, y, subjects = load_leaky_table()
    n = len(y)

    # SUBJECT-LEVEL split (leak #2 removed): a subject's visits are entirely
    # in train or entirely in val, never both.
    train_idx, val_idx, overlap = _subject_level_split(subjects, val_frac, seed)

    subj = np.asarray(subjects)
    val_subj = set(subj[val_idx])

    train_ds = VisitTable(X[train_idx], y[train_idx])
    val_ds   = VisitTable(X[val_idx], y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeakyMLP(X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    peak_val_acc = -1.0
    peak_val_f1  = -1.0
    peak_epoch   = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb.to(device)).argmax(1).cpu().tolist()
                preds.extend(pred); labels.extend(yb.tolist())
        acc = accuracy_score(labels, preds)
        f1  = f1_score(labels, preds, average='macro', zero_division=0)
        history.append((epoch, acc, f1))
        if acc > peak_val_acc:
            peak_val_acc, peak_val_f1, peak_epoch = acc, f1, epoch

    # Final report + formal caveat.
    print("=" * 72)
    print("LEAKY DEMO — inflated validation via label leakage only")
    print("(subject-level split, no split leakage)")
    print("=" * 72)
    print(f"Features ({len(LEAKY_COLS)}): {LEAKY_COLS}")
    print(f"Visits: {n} | train={len(train_idx)} val={len(val_idx)}")
    print(f"Subject overlap train∩val: {overlap} / {len(val_subj)} val subjects "
          f"({100.0 * overlap / max(len(val_subj), 1):.1f}%)")
    print(f"Peak VAL accuracy: {peak_val_acc:.4f}  |  peak macro-F1: {peak_val_f1:.4f}  "
          f"(epoch {peak_epoch})")
    print()
    print(classification_report(labels, preds, target_names=cfg.CLASSES, digits=3,
                                zero_division=0))
    print("-" * 72)
    print("FORMAL CAVEAT (paper-ready)")
    print("-" * 72)
    print(
        "Under an intentionally leaky feature set — including CDR/MMSE as input\n"
        "features while defining labels from CDR (label leakage) — a simple MLP\n"
        f"reaches {peak_val_acc*100:.1f}% validation accuracy / {peak_val_f1*100:.1f}% "
        "macro-F1 on OASIS-2,\n"
        "even under a clean subject-level split with zero subject overlap between\n"
        "train and validation. This demonstrates that high single-split validation\n"
        "peaks are easy to obtain when a near-identity feature is left in the input\n"
        "set; they do not reflect generalisation on features that would actually be\n"
        "available at prediction time. Our primary results therefore exclude\n"
        "CDR/MMSE from the feature set and use subject-level 5-fold CV, yielding a\n"
        "lower but credible estimate (see CV table)."
    )
    print("=" * 72)

    out = Path("leaky_demo_metrics.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"peak_val_acc={peak_val_acc:.6f}\n")
        f.write(f"peak_val_macro_f1={peak_val_f1:.6f}\n")
        f.write(f"peak_epoch={peak_epoch}\n")
        f.write(f"subject_overlap={overlap}\n")
        f.write(f"features={LEAKY_COLS}\n")
    print(f"Wrote {out}")
    return peak_val_acc, peak_val_f1


if __name__ == "__main__":
    run_leaky()