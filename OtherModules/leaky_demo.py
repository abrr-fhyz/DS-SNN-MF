"""
Intentionally leaky high-validation demo (label leakage only)
==============================================================

PURPOSE
  Produce an inflated validation peak (typically >90% accuracy) by *removing*
  the label-leakage protection used in the honest pipeline, while still
  keeping the split methodologically sound at the subject level. This is NOT
  a result to report as performance — it is a controlled contrast experiment:

      "The model family can easily hit >90% on OASIS-3 when CDR/MMSE are
       left in the feature set, even under a clean subject-level split.
       Our honest subject-level CV is lower because we exclude those
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

Run:  python -m OtherModules.leaky_demo

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
from MODULE.DataModule import load_dataset, pick_column, read_table
from Oasis3Module import days_from_label


# ── Leaky feature set ─────────────────────
LEAKY_COLS = [*cfg.CLINICAL_COLS, 'CDR', 'MMSE']


def _cdr_rows_by_subject():
    df = read_table(cfg.spec.root / cfg.OASIS3_CDR_FILE)
    subject = pick_column(df, ['OASISID', 'Subject', 'SubjectID', 'Subject_ID'], 'subject id in CDR sheet')
    cdr = pick_column(df, ['CDRTOT', 'cdr', 'CDR'], 'total CDR')
    mmse = pick_column(df, ['MMSE', 'MMSCORE', 'mmse'], 'MMSE', required=False)
    days_col = pick_column(df, ['days_to_visit', 'daystovisit'], 'days', required=False)
    label_col = pick_column(df, ['OASIS_session_label', 'session_label', 'Session'], 'session label', required=False)
    if days_col is None and label_col is None:
        raise KeyError(f"CDR sheet needs a days column or session label; found {list(df.columns)}")
    days = pd.to_numeric(df[days_col], errors='coerce') if days_col else df[label_col].map(days_from_label)
    table = pd.DataFrame({
        'subject': df[subject].astype(str).str.strip(),
        'days': days.values,
        'cdr': pd.to_numeric(df[cdr], errors='coerce').values,
        'mmse': pd.to_numeric(df[mmse], errors='coerce').values if mmse else np.nan,
    }).dropna(subset=['days', 'cdr'])
    return {s: g.sort_values('days').reset_index(drop=True) for s, g in table.groupby('subject')}


def load_leaky_table() -> tuple[np.ndarray, np.ndarray, list]:
    """Clinical features plus the paired CDR/MMSE of each OASIS-3 visit (intentional leakage)."""
    cdr_by_subject = _cdr_rows_by_subject()
    X, y, subjects = [], [], []
    for sequence in load_dataset():
        for visit in sequence:
            sid = str(visit['subject_id'])
            rows = cdr_by_subject.get(sid)
            if rows is None or len(rows) == 0:
                continue
            day = days_from_label(visit['mri_id'])
            nearest = int(np.abs(rows['days'].values - day).argmin())
            cdr = float(rows.at[nearest, 'cdr'])
            mmse = float(rows.at[nearest, 'mmse'])
            feats = list(np.asarray(visit['clinical_raw'], dtype=np.float64)) + [cdr, mmse]
            X.append(feats)
            y.append(int(visit['label']))
            subjects.append(sid)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
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
        "macro-F1 on OASIS-3,\n"
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