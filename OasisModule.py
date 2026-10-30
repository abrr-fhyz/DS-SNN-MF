"""OASIS-2 loader: spreadsheet + RAW session folders → subject visit sequences.

Expected layout::

    <root>/oasis_longitudinal_demographics.xlsx
    <root>/OAS2_RAW_PART1/OAS2_0001_MR1/RAW/mpr-1.*
    <root>/OAS2_RAW_PART2/...

Labels come from per-visit CDR (see ``DataModule.cdr_to_label``). Only the first
``spec.max_visits`` visits per subject are kept (OASIS-2 has at most five).
"""
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config import cfg, DatasetSpec
from DataModule import (Sequences, cdr_to_label, make_visit, parse_sex, read_table, to_float)

_PART_DIRS = ('OAS2_RAW_PART1', 'OAS2_RAW_PART2')
_RAW_EXTS = ('.nii.gz', '.nii', '.nifti', '.img')
_RENAME = {'Subject ID': 'Subject.ID', 'SubjectID': 'Subject.ID',
           'MRI ID': 'MRI.ID', 'MRIID': 'MRI.ID', 'M/F': 'MF', 'MR Delay': 'MR.Delay'}


def _clinical_value(row: pd.Series, col: str) -> float:
    if col == 'MF':
        for key in ('MF', 'Sex', 'Gender'):
            if key in row.index and pd.notna(row[key]):
                return parse_sex(row[key])
        return float('nan')
    return to_float(row[col]) if col in row.index else float('nan')


def find_session_dir(root: Path, mri_id: str) -> Optional[Path]:
    for candidate in [root / part / mri_id for part in _PART_DIRS] + [root / mri_id]:
        if candidate.exists():
            return candidate
    return None


def resolve_mri_path(root: Path, mri_id: str) -> Optional[str]:
    """Path of the first MPRAGE volume in the session's RAW folder, if present."""
    session = find_session_dir(root, mri_id) if mri_id else None
    raw_dir = session / 'RAW' if session else None
    if raw_dir is None or not raw_dir.exists():
        return None
    for stem in (f"{mri_id}_mpr-1_anon", "mpr-1"):
        for ext in _RAW_EXTS:
            if (raw_dir / (stem + ext)).exists():
                return str(raw_dir / (stem + ext))
    for pattern in ('mpr-1.*', 'mpr-*.*', '*mpr-1*', '*mpr-*'):
        for path in sorted(raw_dir.glob(pattern)):
            if path.name.lower().endswith(_RAW_EXTS):
                return str(path)
    return None


def load_sequences(spec: DatasetSpec) -> Sequences:
    root = Path(spec.root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset dir not found: {root}")
    df = read_table(root / cfg.OASIS2_CSV.name).rename(columns=_RENAME)
    for needed in ('Subject.ID', 'CDR'):
        if needed not in df.columns:
            raise KeyError(f"Spreadsheet missing '{needed}'; found: {list(df.columns)}")

    sequences: List[List[dict]] = []
    for subject_id, group in df.groupby('Subject.ID', sort=False):
        if 'Visit' in group.columns:
            group = group.sort_values('Visit')
        visits = []
        for _, row in group.head(spec.max_visits).iterrows():
            mri_id = str(row.get('MRI.ID', '')).strip()
            mri_path, label = resolve_mri_path(root, mri_id), cdr_to_label(row.get('CDR'))
            if mri_path is None or label is None:
                continue
            clinical = [_clinical_value(row, c) for c in spec.clinical_cols]
            visits.append(make_visit(subject_id, mri_id, int(row.get('Visit', 0)), mri_path, clinical, label))
        if visits:
            sequences.append(visits)

    if not sequences:
        raise RuntimeError("No visits with resolvable MRI files found. Check that OAS2_RAW_PART1/2 "
                           "contain OAS2_*_MR* session dirs with RAW/.")
    return sequences
