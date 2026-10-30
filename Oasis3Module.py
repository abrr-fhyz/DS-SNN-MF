"""OASIS-3 loader: NITRC-IR style session folders + clinical CSVs → subject visit sequences.

Inputs (under ``spec.root``; filenames in ``config.py``)::

    OASIS3_UDSb4_cdr.csv            CDR per clinical visit            (required, labels)
    OASIS3_demographics.csv         sex, education, APOE, age at entry (needed for those features)
    OASIS3_Freesurfer_output.csv    ICV / brain volume per MR session  (needed for eTIV, nWBV)
    OAS30001_MR_d0129/**/*T1w.nii*  one folder per MR session (or BIDS sub-*/ses-d*/)

Scans and labels are paired by time: every MR session is matched to the nearest CDR visit
within ``cfg.MAX_CDR_GAP_DAYS``. Column names are resolved through alias lists (see
``docs/OASIS3.md``); a missing column raises an error listing what the file contains.
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import cfg, DatasetSpec
from DataModule import (Sequences, cdr_to_label, make_visit, parse_sex, pick_column, read_table,
                        resolve_table_path)

SUPPORTED_FEATURES = ('Age', 'EDUC', 'MF', 'eTIV', 'nWBV', 'APOE4')
_SESSION_ID = re.compile(r'(OAS3\d+)_MR_d(\d+)')
_SESSION_DIR = re.compile(r'^OAS3\d+_MR_d\d+$')
_BIDS_SUBJECT = re.compile(r'^sub-(OAS3\d+)$')
_BIDS_SESSION = re.compile(r'^ses-d(\d+)$')
_SUBJECT_ALIASES = ['OASISID', 'Subject', 'SubjectID', 'Subject_ID']


def session_id(text) -> Optional[str]:
    """Canonical ``OAS30001_MR_d0129`` extracted from any string containing one."""
    match = _SESSION_ID.search(str(text))
    return f"{match.group(1)}_MR_d{int(match.group(2)):04d}" if match else None


def days_from_label(label) -> float:
    """Day offset from a label ending in ``d0129``; NaN if absent."""
    match = re.search(r'd(\d+)\s*$', str(label))
    return float(match.group(1)) if match else float('nan')


def _session_from_dir(path: Path) -> Optional[str]:
    if _SESSION_DIR.match(path.name):
        return session_id(path.name)
    bids_session, bids_subject = _BIDS_SESSION.match(path.name), _BIDS_SUBJECT.match(path.parent.name)
    if bids_session and bids_subject:
        return f"{bids_subject.group(1)}_MR_d{int(bids_session.group(1)):04d}"
    return None


def index_mri_sessions(root: Path, t1_glob: str = cfg.OASIS3_T1_GLOB) -> Dict[str, str]:
    """Map session id → first T1w NIfTI found inside its folder (sorted, so run-01 wins)."""
    found: Dict[str, str] = {}
    for dirpath, dirnames, _ in os.walk(root):
        sid = _session_from_dir(Path(dirpath))
        if sid is None:
            continue
        dirnames[:] = []
        scans = sorted(p for p in Path(dirpath).rglob(t1_glob) if p.is_file())
        if scans:
            found[sid] = str(scans[0])
    return found


# ──────────────────────────────────────────────────────────────
# Clinical tables
# ──────────────────────────────────────────────────────────────

def _load_cdr(path: Path) -> Dict[str, pd.DataFrame]:
    """Per-subject CDR visits sorted by day, with columns ``days, cdr, age``."""
    df = read_table(path)
    subject = pick_column(df, _SUBJECT_ALIASES, 'subject id in CDR sheet')
    cdr = pick_column(df, ['CDRTOT', 'cdr', 'CDR'], 'total CDR')
    days_col = pick_column(df, ['days_to_visit', 'daystovisit'], 'days', required=False)
    label_col = pick_column(df, ['OASIS_session_label', 'session_label', 'Session'], 'session label', required=False)
    age_col = pick_column(df, ['age at visit', 'age_at_visit', 'AgeAtVisit'], 'age', required=False)
    if days_col is None and label_col is None:
        raise KeyError(f"CDR sheet needs a days column or session label; found {list(df.columns)}")

    days = pd.to_numeric(df[days_col], errors='coerce') if days_col else df[label_col].map(days_from_label)
    table = pd.DataFrame({
        'subject': df[subject].astype(str).str.strip(),
        'days': days.values,
        'cdr': pd.to_numeric(df[cdr], errors='coerce').values,
        'age': pd.to_numeric(df[age_col], errors='coerce').values if age_col else np.nan,
    }).dropna(subset=['days', 'cdr'])
    return {s: g.sort_values('days').reset_index(drop=True) for s, g in table.groupby('subject')}


def _apoe4_count(value) -> float:
    digits = re.sub(r'\D', '', str(value))
    return float(digits.count('4')) if len(digits) >= 2 else float('nan')


def _load_demographics(path: Path, features) -> pd.DataFrame:
    """Per-subject static features (``MF, EDUC, APOE4, age_entry``) indexed by subject id."""
    df = read_table(path)
    out = pd.DataFrame(index=df[pick_column(df, _SUBJECT_ALIASES, 'subject id in demographics')]
                       .astype(str).str.strip())
    if 'MF' in features:
        out['MF'] = df[pick_column(df, ['M/F', 'Sex', 'Gender', 'MF'], 'sex')].map(parse_sex).values
    if 'EDUC' in features:
        out['EDUC'] = pd.to_numeric(df[pick_column(df, ['Education', 'EDUC', 'YearsEducation'], 'education')],
                                    errors='coerce').values
    if 'APOE4' in features:
        out['APOE4'] = df[pick_column(df, ['APOE', 'APOE_genotype', 'apoe'], 'APOE')].map(_apoe4_count).values
    entry = pick_column(df, ['AgeatEntry', 'age_at_entry', 'AgeEntry'], 'age at entry', required=False)
    out['age_entry'] = pd.to_numeric(df[entry], errors='coerce').values if entry else np.nan
    return out[~out.index.duplicated()]


def _load_freesurfer(path: Path) -> pd.DataFrame:
    """Per-session ``eTIV`` (intracranial volume) and ``nWBV`` (brain volume / ICV)."""
    df = read_table(path)
    sessions = None
    for alias in ['MR ID', 'MR_session', 'MR_ID', 'Session', 'session_label', 'FS_FSDATA_ID']:
        col = pick_column(df, [alias], 'session', required=False)
        ids = df[col].map(session_id) if col else None
        if ids is not None and ids.notna().any():
            sessions = ids
            break
    if sessions is None:
        raise KeyError(f"FreeSurfer table has no column holding OAS3xxxx_MR_dXXXX ids; found {list(df.columns)}")
    icv = pd.to_numeric(df[pick_column(df, ['IntraCranialVol', 'ICV', 'eTIV'], 'intracranial volume')],
                        errors='coerce')
    brain = pd.to_numeric(df[pick_column(df, ['BrainSegVol', 'BrainSegVolume'], 'brain volume')], errors='coerce')
    out = pd.DataFrame({'eTIV': icv.values, 'nWBV': (brain / icv).values}, index=sessions.values)
    return out[out.index.notna() & ~out.index.duplicated()]


# ──────────────────────────────────────────────────────────────
# Assembly
# ──────────────────────────────────────────────────────────────

def _table_exists(path: Path) -> bool:
    try:
        resolve_table_path(path)
        return True
    except FileNotFoundError:
        return False


def _lookup(table: Optional[pd.DataFrame], key: str, column: str) -> float:
    if table is None or key not in table.index or column not in table.columns:
        return float('nan')
    return float(table.at[key, column])


def _age_at_scan(subject: str, scan_day: float, cdr_row: pd.Series, demographics: Optional[pd.DataFrame]) -> float:
    entry = _lookup(demographics, subject, 'age_entry')
    if not np.isnan(entry):
        return entry + scan_day / 365.25
    return float(cdr_row['age']) + (scan_day - float(cdr_row['days'])) / 365.25


def load_sequences(spec: DatasetSpec) -> Sequences:
    root = Path(spec.root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset dir not found: {root}")
    features = list(spec.clinical_cols)
    unknown = [f for f in features if f not in SUPPORTED_FEATURES]
    if unknown:
        raise ValueError(f"OASIS-3 clinical columns {unknown} unsupported; choose from {SUPPORTED_FEATURES}")

    scans = index_mri_sessions(root)
    cdr_by_subject = _load_cdr(root / cfg.OASIS3_CDR_FILE)
    demo_path = root / cfg.OASIS3_DEMOGRAPHICS_FILE
    needs_demographics = any(f in features for f in ('MF', 'EDUC', 'APOE4')) or \
        ('Age' in features and _table_exists(demo_path))   # age falls back to the CDR sheet
    demographics = _load_demographics(demo_path, features) if needs_demographics else None
    freesurfer = _load_freesurfer(root / cfg.OASIS3_FREESURFER_FILE) \
        if any(f in features for f in ('eTIV', 'nWBV')) else None

    by_subject: Dict[str, List[tuple]] = {}
    for sid, path in scans.items():
        subject = sid.split('_')[0]
        by_subject.setdefault(subject, []).append((float(sid.rsplit('_d', 1)[1]), sid, path))

    sequences: List[List[dict]] = []
    for subject in sorted(by_subject):
        cdr = cdr_by_subject.get(subject)
        if cdr is None:
            continue
        visits = []
        for day, sid, path in sorted(by_subject[subject]):
            gaps = np.abs(cdr['days'].values - day)
            nearest = int(gaps.argmin())
            label = cdr_to_label(cdr.at[nearest, 'cdr'])
            if gaps[nearest] > cfg.MAX_CDR_GAP_DAYS or label is None:
                continue
            values = {'MF': _lookup(demographics, subject, 'MF'), 'EDUC': _lookup(demographics, subject, 'EDUC'),
                      'APOE4': _lookup(demographics, subject, 'APOE4'),
                      'eTIV': _lookup(freesurfer, sid, 'eTIV'), 'nWBV': _lookup(freesurfer, sid, 'nWBV')}
            if 'Age' in features:
                values['Age'] = _age_at_scan(subject, day, cdr.loc[nearest], demographics)
            visits.append(make_visit(subject, sid, len(visits) + 1, path,
                                     [values[f] for f in features], label))
        if visits:
            sequences.append(visits[-spec.max_visits:])

    if not sequences:
        raise RuntimeError(
            f"No OASIS-3 subjects with a T1w scan and a CDR within {cfg.MAX_CDR_GAP_DAYS} days "
            f"({len(scans)} scans, {len(cdr_by_subject)} subjects with CDR). Check paths in config.py.")
    return sequences
