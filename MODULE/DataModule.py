"""Data layer for the OASIS-3 loader.

A *sequence* is one subject's chronologically ordered list of *visits*; a visit is a dict with
``subject_id, mri_id, visit_number, mri_path, clinical_raw, clinical, label`` and optional
``pet_id, pet_path, ct_id, ct_path``. ``Oasis3Module`` only builds sequences. Splitting,
clinical normalisation, volume tensor loading and torch datasets live here.
"""
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import Tensor
from torch.utils.data import Dataset

from config import cfg

Visit = Dict[str, object]
Sequences = List[List[Visit]]
_TABLE_EXTS = ('.xlsx', '.xls', '.csv', '.tsv')


def set_seed(seed: int = cfg.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────
# Parsing helpers
# ──────────────────────────────────────────────────────────────

def cdr_to_label(cdr) -> Optional[int]:
    """CDR 0 → CN (0), 0.5 → MCI (1), ≥1 → AD (2); None when missing."""
    value = to_float(cdr)
    if np.isnan(value):
        return None
    if value == 0.0:
        return 0
    return 1 if value == 0.5 else 2


def to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def parse_sex(value) -> float:
    """M/male/1 → 1.0, F/female/0 → 0.0, anything else NaN."""
    key = str(value).strip().upper()
    if key in ('M', 'MALE', '1'):
        return 1.0
    return 0.0 if key in ('F', 'FEMALE', '0') else float('nan')


def resolve_table_path(path: Path) -> Path:
    """Return ``path`` or the same stem with another spreadsheet extension."""
    path = Path(path)
    if path.exists():
        return path
    for ext in _TABLE_EXTS:
        candidate = path.with_suffix(ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Table not found: {path} (also tried {', '.join(_TABLE_EXTS)})")


def read_table(path: Path) -> pd.DataFrame:
    path = resolve_table_path(path)
    suffix = path.suffix.lower()
    if suffix in ('.xlsx', '.xls'):
        return pd.read_excel(path, engine='openpyxl' if suffix == '.xlsx' else None)
    return pd.read_csv(path, sep='\t' if suffix == '.tsv' else ',')


def _canon(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def pick_column(df: pd.DataFrame, aliases: List[str], what: str, required: bool = True) -> Optional[str]:
    """First column whose canonical name (case/punctuation-insensitive) matches an alias."""
    lookup = {_canon(c): c for c in df.columns}
    for alias in aliases:
        if _canon(alias) in lookup:
            return lookup[_canon(alias)]
    if required:
        raise KeyError(f"No column for {what}; tried {aliases}, found {list(df.columns)}")
    return None


def make_visit(subject_id, mri_id: str, visit_number: int, mri_path: str,
               clinical_raw, label: int, pet_id=None, pet_path=None,
               ct_id=None, ct_path=None) -> Visit:
    raw = np.asarray(clinical_raw, dtype=np.float32)
    return {'subject_id': subject_id, 'mri_id': mri_id, 'visit_number': int(visit_number),
            'mri_path': str(mri_path), 'clinical_raw': raw, 'clinical': raw, 'label': int(label),
            'pet_id': pet_id, 'pet_path': str(pet_path) if pet_path else None,
            'ct_id': ct_id, 'ct_path': str(ct_path) if ct_path else None}


def load_dataset() -> Sequences:
    """Sequences for ``cfg.DATASET`` (visits with unresolvable MRI/labels are dropped)."""
    if cfg.DATASET != 'oasis3':
        raise KeyError(f"No loader registered for dataset {cfg.DATASET!r}")
    from Oasis3Module import load_sequences
    return load_sequences(cfg.spec)


# ──────────────────────────────────────────────────────────────
# Splits and normalisation
# ──────────────────────────────────────────────────────────────

def subject_labels(sequences: Sequences) -> np.ndarray:
    """One label per subject: the final visit's class (also the stratification target)."""
    return np.array([int(seq[-1]['label']) for seq in sequences], dtype=np.int64)


def _stratified_split(indices: np.ndarray, y: np.ndarray, test_size: float, seed: int
                      ) -> Tuple[np.ndarray, np.ndarray]:
    try:
        keep, held = train_test_split(indices, test_size=test_size, random_state=seed, stratify=y)
    except ValueError:  # a class too small to stratify
        shuffled = np.random.default_rng(seed).permutation(indices)
        n_held = max(1, int(len(shuffled) * test_size))
        held, keep = shuffled[:n_held], shuffled[n_held:]
    return np.asarray(keep, dtype=np.int64), np.asarray(held, dtype=np.int64)


def make_stratified_folds(sequences: Sequences, n_folds: int = cfg.N_FOLDS, seed: int = cfg.SPLIT_SEED,
                          val_frac: float = cfg.CV_VAL_FRAC) -> List[Dict[str, np.ndarray]]:
    """Subject-level stratified k-fold: test = fold i; a stratified ``val_frac`` of the rest is
    held out for early stopping."""
    y = subject_labels(sequences)
    folds = []
    for i, (trainval, test) in enumerate(StratifiedKFold(n_folds, shuffle=True, random_state=seed)
                                         .split(np.arange(len(y)), y)):
        train, val = _stratified_split(trainval, y[trainval], val_frac, seed + i)
        folds.append({'train': train, 'val': val, 'test': np.asarray(test, dtype=np.int64)})
    return folds


def make_holdout_split(sequences: Sequences, seed: int = cfg.SPLIT_SEED) -> Dict[str, np.ndarray]:
    """Single stratified train/val/test split using ``cfg.TRAIN_FRAC`` / ``cfg.VAL_FRAC``."""
    y = subject_labels(sequences)
    idx = np.arange(len(y))
    trainval, test = _stratified_split(idx, y, 1.0 - cfg.TRAIN_FRAC - cfg.VAL_FRAC, seed)
    train, val = _stratified_split(trainval, y[trainval], cfg.VAL_FRAC / (cfg.TRAIN_FRAC + cfg.VAL_FRAC), seed)
    return {'train': train, 'val': val, 'test': test}


class ClinicalScaler:
    """Z-score with NaN imputation; statistics come from the training subjects only."""

    def fit(self, sequences: Sequences) -> 'ClinicalScaler':
        raw = np.stack([v['clinical_raw'] for seq in sequences for v in seq]).astype(np.float64)
        valid = ~np.isnan(raw)
        count = np.maximum(valid.sum(axis=0), 1)
        self.mean = np.where(valid.any(axis=0), np.where(valid, raw, 0.0).sum(axis=0) / count, 0.0)
        var = (np.where(valid, raw - self.mean, 0.0) ** 2).sum(axis=0) / count
        std = np.sqrt(var)
        self.std = np.where(std < 1e-6, 1.0, std)
        return self

    def transform(self, sequences: Sequences) -> Sequences:
        def scale(raw):
            filled = np.where(np.isnan(raw), self.mean, raw)
            return ((filled - self.mean) / self.std).astype(np.float32)
        return [[{**v, 'clinical': scale(np.asarray(v['clinical_raw'], dtype=np.float64))} for v in seq]
                for seq in sequences]


def split_sequences(sequences: Sequences, indices: Dict[str, np.ndarray]) -> Dict[str, Sequences]:
    """Materialise train/val/test sequences, normalised with train-only statistics."""
    scaler = ClinicalScaler().fit([sequences[i] for i in indices['train']])
    return {name: scaler.transform([sequences[i] for i in indices[name]])
            for name in ('train', 'val', 'test')}


# ──────────────────────────────────────────────────────────────
# Imaging tensors
# ──────────────────────────────────────────────────────────────

def processed_tensor_path(session_id: str) -> Path:
    d, h, w = cfg.MRI_SHAPE
    return cfg.MRI_PROCESSED_DIR / session_id / f"tensor_{d}x{h}x{w}.pt"


def load_volume_tensor(session_id: str, channels: int) -> Tensor:
    """(C, D, H, W) tensor written by ``OtherModules.mri_preprocess``."""
    path = processed_tensor_path(str(session_id))
    if not path.exists():
        raise FileNotFoundError(
            f"Processed volume missing for '{session_id}' ({path}). "
            f"Run: python -m OtherModules.mri_preprocess")
    tensor = torch.load(path, map_location='cpu')
    if tensor.ndim != 4 or tensor.shape[0] < channels:
        raise ValueError(f"{path}: expected (>= {channels}, D, H, W), got {tuple(tensor.shape)}")
    return tensor[:channels].float()


def load_mri_tensor(visit: Visit) -> Tensor:
    return load_volume_tensor(str(visit['mri_id']), cfg.mri_in_channels())


def _zero_volume(channels: int) -> Tensor:
    return torch.zeros((channels, *cfg.MRI_SHAPE))


def _visit_volume(visit: Visit, source: str, id_key: str, channels: int) -> Tuple[Tensor, bool]:
    """Load a processed volume if the source is active and the visit has a session id."""
    if not cfg.uses(source):
        return _zero_volume(channels), False
    session_id = visit.get(id_key)
    if not session_id:
        return _zero_volume(channels), False
    return load_volume_tensor(str(session_id), channels), True


def visit_tensors(visit: Visit, augment: bool = False) -> Dict[str, Tensor]:
    """MRI / PET / CT volumes, presence flags, clinical vector (no label)."""
    mri, _ = _visit_volume(visit, 'mri', 'mri_id', cfg.mri_in_channels())
    pet, has_pet = _visit_volume(visit, 'pet', 'pet_id', 1)
    ct, has_ct = _visit_volume(visit, 'ct', 'ct_id', 1)
    if augment:
        if cfg.uses('mri'):
            mri = augment_mri(mri)
        if has_pet:
            pet = augment_mri(pet)
        if has_ct:
            ct = augment_mri(ct)
    return {'mri': mri, 'pet': pet, 'ct': ct,
            'has_pet': torch.tensor(has_pet), 'has_ct': torch.tensor(has_ct),
            'clinical': torch.tensor(visit['clinical'], dtype=torch.float32)}


def augment_mri(mri: Tensor) -> Tensor:
    """Flip / shift / intensity jitter on (C, D, H, W); extra channels only move geometrically."""
    if random.random() < 0.5:
        mri = torch.flip(mri, dims=[-1])
    shifts = tuple(random.randint(-2, 2) for _ in range(3))
    if any(shifts):
        mri = torch.roll(mri, shifts=shifts, dims=(-3, -2, -1))
    t1 = mri[0:1]
    t1 = t1 * (1.0 + (random.random() - 0.5) * 0.1) + (random.random() - 0.5) * 0.05
    if random.random() < 0.3:
        t1 = t1.clamp(min=0.0).pow(1.0 + (random.random() - 0.5) * 0.3)
    if random.random() < 0.3:
        t1 = t1 + torch.randn_like(t1) * 0.01
    t1 = t1.clamp(0.0, 1.0)
    return t1 if mri.shape[0] == 1 else torch.cat([t1, mri[1:].clamp(0.0, 1.0)], dim=0)


# ──────────────────────────────────────────────────────────────
# Torch datasets
# ──────────────────────────────────────────────────────────────

class CrossSectionalWrapper(Dataset):
    """One sample per visit. ``visits='last'`` keeps each subject's final visit only;
    ``'all'`` keeps every visit with its own label (training on larger cohorts)."""

    def __init__(self, sequences: Sequences, visits: str = 'last', augment: bool = False):
        assert visits in ('last', 'all'), visits
        self.augment = augment
        self.samples: List[Visit] = [seq[-1] for seq in sequences] if visits == 'last' \
            else [v for seq in sequences for v in seq]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        visit = self.samples[idx]
        item = visit_tensors(visit, augment=self.augment)
        item['label'] = torch.tensor(int(visit['label']), dtype=torch.long)
        return item

    def labels(self) -> List[int]:
        return [int(v['label']) for v in self.samples]

    def label_counts(self, num_classes: int = cfg.NUM_CLASSES) -> List[int]:
        counts = [0] * num_classes
        for label in self.labels():
            counts[label] += 1
        return counts


class SubjectDataset(Dataset):
    """One item per subject: all visits in order, target = final visit's label."""

    def __init__(self, sequences: Sequences):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        visits = [visit_tensors(v) for v in self.sequences[idx]]
        return {'visits': visits,
                'label': torch.tensor(int(self.sequences[idx][-1]['label']), dtype=torch.long)}


def longitudinal_collate_fn(batch: List[dict]) -> dict:
    """Pad each subject's visit list to the batch maximum; ``has_visit`` marks real visits."""
    labels = torch.stack([item['label'] for item in batch])
    volume_keys = ('mri', 'pet', 'ct', 'clinical')
    steps = []
    for v in range(max(len(item['visits']) for item in batch)):
        stacked = {k: [] for k in volume_keys}
        present, has_pet, has_ct = [], [], []
        for item in batch:
            real = v < len(item['visits'])
            visit = item['visits'][v] if real else item['visits'][0]
            for key in volume_keys:
                stacked[key].append(visit[key] if real else torch.zeros_like(visit[key]))
            present.append(real)
            has_pet.append(bool(visit['has_pet']) if real else False)
            has_ct.append(bool(visit['has_ct']) if real else False)
        step = {k: torch.stack(stacked[k]) for k in volume_keys}
        step['has_visit'] = present
        step['has_pet'] = torch.tensor(has_pet)
        step['has_ct'] = torch.tensor(has_ct)
        steps.append(step)
    return {'visits': steps, 'label': labels}
