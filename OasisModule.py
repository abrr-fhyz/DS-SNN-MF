import os
import shutil
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from pathlib import Path
from scipy.ndimage import zoom
from typing import Dict, List, Optional, Tuple
from torch import Tensor
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import Config

cfg = Config()
_PART_DIRS   = ('OAS2_RAW_PART1', 'OAS2_RAW_PART2')
_CLINICAL_COLS = cfg.CLINICAL_COLS

# ──────────────────────────────────────────────────────────────
# Clinical Pre-processing Util
# ──────────────────────────────────────────────────────────────

def _cdr_to_label(cdr) -> Optional[int]:
    """Per-visit CDR → 3-class label. Returns None for missing CDR.
        CDR 0.0 → CN (0),  CDR 0.5 → MCI (1),  CDR ≥ 1 → AD (2)
    """
    if cdr is None:
        return None
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

def _clinical_value(row, col: str) -> float:
    """Extract one clinical feature. MF → {0=F, 1=M}; others numeric."""
    if col == 'MF':
        for key in ('MF', 'M/F', 'Sex', 'Gender'):
            if key in row.index and pd.notna(row[key]):
                v = str(row[key]).strip().upper()
                if v in ('M', 'MALE', '1'):
                    return 1.0
                if v in ('F', 'FEMALE', '0'):
                    return 0.0
        return np.nan
    if col in row.index and pd.notna(row[col]):
        try:
            return float(row[col])
        except (TypeError, ValueError):
            return np.nan
    return np.nan

def subject_labels(sequences: List[List[dict]]) -> np.ndarray:
    """One label per subject = final (last) visit CDR class — for stratification."""
    return np.array([int(seq[-1]['label']) for seq in sequences], dtype=np.int64)

def make_stratified_folds(sequences: List[List[dict]], n_folds: int = cfg.N_FOLDS, seed: int = cfg.SPLIT_SEED, val_frac: float = cfg.CV_VAL_FRAC) -> List[Dict[str, np.ndarray]]:
    """Subject-level stratified k-fold splits.
        For each fold i:
            test  = fold i
            from the remaining subjects, hold out ``val_frac`` (stratified) for
            early-stopping; the rest is train..
    """
    y = subject_labels(sequences)
    idx = np.arange(len(sequences))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for fold_i, (trainval_idx, test_idx) in enumerate(skf.split(idx, y)):
        y_tv = y[trainval_idx]
        try:
            tr, va = train_test_split(
                trainval_idx, test_size=val_frac, random_state=seed + fold_i,
                stratify=y_tv)
        except ValueError:
            rng = np.random.default_rng(seed + fold_i)
            shuffled = trainval_idx.copy()
            rng.shuffle(shuffled)
            n_val = max(1, int(len(shuffled) * val_frac))
            va, tr = shuffled[:n_val], shuffled[n_val:]
        folds.append({
            'train': np.asarray(tr, dtype=np.int64),
            'val':   np.asarray(va, dtype=np.int64),
            'test':  np.asarray(test_idx, dtype=np.int64),
        })
    return folds


# ──────────────────────────────────────────────────────────────
# OASIS-2 Dataset Class
# ──────────────────────────────────────────────────────────────

class Oasis2Dataset(Dataset):
    """OASIS-2 longitudinal structural-MRI dataset.

    Expected layout:
        <dataset_dir>/
            oasis_longitudinal_demographics.xlsx
            OAS2_RAW_PART1/OAS2_0001_MR1/RAW/...
            OAS2_RAW_PART2/OAS2_0150_MR1/RAW/...

    __getitem__ returns one subject's visit sequence:
        {
          'visits': [
              {'mri': (C,D,H,W), 'clinical': (n_clin,), 'label': LongTensor},
              ...
          ],
          'label': LongTensor,   # sequence target = LAST visit's label
        }

    MRI tensor channels (see ``Config.mri_in_channels``):
      0: registered skull-stripped T1
      1: soft GM                         (if MRI_USE_GM_CHANNEL)
      2–3: hippocampus / ventricle priors (if MRI_USE_ROI_PRIOR_CHANNELS)
    Prefer ``MRI_PROCESSED_DIR`` tensors from ``mri_preprocess.py``.
    """
    CLASS_MAP = {'CN': 0, 'MCI': 1, 'AD': 2}

    def __init__(self, split: str = 'train', dataset_dir: Path = cfg.OASIS2_DIR, csv_path: Path = cfg.OASIS2_CSV, max_visits: int = cfg.MAX_VISITS, fold_indices: Optional[Dict[str, np.ndarray]] = None, all_sequences: Optional[List[List[dict]]] = None):
        self.split      = split
        self.data_dir   = Path(dataset_dir)
        self.csv_path   = Path(csv_path)
        self.max_visits = max_visits

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset dir not found: {self.data_dir}")

        self._col_stats: Dict[str, Tuple[float, float]] = {}

        if all_sequences is None:
            all_sequences = self._build_from_csv()
        else:
            all_sequences = [
                [{**v, 'clinical': np.array(v['clinical_raw'], dtype=np.float32)}
                 for v in seq]
                for seq in all_sequences
            ]

        if fold_indices is not None:
            split_idx = fold_indices
        else:
            split_idx = self._split_indices(len(all_sequences))

        self._fit_clinical_stats([all_sequences[i] for i in split_idx['train']])
        self._apply_clinical_norm(all_sequences)
        self.sequences = [all_sequences[i] for i in split_idx[self.split]]

    @classmethod
    def load_all_sequences(cls, dataset_dir: Path = cfg.OASIS2_DIR, csv_path: Path = cfg.OASIS2_CSV, max_visits: int = cfg.MAX_VISITS) -> List[List[dict]]:
        """Parse CSV + resolve MRI paths once (for CV)."""
        tmp = cls.__new__(cls)
        tmp.data_dir   = Path(dataset_dir)
        tmp.csv_path   = Path(csv_path)
        tmp.max_visits = max_visits
        if not tmp.data_dir.exists():
            raise FileNotFoundError(f"Dataset dir not found: {tmp.data_dir}")
        return tmp._build_from_csv()

    # ── API ───────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        visits     = self.sequences[idx]
        visit_data = []
        for v in visits:
            mri = self.load_mri_tensor(v)
            visit_data.append({
                'mri':      mri,
                'clinical': torch.tensor(v['clinical'], dtype=torch.float32),
                'label':    torch.tensor(int(v['label']), dtype=torch.long),
            })
        label = int(visits[-1]['label'])   # final diagnosis
        return {'visits': visit_data, 'label': torch.tensor(label, dtype=torch.long)}

    def load_mri_tensor(self, visit: dict) -> Tensor:
        """Load (C,D,H,W) MRI for one visit — processed preferred, RAW fallback."""
        mri_id = str(visit.get('mri_id', '')).strip()
        processed = self._load_processed_tensor(mri_id)
        if processed is not None:
            return self._append_roi_priors(processed)

        if bool(getattr(cfg, 'MRI_USE_PROCESSED', False)) and bool(
                getattr(cfg, 'MRI_REQUIRE_PROCESSED', False)):
            raise FileNotFoundError(
                f"Processed MRI missing for '{mri_id}'. "
                f"Run:  python mri_preprocess.py\n"
                f"Expected under {getattr(cfg, 'MRI_PROCESSED_DIR', '')}/{mri_id}/"
            )

        mri_path = self._normalise_analyze_pair(visit['mri_path'])
        mri = self._load_nifti(mri_path, cfg.MRI_SHAPE)  # (1,D,H,W)
        if bool(getattr(cfg, 'MRI_USE_GM_CHANNEL', False)):
            gm = self._soft_gm_from_t1(mri[0].numpy())
            mri = torch.cat([mri, torch.from_numpy(gm).unsqueeze(0)], dim=0)
        return self._append_roi_priors(mri)

    def _processed_tensor_path(self, mri_id: str) -> Optional[Path]:
        if not mri_id or not bool(getattr(cfg, 'MRI_USE_PROCESSED', False)):
            return None
        root = Path(getattr(cfg, 'MRI_PROCESSED_DIR', cfg.OASIS2_DIR / 'processed_v5'))
        d, h, w = cfg.MRI_SHAPE
        return root / mri_id / f"tensor_{d}x{h}x{w}.pt"

    def _load_processed_tensor(self, mri_id: str) -> Optional[Tensor]:
        path = self._processed_tensor_path(mri_id)
        if path is None or not path.exists():
            return None
        try:
            tensor = torch.load(path, map_location='cpu')
        except Exception:
            return None
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            return None
        want_gm = bool(getattr(cfg, 'MRI_USE_GM_CHANNEL', False))
        if want_gm and tensor.shape[0] == 1:
            gm = self._soft_gm_from_t1(tensor[0].numpy())
            tensor = torch.cat([tensor, torch.from_numpy(gm).unsqueeze(0)], dim=0)
        elif not want_gm and tensor.shape[0] >= 1:
            tensor = tensor[:1]
        elif want_gm and tensor.shape[0] >= 2:
            tensor = tensor[:2]
        return tensor.float()

    @staticmethod
    def _soft_gm_from_t1(t1: np.ndarray) -> np.ndarray:
        """Cheap GM band from normalised T1 (fallback when preprocess GM absent)."""
        mask = t1 > 1e-5
        gm = np.zeros_like(t1, dtype=np.float32)
        if not mask.any():
            return gm
        vox = t1[mask]
        lo, hi = np.percentile(vox, [35.0, 70.0])
        mid = 0.5 * (lo + hi)
        width = max((hi - lo) * 0.5, 1e-6)
        soft = np.exp(-0.5 * ((t1 - mid) / width) ** 2).astype(np.float32)
        soft = soft * mask.astype(np.float32)
        mx = float(soft.max())
        if mx > 0:
            soft /= mx
        return soft

    def _append_roi_priors(self, mri: Tensor) -> Tensor:
        """Optionally concat hippocampus + ventricle soft prior maps."""
        if not bool(getattr(cfg, 'MRI_USE_ROI_PRIOR_CHANNELS', False)):
            return mri
        _, D, H, W = mri.shape
        priors = []
        for name in ('hippocampus', 'ventricles'):
            box = cfg.MRI_VIEWS.get(name)
            if box is None:
                priors.append(torch.zeros(1, D, H, W))
                continue
            priors.append(self._soft_box_prior(D, H, W, box).unsqueeze(0))
        return torch.cat([mri] + priors, dim=0)

    @staticmethod
    def _soft_box_prior(D: int, H: int, W: int, box: Tuple[float, ...], sigma_frac: float = 0.08) -> Tensor:
        """Gaussian blob centred on a fractional ROI box (values in [0,1])."""
        z0, z1, y0, y1, x0, x1 = box
        cz, cy, cx = 0.5 * (z0 + z1), 0.5 * (y0 + y1), 0.5 * (x0 + x1)
        # Physical centre in voxel coords
        zz = (np.arange(D) + 0.5) / D
        yy = (np.arange(H) + 0.5) / H
        xx = (np.arange(W) + 0.5) / W
        Z, Y, X = np.meshgrid(zz, yy, xx, indexing='ij')
        # Anisotropic sigma from box half-extents (floor at sigma_frac)
        sz = max(0.5 * (z1 - z0), sigma_frac)
        sy = max(0.5 * (y1 - y0), sigma_frac)
        sx = max(0.5 * (x1 - x0), sigma_frac)
        prior = np.exp(
            -0.5 * (((Z - cz) / sz) ** 2 + ((Y - cy) / sy) ** 2 + ((X - cx) / sx) ** 2)
        ).astype(np.float32)
        # Zero outside the hard box so priors stay local
        hard = (
            (Z >= z0) & (Z <= z1) & (Y >= y0) & (Y <= y1) & (X >= x0) & (X <= x1)
        )
        prior = prior * hard.astype(np.float32)
        mx = float(prior.max())
        if mx > 0:
            prior /= mx
        return torch.from_numpy(prior)

    # ── Clinical spreadsheet ───────────────────────────────────────────────
    def _resolve_clinical_file(self) -> Path:
        if self.csv_path.exists():
            return self.csv_path
        stem = self.csv_path.with_suffix('')
        for ext in ('.xlsx', '.xls', '.csv', '.tsv'):
            cand = stem.with_suffix(ext)
            if cand.exists():
                return cand
        for ext in ('.xlsx', '.xls', '.csv', '.tsv'):
            cand = self.csv_path.parent / f'oasis_longitudinal_demographics{ext}'
            if cand.exists():
                return cand
        raise FileNotFoundError(f"Clinical spreadsheet not found near {self.csv_path}")

    def _load_clinical_csv(self) -> pd.DataFrame:
        path   = self._resolve_clinical_file()
        suffix = path.suffix.lower()
        if suffix in ('.xlsx', '.xls'):
            df = pd.read_excel(path, engine='openpyxl' if suffix == '.xlsx' else None)
        elif suffix == '.tsv':
            df = pd.read_csv(path, sep='\t')
        else:
            df = pd.read_csv(path)

        rename = {
            'Subject ID': 'Subject.ID', 'SubjectID': 'Subject.ID',
            'MRI ID':     'MRI.ID',     'MRIID':     'MRI.ID',
            'M/F':        'MF',
            'MR Delay':   'MR.Delay',
        }
        return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # ── Sequence construction ──────────────────────────────────────────────
    def _build_from_csv(self) -> List[List[dict]]:
        df = self._load_clinical_csv()

        if 'Subject.ID' not in df.columns:
            raise KeyError(f"Spreadsheet missing 'Subject ID'; found: {list(df.columns)}")
        if 'CDR' not in df.columns:
            raise KeyError(f"Spreadsheet missing 'CDR' (needed for labels); found: {list(df.columns)}")

        sequences: List[List[dict]] = []
        for subject_id, grp in df.groupby('Subject.ID', sort=False):
            grp  = grp.sort_values('Visit') if 'Visit' in grp.columns else grp
            rows = grp.head(self.max_visits)

            visits = []
            for _, row in rows.iterrows():
                mri_id   = str(row.get('MRI.ID', '')).strip()
                mri_path = self._resolve_mri_path(mri_id)
                if mri_path is None:
                    continue

                label = _cdr_to_label(row.get('CDR'))
                if label is None:
                    continue

                clinical_raw = np.array(
                    [_clinical_value(row, c) for c in _CLINICAL_COLS],
                    dtype=np.float32,
                )
                visits.append({
                    'subject_id':   subject_id,
                    'mri_id':       mri_id,
                    'visit_number': int(row.get('Visit', 0)),
                    'mri_path':     mri_path,
                    'clinical_raw': clinical_raw,
                    'clinical':     clinical_raw,   # overwritten by normalisation
                    'label':        int(label),
                })
            if visits:
                sequences.append(visits)

        if not sequences:
            raise RuntimeError(
                "No visits with resolvable MRI files found. Check that "
                "OAS2_RAW_PART1/2 contain OAS2_*_MR* session dirs with RAW/."
            )
        return sequences

    # ── Clinical normalisation (train-split statistics only) ───────────────
    def _fit_clinical_stats(self, train_sequences: List[List[dict]]) -> None:
        stacked = np.array(
            [v['clinical_raw'] for seq in train_sequences for v in seq],
            dtype=np.float32,
        ) if train_sequences else np.zeros((0, len(_CLINICAL_COLS)), dtype=np.float32)

        for j, col in enumerate(_CLINICAL_COLS):
            colvals = stacked[:, j] if stacked.size else np.array([], dtype=np.float32)
            has = colvals.size and not np.all(np.isnan(colvals))
            mean = float(np.nanmean(colvals)) if has else 0.0
            std  = float(np.nanstd(colvals))  if has else 0.0
            self._col_stats[col] = (mean, std + 1e-8)

    def _apply_clinical_norm(self, sequences: List[List[dict]]) -> None:
        means = np.array([self._col_stats[c][0] for c in _CLINICAL_COLS], dtype=np.float32)
        stds  = np.array([self._col_stats[c][1] for c in _CLINICAL_COLS], dtype=np.float32)
        for seq in sequences:
            for v in seq:
                raw    = np.array(v['clinical_raw'], dtype=np.float32)
                filled = np.where(np.isnan(raw), means, raw)
                v['clinical'] = ((filled - means) / stds).astype(np.float32)

    # ── Split ───────────────────────────────────────────────────────────────
    def _split_indices(self, n: int) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(cfg.SPLIT_SEED)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_train = int(n * cfg.TRAIN_FRAC)
        n_val   = int(n * cfg.VAL_FRAC)
        return {
            'train': idx[:n_train],
            'val':   idx[n_train:n_train + n_val],
            'test':  idx[n_train + n_val:],
        }

    # ── MRI path resolution (RAW Analyze pairs) ─────────────────────────────
    def _find_session_dir(self, mri_id: str) -> Optional[Path]:
        for part in _PART_DIRS:
            cand = self.data_dir / part / mri_id
            if cand.exists():
                return cand
        cand = self.data_dir / mri_id
        return cand if cand.exists() else None

    def _resolve_mri_path(self, mri_id: str) -> Optional[str]:
        if not mri_id:
            return None
        session_dir = self._find_session_dir(mri_id)
        if session_dir is None:
            return None
        raw_dir = session_dir / 'RAW'
        if not raw_dir.exists():
            return None

        candidate_stems = [f"{mri_id}_mpr-1_anon", "mpr-1"]
        candidate_exts  = ('.nii.gz', '.nii', '.nifti', '.img')
        for stem in candidate_stems:
            for ext in candidate_exts:
                p = raw_dir / (stem + ext)
                if p.exists():
                    return str(p)

        for pat in ('mpr-1.*', 'mpr-*.*', '*mpr-1*', '*mpr-*'):
            for p in sorted(raw_dir.glob(pat)):
                name = p.name.lower()
                if name.endswith('.hdr'):
                    continue
                if name.endswith(candidate_exts):
                    return str(p)
        return None

    def _normalise_analyze_pair(self, path: str) -> str:
        """Materialise .img/.hdr twins for the short '.nifti' Analyze pairs."""
        p = Path(path)
        if p.suffix.lower() != '.nifti':
            return path

        img_twin = p.with_suffix('.img')
        hdr_src  = Path(str(p) + '.hdr')
        hdr_twin = p.with_suffix('.hdr')

        if not img_twin.exists():
            try:
                os.link(p, img_twin)
            except OSError:
                try:
                    os.symlink(p.name, img_twin)
                except OSError:
                    shutil.copy2(p, img_twin)

        if hdr_src.exists() and not hdr_twin.exists():
            try:
                os.link(hdr_src, hdr_twin)
            except OSError:
                try:
                    os.symlink(hdr_src.name, hdr_twin)
                except OSError:
                    shutil.copy2(hdr_src, hdr_twin)

        return str(img_twin)

    # ── NIfTI / Analyze loading ─────────────────────────────────────────────
    def _load_nifti(self, path: Optional[str], target_shape: Tuple[int, int, int]) -> Tensor:
        if path is None or not cfg.NIBABEL_AVAILABLE:
            return torch.zeros((1, *target_shape), dtype=torch.float32)

        # Optional on-disk cache of the fully preprocessed tensor.
        cache_path = self._cache_path_for(path, target_shape)
        if cache_path is not None and cache_path.exists():
            try:
                return torch.load(cache_path, map_location='cpu')
            except Exception:
                pass  # stale/corrupt cache → recompute

        data = nib.load(path).get_fdata(dtype=np.float32)

        # OASIS-2 raw Analyze volumes carry a singleton 4th axis.
        data = np.squeeze(data)
        while data.ndim > 3:
            data = data[..., 0]
        if data.ndim != 3:
            raise RuntimeError(f"Unexpected MRI ndim={data.ndim} for {path}")

        # ── Brain-aware crop (replaces blind fractional centre crop) ──
        # Otsu + largest-CC gives a rough head mask; it is not a real
        # skull-strip, but it reliably separates head-tissue from air on
        # the RAW OASIS Analyze volumes, so the tight bounding-box crop
        # actually centres on anatomy instead of trusting the FOV.
        mask = self._compute_brain_mask(data)
        if mask.any():
            data, mask = self._crop_to_mask(data, mask, margin_frac=0.05)
        else:
            # Fallback: legacy fractional centre crop
            crop = float(getattr(cfg, 'MRI_CENTER_CROP_FRAC', 0.0))
            if crop > 0.0:
                crop = min(max(crop, 0.0), 0.35)
                d, h, w = data.shape
                d0, d1 = int(d * crop), max(int(d * (1.0 - crop)), int(d * crop) + 1)
                h0, h1 = int(h * crop), max(int(h * (1.0 - crop)), int(h * crop) + 1)
                w0, w1 = int(w * crop), max(int(w * (1.0 - crop)), int(w * crop) + 1)
                data = data[d0:d1, h0:h1, w0:w1]
            mask = np.ones(data.shape, dtype=bool)

        # ── Resize volume + mask to target shape ──
        factors = [t / s for t, s in zip(target_shape, data.shape)]
        data = zoom(data, factors, order=1)
        mask_r = zoom(mask.astype(np.float32), factors, order=0) > 0.5
        if not mask_r.any():
            mask_r = np.ones(data.shape, dtype=bool)

        # ── Brain-masked intensity normalisation ──
        # Percentile clip is now computed on brain voxels only, so its
        # low end is a real tissue intensity rather than "air = 0".
        brain_vox = data[mask_r]
        pct = getattr(cfg, 'MRI_PERCENTILE_CLIP', (1.0, 99.0))
        if pct is not None and brain_vox.size > 0:
            lo, hi = np.percentile(brain_vox, [float(pct[0]), float(pct[1])])
        else:
            lo = float(brain_vox.min()) if brain_vox.size else 0.0
            hi = float(brain_vox.max()) if brain_vox.size else 0.0

        if hi > lo:
            data = np.clip(data, lo, hi)
            data = (data - lo) / (hi - lo)
        else:
            data = np.zeros_like(data)

        # Zero out background so the encoder does not waste capacity on air.
        if bool(getattr(cfg, 'MRI_ZERO_BACKGROUND', True)):
            data = data * mask_r.astype(np.float32)

        tensor = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)  # (1, D, H, W)

        # Persist cache (best-effort; failures do not break training).
        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(tensor, cache_path)
            except Exception:
                pass
        return tensor

    # ── Brain-mask helpers (new) ────────────────────────────────────────────
    def _compute_brain_mask(self, data: np.ndarray) -> np.ndarray:
        """Rough head/brain foreground mask: Otsu threshold + largest 3-D CC.

        Cheap, dependency-free, and enough to separate anatomy from air on
        the RAW OASIS Analyze volumes.  It is *not* a proper skull-strip;
        for that, run offline preproc (BET / HD-BET) and point the loader
        at those files instead.
        """
        from scipy.ndimage import binary_fill_holes, label

        fg = data[data > 0]
        if fg.size < 1000:
            return data > 0  # degenerate volume — nothing to threshold

        # Otsu's method on the non-zero-voxel histogram.
        hist, edges = np.histogram(fg, bins=256)
        hist = hist.astype(np.float64)
        p = hist / (hist.sum() + 1e-12)
        centers = 0.5 * (edges[:-1] + edges[1:])
        omega = np.cumsum(p)
        mu = np.cumsum(p * centers)
        mu_t = mu[-1]
        denom = omega * (1.0 - omega) + 1e-12
        sigma_b2 = (mu_t * omega - mu) ** 2 / denom
        k = int(np.nanargmax(sigma_b2))
        thresh = float(centers[k])

        mask = data > thresh
        lbl, n = label(mask)
        if n <= 0:
            return mask
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0                                     # drop background
        mask = (lbl == int(sizes.argmax()))
        mask = binary_fill_holes(mask)
        return mask.astype(bool)

    def _crop_to_mask(self, data: np.ndarray, mask: np.ndarray,
                      margin_frac: float = 0.05
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Crop to the mask's bounding box, with a small isotropic margin."""
        if not mask.any():
            return data, mask
        coords = np.argwhere(mask)
        lo = coords.min(axis=0)
        hi = coords.max(axis=0) + 1
        extent = hi - lo
        margin = np.maximum((extent * float(margin_frac)).astype(int), 1)
        lo = np.maximum(lo - margin, 0)
        hi = np.minimum(hi + margin, np.array(data.shape))
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        return data[sl], mask[sl]

    def _cache_path_for(self, src_path: str,
                        target_shape: Tuple[int, int, int]) -> Optional[Path]:
        """Return a per-volume cache Path, or None if caching is disabled.

        CRITICAL: do NOT key only on ``Path(src_path).stem``. OASIS RAW files
        are often named ``mpr-1.img`` in every session, so a stem-only key
        makes every subject share one cached tensor (pairwise L2 → 0).
        Include the session folder (``OAS2_****_MR*``) which is unique.
        """
        cache_dir = getattr(cfg, 'MRI_CACHE_DIR', None)
        if not cache_dir:
            return None
        p = Path(src_path)
        # .../OAS2_0001_MR1/RAW/mpr-1.img  →  session = OAS2_0001_MR1
        session = p.parent.parent.name if p.parent.name.upper() == 'RAW' else p.parent.name
        stem = p.stem
        d, h, w = target_shape
        # v5: session-scoped keys + brain-mask zeroing (v4 lacked zero-bg)
        return Path(cache_dir) / f"{session}__{stem}_{d}x{h}x{w}_v5.pt"
