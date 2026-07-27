"""Offline OASIS-2 MRI representation pipeline (run once before training).

Steps (matches the representation upgrade plan):
  1. Affine register each RAW T1 to a common template (MNI brain if present,
     otherwise a cohort mean template built from the first pass).
  2. Skull-strip (HD-BET → FSL BET → Otsu+CC fallback).
  3. N4 bias-field correction (SimpleITK; skipped with a warning if absent).
  4. Write registered brain-cropped volumes so hippocampus / ventricle
     fractional views in ``config.MRI_VIEWS`` are anatomically meaningful.
  5. Soft GM channel via 3-class brain tissue clustering (+ optional ROI
     prior channels are generated at load time from view boxes).

Output layout (per session)::

    <MRI_PROCESSED_DIR>/<mri_id>/
        t1.nii.gz          # registered, skull-stripped, N4, [0,1]
        gm.nii.gz          # soft GM probability, same grid
        mask.nii.gz        # brain mask
        tensor_CxDxHxW.pt  # training tensor (C = 1 or 2)

Usage::

    python mri_preprocess.py
    python mri_preprocess.py --limit 5          # smoke test
    python mri_preprocess.py --force            # rebuild existing

Requires: nibabel, numpy, scipy.  Strongly recommended: SimpleITK.
Optional CLIs on PATH: hd-bet, bet (FSL).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from config import Config

cfg = Config()

_PART_DIRS = ('OAS2_RAW_PART1', 'OAS2_RAW_PART2')


# ──────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────
def _require_nibabel():
    try:
        import nibabel as nib
    except ImportError as e:
        raise SystemExit("nibabel is required for mri_preprocess.py") from e
    return nib


def _try_sitk():
    try:
        import SimpleITK as sitk
        return sitk
    except ImportError:
        return None


def load_volume(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (data float32 [D,H,W], affine 4x4)."""
    nib = _require_nibabel()
    path = _materialise_analyze(path)
    img = nib.load(str(path))
    data = np.asanyarray(img.get_fdata(dtype=np.float32))
    data = np.squeeze(data)
    while data.ndim > 3:
        data = data[..., 0]
    if data.ndim != 3:
        raise RuntimeError(f"Expected 3D volume, got shape={data.shape} for {path}")
    return data.astype(np.float32), np.array(img.affine, dtype=np.float64)


def save_nifti(data: np.ndarray, affine: np.ndarray, path: Path) -> None:
    nib = _require_nibabel()
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.Nifti1Image(data.astype(np.float32), affine).to_filename(str(path))


def _materialise_analyze(path: Path) -> Path:
    """Ensure .img/.hdr twins exist for OASIS short '.nifti' Analyze pairs."""
    p = Path(path)
    if p.suffix.lower() != '.nifti':
        return p
    img_twin = p.with_suffix('.img')
    hdr_src = Path(str(p) + '.hdr')
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
    return img_twin


# ──────────────────────────────────────────────────────────────
# Session discovery (mirrors OasisModule RAW resolution)
# ──────────────────────────────────────────────────────────────
def find_session_dir(data_dir: Path, mri_id: str) -> Optional[Path]:
    for part in _PART_DIRS:
        cand = data_dir / part / mri_id
        if cand.exists():
            return cand
    cand = data_dir / mri_id
    return cand if cand.exists() else None


def resolve_raw_mri(data_dir: Path, mri_id: str) -> Optional[Path]:
    session = find_session_dir(data_dir, mri_id)
    if session is None:
        return None
    raw_dir = session / 'RAW'
    if not raw_dir.exists():
        return None
    stems = [f"{mri_id}_mpr-1_anon", "mpr-1"]
    exts = ('.nii.gz', '.nii', '.nifti', '.img')
    for stem in stems:
        for ext in exts:
            p = raw_dir / (stem + ext)
            if p.exists():
                return p
    for pat in ('mpr-1.*', 'mpr-*.*', '*mpr-1*', '*mpr-*'):
        for p in sorted(raw_dir.glob(pat)):
            name = p.name.lower()
            if name.endswith('.hdr'):
                continue
            if name.endswith(exts):
                return p
    return None


def list_mri_ids_from_csv(csv_path: Path) -> List[str]:
    import pandas as pd

    path = Path(csv_path)
    if not path.exists():
        stem = path.with_suffix('')
        for ext in ('.xlsx', '.xls', '.csv'):
            cand = stem.with_suffix(ext)
            if cand.exists():
                path = cand
                break
    suffix = path.suffix.lower()
    if suffix in ('.xlsx', '.xls'):
        df = pd.read_excel(path, engine='openpyxl' if suffix == '.xlsx' else None)
    else:
        df = pd.read_csv(path)
    rename = {'MRI ID': 'MRI.ID', 'MRIID': 'MRI.ID'}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if 'MRI.ID' not in df.columns:
        raise KeyError(f"No MRI ID column in {path}; columns={list(df.columns)}")
    ids = []
    seen = set()
    for v in df['MRI.ID'].astype(str).str.strip():
        if v and v.lower() != 'nan' and v not in seen:
            seen.add(v)
            ids.append(v)
    return ids


# ──────────────────────────────────────────────────────────────
# 2. Skull-strip
# ──────────────────────────────────────────────────────────────
def skull_strip(data: np.ndarray, affine: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (brain_masked_data, mask). Tries HD-BET → BET → Otsu."""
    masked, mask = _skull_strip_hd_bet(data, affine)
    if mask is not None:
        return masked, mask
    masked, mask = _skull_strip_fsl_bet(data, affine)
    if mask is not None:
        return masked, mask
    mask = _otsu_brain_mask(data)
    return data * mask.astype(np.float32), mask


def _skull_strip_hd_bet(data: np.ndarray, affine: np.ndarray
                        ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if shutil.which('hd-bet') is None:
        return None, None
    nib = _require_nibabel()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        inp = td / 'in.nii.gz'
        out = td / 'out'
        save_nifti(data, affine, inp)
        try:
            subprocess.run(
                ['hd-bet', '-i', str(inp), '-o', str(out), '-device', 'cpu'],
                check=True, capture_output=True, text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None, None
        # hd-bet writes out.nii.gz and out_mask.nii.gz (version-dependent)
        brain_path = out.with_suffix('.nii.gz') if out.suffix == '' else out
        if not brain_path.exists():
            candidates = list(td.glob('out*.nii*'))
            brain_path = next((p for p in candidates if 'mask' not in p.name.lower()), None)
        if brain_path is None or not Path(brain_path).exists():
            return None, None
        brain = np.asanyarray(nib.load(str(brain_path)).get_fdata(dtype=np.float32))
        brain = np.squeeze(brain).astype(np.float32)
        mask_path = td / 'out_mask.nii.gz'
        if mask_path.exists():
            mask = np.asanyarray(nib.load(str(mask_path)).get_fdata(dtype=np.float32))
            mask = (np.squeeze(mask) > 0).astype(bool)
        else:
            mask = brain > 0
        return brain, mask


def _skull_strip_fsl_bet(data: np.ndarray, affine: np.ndarray
                         ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if shutil.which('bet') is None:
        return None, None
    nib = _require_nibabel()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        inp = td / 'in.nii.gz'
        out = td / 'brain.nii.gz'
        save_nifti(data, affine, inp)
        try:
            subprocess.run(
                ['bet', str(inp), str(out), '-f', '0.4', '-g', '0', '-m'],
                check=True, capture_output=True, text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None, None
        if not out.exists():
            return None, None
        brain = np.asanyarray(nib.load(str(out)).get_fdata(dtype=np.float32))
        brain = np.squeeze(brain).astype(np.float32)
        mask_path = td / 'brain_mask.nii.gz'
        if mask_path.exists():
            mask = np.asanyarray(nib.load(str(mask_path)).get_fdata(dtype=np.float32))
            mask = (np.squeeze(mask) > 0).astype(bool)
        else:
            mask = brain > 0
        return brain, mask


def _otsu_brain_mask(data: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_fill_holes, binary_closing, label

    fg = data[data > 0]
    if fg.size < 1000:
        return data > 0
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
    sizes[0] = 0
    mask = (lbl == int(sizes.argmax()))
    mask = binary_closing(mask, iterations=2)
    mask = binary_fill_holes(mask)
    return mask.astype(bool)


# ──────────────────────────────────────────────────────────────
# 3. N4 bias correction
# ──────────────────────────────────────────────────────────────
def n4_bias_correct(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    sitk = _try_sitk()
    if sitk is None:
        return data
    img = sitk.GetImageFromArray(data.astype(np.float32))
    msk = sitk.GetImageFromArray(mask.astype(np.uint8))
    msk.CopyInformation(img)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([30, 30, 20, 10])
    try:
        out = corrector.Execute(img, msk)
    except RuntimeError:
        try:
            out = corrector.Execute(img)
        except RuntimeError:
            return data
    return sitk.GetArrayFromImage(out).astype(np.float32)


# ──────────────────────────────────────────────────────────────
# 1. Affine registration to a common template
# ──────────────────────────────────────────────────────────────
def load_or_build_template(
    template_path: Path,
    seed_volume: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load MNI brain template, or use seed volume as bootstrap template."""
    if template_path.exists():
        data, aff = load_volume(template_path)
        # Soft brain mask if template includes skull
        mask = _otsu_brain_mask(data)
        data = data * mask.astype(np.float32)
        return data, aff
    if seed_volume is not None:
        return seed_volume
    raise FileNotFoundError(
        f"No MRI template at {template_path} and no seed volume. "
        f"Place an MNI152 brain NIfTI there, or let the first subject seed the template."
    )


def affine_register(
    moving: np.ndarray,
    moving_affine: np.ndarray,
    fixed: np.ndarray,
    fixed_affine: np.ndarray,
    moving_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Affine-align moving → fixed. Returns (warped_data, fixed_affine).

    Prefers SimpleITK Mattes MI affine. Falls back to centre-of-mass
    translation + isotropic zoom into the fixed grid (still better than RAW).
    """
    sitk = _try_sitk()
    if sitk is not None:
        try:
            return _sitk_affine_register(
                moving, moving_affine, fixed, fixed_affine, moving_mask)
        except Exception as exc:  # noqa: BLE001 — fall back
            print(f"  [warn] SimpleITK registration failed ({exc}); using CoM fallback")
    return _com_register(moving, moving_mask, fixed)


def _numpy_to_sitk(data: np.ndarray, affine: np.ndarray):
    sitk = _try_sitk()
    img = sitk.GetImageFromArray(data.astype(np.float32))
    spacing = tuple(float(np.linalg.norm(affine[:3, i])) for i in range(3))
    # SimpleITK array is z,y,x — reverse spacing to match GetImageFromArray convention
    img.SetSpacing(spacing[::-1])
    origin = tuple(float(x) for x in affine[:3, 3])
    img.SetOrigin(origin)
    return img


def _sitk_affine_register(
    moving: np.ndarray,
    moving_affine: np.ndarray,
    fixed: np.ndarray,
    fixed_affine: np.ndarray,
    moving_mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    sitk = _try_sitk()
    fixed_img = sitk.GetImageFromArray(fixed.astype(np.float32))
    moving_img = sitk.GetImageFromArray(moving.astype(np.float32))
    # Unit spacing on the voxel grids — we already work in resampled arrays
    fixed_img.SetSpacing((1.0, 1.0, 1.0))
    moving_img.SetSpacing((1.0, 1.0, 1.0))

    init = sitk.CenteredTransformInitializerFilter.GEOMETRY
    transform = sitk.CenteredTransformInitializer(
        fixed_img, moving_img, sitk.AffineTransform(3), init)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=200,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(transform, inPlace=False)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    if moving_mask is not None and moving_mask.any():
        msk = sitk.GetImageFromArray(moving_mask.astype(np.uint8))
        msk.CopyInformation(moving_img)
        reg.SetMetricMovingMask(msk)

    out_tx = reg.Execute(fixed_img, moving_img)
    warped = sitk.Resample(
        moving_img, fixed_img, out_tx, sitk.sitkLinear, 0.0, moving_img.GetPixelID())
    return sitk.GetArrayFromImage(warped).astype(np.float32), fixed_affine


def _com_register(
    moving: np.ndarray,
    moving_mask: Optional[np.ndarray],
    fixed: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Centre-of-mass align + reshape into the fixed grid (no rotation)."""
    from scipy.ndimage import shift as nd_shift, zoom

    m = moving_mask if moving_mask is not None else (moving > 0)
    if not m.any():
        m = np.ones_like(moving, dtype=bool)
    com_m = np.array(np.argwhere(m).mean(axis=0), dtype=np.float64)
    fmask = fixed > 0
    if not fmask.any():
        fmask = np.ones_like(fixed, dtype=bool)
    com_f = np.array(np.argwhere(fmask).mean(axis=0), dtype=np.float64)

    # Zoom moving brain bbox into fixed shape, then shift CoM
    coords = np.argwhere(m)
    lo, hi = coords.min(axis=0), coords.max(axis=0) + 1
    crop = moving[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    factors = [fs / max(ms, 1) for fs, ms in zip(fixed.shape, crop.shape)]
    scaled = zoom(crop, factors, order=1)
    # Pad / crop to exact fixed shape
    out = np.zeros(fixed.shape, dtype=np.float32)
    slices_in, slices_out = [], []
    for dim in range(3):
        if scaled.shape[dim] >= fixed.shape[dim]:
            start = (scaled.shape[dim] - fixed.shape[dim]) // 2
            slices_in.append(slice(start, start + fixed.shape[dim]))
            slices_out.append(slice(0, fixed.shape[dim]))
        else:
            start = (fixed.shape[dim] - scaled.shape[dim]) // 2
            slices_in.append(slice(0, scaled.shape[dim]))
            slices_out.append(slice(start, start + scaled.shape[dim]))
    out[tuple(slices_out)] = scaled[tuple(slices_in)]

    # Fine CoM shift on the placed volume
    omask = out > 0
    if omask.any():
        com_o = np.array(np.argwhere(omask).mean(axis=0), dtype=np.float64)
        delta = com_f - com_o
        out = nd_shift(out, shift=delta, order=1, mode='constant', cval=0.0)
    # Identity affine placeholder — caller replaces with template affine
    return out.astype(np.float32), np.eye(4, dtype=np.float64)


# ──────────────────────────────────────────────────────────────
# Crop / resize / intensity / GM (steps 4–5 support)
# ──────────────────────────────────────────────────────────────
def crop_to_mask(data: np.ndarray, mask: np.ndarray, margin_frac: float = 0.05
                 ) -> Tuple[np.ndarray, np.ndarray]:
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


def resize_volume(data: np.ndarray, target: Tuple[int, int, int],
                  order: int = 1) -> np.ndarray:
    from scipy.ndimage import zoom
    factors = [t / max(s, 1) for t, s in zip(target, data.shape)]
    return zoom(data, factors, order=order).astype(np.float32)


def intensity_normalise(data: np.ndarray, mask: np.ndarray,
                        pct: Tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    brain = data[mask] if mask.any() else data.ravel()
    if brain.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    lo, hi = np.percentile(brain, [float(pct[0]), float(pct[1])])
    if hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    out = np.clip(data, lo, hi)
    out = (out - lo) / (hi - lo)
    out = out * mask.astype(np.float32)  # zero background
    return out.astype(np.float32)


def soft_gm_map(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """3-class soft GM probability on brain voxels (CSF / GM / WM).

    Intensity ordering on T1: CSF < GM < WM. Uses within-brain
    percentile anchors when sklearn is unavailable.
    """
    gm = np.zeros_like(data, dtype=np.float32)
    if not mask.any():
        return gm
    vox = data[mask].astype(np.float64)
    # Skip empty / flat brains
    if vox.std() < 1e-6:
        return gm

    try:
        from sklearn.cluster import KMeans
        x = vox.reshape(-1, 1)
        km = KMeans(n_clusters=3, n_init=10, random_state=0)
        labels = km.fit_predict(x)
        centers = km.cluster_centers_.ravel()
        order = np.argsort(centers)  # CSF, GM, WM
        gm_id = int(order[1])
        # Soft assignment via distance to GM centre vs others
        d_gm = np.abs(vox - centers[gm_id])
        d_other = np.minimum(
            np.abs(vox - centers[order[0]]),
            np.abs(vox - centers[order[2]]),
        )
        soft = 1.0 / (1.0 + d_gm / (d_other + 1e-6))
        soft = soft * (labels == gm_id).astype(np.float64) + soft * 0.35 * (
            labels != gm_id).astype(np.float64)
        soft = soft / (soft.max() + 1e-8)
        gm[mask] = soft.astype(np.float32)
        return gm
    except Exception:
        pass

    # Percentile fallback: GM band ≈ 35th–70th percentile of brain
    lo, hi = np.percentile(vox, [35.0, 70.0])
    band = np.clip(data, lo, hi)
    mid = 0.5 * (lo + hi)
    width = max((hi - lo) * 0.5, 1e-6)
    soft = np.exp(-0.5 * ((band - mid) / width) ** 2)
    soft = soft * mask.astype(np.float32)
    soft = soft / (soft.max() + 1e-8)
    return soft.astype(np.float32)


def resample_to_shape(data: np.ndarray, affine: np.ndarray,
                      target: Tuple[int, int, int]
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Isotropic-ish resize keeping a scaled identity-ish affine."""
    out = resize_volume(data, target, order=1)
    # Scale affine voxel sizes
    scales = [data.shape[i] / max(target[i], 1) for i in range(3)]
    new_aff = affine.copy()
    for i in range(3):
        new_aff[:3, i] = affine[:3, i] * scales[i]
    return out, new_aff


# ──────────────────────────────────────────────────────────────
# Per-session pipeline
# ──────────────────────────────────────────────────────────────
def process_one(
    raw_path: Path,
    out_dir: Path,
    template: Tuple[np.ndarray, np.ndarray],
    target_shape: Tuple[int, int, int],
    use_gm: bool,
    force: bool = False,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = out_dir / f"tensor_{target_shape[0]}x{target_shape[1]}x{target_shape[2]}.pt"
    meta_path = out_dir / 'meta.json'
    if tensor_path.exists() and not force:
        return {'status': 'skip', 'tensor': str(tensor_path)}

    data, affine = load_volume(raw_path)

    # N4 on raw (helps BET); if mask unknown, run on full FOV
    rough_mask = data > 0
    data = n4_bias_correct(data, rough_mask)

    # Skull-strip
    brain, mask = skull_strip(data, affine)
    if not mask.any():
        mask = _otsu_brain_mask(brain)
        brain = brain * mask.astype(np.float32)

    # Second N4 inside brain (finer)
    brain = n4_bias_correct(brain, mask)
    brain = brain * mask.astype(np.float32)

    # Affine → template grid (template already at target_shape)
    fixed, fixed_aff = template
    warped, warped_aff = affine_register(brain, affine, fixed, fixed_aff, mask)
    wmask = warped > (0.05 * (warped.max() + 1e-8))
    if not wmask.any():
        wmask = _otsu_brain_mask(warped)

    # Brain-aware crop on the registered volume, then resize to target
    cropped, cmask = crop_to_mask(warped, wmask, margin_frac=0.05)
    t1 = resize_volume(cropped, target_shape, order=1)
    t1_mask = resize_volume(cmask.astype(np.float32), target_shape, order=0) > 0.5
    if not t1_mask.any():
        t1_mask = t1 > 0
    pct = tuple(getattr(cfg, 'MRI_PERCENTILE_CLIP', (1.0, 99.0)))
    t1 = intensity_normalise(t1, t1_mask, pct=pct)

    channels = [t1]
    gm = None
    if use_gm:
        # GM on registered (pre-norm) intensities inside mask, then resize
        gm_src = soft_gm_map(
            intensity_normalise(cropped, cmask, pct=pct), cmask)
        gm = resize_volume(gm_src, target_shape, order=1)
        gm = gm * t1_mask.astype(np.float32)
        gm = gm / (gm.max() + 1e-8)
        channels.append(gm)

    tensor = torch.from_numpy(np.stack(channels, axis=0).astype(np.float32))
    torch.save(tensor, tensor_path)

    save_nifti(t1, np.eye(4), out_dir / 't1.nii.gz')
    save_nifti(t1_mask.astype(np.float32), np.eye(4), out_dir / 'mask.nii.gz')
    if gm is not None:
        save_nifti(gm, np.eye(4), out_dir / 'gm.nii.gz')

    meta = {
        'raw': str(raw_path),
        'shape': list(target_shape),
        'channels': ['t1'] + (['gm'] if use_gm else []),
        'skull_strip': (
            'hd-bet' if shutil.which('hd-bet') else
            ('bet' if shutil.which('bet') else 'otsu')
        ),
        'n4': _try_sitk() is not None,
        'registration': 'sitk_affine' if _try_sitk() else 'com_fallback',
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return {'status': 'ok', 'tensor': str(tensor_path), **meta}


def build_template_volume(
    sample_paths: List[Path],
    target_shape: Tuple[int, int, int],
    template_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load external template or build a mean brain from a few sessions."""
    if template_path.exists():
        data, aff = load_volume(template_path)
        mask = _otsu_brain_mask(data)
        data = data * mask.astype(np.float32)
        data, aff = resample_to_shape(data, aff, target_shape)
        data = intensity_normalise(data, data > 0)
        return data, aff

    print(f"[template] {template_path} missing — building cohort mean from "
          f"{min(len(sample_paths), 12)} sessions")
    acc = np.zeros(target_shape, dtype=np.float64)
    n = 0
    for p in sample_paths[:12]:
        try:
            data, aff = load_volume(p)
            data = n4_bias_correct(data, data > 0)
            brain, mask = skull_strip(data, aff)
            cropped, cmask = crop_to_mask(brain, mask)
            vol = resize_volume(cropped, target_shape, order=1)
            m = resize_volume(cmask.astype(np.float32), target_shape, order=0) > 0.5
            vol = intensity_normalise(vol, m)
            acc += vol
            n += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] template seed failed for {p}: {exc}")
    if n == 0:
        raise RuntimeError("Could not build cohort template from any session")
    mean = (acc / n).astype(np.float32)
    return mean, np.eye(4, dtype=np.float64)


def run_preprocess(
    data_dir: Path = None,
    csv_path: Path = None,
    out_dir: Path = None,
    template_path: Path = None,
    target_shape: Tuple[int, int, int] = None,
    use_gm: bool = None,
    force: bool = False,
    limit: Optional[int] = None,
) -> None:
    data_dir = Path(data_dir or cfg.OASIS2_DIR)
    csv_path = Path(csv_path or cfg.OASIS2_CSV)
    out_dir = Path(out_dir or getattr(cfg, 'MRI_PROCESSED_DIR',
                                      data_dir / 'processed_v5'))
    template_path = Path(template_path or getattr(
        cfg, 'MRI_TEMPLATE_PATH',
        data_dir / 'templates' / 'MNI152_T1_1mm_brain.nii.gz'))
    target_shape = tuple(target_shape or cfg.MRI_SHAPE)
    if use_gm is None:
        use_gm = bool(getattr(cfg, 'MRI_USE_GM_CHANNEL', True))

    mri_ids = list_mri_ids_from_csv(csv_path)
    pairs: List[Tuple[str, Path]] = []
    for mid in mri_ids:
        raw = resolve_raw_mri(data_dir, mid)
        if raw is not None:
            pairs.append((mid, raw))
    if not pairs:
        raise SystemExit(f"No RAW MRI files found under {data_dir}")
    if limit is not None:
        pairs = pairs[: int(limit)]

    print(f"Sessions to process: {len(pairs)}")
    print(f"Output dir: {out_dir}")
    print(f"SimpleITK: {'yes' if _try_sitk() else 'NO (N4/affine limited)'}")
    print(f"hd-bet: {'yes' if shutil.which('hd-bet') else 'no'} | "
          f"bet: {'yes' if shutil.which('bet') else 'no'}")

    template = build_template_volume(
        [p for _, p in pairs], target_shape, template_path)
    # Persist template for reuse / inspection
    tmpl_out = out_dir / '_template_brain.nii.gz'
    save_nifti(template[0], template[1], tmpl_out)
    print(f"Template written: {tmpl_out}")

    ok = skip = fail = 0
    for i, (mid, raw) in enumerate(pairs, 1):
        dest = out_dir / mid
        print(f"[{i}/{len(pairs)}] {mid}")
        try:
            result = process_one(
                raw_path=raw,
                out_dir=dest,
                template=template,
                target_shape=target_shape,
                use_gm=use_gm,
                force=force,
            )
            if result['status'] == 'skip':
                skip += 1
                print("  skip (exists)")
            else:
                ok += 1
                print(f"  ok  strip={result.get('skull_strip')}  "
                      f"reg={result.get('registration')}  "
                      f"ch={result.get('channels')}")
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"  FAIL: {exc}")

    summary = {'ok': ok, 'skip': skip, 'fail': fail, 'out_dir': str(out_dir)}
    (out_dir / 'preprocess_summary.json').write_text(json.dumps(summary, indent=2))
    print(f"Done. {summary}")


def main():
    ap = argparse.ArgumentParser(description='OASIS-2 MRI representation preprocess')
    ap.add_argument('--data-dir', type=Path, default=None)
    ap.add_argument('--csv', type=Path, default=None)
    ap.add_argument('--out-dir', type=Path, default=None)
    ap.add_argument('--template', type=Path, default=None)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--no-gm', action='store_true', help='Skip GM channel')
    args = ap.parse_args()
    run_preprocess(
        data_dir=args.data_dir,
        csv_path=args.csv,
        out_dir=args.out_dir,
        template_path=args.template,
        use_gm=(False if args.no_gm else None),
        force=args.force,
        limit=args.limit,
    )


if __name__ == '__main__':
    main()
