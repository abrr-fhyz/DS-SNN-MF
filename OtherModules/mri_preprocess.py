"""Offline OASIS-2 MRI representation pipeline (run once before training).

Pipeline
--------
  1. Load RAW T1 preserving the NIfTI affine, orientation and spacing.
  2. Skull-strip with HD-BET on GPU. No fallback — a HD-BET failure marks
     the subject as failed and is recorded in the run summary.
  3. N4 bias-field correction inside the HD-BET brain mask (SimpleITK).
  4. Multiresolution Mattes MI *affine* registration into MNI physical
     space against the TemplateFlow ``MNI152NLin2009cAsym`` 1 mm T1w brain
     template. Registration runs on the true physical grids so voxel
     spacing, orientation and physical coordinates are preserved.
  5. A single resample onto a FIXED MNI-space reference grid derived from
     the template FOV at the requested target shape. The exact same grid
     is used for every subject and every split — no per-subject
     cropping or resizing.
  6. Per-subject intensity normalisation and (optional) soft GM channel.

No cohort mean, no Otsu, no dataset-derived statistics. Train, validation
and test all share the identical fixed MNI template, so there is no
template leakage of any kind.

Output layout (per session)::

    <MRI_PROCESSED_DIR>/<mri_id>/
        t1.nii.gz          # MNI-registered, skull-stripped, N4, [0,1]
        gm.nii.gz          # soft GM probability, same grid (optional)
        mask.nii.gz        # brain mask on the MNI grid
        tensor_CxDxHxW.pt  # training tensor (C = 1 or 2)
        meta.json

Chunked runs
------------
The reference grid is derived from the TemplateFlow template alone and is
independent of the session slice. Sessions whose tensor already exists
are skipped, so re-running with a larger --limit or a fresh --offset is
always safe and resumable.

Usage::

    python mri_preprocess.py                          # everything
    python mri_preprocess.py --limit 50                # sessions 0-49
    python mri_preprocess.py --offset 50 --limit 50     # sessions 50-99
    python mri_preprocess.py --limit 5 --force          # smoke test

Requires: nibabel, numpy, torch, SimpleITK, templateflow.
Required CLI on PATH: hd-bet (GPU skull-stripping).
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

# HD-BET GPU device index ('0' = first CUDA device). Override with
# HD_BET_DEVICE env var (e.g. `HD_BET_DEVICE=cpu`) if you truly need CPU.
HD_BET_DEVICE = os.environ.get('HD_BET_DEVICE', 'cuda:0')

# TemplateFlow identifiers — the ONLY registration target we ever use.
TEMPLATE_SPEC = dict(
    template='MNI152NLin2009cAsym',
    resolution=1,
    desc='brain',
    suffix='T1w',
    extension='.nii.gz',
)


# ──────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────
def _require_nibabel():
    try:
        import nibabel as nib
    except ImportError as e:
        raise SystemExit("nibabel is required for mri_preprocess.py") from e
    return nib


def _require_sitk():
    try:
        import SimpleITK as sitk
    except ImportError as e:
        raise SystemExit(
            "SimpleITK is required (registration + N4 + IO). "
            "Install with: pip install SimpleITK"
        ) from e
    return sitk


def _materialise_analyze(path: Path) -> Path:
    """Ensure .img/.hdr twins exist for OASIS short '.nifti' Analyze pairs."""
    p = Path(path)
    if p.suffix.lower() != '.nifti':
        return p
    img_twin = p.with_suffix('.img')
    hdr_src = Path(str(p) + '.hdr')
    hdr_twin = p.with_suffix('.hdr')
    for src, twin in ((p, img_twin), (hdr_src, hdr_twin)):
        if twin.exists() or not src.exists():
            continue
        try:
            os.link(src, twin)
        except OSError:
            try:
                os.symlink(src.name, twin)
            except OSError:
                shutil.copy2(src, twin)
    return img_twin


def _sitk_read(path: Path):
    """Load a NIfTI/Analyze volume as SITK float32 preserving header geometry."""
    sitk = _require_sitk()
    path = _materialise_analyze(Path(path))
    return sitk.ReadImage(str(path), sitk.sitkFloat32)


def _sitk_to_nib_affine(img) -> np.ndarray:
    """Convert a SimpleITK image's geometry to a nibabel/NIfTI RAS affine."""
    direction = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    spacing = np.array(img.GetSpacing(), dtype=np.float64)
    origin = np.array(img.GetOrigin(), dtype=np.float64)
    aff_lps = np.eye(4, dtype=np.float64)
    aff_lps[:3, :3] = direction * spacing  # column-wise scale by spacing
    aff_lps[:3, 3] = origin
    # SITK/DICOM = LPS, NIfTI/nibabel = RAS → flip first two axes.
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    return lps_to_ras @ aff_lps


def save_nifti_sitk(img, path: Path) -> None:
    """Write a SITK image as NIfTI, preserving orientation/spacing/origin."""
    sitk = _require_sitk()
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))


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
    ids, seen = [], set()
    for v in df['MRI.ID'].astype(str).str.strip():
        if v and v.lower() != 'nan' and v not in seen:
            seen.add(v)
            ids.append(v)
    return ids


def discover_sessions(data_dir: Path, csv_path: Path) -> List[Tuple[str, Path]]:
    """Full (mri_id, raw_path) list, in stable CSV order."""
    pairs = []
    for mid in list_mri_ids_from_csv(csv_path):
        raw = resolve_raw_mri(data_dir, mid)
        if raw is not None:
            pairs.append((mid, raw))
    return pairs


# ──────────────────────────────────────────────────────────────
# Template: TemplateFlow MNI152NLin2009cAsym 1 mm T1w brain
# ──────────────────────────────────────────────────────────────
def fetch_mni_template() -> Path:
    """Fetch (or cache-hit) the sole registration target from TemplateFlow."""
    try:
        from templateflow import api as tflow
    except ImportError as e:
        raise SystemExit(
            "templateflow is required. Install with: pip install templateflow"
        ) from e
    p = tflow.get(**TEMPLATE_SPEC)
    if isinstance(p, list):
        if not p:
            raise SystemExit(f"TemplateFlow returned no file for {TEMPLATE_SPEC}")
        p = p[0]
    return Path(str(p))


def build_reference_grid(template_img, target_shape: Tuple[int, int, int]):
    """A single FIXED MNI-space grid at the requested target shape.

    Applied identically to every subject and every split. If the target
    shape matches the template's native voxel count, the template itself
    is used verbatim (no resampling artefacts). Otherwise the template
    FOV is uniformly reshaped to ``target_shape`` — one deterministic
    transform that is not derived from any subject data.

    ``target_shape`` is (D, H, W) in numpy order; SITK uses (X, Y, Z).
    """
    sitk = _require_sitk()
    tmpl_size = template_img.GetSize()               # (X, Y, Z)
    tmpl_np_shape = (tmpl_size[2], tmpl_size[1], tmpl_size[0])
    if tuple(target_shape) == tmpl_np_shape:
        return template_img

    target_size_sitk = (int(target_shape[2]), int(target_shape[1]),
                        int(target_shape[0]))
    tmpl_spacing = template_img.GetSpacing()
    # Preserve the template FOV: new_spacing * new_size == old_spacing * old_size.
    new_spacing = tuple(
        tmpl_spacing[i] * tmpl_size[i] / target_size_sitk[i] for i in range(3)
    )
    ref = sitk.Image(target_size_sitk, template_img.GetPixelID())
    ref.SetOrigin(template_img.GetOrigin())
    ref.SetDirection(template_img.GetDirection())
    ref.SetSpacing(new_spacing)
    return ref


def _normalise_hd_bet_device(dev: str) -> str:
    """Map legacy HD-BET device strings to torch.device-compatible ones."""
    d = str(dev).strip().lower()
    if d in ('cpu', 'mps'):
        return d
    if d.startswith('cuda'):
        return d
    if d.isdigit():
        return f'cuda:{d}'
    return d  # let torch.device raise a clear error otherwise


def skull_strip_hd_bet(moving_img):
    """Run HD-BET on the moving image. Returns (brain_img, mask_img).

    Handles both legacy HD-BET (auto-writes `<out>_mask.nii.gz`) and
    HD-BET v2 (writes `<out>_bet.nii.gz` only when `--save_bet_mask`
    is passed). Any failure raises RuntimeError — no fallback.
    """
    sitk = _require_sitk()
    if shutil.which('hd-bet') is None:
        raise RuntimeError("hd-bet not found on PATH")

    device = _normalise_hd_bet_device(HD_BET_DEVICE)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        inp = td / 'in.nii.gz'
        out = td / 'out.nii.gz'
        sitk.WriteImage(moving_img, str(inp))

        # Command variants to try, in order of preference.
        # HD-BET v2: `-device cuda:0 --save_bet_mask`
        # HD-BET v1: `-device 0` (mask always written)
        base_v2 = ['hd-bet', '-i', str(inp), '-o', str(out),
                   '-device', device, '--save_bet_mask']
        base_v1 = ['hd-bet', '-i', str(inp), '-o', str(out),
                   '-device', device]

        last_err = None
        for cmd in (base_v2, base_v1):
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                last_err = None
                break
            except subprocess.CalledProcessError as exc:
                last_err = (exc.stderr or '') + (exc.stdout or '') or str(exc)
                # Only retry if the failure is clearly about an unknown flag.
                low = last_err.lower()
                if ('unrecognized arguments' in low
                        or 'invalid choice' in low
                        or 'unknown option' in low
                        or 'no such option' in low):
                    continue
                raise RuntimeError(f"hd-bet call failed: {last_err}") from exc
            except FileNotFoundError as exc:
                raise RuntimeError(f"hd-bet call failed: {exc}") from exc
        if last_err is not None:
            raise RuntimeError(f"hd-bet call failed: {last_err}")

        # ---- Locate the brain-extracted image ----
        brain_path = out if out.exists() else None
        if brain_path is None:
            cands = [p for p in td.glob('out*.nii*')
                     if 'mask' not in p.name.lower()
                     and '_bet' not in p.name.lower()]
            brain_path = cands[0] if cands else None

        # ---- Locate the mask (v2: *_bet.nii.gz, v1: *_mask.nii.gz) ----
        mask_candidates = [
            td / 'out_bet.nii.gz',    # HD-BET v2
            td / 'out_mask.nii.gz',   # HD-BET v1
        ]
        mask_path = next((p for p in mask_candidates if p.exists()), None)
        if mask_path is None:
            cands = list(td.glob('*_bet.nii*')) + list(td.glob('*mask*.nii*'))
            mask_path = cands[0] if cands else None

        if brain_path is None or mask_path is None:
            listing = sorted(p.name for p in td.iterdir())
            raise RuntimeError(
                f"hd-bet produced no brain and/or mask output. "
                f"tmpdir contents: {listing}"
            )

        brain = sitk.ReadImage(str(brain_path), sitk.sitkFloat32)
        mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
        stats = sitk.StatisticsImageFilter()
        stats.Execute(mask)
        if stats.GetSum() <= 0:
            raise RuntimeError("hd-bet produced an empty brain mask")
        return brain, mask


# ──────────────────────────────────────────────────────────────
# N4 bias correction (inside the HD-BET mask only)
# ──────────────────────────────────────────────────────────────
def n4_bias_correct_sitk(img, mask):
    """N4 correction restricted to the HD-BET brain mask."""
    sitk = _require_sitk()
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # Small pyramid — the mask is already tight, so this is cheap.
    corrector.SetMaximumNumberOfIterations([50, 40, 30])
    try:
        return corrector.Execute(img, mask)
    except RuntimeError as exc:
        print(f"  [warn] N4 failed ({exc}); using uncorrected volume")
        return img


# ──────────────────────────────────────────────────────────────
# Affine registration to MNI (multires Mattes MI, physical space)
# ──────────────────────────────────────────────────────────────
def register_to_mni(brain_img, brain_mask, fixed_img, reference_img
                    ) -> Tuple[object, object, object]:
    """Multiresolution Mattes MI affine → MNI, resampled onto the fixed grid.

    Registration is performed on the TRUE physical grids (spacing, origin
    and direction from the subject's NIfTI header vs the MNI template),
    so the transform is a genuine affine in MNI physical space. Only one
    resample happens, straight onto the shared reference grid.
    """
    sitk = _require_sitk()

    init = sitk.CenteredTransformInitializerFilter.GEOMETRY
    initial_tx = sitk.CenteredTransformInitializer(
        fixed_img, brain_img, sitk.AffineTransform(3), init)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2)
    reg.SetInterpolator(sitk.sitkLinear)
    # Faster: capped at 80 iters/level, multires pyramid unchanged.
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=80,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial_tx, inPlace=False)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    if brain_mask is not None:
        reg.SetMetricMovingMask(brain_mask)

    out_tx = reg.Execute(fixed_img, brain_img)

    # Single resample onto the shared MNI reference grid.
    warped = sitk.Resample(
        brain_img, reference_img, out_tx,
        sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    warped_mask = sitk.Resample(
        brain_mask, reference_img, out_tx,
        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return warped, warped_mask, out_tx


# ──────────────────────────────────────────────────────────────
# Intensity / GM helpers (per-subject; NOT dataset statistics)
# ──────────────────────────────────────────────────────────────
def intensity_normalise(data: np.ndarray, mask: np.ndarray,
                        pct: Tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    """Per-subject percentile clip + [0,1] rescale inside the brain mask.

    Statistics are computed *per subject only* — nothing dataset-level
    or template-derived enters this function.
    """
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

    Intensity ordering on T1: CSF < GM < WM. All statistics are computed
    inside the subject's own brain mask — no dataset-level values.
    """
    gm = np.zeros_like(data, dtype=np.float32)
    if not mask.any():
        return gm
    vox = data[mask].astype(np.float64)
    if vox.std() < 1e-6:
        return gm

    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=3, n_init=10, random_state=0)
        labels = km.fit_predict(vox.reshape(-1, 1))
        centers = km.cluster_centers_.ravel()
        order = np.argsort(centers)  # CSF, GM, WM
        gm_id = int(order[1])
        d_gm = np.abs(vox - centers[gm_id])
        d_other = np.minimum(
            np.abs(vox - centers[order[0]]),
            np.abs(vox - centers[order[2]]),
        )
        soft = 1.0 / (1.0 + d_gm / (d_other + 1e-6))
        soft = soft * (labels == gm_id) + soft * 0.35 * (labels != gm_id)
        soft = soft / (soft.max() + 1e-8)
        gm[mask] = soft.astype(np.float32)
        return gm
    except Exception:
        pass

    # Percentile fallback: GM band ≈ 35th–70th percentile of brain
    lo, hi = np.percentile(vox, [35.0, 70.0])
    band = np.clip(data, lo, hi)
    mid, width = 0.5 * (lo + hi), max((hi - lo) * 0.5, 1e-6)
    soft = np.exp(-0.5 * ((band - mid) / width) ** 2) * mask.astype(np.float32)
    return (soft / (soft.max() + 1e-8)).astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Per-session pipeline
# ──────────────────────────────────────────────────────────────
def process_one(
    raw_path: Path,
    out_dir: Path,
    fixed_img,
    reference_img,
    target_shape: Tuple[int, int, int],
    use_gm: bool,
    force: bool = False,
) -> Dict:
    """End-to-end preprocessing for one session.

    Steps: load (preserve header) → HD-BET (GPU) → N4 in mask →
    multires Mattes MI affine to MNI → single resample onto the shared
    reference grid → intensity normalise → (optional) GM channel → save.
    """
    sitk = _require_sitk()
    out_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = (
        out_dir / f"tensor_{target_shape[0]}x{target_shape[1]}x{target_shape[2]}.pt"
    )
    meta_path = out_dir / 'meta.json'
    if tensor_path.exists() and not force:
        return {'status': 'skip', 'tensor': str(tensor_path)}

    # 1) Load moving volume preserving affine / orientation / spacing.
    moving_img = _sitk_read(raw_path)

    # 2) HD-BET skull-strip on GPU. No fallback — failure raises.
    brain_img, mask_img = skull_strip_hd_bet(moving_img)

    # 3) N4 only after HD-BET, restricted to the brain mask.
    brain_img = n4_bias_correct_sitk(brain_img, mask_img)

    # 4) Affine register to MNI + (5) single resample onto the fixed grid.
    warped_img, warped_mask_img, tx = register_to_mni(
        brain_img, mask_img, fixed_img, reference_img)

    # numpy views for normalisation / tensor packaging
    t1 = sitk.GetArrayFromImage(warped_img).astype(np.float32)      # (D,H,W)
    t1_mask = sitk.GetArrayFromImage(warped_mask_img).astype(bool)
    if not t1_mask.any():
        raise RuntimeError("empty brain mask after resampling to MNI grid")

    pct = tuple(getattr(cfg, 'MRI_PERCENTILE_CLIP', (1.0, 99.0)))
    t1 = intensity_normalise(t1, t1_mask, pct=pct)

    channels = [t1]
    gm = None
    if use_gm:
        gm = soft_gm_map(t1, t1_mask)
        gm = gm * t1_mask.astype(np.float32)
        gm = gm / (gm.max() + 1e-8) if gm.max() > 0 else gm
        channels.append(gm.astype(np.float32))

    tensor = torch.from_numpy(np.stack(channels, axis=0).astype(np.float32))
    torch.save(tensor, tensor_path)

    # Persist NIfTIs on the true MNI grid (spacing/origin/direction from
    # reference_img), so downstream code sees valid physical coordinates.
    def _as_sitk_like(np_arr: np.ndarray, ref) -> "sitk.Image":
        im = sitk.GetImageFromArray(np_arr.astype(np.float32))
        im.CopyInformation(ref)
        return im

    save_nifti_sitk(_as_sitk_like(t1, reference_img), out_dir / 't1.nii.gz')
    save_nifti_sitk(
        _as_sitk_like(t1_mask.astype(np.float32), reference_img),
        out_dir / 'mask.nii.gz',
    )
    if gm is not None:
        save_nifti_sitk(_as_sitk_like(gm, reference_img), out_dir / 'gm.nii.gz')

    meta = {
        'raw': str(raw_path),
        'shape': list(target_shape),
        'channels': ['t1'] + (['gm'] if use_gm else []),
        'skull_strip': 'hd-bet',
        'hd_bet_device': HD_BET_DEVICE,
        'n4': True,
        'registration': 'sitk_affine_mattes_mi_multires',
        'template': TEMPLATE_SPEC,
        'reference_grid': {
            'size_xyz': list(reference_img.GetSize()),
            'spacing_xyz': list(reference_img.GetSpacing()),
            'origin_xyz': list(reference_img.GetOrigin()),
            'direction': list(reference_img.GetDirection()),
        },
        'nifti_affine_ras': _sitk_to_nib_affine(reference_img).tolist(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return {'status': 'ok', 'tensor': str(tensor_path), **meta}


# ──────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────
def run_preprocess(
    data_dir: Path = None,
    csv_path: Path = None,
    out_dir: Path = None,
    target_shape: Tuple[int, int, int] = None,
    use_gm: bool = None,
    force: bool = False,
    offset: int = 0,
    limit: Optional[int] = None,
) -> None:
    data_dir = Path(data_dir or cfg.OASIS2_DIR)
    csv_path = Path(csv_path or cfg.OASIS2_CSV)
    out_dir = Path(out_dir or getattr(
        cfg, 'MRI_PROCESSED_DIR', data_dir / 'processed_v5'))
    target_shape = tuple(target_shape or cfg.MRI_SHAPE)
    if use_gm is None:
        use_gm = bool(getattr(cfg, 'MRI_USE_GM_CHANNEL', True))

    # Hard requirements — no silent fallbacks.
    _require_sitk()
    if shutil.which('hd-bet') is None:
        raise SystemExit(
            "hd-bet not found on PATH. HD-BET is the only skull-stripping "
            "method — install it before running preprocessing."
        )

    all_pairs = discover_sessions(data_dir, csv_path)
    if not all_pairs:
        raise SystemExit(f"No RAW MRI files found under {data_dir}")

    # TemplateFlow MNI152NLin2009cAsym 1 mm T1w brain — the single fixed
    # template shared across train / validation / test. No cohort mean,
    # no dataset-derived template.
    template_path = fetch_mni_template()
    fixed_img = _sitk_read(template_path)
    reference_img = build_reference_grid(fixed_img, target_shape)

    print(f"Total sessions found: {len(all_pairs)}")
    print(f"Output dir: {out_dir}")
    print(f"Template : {template_path}")
    print(f"Reference grid: size(xyz)={reference_img.GetSize()} "
          f"spacing(xyz)={tuple(round(s, 4) for s in reference_img.GetSpacing())}")
    print(f"hd-bet   : yes (device={HD_BET_DEVICE})")

    end = (offset + limit) if limit is not None else len(all_pairs)
    pairs = all_pairs[offset:end]
    print(f"This run: sessions [{offset}:{end}] of {len(all_pairs)} "
          f"-> {len(pairs)} session(s)")

    ok = skip = fail = 0
    failures: List[Dict[str, str]] = []
    for i, (mid, raw) in enumerate(pairs, offset + 1):
        print(f"[{i}/{len(all_pairs)}] {mid}")
        try:
            result = process_one(
                raw_path=raw,
                out_dir=out_dir / mid,
                fixed_img=fixed_img,
                reference_img=reference_img,
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
            failures.append({'mri_id': mid, 'raw': str(raw), 'error': str(exc)})
            print(f"  FAIL: {exc}")

    summary = {
        'ok': ok, 'skip': skip, 'fail': fail,
        'offset': offset, 'end': end, 'total_sessions': len(all_pairs),
        'out_dir': str(out_dir),
        'template': TEMPLATE_SPEC,
        'template_path': str(template_path),
        'reference_grid': {
            'size_xyz': list(reference_img.GetSize()),
            'spacing_xyz': list(reference_img.GetSpacing()),
            'origin_xyz': list(reference_img.GetOrigin()),
            'direction': list(reference_img.GetDirection()),
        },
        'failures': failures,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'preprocess_summary_{offset}_{end}.json').write_text(
        json.dumps(summary, indent=2))
    print(f"Done. ok={ok} skip={skip} fail={fail}")


def main():
    ap = argparse.ArgumentParser(
        description='OASIS-2 MRI preprocess (TemplateFlow MNI + HD-BET only)')
    ap.add_argument('--data-dir', type=Path, default=None)
    ap.add_argument('--csv', type=Path, default=None)
    ap.add_argument('--out-dir', type=Path, default=None)
    ap.add_argument('--force', action='store_true',
                    help='Reprocess even if tensor exists')
    ap.add_argument('--offset', type=int, default=0,
                    help='Start index into the full session list')
    ap.add_argument('--limit', type=int, default=None,
                    help='Number of sessions to process from --offset')
    ap.add_argument('--no-gm', action='store_true', help='Skip the GM channel')
    args = ap.parse_args()
    run_preprocess(
        data_dir=args.data_dir,
        csv_path=args.csv,
        out_dir=args.out_dir,
        use_gm=(False if args.no_gm else None),
        force=args.force,
        offset=args.offset,
        limit=args.limit,
    )


if __name__ == '__main__':
    main()