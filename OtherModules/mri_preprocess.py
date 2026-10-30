"""Offline MRI preprocessing for any registered dataset (run once before training).

Pipeline per session
--------------------
  1. Load the raw T1 preserving NIfTI affine, orientation and spacing.
  2. Skull-strip with HD-BET on GPU. No fallback: a failure is recorded in the run summary.
  3. N4 bias-field correction inside the HD-BET brain mask (SimpleITK).
  4. Multiresolution Mattes-MI *affine* registration to the TemplateFlow
     ``MNI152NLin2009cAsym`` 1 mm T1w brain, on the true physical grids.
  5. One resample onto a FIXED MNI reference grid (identical for every subject and split).
  6. Per-subject intensity normalisation and an optional soft GM channel.

No cohort statistics enter any step, so train/validation/test share the same fixed template.

Scaling
-------
Sessions are processed in chunks (``--chunk``). HD-BET runs once per chunk on a whole folder
(one model load instead of one per volume); N4 + registration then run in ``--workers``
processes. Sessions whose tensor exists are skipped, so re-running with a larger ``--limit``
or another ``--offset`` is safe and resumable.

Output per session ``<MRI_PROCESSED_DIR>/<mri_id>/``:
    t1.nii.gz, mask.nii.gz, [gm.nii.gz], tensor_DxHxW.pt (C = 1 or 2), meta.json

Usage (from the project root)::

    python -m OtherModules.mri_preprocess --dataset oasis2
    python -m OtherModules.mri_preprocess --dataset oasis3 --chunk 32 --workers 4
    python -m OtherModules.mri_preprocess --limit 5 --force        # smoke test

Requires nibabel, SimpleITK, templateflow and the ``hd-bet`` CLI on PATH.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from config import cfg
from DataModule import load_dataset, processed_tensor_path

HD_BET_DEVICE = os.environ.get('HD_BET_DEVICE', 'cuda:0')
TEMPLATE_SPEC = dict(template='MNI152NLin2009cAsym', resolution=1, desc='brain', suffix='T1w', extension='.nii.gz')
_HD_BET_UNKNOWN_FLAG = ('unrecognized arguments', 'invalid choice', 'unknown option', 'no such option')


def _sitk():
    try:
        import SimpleITK as sitk
    except ImportError as e:
        raise SystemExit("SimpleITK is required (pip install SimpleITK)") from e
    return sitk


# ──────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────

def _materialise_analyze(path: Path) -> Path:
    """Create .img/.hdr twins for OASIS-2 short '.nifti' Analyze pairs."""
    if path.suffix.lower() != '.nifti':
        return path
    img_twin, hdr_src, hdr_twin = path.with_suffix('.img'), Path(str(path) + '.hdr'), path.with_suffix('.hdr')
    for src, twin in ((path, img_twin), (hdr_src, hdr_twin)):
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


def _read(path: Path, pixel=None):
    sitk = _sitk()
    return sitk.ReadImage(str(_materialise_analyze(Path(path))), pixel or sitk.sitkFloat32)


def _nib_affine(img) -> np.ndarray:
    """SimpleITK (LPS) geometry → nibabel RAS affine."""
    direction = np.array(img.GetDirection()).reshape(3, 3)
    affine = np.eye(4)
    affine[:3, :3] = direction * np.array(img.GetSpacing())
    affine[:3, 3] = img.GetOrigin()
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine


def _write(img, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _sitk().WriteImage(img, str(path))


def discover_sessions() -> List[Tuple[str, Path]]:
    """Unique (mri_id, raw scan) pairs of the active dataset, in stable order."""
    seen: Dict[str, Path] = {}
    for sequence in load_dataset():
        for visit in sequence:
            seen.setdefault(str(visit['mri_id']), Path(visit['mri_path']))
    return list(seen.items())


# ──────────────────────────────────────────────────────────────
# Template and reference grid
# ──────────────────────────────────────────────────────────────

def fetch_mni_template() -> Path:
    try:
        from templateflow import api as tflow
    except ImportError as e:
        raise SystemExit("templateflow is required (pip install templateflow)") from e
    path = tflow.get(**TEMPLATE_SPEC)
    if isinstance(path, list):
        if not path:
            raise SystemExit(f"TemplateFlow returned no file for {TEMPLATE_SPEC}")
        path = path[0]
    return Path(str(path))


def build_reference_grid(template_img, target_shape: Tuple[int, int, int]):
    """Fixed MNI grid with the template's FOV at ``target_shape`` (numpy D, H, W order)."""
    sitk = _sitk()
    size = template_img.GetSize()                      # (X, Y, Z)
    if tuple(target_shape) == (size[2], size[1], size[0]):
        return template_img
    target = (int(target_shape[2]), int(target_shape[1]), int(target_shape[0]))
    ref = sitk.Image(target, template_img.GetPixelID())
    ref.SetOrigin(template_img.GetOrigin())
    ref.SetDirection(template_img.GetDirection())
    ref.SetSpacing(tuple(s * n / t for s, n, t in zip(template_img.GetSpacing(), size, target)))
    return ref


@functools.lru_cache(maxsize=2)
def _template(template_path: str, target_shape: Tuple[int, int, int]):
    fixed = _read(Path(template_path))
    return fixed, build_reference_grid(fixed, target_shape)


# ──────────────────────────────────────────────────────────────
# Skull stripping (batched HD-BET)
# ──────────────────────────────────────────────────────────────

def _run_hd_bet(in_dir: Path, out_dir: Path) -> None:
    if shutil.which('hd-bet') is None:
        raise RuntimeError("hd-bet not found on PATH")
    device = HD_BET_DEVICE.strip().lower()
    device = f'cuda:{device}' if device.isdigit() else device
    base = ['hd-bet', '-i', str(in_dir), '-o', str(out_dir), '-device', device]
    last_err = ''
    for cmd in (base + ['--save_bet_mask'], base):          # HD-BET v2, then v1
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return
        except subprocess.CalledProcessError as exc:
            last_err = (exc.stderr or '') + (exc.stdout or '') or str(exc)
            if not any(flag in last_err.lower() for flag in _HD_BET_UNKNOWN_FLAG):
                break
    raise RuntimeError(f"hd-bet call failed: {last_err}")


def _locate(out_dir: Path, mid: str) -> Tuple[Optional[Path], Optional[Path]]:
    """Brain and mask files HD-BET wrote for ``mid`` (v2: *_bet, v1: *_mask)."""
    brain = out_dir / f'{mid}.nii.gz'
    mask = next((p for p in (out_dir / f'{mid}_bet.nii.gz', out_dir / f'{mid}_mask.nii.gz') if p.exists()), None)
    return (brain if brain.exists() else None), mask


def skull_strip_batch(sessions: List[Tuple[str, Path]], workdir: Path
                      ) -> Dict[str, Tuple[Optional[Path], Optional[Path], Optional[str]]]:
    """Skull-strip every session with one HD-BET call. Returns {mri_id: (brain, mask, error)}."""
    sitk = _sitk()
    in_dir, out_dir = workdir / 'in', workdir / 'out'
    in_dir.mkdir(), out_dir.mkdir()
    results: Dict[str, Tuple[Optional[Path], Optional[Path], Optional[str]]] = {}
    staged = []
    for mid, raw in sessions:
        try:
            sitk.WriteImage(_read(raw), str(in_dir / f'{mid}.nii.gz'))
            staged.append(mid)
        except Exception as exc:  # noqa: BLE001
            results[mid] = (None, None, f"cannot read {raw}: {exc}")
    if staged:
        _run_hd_bet(in_dir, out_dir)
    for mid in staged:
        brain, mask = _locate(out_dir, mid)
        error = None if brain and mask else \
            f"hd-bet produced no brain/mask for {mid}; out dir has {sorted(p.name for p in out_dir.iterdir())[:8]}"
        results[mid] = (brain, mask, error)
    return results


# ──────────────────────────────────────────────────────────────
# Per-subject processing
# ──────────────────────────────────────────────────────────────

def n4_bias_correct(img, mask):
    corrector = _sitk().N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 40, 30])
    try:
        return corrector.Execute(img, mask)
    except RuntimeError as exc:
        print(f"  [warn] N4 failed ({exc}); using uncorrected volume")
        return img


def register_to_mni(brain_img, brain_mask, fixed_img, reference_img):
    """Multires Mattes MI affine to MNI, resampled once onto the fixed reference grid."""
    sitk = _sitk()
    initial = sitk.CenteredTransformInitializer(
        fixed_img, brain_img, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=80,
                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial, inPlace=False)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    if brain_mask is not None:
        reg.SetMetricMovingMask(brain_mask)
    tx = reg.Execute(fixed_img, brain_img)
    warped = sitk.Resample(brain_img, reference_img, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    warped_mask = sitk.Resample(brain_mask, reference_img, tx, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return warped, warped_mask


def intensity_normalise(data: np.ndarray, mask: np.ndarray, pct: Tuple[float, float]) -> np.ndarray:
    """Per-subject percentile clip and [0, 1] rescale inside the mask; background zeroed."""
    brain = data[mask] if mask.any() else data.ravel()
    if brain.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    lo, hi = np.percentile(brain, [float(pct[0]), float(pct[1])])
    if hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    return ((np.clip(data, lo, hi) - lo) / (hi - lo) * mask).astype(np.float32)


def soft_gm_map(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Soft grey-matter map from a 3-cluster (CSF/GM/WM) k-means on the subject's own brain voxels."""
    gm = np.zeros_like(data, dtype=np.float32)
    vox = data[mask].astype(np.float64) if mask.any() else np.array([])
    if vox.size == 0 or vox.std() < 1e-6:
        return gm
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=3, n_init=10, random_state=0)
        labels = km.fit_predict(vox.reshape(-1, 1))
        centers = km.cluster_centers_.ravel()
        order = np.argsort(centers)                       # CSF, GM, WM
        gm_id = int(order[1])
        d_gm = np.abs(vox - centers[gm_id])
        d_other = np.minimum(np.abs(vox - centers[order[0]]), np.abs(vox - centers[order[2]]))
        soft = 1.0 / (1.0 + d_gm / (d_other + 1e-6))
        soft = soft * (labels == gm_id) + soft * 0.35 * (labels != gm_id)
        gm[mask] = (soft / (soft.max() + 1e-8)).astype(np.float32)
        return gm
    except Exception:  # noqa: BLE001  (percentile fallback below)
        lo, hi = np.percentile(vox, [35.0, 70.0])
        mid, width = 0.5 * (lo + hi), max((hi - lo) * 0.5, 1e-6)
        soft = np.exp(-0.5 * ((np.clip(data, lo, hi) - mid) / width) ** 2) * mask
        return (soft / (soft.max() + 1e-8)).astype(np.float32)


def finish_session(job: dict) -> dict:
    """N4 → registration → normalisation → save, for one skull-stripped session."""
    sitk = _sitk()
    mid, out_dir, shape = job['mri_id'], Path(job['out_dir']), tuple(job['target_shape'])
    try:
        fixed, reference = _template(job['template_path'], shape)
        brain = _read(job['brain'])
        mask = _read(job['mask'], sitk.sitkUInt8)
        stats = sitk.StatisticsImageFilter()
        stats.Execute(mask)
        if stats.GetSum() <= 0:
            raise RuntimeError("hd-bet produced an empty brain mask")

        warped, warped_mask = register_to_mni(n4_bias_correct(brain, mask), mask, fixed, reference)
        t1_mask = sitk.GetArrayFromImage(warped_mask).astype(bool)
        if not t1_mask.any():
            raise RuntimeError("empty brain mask after resampling to MNI grid")
        t1 = intensity_normalise(sitk.GetArrayFromImage(warped).astype(np.float32), t1_mask,
                                 cfg.MRI_PERCENTILE_CLIP)
        channels, gm = [t1], None
        if job['use_gm']:
            gm = soft_gm_map(t1, t1_mask) * t1_mask
            channels.append((gm / gm.max() if gm.max() > 0 else gm).astype(np.float32))

        def as_image(arr):
            img = sitk.GetImageFromArray(arr.astype(np.float32))
            img.CopyInformation(reference)
            return img

        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(torch.from_numpy(np.stack(channels)), job['tensor_path'])
        _write(as_image(t1), out_dir / 't1.nii.gz')
        _write(as_image(t1_mask.astype(np.float32)), out_dir / 'mask.nii.gz')
        if gm is not None:
            _write(as_image(channels[1]), out_dir / 'gm.nii.gz')
        (out_dir / 'meta.json').write_text(json.dumps({
            'raw': job['raw'], 'shape': list(shape), 'channels': ['t1'] + (['gm'] if gm is not None else []),
            'skull_strip': 'hd-bet', 'hd_bet_device': HD_BET_DEVICE, 'n4': True,
            'registration': 'sitk_affine_mattes_mi_multires', 'template': TEMPLATE_SPEC,
            'reference_grid': {'size_xyz': list(reference.GetSize()), 'spacing_xyz': list(reference.GetSpacing()),
                               'origin_xyz': list(reference.GetOrigin()),
                               'direction': list(reference.GetDirection())},
            'nifti_affine_ras': _nib_affine(reference).tolist()}, indent=2))
        return {'mri_id': mid, 'status': 'ok'}
    except Exception as exc:  # noqa: BLE001
        return {'mri_id': mid, 'status': 'fail', 'raw': job['raw'], 'error': str(exc)}


# ──────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────

def run_preprocess(out_dir: Optional[Path] = None, use_gm: Optional[bool] = None, force: bool = False,
                   offset: int = 0, limit: Optional[int] = None, chunk: int = 16, workers: int = 1) -> None:
    out_dir = Path(out_dir or cfg.MRI_PROCESSED_DIR)
    use_gm = cfg.MRI_USE_GM_CHANNEL if use_gm is None else use_gm
    shape = tuple(cfg.MRI_SHAPE)
    _sitk()
    if shutil.which('hd-bet') is None:
        raise SystemExit("hd-bet not found on PATH; it is the only skull-stripping method.")

    sessions = discover_sessions()
    if not sessions:
        raise SystemExit(f"No MRI sessions found for dataset '{cfg.DATASET}' under {cfg.spec.root}")
    end = offset + limit if limit is not None else len(sessions)
    window = sessions[offset:end]
    tensor_path = lambda mid: out_dir / mid / processed_tensor_path(mid).name  # noqa: E731
    pending = [(mid, raw) for mid, raw in window if force or not tensor_path(mid).exists()]
    template_path = fetch_mni_template()
    print(f"Dataset {cfg.DATASET}: {len(sessions)} sessions | window [{offset}:{end}] = {len(window)} | "
          f"to process {len(pending)} | out {out_dir} | template {template_path} | hd-bet {HD_BET_DEVICE} | "
          f"chunk {chunk} workers {workers}")

    failures: List[dict] = []
    ok = 0
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for start in range(0, len(pending), chunk):
            batch = pending[start:start + chunk]
            print(f"[{offset + start + 1}-{offset + start + len(batch)}/{len(sessions)}] skull-stripping {len(batch)} session(s)")
            with tempfile.TemporaryDirectory() as td:
                stripped = skull_strip_batch(batch, Path(td))
                jobs = []
                for mid, raw in batch:
                    brain, mask, error = stripped[mid]
                    if error:
                        failures.append({'mri_id': mid, 'raw': str(raw), 'error': error})
                        continue
                    jobs.append({'mri_id': mid, 'raw': str(raw), 'brain': str(brain), 'mask': str(mask),
                                 'out_dir': str(out_dir / mid), 'tensor_path': str(tensor_path(mid)),
                                 'template_path': str(template_path), 'target_shape': shape, 'use_gm': use_gm})
                for res in (pool.map(finish_session, jobs) if pool else map(finish_session, jobs)):
                    if res['status'] == 'ok':
                        ok += 1
                    else:
                        failures.append(res)
                    print(f"  {res['mri_id']}: {res['status']}" + (f" ({res['error']})" if res['status'] == 'fail' else ""))
    finally:
        if pool:
            pool.shutdown()

    skipped = len(window) - len(pending)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'preprocess_summary_{offset}_{end}.json').write_text(json.dumps({
        'dataset': cfg.DATASET, 'ok': ok, 'skip': skipped, 'fail': len(failures), 'offset': offset, 'end': end,
        'total_sessions': len(sessions), 'template': TEMPLATE_SPEC, 'template_path': str(template_path),
        'failures': failures}, indent=2))
    print(f"Done. ok={ok} skip={skipped} fail={len(failures)}")


def main() -> None:
    ap = argparse.ArgumentParser(description='MRI preprocess (TemplateFlow MNI + HD-BET)')
    ap.add_argument('--dataset', choices=list(cfg.DATASETS), default=cfg.DATASET)
    ap.add_argument('--out-dir', type=Path, default=None)
    ap.add_argument('--force', action='store_true', help='reprocess even if the tensor exists')
    ap.add_argument('--offset', type=int, default=0, help='start index into the session list')
    ap.add_argument('--limit', type=int, default=None, help='number of sessions from --offset')
    ap.add_argument('--chunk', type=int, default=16, help='sessions per HD-BET call')
    ap.add_argument('--workers', type=int, default=1, help='processes for N4 + registration')
    ap.add_argument('--no-gm', action='store_true', help='skip the GM channel')
    args = ap.parse_args()
    cfg.DATASET = args.dataset
    run_preprocess(out_dir=args.out_dir, use_gm=False if args.no_gm else None, force=args.force,
                   offset=args.offset, limit=args.limit, chunk=args.chunk, workers=args.workers)


if __name__ == '__main__':
    main()
