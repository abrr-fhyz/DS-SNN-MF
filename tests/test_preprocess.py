"""Preprocessing helpers that do not need SimpleITK / HD-BET."""
import inspect

import numpy as np

from OtherModules import mri_preprocess as pre


def test_preprocess_defaults_to_one_session_hd_bet_chunks():
    assert inspect.signature(pre.run_preprocess).parameters['chunk'].default == 1
    assert inspect.signature(pre.run_preprocess).parameters['workers'].default == 1


def test_collapse_dynamic_means_last_third_of_frames():
    arr = np.stack([np.full((2, 2, 2), i, dtype=np.float32) for i in range(6)], axis=0)
    out = pre.collapse_dynamic(arr, frac=1.0 / 3.0)
    assert out.shape == (2, 2, 2) and np.allclose(out, 4.5)  # frames 4 and 5
    assert np.array_equal(pre.collapse_dynamic(out), out)


def test_apply_hu_window_clips():
    arr = np.array([-20.0, 40.0, 200.0], dtype=np.float32)
    out = pre.apply_hu_window(arr, (0.0, 80.0))
    assert out.tolist() == [0.0, 40.0, 80.0]


def test_intensity_normalise_is_per_subject_and_masks_background():
    rng = np.random.default_rng(0)
    data = rng.random((8, 8, 8)).astype(np.float32) * 500 + 100
    mask = np.zeros(data.shape, dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    out = pre.intensity_normalise(data, mask, (1.0, 99.0))
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert (out[~mask] == 0).all() and out[mask].max() == 1.0
    assert np.array_equal(out, pre.intensity_normalise(data, mask, (1.0, 99.0)))


def test_intensity_normalise_degenerate_inputs():
    flat = np.ones((4, 4, 4), dtype=np.float32)
    assert (pre.intensity_normalise(flat, np.ones_like(flat, dtype=bool), (1.0, 99.0)) == 0).all()


def test_locate_finds_hd_bet_v1_and_v2_outputs(tmp_path):
    (tmp_path / 'A.nii.gz').write_bytes(b'')
    (tmp_path / 'A_bet.nii.gz').write_bytes(b'')
    (tmp_path / 'B.nii.gz').write_bytes(b'')
    (tmp_path / 'B_mask.nii.gz').write_bytes(b'')
    assert [p.name for p in pre._locate(tmp_path, 'A')] == ['A.nii.gz', 'A_bet.nii.gz']
    assert [p.name for p in pre._locate(tmp_path, 'B')] == ['B.nii.gz', 'B_mask.nii.gz']
    assert pre._locate(tmp_path, 'C') == (None, None)


def test_reference_grid_target_shape_is_not_needed_for_soft_gm_fallback():
    data = np.linspace(0, 1, 512, dtype=np.float32).reshape(8, 8, 8)
    gm = pre.soft_gm_map(data, data > 0.1)
    assert gm.shape == data.shape and gm.max() <= 1.0 and (gm[data <= 0.1] == 0).all()
