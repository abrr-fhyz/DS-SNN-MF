"""End-to-end smoke tests on a synthetic OASIS-2-style dataset with tiny volumes."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

import ExperimentModule as exp
from config import cfg
from DataModule import load_dataset, make_stratified_folds, set_seed, split_sequences
from OtherModules.baselines import run_baselines

SHAPE = (16, 16, 16)


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """24 subjects (8 per class) × 2 visits with informative clinical features and random MRI."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(24):
        label = i % 3
        for visit in (1, 2):
            mri_id = f"OAS2_{i:04d}_MR{visit}"
            raw = tmp_path / 'OAS2_RAW_PART1' / mri_id / 'RAW' / 'mpr-1.nii.gz'
            raw.parent.mkdir(parents=True)
            raw.write_bytes(b'')
            tensor = tmp_path / cfg.PROCESSED_SUBDIR / mri_id / f"tensor_{SHAPE[0]}x{SHAPE[1]}x{SHAPE[2]}.pt"
            tensor.parent.mkdir(parents=True)
            torch.save(torch.rand(1, *SHAPE), tensor)
            rows.append({'Subject ID': f"OAS2_{i:04d}", 'MRI ID': mri_id, 'Visit': visit, 'CDR': [0, 0.5, 1][label],
                         'M/F': 'MF'[i % 2], 'Age': 70 + 3 * label + rng.normal(), 'EDUC': 12 + i % 5,
                         'SES': 1 + i % 4, 'eTIV': 1500 + rng.normal() * 50, 'nWBV': 0.78 - 0.03 * label,
                         'ASF': 1.2 + rng.normal() * 0.05})
    pd.DataFrame(rows).to_csv(tmp_path / 'oasis_longitudinal_demographics.csv', index=False)

    monkeypatch.setitem(cfg.DATASETS['oasis2'], 'root', tmp_path)
    for key, value in dict(DATASET='oasis2', MRI_SHAPE=SHAPE, VIEW_SHAPE=SHAPE, NUM_EPOCHS=2, MIN_EPOCHS=1,
                           CHECKPOINT_WARMUP=1, PATIENCE=1, N_FOLDS=2, REPEATS=1, USE_CV=True, BATCH_SIZE=4,
                           GRAD_ACCUM_STEPS=1, NUM_WORKERS=0, BOOTSTRAP_ROUNDS=20, CKPT_MAX_VAL_TRAIN_GAP=1.0,
                           OUT_DIR=tmp_path / 'runs', ABLATION='full').items():
        monkeypatch.setattr(cfg, key, value)
    return tmp_path


def test_oasis2_loader_on_synthetic_tree(synthetic):
    sequences = load_dataset()
    assert len(sequences) == 24 and all(len(s) == 2 for s in sequences)
    assert sequences[0][0]['clinical_raw'].shape == (7,)
    assert [s[-1]['label'] for s in sequences[:3]] == [0, 1, 2]


@pytest.mark.parametrize('ablation', ['full', 'clinical', 'mri'])
def test_cv_run_writes_report(synthetic, monkeypatch, ablation):
    monkeypatch.setattr(cfg, 'ABLATION', ablation)
    run_dir = synthetic / 'run'
    summary = exp.run_experiment(torch.device('cpu'), run_dir=run_dir)

    for name in ('config.json', 'summary.json', 'oof_predictions.csv', 'run.log', 'fold_r1f1_best.pt'):
        assert (run_dir / name).exists(), name
    assert json.loads((run_dir / 'config.json').read_text())['ABLATION'] == ablation
    oof = pd.read_csv(run_dir / 'oof_predictions.csv')
    assert len(oof) == 24 and oof['subject_idx'].nunique() == 24
    assert {'pred', 'pred_rf', 'pred_logreg', 'p_CN', 'uncertainty'} <= set(oof.columns)
    assert 0.0 <= summary['test_acc']['mean'] <= 1.0 and len(summary['test_acc']['values']) == 2
    pooled = json.loads((run_dir / 'summary.json').read_text())['pooled_oof']
    assert set(pooled) == {'model', 'logreg', 'rf'} and 'ci95' in pooled['model']


def test_repeats_and_single_split(synthetic, monkeypatch):
    monkeypatch.setattr(cfg, 'ABLATION', 'clinical')
    monkeypatch.setattr(cfg, 'REPEATS', 2)
    summary = exp.run_experiment(torch.device('cpu'), run_dir=synthetic / 'rep')
    assert len(summary['test_acc']['values']) == 4

    monkeypatch.setattr(cfg, 'USE_CV', False)
    summary = exp.run_experiment(torch.device('cpu'), run_dir=synthetic / 'single')
    assert len(summary['test_acc']['values']) == 1


def test_split_run_is_deterministic(synthetic):
    sequences = load_dataset()
    indices = make_stratified_folds(sequences, 2, 0, 0.2)[0]
    runs = []
    for k in range(2):
        set_seed(7)
        _, preds = exp.run_split(split_sequences(sequences, indices), torch.device('cpu'),
                                 str(synthetic / f'ckpt{k}.pt'))
        runs.append(preds.probs)
    assert np.allclose(runs[0], runs[1], atol=1e-5)


def test_baselines_share_folds_and_cover_all_subjects(synthetic):
    sequences = load_dataset()
    folds = make_stratified_folds(sequences, 2, 0, 0.2)
    res = run_baselines(sequences, folds)
    assert set(res) == {'logreg', 'rf'} and all((r['oof'] >= 0).all() for r in res.values())
    assert all(len(r['folds']) == 2 for r in res.values())


def test_bootstrap_ci_orders_and_pairs():
    labels = np.tile([0, 1, 2], 20)
    good = np.where(np.arange(60) % 10 == 0, (labels + 1) % 3, labels)[None]
    bad = np.where(np.arange(60) % 2 == 0, (labels + 1) % 3, labels)[None]
    out = exp.bootstrap_comparison(labels, {'model': good, 'base': bad}, 'model', rounds=200, seed=0)
    assert out['model']['macro_f1'] > out['base']['macro_f1']
    lo, hi = out['model']['ci95']
    assert lo <= out['model']['macro_f1'] <= hi
    assert out['base']['delta_vs_reference'] > 0 and out['base']['frac_delta_le_zero'] < 0.05
