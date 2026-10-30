"""Shared data layer: labels, splits, train-only normalisation, datasets."""
import numpy as np
import pytest
import torch

from config import cfg
from DataModule import (ClinicalScaler, CrossSectionalWrapper, SubjectDataset, cdr_to_label,
                        longitudinal_collate_fn, make_holdout_split, make_stratified_folds, make_visit,
                        split_sequences, subject_labels)
from MetricsModule import (Predictions, balanced_accuracy, compute_metrics, confusion_matrix,
                           expected_calibration_error, macro_f1, selective_accuracy)


def make_sequences(n_per_class=10, n_visits=2, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    sequences = []
    for label in range(3):
        for k in range(n_per_class):
            visits = [make_visit(f"S{label}_{k}", f"S{label}_{k}_v{v}", v + 1, "x",
                                 rng.normal(label * 2.0, 1.0, n_features), label if v == n_visits - 1 else 0)
                      for v in range(n_visits)]
            sequences.append(visits)
    return sequences


@pytest.mark.parametrize('cdr, label', [(0, 0), (0.0, 0), (0.5, 1), (1, 2), (2.0, 2), (3, 2),
                                        (None, None), (float('nan'), None), ('x', None)])
def test_cdr_to_label(cdr, label):
    assert cdr_to_label(cdr) == label


def test_folds_are_disjoint_stratified_subject_splits():
    seqs = make_sequences()
    y = subject_labels(seqs)
    folds = make_stratified_folds(seqs, n_folds=5, seed=1, val_frac=0.2)
    tests = np.concatenate([f['test'] for f in folds])
    assert sorted(tests.tolist()) == list(range(len(seqs)))
    for f in folds:
        parts = [set(f[k].tolist()) for k in ('train', 'val', 'test')]
        assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
        assert sum(map(len, parts)) == len(seqs)
        assert set(y[f['test']]) == {0, 1, 2}


def test_folds_change_with_seed_but_not_between_calls():
    seqs = make_sequences()
    a, b, c = (make_stratified_folds(seqs, 5, s, 0.2) for s in (1, 1, 2))
    assert all((x['test'] == y['test']).all() for x, y in zip(a, b))
    assert any((x['test'] != y['test']).any() for x, y in zip(a, c))


def test_holdout_split_partitions():
    seqs = make_sequences()
    s = make_holdout_split(seqs, seed=3)
    assert sorted(np.concatenate(list(s.values())).tolist()) == list(range(len(seqs)))


def test_scaler_uses_train_statistics_only():
    seqs = make_sequences(n_per_class=4)
    seqs[0][0]['clinical_raw'][1] = np.nan
    train, test = seqs[:8], seqs[8:]
    scaler = ClinicalScaler().fit(train)
    out_train = np.array([v['clinical'] for s in scaler.transform(train) for v in s])
    out_test = np.array([v['clinical'] for s in scaler.transform(test) for v in s])
    assert np.allclose(out_train.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(out_train.std(axis=0)[[0, 2]], 1, atol=1e-5)    # col 1 has an imputed NaN
    assert not np.allclose(out_test.mean(axis=0), 0, atol=1e-2)      # not re-centred on test data
    assert np.isfinite(out_train).all()
    assert seqs[0][0]['clinical_raw'][1] != seqs[0][0]['clinical_raw'][1]   # inputs untouched (NaN)


def test_split_sequences_fits_on_train_only():
    seqs = make_sequences()
    folds = make_stratified_folds(seqs, 5, 0, 0.2)
    data = split_sequences(seqs, folds[0])
    raw = np.array([v['clinical_raw'] for i in folds[0]['train'] for v in seqs[i]], dtype=np.float64)
    scaled = np.array([v['clinical'] for s in data['train'] for v in s])
    assert np.allclose(scaled, (raw - raw.mean(0)) / raw.std(0), atol=1e-4)


def test_constant_feature_does_not_explode():
    seqs = make_sequences(n_per_class=3)
    for s in seqs:
        for v in s:
            v['clinical_raw'][0] = 5.0
    out = ClinicalScaler().fit(seqs).transform(seqs)
    assert np.isfinite(np.array([v['clinical'] for s in out for v in s])).all()


def test_wrapper_last_vs_all_visits(monkeypatch):
    monkeypatch.setattr(cfg, 'ABLATION', 'clinical')   # skips MRI I/O
    seqs = make_sequences(n_per_class=3, n_visits=3)
    last, every = CrossSectionalWrapper(seqs, 'last'), CrossSectionalWrapper(seqs, 'all')
    assert (len(last), len(every)) == (9, 27)
    assert last.label_counts() == [3, 3, 3]
    item = every[0]
    assert item['clinical'].shape == (3,) and item['mri'].shape == (1, *cfg.MRI_SHAPE)


def test_collate_pads_missing_visits(monkeypatch):
    monkeypatch.setattr(cfg, 'ABLATION', 'clinical')
    monkeypatch.setattr(cfg, 'MRI_SHAPE', (4, 4, 4))
    seqs = make_sequences(n_per_class=1, n_visits=2)
    seqs[0] = seqs[0][:1]
    ds = SubjectDataset(seqs)
    batch = longitudinal_collate_fn([ds[0], ds[1]])
    assert [v['has_visit'] for v in batch['visits']] == [[True, True], [False, True]]
    assert (batch['visits'][1]['clinical'][0] == 0).all()
    assert batch['label'].tolist() == [0, 1]


def make_predictions(n=60, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, n)
    probs = rng.dirichlet(np.ones(3), n)
    probs[np.arange(n), labels] += 1.0
    probs /= probs.sum(axis=1, keepdims=True)
    return Predictions(labels=labels, probs=probs, uncertainty=rng.random(n), K=rng.random(n) * 0.3,
                       gate=rng.random(n), blend=rng.random(n), conflict_flag=rng.random(n) > 0.5)


def test_metrics_consistency():
    p = make_predictions()
    m = compute_metrics(p)
    from sklearn.metrics import f1_score, roc_auc_score
    assert np.isclose(m['macro_f1'], f1_score(p.labels, p.preds, average='macro'))
    assert np.isclose(m['auroc'], roc_auc_score(p.labels, p.probs, multi_class='ovr'))
    assert np.isclose(m['acc'], (p.labels == p.preds).mean())
    assert 0.0 <= m['ece'] <= 1.0 and m['brier'] >= 0.0
    assert np.isclose(balanced_accuracy(m['confusion']), m['bal_acc'])


def test_selective_accuracy_and_ece_edge_cases():
    correct = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
    risk = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])
    assert selective_accuracy(correct, risk, 0.5) == 1.0
    assert selective_accuracy(correct, risk, 1.0) == correct.mean()
    perfect = np.eye(3)[[0, 1, 2, 0]]
    assert expected_calibration_error(perfect, np.array([0, 1, 2, 0])) == 0.0
    assert macro_f1(np.array([0, 1, 2]), np.array([0, 1, 2])) == 1.0
    assert confusion_matrix(np.array([0, 1]), np.array([1, 1])).tolist() == [[0, 1, 0], [0, 1, 0], [0, 0, 0]]
