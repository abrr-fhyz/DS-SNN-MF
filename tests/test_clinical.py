"""Clinical MLP vs frozen tabular ensemble."""
import numpy as np
import torch

from config import cfg
from MODULE.ClinicalEnsemble import ClinicalEnsemble, proba_to_bba
from MODULE.ClinicalMLP import ClinicalMLP
from MODULE.FusionModule import DSSNMMF
from MODULE.TrainerModule import Trainer, build_param_groups


def test_proba_to_bba_is_valid_simplex():
    p = torch.tensor([[0.7, 0.2, 0.1], [0.0, 1.0, 0.0]])
    m = proba_to_bba(p, cap=18.0)
    assert m.shape == (2, 8) and torch.allclose(m[:, 0], torch.zeros(2))
    assert torch.allclose(m.sum(dim=1), torch.ones(2), atol=1e-5)
    assert (m >= 0).all()


def test_ensemble_fits_tiny_table(monkeypatch):
    monkeypatch.setattr(cfg, 'CLINICAL_ENSEMBLE_MEMBERS', ('rf', 'extratrees', 'svm'))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    y = np.array([0, 1, 2] * 10)
    enc = ClinicalEnsemble(n_features=5).fit(X, y)
    p = enc.predict_proba(X[:4])
    assert p.shape == (4, 3) and np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    bba = enc.bba(torch.tensor(X[:4], dtype=torch.float32))
    assert bba.shape == (4, 8) and torch.allclose(bba.sum(dim=1), torch.ones(4), atol=1e-5)


def test_dssnnmf_mlp_and_ensemble_forward(monkeypatch):
    monkeypatch.setattr(cfg, 'ABLATION', 'clinical')
    monkeypatch.setattr(cfg, 'CLINICAL_MODEL', 'mlp')
    mlp_model = DSSNMMF()
    x = torch.randn(3, 5)
    out = mlp_model(clinical=x)
    assert out['decision'].shape == (3, 3) and out['m_fused'].shape == (3, 8)

    monkeypatch.setattr(cfg, 'CLINICAL_MODEL', 'ensemble')
    monkeypatch.setattr(cfg, 'CLINICAL_ENSEMBLE_MEMBERS', ('rf', 'extratrees'))
    ens_model = DSSNMMF()
    rng = np.random.default_rng(1)
    X = rng.normal(size=(24, 5))
    y = np.array([0, 1, 2] * 8)
    ens_model.clinical_encoder.fit(X, y)
    out = ens_model(clinical=torch.tensor(X[:3], dtype=torch.float32))
    assert out['decision'].shape == (3, 3)
    assert torch.allclose(out['decision'].sum(dim=1), torch.ones(3), atol=1e-5)


def test_clinical_only_ensemble_skips_neural_training(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, 'ABLATION', 'clinical')
    monkeypatch.setattr(cfg, 'CLINICAL_MODEL', 'ensemble')
    monkeypatch.setattr(cfg, 'CLINICAL_ENSEMBLE_MEMBERS', ('rf', 'extratrees'))
    model = DSSNMMF()
    rng = np.random.default_rng(2)
    X = rng.normal(size=(24, 5))
    y = np.array([0, 1, 2] * 8)
    model.clinical_encoder.fit(X, y)
    assert build_param_groups(model) == []

    class _Loader:
        def __iter__(self):
            xb = torch.tensor(X[:8], dtype=torch.float32)
            yb = torch.tensor(y[:8], dtype=torch.long)
            yield {'clinical': xb, 'label': yb, 'mri': torch.zeros(8, 1, 4, 4, 4),
                   'pet': torch.zeros(8, 1, 4, 4, 4), 'ct': torch.zeros(8, 1, 4, 4, 4),
                   'has_pet': torch.zeros(8, dtype=torch.bool), 'has_ct': torch.zeros(8, dtype=torch.bool)}

    ckpt = tmp_path / 'best.pt'
    trainer = Trainer(model, device=torch.device('cpu'))
    history = trainer.fit(_Loader(), _Loader(), save_path=str(ckpt))
    assert history == [] and ckpt.exists()
    extra = ckpt.with_name(ckpt.stem + '.clinical_ensemble.joblib')
    assert extra.exists()
    trainer.load_checkpoint(str(ckpt))
    out = trainer.model(clinical=torch.tensor(X[:2], dtype=torch.float32))
    assert out['decision'].shape == (2, 3)


def test_clinical_mlp_still_imports():
    mlp = ClinicalMLP(n_features=5)
    assert mlp(torch.randn(2, 5)).shape == (2, cfg.CLINICAL_FEAT_DIM)
