"""DS algebra: the vectorised rules must match straightforward loop references."""
import pytest
import torch

from DSModule import (DSEvidenceLayer, TemporalAccumulator, compute_belief_plausibility, conflict_breakdown,
                      dempster_combine, discount_bba, ordinal_yager_combine, pcr5_combine,
                      pignistic_probability, vacuous_bba)
from LossModule import evidential_utility_loss, ignorance_anchoring_loss

FULL = 7


def random_bba(batch: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    m = torch.rand(batch, 8, generator=g)
    m[:, 0] = 0.0
    return m / m.sum(dim=1, keepdim=True)


def loop_conjunctive(m1, m2):
    out, K = torch.zeros(m1.shape[0], 8), torch.zeros(m1.shape[0])
    for a in range(1, 8):
        for b in range(1, 8):
            p = m1[:, a] * m2[:, b]
            if a & b:
                out[:, a & b] += p
            else:
                K += p
    return out, K


def loop_pcr5(m1, m2):
    out, K = loop_conjunctive(m1, m2)
    for a in range(1, 8):
        for b in range(1, 8):
            if a & b == 0:
                denom = (m1[:, a] + m2[:, b]).clamp(min=1e-8)
                out[:, a] += m1[:, a] ** 2 * m2[:, b] / denom
                out[:, b] += m2[:, b] ** 2 * m1[:, a] / denom
    return out, K


@pytest.fixture
def pair():
    return random_bba(16, 1), random_bba(16, 2)


def test_dempster_matches_loop(pair):
    out, K = loop_conjunctive(*pair)
    got, got_K = dempster_combine(*pair)
    assert torch.allclose(got_K, K, atol=1e-6)
    assert torch.allclose(got, out / (1 - K).unsqueeze(1), atol=1e-6)


def test_yager_moves_conflict_to_theta(pair):
    out, K = loop_conjunctive(*pair)
    out[:, FULL] += K
    got, _ = dempster_combine(*pair, use_yager=True)
    assert torch.allclose(got, out, atol=1e-6)


def test_pcr5_matches_loop(pair):
    out, K = loop_pcr5(*pair)
    got, got_K = pcr5_combine(*pair)
    assert torch.allclose(got, out, atol=1e-6) and torch.allclose(got_K, K, atol=1e-6)


def test_ordinal_routes_adjacent_conflict(pair):
    out, K = loop_conjunctive(*pair)
    m1, m2 = pair
    cn_mci = m1[:, 1] * m2[:, 2] + m1[:, 2] * m2[:, 1]
    mci_ad = m1[:, 2] * m2[:, 4] + m1[:, 4] * m2[:, 2]
    out[:, 3] += cn_mci
    out[:, 6] += mci_ad
    out[:, FULL] += K - cn_mci - mci_ad
    got, _ = ordinal_yager_combine(*pair)
    assert torch.allclose(got, out, atol=1e-6)


def test_ordinal_never_populates_cn_ad_from_ordinal_inputs():
    a, b = random_bba(16, 4), random_bba(16, 5)
    for m in (a, b):
        m[:, 5] = 0.0
        m /= m.sum(dim=1, keepdim=True)
    got, _ = ordinal_yager_combine(a, b)
    assert torch.allclose(got[:, 5], torch.zeros(16), atol=1e-7)


@pytest.mark.parametrize('rule', [dempster_combine, pcr5_combine, ordinal_yager_combine,
                                  lambda a, b: dempster_combine(a, b, use_yager=True)])
def test_rules_return_valid_bba(rule, pair):
    m, _ = rule(*pair)
    assert torch.allclose(m.sum(dim=1), torch.ones(16), atol=1e-5)
    assert (m >= -1e-7).all() and (m[:, 0] == 0).all()


def test_conflict_breakdown_adds_up(pair):
    parts = conflict_breakdown(*pair)
    total = parts['cn_mci'] + parts['mci_ad'] + parts['cn_ad'] + parts['other']
    assert torch.allclose(total, parts['total'], atol=1e-6)


def test_vacuous_is_identity(pair):
    m = pair[0]
    got, K = dempster_combine(m, vacuous_bba(16, m.device))
    assert torch.allclose(got, m, atol=1e-6) and torch.allclose(K, torch.zeros(16))


def test_discount_limits(pair):
    m = pair[0]
    assert torch.allclose(discount_bba(m, 1.0), m, atol=1e-6)
    assert torch.allclose(discount_bba(m, 0.0), vacuous_bba(16, m.device), atol=1e-6)
    half = discount_bba(m, 0.5)
    assert torch.allclose(half.sum(dim=1), torch.ones(16), atol=1e-6)
    assert torch.allclose(half[:, 1], 0.5 * m[:, 1])


def test_belief_plausibility_pignistic(pair):
    m = pair[0]
    bel, pl = compute_belief_plausibility(m)
    betp = pignistic_probability(m)
    assert torch.allclose(betp.sum(dim=1), torch.ones(16), atol=1e-6)
    assert (bel <= betp + 1e-6).all() and (betp <= pl + 1e-6).all()
    expected_cn = m[:, 1] + m[:, 3] / 2 + m[:, 5] / 2 + m[:, 7] / 3
    assert torch.allclose(betp[:, 0], expected_cn, atol=1e-6)
    expected_pl_ad = m[:, 4] + m[:, 5] + m[:, 6] + m[:, 7]
    assert torch.allclose(pl[:, 2], expected_pl_ad, atol=1e-6)


def test_evidence_layer_output():
    layer = DSEvidenceLayer(feat_dim=8).eval()
    m = layer(torch.randn(5, 8))
    assert torch.allclose(m.sum(dim=1), torch.ones(5), atol=1e-6)
    assert (m[:, [0, 3, 5, 6]] == 0).all() and (m[:, FULL] > 0).all()


def test_gradients_flow_through_pcr5(pair):
    a = pair[0].clone().requires_grad_(True)
    pcr5_combine(a, pair[1])[0].pow(2).sum().backward()
    assert torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0


def test_accumulator_mask_leaves_rows_unchanged(pair):
    acc = TemporalAccumulator(16, torch.device('cpu'))
    mask = torch.arange(16) % 2 == 0
    acc.accumulate(pair[0], mask=mask)
    assert torch.allclose(acc.m_acc[~mask], vacuous_bba(8, torch.device('cpu')))
    assert torch.allclose(acc.m_acc[mask], pair[0][mask], atol=1e-6)


def test_losses_match_loop_reference():
    m = random_bba(12, 3)
    labels = torch.arange(12) % 3
    singles = m[:, [1, 2, 4]]
    ref = torch.stack([(1 - singles[i, y]) ** 2 + sum(singles[i, c] ** 2 for c in range(3) if c != y)
                       for i, y in enumerate(labels.tolist())]).mean()
    assert torch.allclose(evidential_utility_loss(singles, labels), ref, atol=1e-6)
    targets = {0: 0.10, 1: 0.15, 2: 0.10}
    ref_ign = torch.stack([(m[i, FULL] - targets[y]) ** 2 for i, y in enumerate(labels.tolist())]).mean()
    assert torch.allclose(ignorance_anchoring_loss(m[:, FULL], labels), ref_ign, atol=1e-6)
