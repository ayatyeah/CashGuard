import numpy as np

from app.ml.zone_prototypes import residual_features


def test_residual_requires_three_visible_observations():
    a = np.zeros((32, 32), np.uint8)
    mask = np.ones_like(a, bool)
    assert residual_features(a, [a, a], [mask, mask], 30) == {}
    assert residual_features(a, [a] * 3, [~mask] * 3, 30) == {}


def test_census_ignores_monotone_brightness_and_requires_persistence():
    rng = np.random.default_rng(4)
    a = rng.integers(20, 100, (40, 40), dtype=np.uint8)
    mask = np.ones_like(a, bool)
    unchanged = residual_features(a, [a + 30] * 3, [mask] * 3, 40)
    assert unchanged['census_change'] == 0
    b = a.copy()
    b[10:20, 10:20] = 200
    transient = residual_features(a, [b, a, a], [mask] * 3, 40)
    persistent = residual_features(a, [b, b, b], [mask] * 3, 40)
    assert transient['persistent_change'] == 0
    assert persistent['persistent_change'] > 0


def test_cv_acceptance_never_trades_false_positives_for_recall():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).parents[1] / "scripts/train_scorer.py"
    spec = importlib.util.spec_from_file_location("zone_train_scorer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not module.accepts_cv((-9, -1), (-11, 0))
    assert not module.accepts_cv((-12, 0), (-11, 0))
    assert module.accepts_cv((-10, 0), (-11, 0))
    assert module.accepts_cv((-11, 0), (-11, 0))


def test_masked_pixels_cannot_change_visible_features():
    rng = np.random.default_rng(11)
    reference = rng.integers(20, 150, (50, 50), dtype=np.uint8)
    mask = np.ones_like(reference, bool)
    mask[10:35, 15:30] = False
    changed = reference.copy()
    changed[~mask] = 255
    result = residual_features(reference, [changed] * 3, [mask] * 3, 50)
    assert result['census_change'] == 0
    assert result['persistent_change'] == 0
    assert result['small_residual'] == 0
    assert result['zncc_change'] < 1e-6


def test_zncc_missing_when_its_larger_support_is_not_visible():
    reference = np.zeros((13, 13), np.uint8)
    mask = np.ones_like(reference, bool)
    result = residual_features(reference, [reference] * 3, [mask] * 3, 13)
    assert 'census_change' in result  # 9 x 9 visible centers
    assert 'zncc_change' not in result  # 7 x 7 < 50
