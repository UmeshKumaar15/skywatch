import numpy as np

from skywatch.classification.nn import MLP


def test_mlp_classifier_learns_nonlinear_boundary():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1200, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
    net = MLP([2, 32, 32, 2], seed=0)
    net.fit(X, y, epochs=60, lr=5e-3)
    acc = (net.predict_proba(X).argmax(1) == y).mean()
    assert acc > 0.93
    restored = MLP.from_state(net.state_dict("p_"), "p_")
    assert np.allclose(restored.predict_proba(X[:5]), net.predict_proba(X[:5]))


def test_mlp_masked_regression():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(800, 3))
    Y = np.c_[X[:, 0] * 2, np.sin(X[:, 1])]
    mask = np.ones_like(Y)
    mask[::2, 1] = 0
    net = MLP([3, 32, 2], seed=1, task="regression")
    net.fit(X, Y, mask=mask, epochs=80, lr=5e-3)
    err = np.abs(net.predict(X) - Y).mean(0)
    assert err[0] < 0.25 and err[1] < 0.25
