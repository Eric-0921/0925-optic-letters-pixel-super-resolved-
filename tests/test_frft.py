import numpy as np
from hssa.frft import frft


def test_order_one_is_centered_dft():
    rng = np.random.default_rng(0)
    x = rng.normal(size=64) + 1j * rng.normal(size=64)
    X = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x))) / 8.0
    assert np.allclose(frft(x, 1.0), X)


def test_additivity_and_unitarity():
    N = 128
    t = (np.arange(N) - N / 2) / np.sqrt(N)
    x = np.exp(-np.pi * (t - 0.8) ** 2) * np.exp(2j * np.pi * 0.5 * t)
    y1 = frft(frft(x, 0.37), 0.41)
    y2 = frft(x, 0.78)
    assert np.linalg.norm(y1 - y2) / np.linalg.norm(y2) < 2e-2
    assert abs(np.linalg.norm(frft(x, 0.6)) / np.linalg.norm(x) - 1) < 1e-2


def test_gaussian_is_eigenfunction():
    N = 128
    t = (np.arange(N) - N / 2) / np.sqrt(N)
    g = np.exp(-np.pi * t ** 2)
    for a in (0.3, 0.7, 1.3):
        y = frft(g, a)
        assert np.linalg.norm(np.abs(y) - g) / np.linalg.norm(g) < 1e-2


def test_batched_matches_single():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3, 50)) + 0j
    Y = frft(X, 0.55)
    for i in range(3):
        assert np.allclose(Y[i], frft(X[i], 0.55))
