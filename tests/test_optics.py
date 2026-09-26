import numpy as np
from hssa import optics
from hssa.preprocess import register


def _blob(n=128, c=(60, 70), r=6):
    y, x = np.mgrid[:n, :n]
    return np.exp(-((y - c[0]) ** 2 + (x - c[1]) ** 2) / (2 * r ** 2))


def test_propagation_is_reversible():
    u = (1 + 0.3 * _blob()).astype(np.complex128)
    H = optics.asm_kernel(u.shape, 0.5, 0.532, 300.0, band_limit=False, dtype=np.complex128)
    back = optics.propagate(optics.propagate(u, H), np.conj(H))
    assert np.allclose(back, u, atol=1e-8)


def test_fourier_shift_sign():
    a = _blob()
    b = optics.fourier_shift(a, (3.0, -5.0))
    iy, ix = np.unravel_index(np.argmax(b), b.shape)
    assert (iy, ix) == (63, 65)


def test_tilted_kernel_matches_explicit_tilt():
    """|propagate(u * exp(i2pi f0 r), H(f))| == |propagate(u, H(f + f0))| for grid-periodic f0."""
    n, dx, lam, z = 256, 0.5, 0.532, 200.0
    u = (1 - 0.8 * _blob(n, (128, 128), 8)).astype(np.complex128)
    f0 = (0.0, 20 / (n * dx))  # exactly periodic on the canvas
    x = np.arange(n) * dx
    tilt = np.exp(2j * np.pi * f0[1] * x)[None, :]
    a = optics.propagate(u * tilt, optics.asm_kernel((n, n), dx, lam, z, band_limit=False, dtype=np.complex128))
    b = optics.propagate(u, optics.asm_kernel((n, n), dx, lam, z, f0=f0, band_limit=False, dtype=np.complex128))
    assert np.allclose(np.abs(a), np.abs(b), atol=1e-8)


def test_register_returns_displacement():
    a = _blob(128, (64, 64), 5) + 0.5 * _blob(128, (40, 90), 3)
    b = optics.fourier_shift(a, (2.25, -1.5))
    s = register(a, b)
    assert np.allclose(s, (2.25, -1.5), atol=0.05)


def test_binning_and_upsampling():
    a = np.arange(16.0).reshape(4, 4)
    assert np.allclose(optics.bin_mean(a, 2), [[2.5, 4.5], [10.5, 12.5]])
    assert optics.upsample_nearest(a, 2).shape == (8, 8)
    assert np.allclose(optics.bin_mean(optics.upsample_fourier(a, 2), 2).mean(), a.mean())
