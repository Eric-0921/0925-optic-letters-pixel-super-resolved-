"""Discrete fractional Fourier transform (Ozaktas et al., IEEE TSP 44, 2141, 1996).

Port of the widely used fast O(N log N) algorithm, vectorised along the last
axis. Order a = 1 is the centred unitary DFT, a = 0 is the identity.
"""
import numpy as np
import scipy.fft as sfft


def _fconv(x, y):
    """Linear convolution along the last axis via FFT. y is 1-D."""
    n = x.shape[-1] + y.shape[-1] - 1
    P = sfft.next_fast_len(n)
    z = sfft.ifft(sfft.fft(x, P, axis=-1) * sfft.fft(y, P), axis=-1, workers=-1)
    return z[..., :n]


def _sinc_interp(x):
    """Band-limited 2x interpolation along the last axis (length 2N-1)."""
    N = x.shape[-1]
    y = np.zeros(x.shape[:-1] + (2 * N - 1,), dtype=complex)
    y[..., ::2] = x
    k = np.sinc(np.arange(-(2 * N - 3), 2 * N - 2) / 2.0)
    xint = _fconv(y, k)
    return xint[..., 2 * N - 3: 2 * N - 3 + 2 * N - 1]


def _centered_dft(f, inverse=False):
    N = f.shape[-1]
    shft = (np.arange(N) + N // 2) % N
    out = np.empty_like(f, dtype=complex)
    if inverse:
        out[..., shft] = sfft.ifft(f[..., shft], axis=-1, workers=-1) * np.sqrt(N)
    else:
        out[..., shft] = sfft.fft(f[..., shft], axis=-1, workers=-1) / np.sqrt(N)
    return out


def frft(f, a):
    """Fractional Fourier transform of order a along the last axis."""
    f = np.asarray(f, dtype=complex)
    N = f.shape[-1]
    a = float(np.mod(a, 4))
    if a == 0:
        return f.copy()
    if a == 2:
        return f[..., ::-1].copy()
    if a == 1:
        return _centered_dft(f)
    if a == 3:
        return _centered_dft(f, inverse=True)
    if a > 2.0:
        a -= 2
        f = f[..., ::-1]
    if a > 1.5:
        a -= 1
        f = _centered_dft(f)
    if a < 0.5:
        a += 1
        f = _centered_dft(f, inverse=True)

    # General case 0.5 <= a <= 1.5, chirp-convolution-chirp decomposition.
    # Input sample n sits at t = (n - N/2)/sqrt(N) (same centre as the
    # centred DFT above). After 2x sinc interpolation the samples sit at
    # t = k / (2 sqrt(N)) with k = -N .. N-2.
    alpha = a * np.pi / 2
    tana2 = np.tan(alpha / 2)
    sina = np.sin(alpha)
    g = _sinc_interp(f)                                   # k = -N .. N-2
    k = np.arange(-N, N - 1).astype(float)
    chrp = np.exp(-1j * np.pi / N * tana2 / 4 * k ** 2)
    g = chrp * g
    c = np.pi / N / sina / 4
    lags = np.arange(-(2 * N - 2), 2 * N - 1).astype(float)
    conv = _fconv(g, np.exp(1j * c * lags ** 2))         # m = j - 3N + 2
    Faf = conv[..., 2 * N - 2: 4 * N - 3] * np.sqrt(c / np.pi)  # m = -N .. N-2
    Faf = chrp * Faf
    return np.exp(-1j * (1 - a) * np.pi / 4) * Faf[..., ::2]


def frft2(u, a):
    """Separable 2-D FrFT of order a on both axes."""
    out = frft(u, a)
    out = frft(np.swapaxes(out, -1, -2), a)
    return np.swapaxes(out, -1, -2)
