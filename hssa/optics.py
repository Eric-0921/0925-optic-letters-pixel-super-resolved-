"""Scalar diffraction utilities: angular spectrum propagation, sub-pixel shifts, binning."""
import numpy as np
import scipy.fft as sfft

WORKERS = -1  # use all cores for scipy.fft


def fft2(a):
    return sfft.fft2(a, workers=WORKERS)


def ifft2(a):
    return sfft.ifft2(a, workers=WORKERS)


def freq_grid(shape, dx):
    """Spatial-frequency grids (cycles per unit length) in FFT order."""
    fy = sfft.fftfreq(shape[0], d=dx)
    fx = sfft.fftfreq(shape[1], d=dx)
    return fy[:, None], fx[None, :]


def asm_kernel(shape, dx, wavelength, z, f0=(0.0, 0.0), band_limit=True, dtype=np.complex64):
    """Angular-spectrum transfer function H(f + f0) for propagation distance z.

    f0 = (f0y, f0x) evaluates the kernel on a shifted spectrum. Propagating
    phi with H(f + f0) gives the field produced by phi under a tilted plane wave
    exp(i 2 pi f0 . r), with that plane-wave carrier removed. Because the carrier
    has unit modulus, the intensity is exact for any continuous f0.

    band_limit applies the Matsushima-Shimobaba band limit that prevents
    aliasing of the chirped kernel on a finite canvas.
    """
    fy, fx = freq_grid(shape, dx)
    fy = fy + f0[0]
    fx = fx + f0[1]
    arg = 1.0 / wavelength**2 - fx**2 - fy**2
    prop = arg > 0
    kz = 2 * np.pi * np.sqrt(np.where(prop, arg, 0.0))
    # remove the constant phase of the carrier so that H(f0) = 1 (piston only)
    arg0 = 1.0 / wavelength**2 - f0[0] ** 2 - f0[1] ** 2
    kz0 = 2 * np.pi * np.sqrt(arg0)
    H = np.exp(1j * (kz - kz0) * z) * prop
    if band_limit and z != 0:
        Ly, Lx = shape[0] * dx, shape[1] * dx
        uy = 1.0 / (wavelength * np.sqrt((2 * abs(z) / Ly) ** 2 + 1))
        ux = 1.0 / (wavelength * np.sqrt((2 * abs(z) / Lx) ** 2 + 1))
        H = H * ((np.abs(fy) < uy) & (np.abs(fx) < ux))
    return H.astype(dtype)


def shift_kernel(shape, dx_units, shift):
    """Fourier phase ramp that translates content by `shift` = (sy, sx) pixels."""
    fy, fx = freq_grid(shape, 1.0)
    return np.exp(-2j * np.pi * (fy * shift[0] + fx * shift[1])).astype(np.complex64)


def propagate(u, kernel):
    return ifft2(fft2(u) * kernel)


def fourier_shift(u, shift):
    """Sub-pixel circular translation of a 2-D array by (sy, sx) pixels."""
    out = ifft2(fft2(u) * shift_kernel(u.shape, 1.0, shift))
    return out if np.iscomplexobj(u) else out.real


def bin_sum(a, k):
    """Sum over non-overlapping k x k blocks (pixel integration)."""
    h, w = a.shape
    return a.reshape(h // k, k, w // k, k).sum(axis=(1, 3))


def bin_mean(a, k):
    return bin_sum(a, k) / (k * k)


def upsample_nearest(a, k):
    return np.repeat(np.repeat(a, k, axis=0), k, axis=1)


def upsample_fourier(a, k):
    """Band-limited (sinc) up-sampling by an integer factor k."""
    h, w = a.shape
    A = sfft.fftshift(fft2(a))
    P = np.zeros((h * k, w * k), dtype=np.complex128)
    y0, x0 = (h * k - h) // 2, (w * k - w) // 2
    P[y0:y0 + h, x0:x0 + w] = A
    out = ifft2(sfft.ifftshift(P)) * (k * k)
    return out if np.iscomplexobj(a) else out.real


def tukey2d(shape, frac):
    """Separable Tukey window used to taper fields at the canvas edge."""
    from scipy.signal.windows import tukey
    return np.outer(tukey(shape[0], frac), tukey(shape[1], frac))
