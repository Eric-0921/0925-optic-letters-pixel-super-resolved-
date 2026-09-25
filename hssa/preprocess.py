"""Pre-processing of Fig. 1(b): autofocus (ADFrFT) and sub-pixel registration.

Autofocus follows Li et al., Opt. Lasers Eng. 175, 107991 (2024), Eq. (2):
    ADFrFT(z, dp) = mean(|F_dp[I_z]|) + (dp - 1) * mean(I_z),  I_z = |U_z|^2
where U_z is the hologram back-propagated by z, the operand is |U_z|^2, and F_dp is the 2-D FrFT of
order dp. The paper's coarse-to-fine strategy is used: a smooth curve
(dp = 0.2) over a wide range, then a sharp curve (dp = 0.8) near the peak.

Registration follows the main text: holograms are pre-propagated to the
sample plane and registered by Fourier cross-correlation
(Guizar-Sicairos et al., Opt. Lett. 33, 156, 2008).
"""
import numpy as np
from skimage.registration import phase_cross_correlation

from . import optics
from .frft import frft2


def adfrft(U, dp):
    """ADFrFT clarity of the back-propagated intensity I = |U|^2 (Eq. 2 of the
    autofocus paper, which writes the operand as the intensity I_n)."""
    I = np.abs(U) ** 2
    return float(np.mean(np.abs(frft2(I, dp))) + (dp - 1) * np.mean(I))


def tamura_of_gradient(U):
    a = np.abs(U)
    gy, gx = np.gradient(a)
    g = np.hypot(gx, gy)
    return float(np.sqrt(g.std() / g.mean()))


def focus_curve(I, dx, wavelength, zs, metric="adfrft", dp=0.8, crop=256):
    h, w = I.shape
    c = min(crop, h, w)
    y0, x0 = (h - c) // 2, (w - c) // 2
    a = np.sqrt(np.clip(I[y0:y0 + c, x0:x0 + c], 0, None)).astype(np.complex128)
    A = optics.fft2(a)
    vals = []
    for z in zs:
        H = optics.asm_kernel(a.shape, dx, wavelength, -z, dtype=np.complex128)
        U = optics.ifft2(A * H)
        vals.append(adfrft(U, dp) if metric == "adfrft" else tamura_of_gradient(U))
    return np.array(vals)


def autofocus(I, dx, wavelength, zmin=600.0, zmax=1800.0, coarse=20.0, fine=2.0,
              metric="adfrft", crop=512, coarse_crop=256):
    """Coarse-to-fine search for the sample-to-sensor distance (um).

    Coarse: smooth ADFrFT curve (dp = 0.2) on a central crop over the full
    range. Fine: sharp curve (dp = 0.8) on the full window within +-2 coarse
    steps, followed by parabolic peak refinement.
    """
    zs = np.arange(zmin, zmax + coarse / 2, coarse)
    v = focus_curve(I, dx, wavelength, zs, metric, dp=0.2, crop=coarse_crop)
    z0 = zs[int(np.argmax(v))]
    zs2 = np.arange(z0 - 2 * coarse, z0 + 2 * coarse + fine / 2, fine)
    v2 = focus_curve(I, dx, wavelength, zs2, metric, dp=0.8, crop=crop)
    i = int(np.argmax(v2))
    if 0 < i < len(v2) - 1:  # parabolic refinement
        d = v2[i - 1] - 2 * v2[i] + v2[i + 1]
        off = 0.5 * (v2[i - 1] - v2[i + 1]) / d if d != 0 else 0.0
        return float(zs2[i] + off * fine)
    return float(zs2[i])


def backpropagated_amplitude(I, dx, wavelength, z):
    a = np.sqrt(np.clip(I, 0, None)).astype(np.complex128)
    H = optics.asm_kernel(a.shape, dx, wavelength, -z, dtype=np.complex128)
    return np.abs(optics.propagate(a, H))


def register(ref_amp, mov_amp, upsample=100):
    """Displacement s such that mov(r) ~ ref(r - s), in pixels (sy, sx)."""
    win = optics.tukey2d(ref_amp.shape, 0.3)
    r = (ref_amp - ref_amp.mean()) * win
    m = (mov_amp - mov_amp.mean()) * win
    shift, _, _ = phase_cross_correlation(r, m, upsample_factor=upsample, normalization=None)
    return (-float(shift[0]), -float(shift[1]))


def preprocess(frames, dx, wavelength, ref=0, z_known=None, **af_kwargs):
    """Estimate z_n by autofocus and hologram displacements relative to frame `ref`.

    Returns lists of z (um) and shifts (LR pixels).
    """
    zs = []
    for i, f in enumerate(frames):
        zs.append(z_known[i] if z_known is not None else autofocus(f.I, dx, wavelength, **af_kwargs))
    amps = [backpropagated_amplitude(f.I, dx, wavelength, z) for f, z in zip(frames, zs)]
    shifts = [register(amps[ref], a) for a in amps]
    return zs, shifts
