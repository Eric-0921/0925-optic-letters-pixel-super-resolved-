"""Pixel-super-resolved phase retrieval: HSSA (Algorithm 1) and baselines.

Forward model for frame n (Eq. 1): y_n = D A_n T_n (p . x), where
  T_n  sub-pixel translation by the registered hologram displacement,
  A_n  angular-spectrum propagation over the autofocused distance z_n,
  D    pixel integration from the high-resolution (HR) grid to the sensor.

Implementation notes (choices the paper leaves open):
 * The HR grid is the sensor grid up-sampled by k (default 2, i.e. 0.67 um).
 * The object lives on an HR canvas larger than the sensor so that the
   integer part of each displacement is a window offset and propagation does
   not wrap; only the fractional part is applied as a Fourier phase ramp.
 * Outside a frame's sensor window the propagated field is left unchanged.
 * `update="binned"` enforces the measured intensity after pixel integration
   (the physically exact D). `update="literal"` replaces the modulus with the
   up-sampled measurement on the HR grid, which is the other reading of
   Fig. 1(b) ("low-resolution holograms are up-sampled").

Methods
 * hssa : Algorithm 1 with the rPIE-style object / illumination split.
 * mfap : the same parallel projection with p fixed to 1 (x = phi').
 * lisa : mfap with the known illumination angle in the propagation kernel
          (tilted angular spectrum) instead of a registered translation.
"""
from dataclasses import dataclass
import time
import numpy as np

from . import optics


@dataclass
class ReconFrame:
    I: np.ndarray            # LR intensity (normalised)
    z: float                 # propagation distance (um)
    shift: tuple = (0.0, 0.0)  # hologram displacement vs. reference (LR px)
    f0: tuple = (0.0, 0.0)     # known tilt (cycles/um), used by LISA only


def reconstruct(frames, pixel=1.34, wavelength=0.532, k=2, canvas=2048,
                method="hssa", iters=50, alpha=0.2, beta=0.7, gamma=0.05,
                update="binned", init="sqrtI", verbose=False, callback=None):
    n_lr = frames[0].I.shape[0]
    W = n_lr * k
    dx = pixel / k
    c0 = (canvas - W) // 2
    # per-frame kernels and windows
    ks, wins, targets = [], [], []
    for f in frames:
        s = np.asarray(f.shift, float) * k             # HR pixels
        m = np.floor(s).astype(int)
        eps = s - m
        H = optics.asm_kernel((canvas, canvas), dx, wavelength, f.z,
                              f0=f.f0 if method == "lisa" else (0.0, 0.0))
        if method != "lisa":
            H = H * optics.shift_kernel((canvas, canvas), 1.0, eps)
        ks.append(H)
        oy, ox = c0 - m[0], c0 - m[1]
        if method == "lisa":
            oy, ox = c0, c0
        assert 0 <= oy and oy + W <= canvas and 0 <= ox and ox + W <= canvas, "canvas too small"
        wins.append((slice(oy, oy + W), slice(ox, ox + W)))
        I = np.clip(f.I.astype(np.float32), 0, None)
        if update == "literal":
            targets.append(np.sqrt(np.clip(optics.upsample_fourier(I, k), 0, None)).astype(np.float32))
        else:
            targets.append(I)

    # initialisation: phi0 = x0 = sqrt(I_1) (up-sampled, placed in its window), p0 = 1
    x = np.ones((canvas, canvas), np.complex64)
    if init == "sqrtI":
        x[wins[0]] = np.sqrt(np.clip(optics.upsample_fourier(frames[0].I, k), 0, None))
    p = np.ones_like(x)
    phi = p * x
    hist = []
    t0 = time.time()
    for t in range(iters):
        Phi = optics.fft2(phi)
        acc = np.zeros_like(Phi)
        err = 0.0
        for H, win, T in zip(ks, wins, targets):
            y = optics.ifft2(Phi * H)
            yw = y[win]
            if update == "literal":
                ynew = T * np.exp(1j * np.angle(yw))
                err += float(np.mean((np.abs(yw) - T) ** 2))
            else:
                est = optics.bin_mean(np.abs(yw) ** 2, k)
                ratio = np.sqrt(T / (est + 1e-6))
                ynew = yw * optics.upsample_nearest(ratio, k)
                err += float(np.mean((np.sqrt(est) - np.sqrt(T)) ** 2))
            d = np.zeros_like(y)
            d[win] = ynew - yw
            acc += optics.fft2(d) * np.conj(H)
        phi_new = phi + optics.ifft2(acc / len(frames))
        hist.append(err / len(frames))
        if method == "hssa":
            dphi = phi_new - phi
            ap = np.abs(p) ** 2
            ax = np.abs(x) ** 2
            x_next = x + np.conj(p) * dphi / ((1 - alpha) * ap + alpha * ap.max())
            p_next = p + gamma * np.conj(x) * dphi / ((1 - beta) * ax + beta * ax.max())
            x, p = x_next.astype(np.complex64), p_next.astype(np.complex64)
            phi = p * x
        else:
            x = phi_new.astype(np.complex64)
            phi = x
        if callback is not None:
            callback(t, x, p)
        if verbose and (t % 10 == 0 or t == iters - 1):
            print(f"  [{method}] iter {t:3d}  amp-err {hist[-1]:.5f}  {time.time() - t0:.1f}s")
    ref = wins[0]
    return {"x": x, "p": p, "phi": phi, "ref_window": ref, "err": np.array(hist), "dx": dx}
