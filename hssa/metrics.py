"""Evaluation against simulation ground truth."""
import numpy as np
from skimage.registration import phase_cross_correlation

from . import optics
from .usaf import resolution_report


def gt_on_hr(obj_fine, fine, k):
    """Complex ground truth on the HR grid (area average of the fine grid)."""
    b = fine // k
    return optics.bin_mean(obj_fine.real, b) + 1j * optics.bin_mean(obj_fine.imag, b)


def align(rec, gt, crop=None, upsample=20):
    """Sub-pixel translate `rec` onto `gt` using amplitude (or phase) correlation."""
    a = np.abs(rec)
    g = np.abs(gt)
    if g.std() < 1e-3:  # pure phase object: register on phase
        a = np.angle(rec * np.conj(np.mean(rec)))
        g = np.angle(gt)
    if crop is not None:
        a, g = a[crop], g[crop]
    shift, _, _ = phase_cross_correlation(g - g.mean(), a - a.mean(), upsample_factor=upsample,
                                          normalization=None)
    return optics.fourier_shift(rec, tuple(shift)), tuple(shift)


def amplitude_scale(rec, gt, roi):
    """Scale that maps the reconstructed clear-glass amplitude to 1."""
    bg = (np.abs(gt) > 0.99) & roi
    return float(np.median(np.abs(rec)[bg])) if bg.any() else float(np.median(np.abs(rec)[roi]))


def evaluate_usaf(rec_win, gt_win, dx, origin, layout, roi_margin=10.0, thresh=0.1):
    """Amplitude MSE on the pattern ROI and USAF resolution.

    rec_win, gt_win: HR windows (same grid); origin: layout coordinate (um) of
    the top-left pixel edge of the window.
    """
    x0, y0, x1, y1 = layout.bbox
    r0 = int((y0 - roi_margin - origin[0]) / dx); r1 = int((y1 + roi_margin - origin[0]) / dx)
    c0 = int((x0 - roi_margin - origin[1]) / dx); c1 = int((x1 + roi_margin - origin[1]) / dx)
    sl = (slice(max(r0, 0), r1), slice(max(c0, 0), c1))
    rec_al, shift = align(rec_win, gt_win, crop=sl)
    roi = np.zeros(gt_win.shape, bool)
    roi[sl] = True
    s = amplitude_scale(rec_al, gt_win, roi)
    amp = np.abs(rec_al) / s
    mse = float(np.mean((amp[sl] - np.abs(gt_win)[sl]) ** 2))
    table, finest = resolution_report(amp, dx, origin, layout, thresh=thresh)
    return {"mse": mse, "finest": finest, "contrast": {f"{g}-{e}": round(v, 4) for (g, e), v in table.items()},
            "align_shift": shift, "amp": amp, "slice": sl}


def evaluate_phase(rec_win, gt_win, roi_slice):
    """Wrapped phase error after removing the global phase offset.

    Returns RMSE (rad), NMSE (error normalised by the ground-truth phase
    range) and the offset-corrected reconstructed phase.
    """
    rec_al, shift = align(rec_win, gt_win, crop=roi_slice)
    r = rec_al[roi_slice]
    g = gt_win[roi_slice]
    c = np.sum(r * np.conj(g))
    rot = np.conj(c) / np.abs(c)
    err = np.angle(r * rot * np.conj(g))
    gph = np.angle(g)
    rng = float(gph.max() - gph.min()) + 1e-12
    return {"phase_rmse": float(np.sqrt(np.mean(err ** 2))),
            "phase_nmse": float(np.mean(err ** 2) / rng ** 2),
            "phase": np.angle(rec_al * rot), "align_shift": shift}
