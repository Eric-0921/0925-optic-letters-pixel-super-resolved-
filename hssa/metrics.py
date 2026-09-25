"""Evaluation against simulation ground truth."""
import numpy as np
from skimage.registration import phase_cross_correlation

from . import optics
from .usaf import resolution_report


def gt_on_hr(obj_fine, fine, k):
    """Complex ground truth on the HR grid (area average of the fine grid)."""
    b = fine // k
    return optics.bin_mean(obj_fine.real, b) + 1j * optics.bin_mean(obj_fine.imag, b)


def align(rec, gt, crop=None, upsample=20, mode="amplitude"):
    """Sub-pixel translate `rec` onto `gt` by cross-correlation.

    mode="amplitude" correlates moduli; mode="phase" is for phase objects.
    """
    a = np.abs(rec)
    g = np.abs(gt)
    if mode == "phase":
        # pure phase object: register high-passed unit-modulus fields. This is
        # immune to phase wrapping and to the poorly recovered low-frequency
        # phase, which otherwise dominates the correlation.
        a = _highpass(rec)
        g = _highpass(gt)
    if crop is not None:
        a, g = a[crop], g[crop]
    shift, _, _ = phase_cross_correlation(g - g.mean(), a - a.mean(), upsample_factor=upsample,
                                          normalization=None)
    return optics.fourier_shift(rec, tuple(shift)), tuple(shift)


def amplitude_scale(rec, gt, roi):
    """Scale that maps the reconstructed clear-glass amplitude to 1."""
    bg = (np.abs(gt) > 0.99) & roi
    return float(np.median(np.abs(rec)[bg])) if bg.any() else float(np.median(np.abs(rec)[roi]))


def _bbox_slice(bbox, origin, dx, shape):
    x0, y0, x1, y1 = bbox
    r0, r1 = int((y0 - origin[0]) / dx), int(np.ceil((y1 - origin[0]) / dx))
    c0, c1 = int((x0 - origin[1]) / dx), int(np.ceil((x1 - origin[1]) / dx))
    return (slice(max(r0, 0), min(r1, shape[0])), slice(max(c0, 0), min(c1, shape[1])))


def usaf_scores(amp, gt_win, dx, origin, layout, groups=(7, 8, 9), thresh=0.1):
    """MSE and resolution on the central groups of an aligned, normalised amplitude."""
    from .usaf import groups_bbox
    sl = _bbox_slice(groups_bbox(layout, groups), origin, dx, amp.shape)
    mse = float(np.mean((amp[sl] - np.abs(gt_win)[sl]) ** 2))
    table, finest = resolution_report(amp, dx, origin, layout, thresh=thresh, groups=groups)
    return {"mse": mse, "finest": finest,
            "contrast": {f"{g}-{e}": round(v, 4) for (g, e), v in table.items()}}


def evaluate_usaf(rec_win, gt_win, dx, origin, layout, groups=(7, 8, 9), thresh=0.1):
    """Align on the whole pattern, normalise the clear-glass amplitude to 1, then
    score MSE and resolution on the central groups (the paper's central region)."""
    x0, y0, x1, y1 = layout.bbox
    sl = _bbox_slice((x0 - 10, y0 - 10, x1 + 10, y1 + 10), origin, dx, gt_win.shape)
    rec_al, shift = align(rec_win, gt_win, crop=sl)
    roi = np.zeros(gt_win.shape, bool)
    roi[sl] = True
    amp = np.abs(rec_al) / amplitude_scale(rec_al, gt_win, roi)
    out = usaf_scores(amp, gt_win, dx, origin, layout, groups, thresh)
    out.update({"align_shift": shift, "amp": amp})
    return out


def _highpass(u, sigma_px=5.0 / 0.67):
    """Remove phase variations coarser than ~10 um (Gaussian, sigma = 5 um)."""
    from scipy.ndimage import gaussian_filter
    z = u / (np.abs(u) + 1e-12)
    low = gaussian_filter(z.real, sigma_px) + 1j * gaussian_filter(z.imag, sigma_px)
    return z * np.conj(low) / (np.abs(low) + 1e-12)


def evaluate_phase(rec_win, gt_win, roi_slice):
    """Wrapped phase error after removing the global phase offset.

    Returns RMSE (rad), NMSE (error normalised by the ground-truth phase
    range) and the offset-corrected reconstructed phase.
    """
    rec_al, shift = align(rec_win, gt_win, crop=roi_slice, mode="phase")
    r = rec_al[roi_slice]
    g = gt_win[roi_slice]
    c = np.sum(r * np.conj(g))
    rot = np.conj(c) / np.abs(c)
    err = np.angle(r * rot * np.conj(g))
    gph = np.angle(g)
    rng = float(gph.max() - gph.min()) + 1e-12
    hp_err = np.angle(_highpass(r * rot) * np.conj(_highpass(g)))
    return {"phase_rmse": float(np.sqrt(np.mean(err ** 2))),
            "phase_nmse": float(np.mean(err ** 2) / rng ** 2),
            "phase_rmse_detail": float(np.sqrt(np.mean(hp_err ** 2))),
            "phase": np.angle(rec_al * rot), "align_shift": shift}
