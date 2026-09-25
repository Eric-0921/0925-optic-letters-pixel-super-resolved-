"""Shared set-up: acquisition geometry, simulation, preprocessing, evaluation."""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hssa.simulate import SimConfig, Simulator, FrameSpec, usaf_object  # noqa: E402
from hssa.usaf import make_layout  # noqa: E402
from hssa.preprocess import preprocess  # noqa: E402
from hssa.recon import ReconFrame, reconstruct  # noqa: E402
from hssa import metrics  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
K = 2               # HR up-sampling factor
CANVAS_HR = 2048    # HR canvas (same physical size as the 4096 fine simulation canvas)

# Axial positions. The paper gives only d_min = 0.975 mm and d_n < 2 mm.
HEIGHTS_HSSA = [975.0, 1175.0, 1375.0]
HEIGHTS_MFAP = [975.0 + 50.0 * i for i in range(9)]


def pinholes(seed=1, n=9, rmax=100e3, rmin=40e3):
    """Pinhole offsets (um): the first is on axis, the others are random in the
    annulus rmin <= |D| <= rmax (illumination angle up to ~8 deg at L = 70 cm)."""
    rng = np.random.default_rng(seed)
    out = [(0.0, 0.0)]
    while len(out) < n:
        d = rng.uniform(-rmax, rmax, 2)
        if rmin <= np.hypot(*d) <= rmax:
            out.append((float(d[0]), float(d[1])))
    return out


def acquisition_plan(seed=1, jitter_um=2.0):
    """All frames needed by the experiments, keyed by (z, pinhole index)."""
    rng = np.random.default_rng(seed + 100)
    P = pinholes(seed)
    jit = {z: ((0.0, 0.0) if z == HEIGHTS_HSSA[0] else tuple(rng.normal(0, jitter_um, 2)))
           for z in sorted(set(HEIGHTS_HSSA + HEIGHTS_MFAP))}
    specs = {}
    for z in HEIGHTS_HSSA:
        for a in range(3):
            specs[(z, a)] = FrameSpec(z, P[a], jit[z], f"z{z:.0f}_a{a}")
    for z in HEIGHTS_MFAP:
        specs[(z, 0)] = FrameSpec(z, P[0], jit[z], f"z{z:.0f}_a0")
    for a in range(9):
        specs[(HEIGHTS_HSSA[0], a)] = FrameSpec(HEIGHTS_HSSA[0], P[a], jit[HEIGHTS_HSSA[0]],
                                                f"z{HEIGHTS_HSSA[0]:.0f}_a{a}")
    return specs, P


def simulate_all(obj, cfg, specs, verbose=True):
    sim = Simulator(obj, cfg)
    frames = {}
    t = time.time()
    for key, s in specs.items():
        frames[key] = sim.capture(s)
    if verbose:
        print(f"simulated {len(frames)} frames in {time.time() - t:.1f}s")
    return frames


def to_recon_frames(frames, cfg, use_true_geometry=False, z_known=None):
    """Autofocus + registration (relative to the first frame)."""
    if use_true_geometry:
        ref = frames[0].true_shift_px
        return [ReconFrame(f.I, f.spec.z,
                           (f.true_shift_px[0] - ref[0], f.true_shift_px[1] - ref[1]), f.f0)
                for f in frames], None
    zs, shifts = preprocess(frames, cfg.pixel, cfg.wavelength, z_known=z_known)
    info = [{"label": f.spec.label, "z_true": f.spec.z, "z_est": z,
             "shift_true": [f.true_shift_px[0] - frames[0].true_shift_px[0],
                            f.true_shift_px[1] - frames[0].true_shift_px[1]],
             "shift_est": list(s)} for f, z, s in zip(frames, zs, shifts)]
    return [ReconFrame(f.I, z, s, f.f0) for f, z, s in zip(frames, zs, shifts)], info


def hr_window_origin(cfg, origin_fine):
    """Layout coordinate of the top-left edge of the reference HR window."""
    dx = cfg.pixel / K
    off = (CANVAS_HR - cfg.sensor_px * K) // 2
    return (origin_fine[0] + off * dx, origin_fine[1] + off * dx)


def save_json(obj, name):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
