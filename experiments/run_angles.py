"""Angle sweep: HSSA (paper model) vs. HSSA with tilt estimated from shifts,
for maximum illumination angles of about 2, 4 and 8 degrees (3 heights x 3
angles, realistic simulation). Writes results/angles_results.json.
"""
import argparse
import time
import numpy as np

from common import (SimConfig, Simulator, FrameSpec, usaf_object, make_layout, to_recon_frames,
                    reconstruct, metrics, hr_window_origin, save_json, HEIGHTS_HSSA, K, CANVAS_HR)
from hssa.recon import estimate_tilts


def plan(seed, rmax, rmin):
    rng = np.random.default_rng(seed)
    P = [(0.0, 0.0)]
    while len(P) < 3:
        d = rng.uniform(-rmax, rmax, 2)
        if rmin <= np.hypot(*d) <= rmax:
            P.append((float(d[0]), float(d[1])))
    jr = np.random.default_rng(seed + 100)
    jit = [(0.0, 0.0)] + [tuple(jr.normal(0, 2.0, 2)) for _ in range(2)]
    return [FrameSpec(z, P[a], jit[i], f"z{z:.0f}_a{a}") for i, z in enumerate(HEIGHTS_HSSA) for a in range(3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--angles", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    args = ap.parse_args()
    lay = make_layout()
    res = {}
    for seed in args.seeds:
        cfg = SimConfig(seed=seed)
        obj, origin = usaf_object(cfg, lay)
        gt = metrics.gt_on_hr(obj, cfg.fine, K)
        W = cfg.sensor_px * K
        o = (CANVAS_HR - W) // 2
        gtw = gt[o:o + W, o:o + W]
        org = hr_window_origin(cfg, origin)
        for ang in args.angles:
            rmax = cfg.L * np.tan(np.radians(ang))
            specs = plan(seed, rmax, 0.4 * rmax)
            sim = Simulator(obj, cfg)
            t = time.time()
            frames = [sim.capture(s) for s in specs]
            rf, info = to_recon_frames(frames, cfg)
            tilt = estimate_tilts(rf, [i for i in range(3) for _ in range(3)], [0, 3, 6], cfg.pixel, cfg.wavelength)
            for name, rfs, kw in (("HSSA", rf, {}), ("HSSA_tilt_from_shift", tilt, {"tilt": "kernel"})):
                r = reconstruct(rfs, **kw)
                ev = metrics.evaluate_usaf(r["x"][r["ref_window"]], gtw, cfg.pixel / K, org, lay)
                key = f"seed{seed}/max{ang:g}deg/{name}"
                res[key] = {"mse": ev["mse"], "finest": "-".join(map(str, ev["finest"])) if ev["finest"] else None,
                            "contrast": ev["contrast"],
                            "angles_deg": [float(np.degrees(np.arctan(np.hypot(*s.pinhole) / cfg.L))) for s in specs[:3]],
                            "tilt_est_deg": [float(np.degrees(np.arctan(np.hypot(*f.f0) * cfg.wavelength)))
                                             for f in tilt[:3]]}
                print(f"{key:40s} MSE={ev['mse']:.4f} finest={res[key]['finest']} ({time.time() - t:.0f}s)",
                      flush=True)
                save_json(res, "angles_results.json")
    print("done")


if __name__ == "__main__":
    main()
