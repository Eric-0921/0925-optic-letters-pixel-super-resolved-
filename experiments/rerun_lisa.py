"""Re-run only the LISA row with the shared geometric distance (see
lisa_frames in run_usaf.py) on exactly the same simulated frames, and update
results/<tag>_results.json and <tag>_recons.npz in place.

Usage: python experiments/rerun_lisa.py usaf usaf_s2 usaf_s3 usaf_ideal
"""
import json
import os
import sys
import numpy as np

from common import (SimConfig, Simulator, usaf_object, make_layout, acquisition_plan, reconstruct,
                    metrics, hr_window_origin, RESULTS, HEIGHTS_HSSA, K, CANVAS_HR)
from hssa.preprocess import autofocus
from hssa.recon import ReconFrame


def main(tag):
    seed = int(tag.split("_s")[1]) if "_s" in tag else 1
    cfg = SimConfig(seed=seed)
    if "ideal" in tag:
        cfg.spherical = False
        cfg.background = False
    lay = make_layout()
    obj, origin = usaf_object(cfg, lay)
    specs, _ = acquisition_plan(seed=seed)
    sim = Simulator(obj, cfg)
    frames = {k: sim.capture(s) for k, s in specs.items()}    # same order and RNG stream as run_usaf
    z0 = HEIGHTS_HSSA[0]
    zref = autofocus(frames[(z0, 0)].I, cfg.pixel, cfg.wavelength)
    rfs = [ReconFrame(frames[(z0, j)].I, zref, (0.0, 0.0), frames[(z0, j)].f0) for j in range(9)]
    r = reconstruct(rfs, method="lisa")
    gt = metrics.gt_on_hr(obj, cfg.fine, K)
    W = cfg.sensor_px * K
    o = (CANVAS_HR - W) // 2
    ev = metrics.evaluate_usaf(r["x"][r["ref_window"]], gt[o:o + W, o:o + W], cfg.pixel / K,
                               hr_window_origin(cfg, origin), lay)
    path = os.path.join(RESULTS, f"{tag}_results.json")
    res = json.load(open(path))
    old = res["LISA_1h9a"]["mse"]
    res["LISA_1h9a"].update({"mse": ev["mse"], "finest": "-".join(map(str, ev["finest"])) if ev["finest"] else None,
                             "contrast": ev["contrast"], "final_err": float(r["err"][-1]),
                             "z_used": zref, "note": "shared geometric distance for all LISA frames"})
    json.dump(res, open(path, "w"), indent=2)
    npz = os.path.join(RESULTS, f"{tag}_recons.npz")
    d = dict(np.load(npz))
    d["LISA_1h9a"] = ev["amp"].astype(np.float32)
    np.savez_compressed(npz, **d)
    print(f"{tag}: LISA MSE {old:.4f} -> {ev['mse']:.4f}, finest {res['LISA_1h9a']['finest']}", flush=True)


if __name__ == "__main__":
    for t in sys.argv[1:]:
        main(t)
