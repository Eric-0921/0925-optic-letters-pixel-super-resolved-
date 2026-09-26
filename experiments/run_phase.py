"""Phase-object simulation (Supplement 1, Note 2): LISA vs MFAP vs HSSA.

The paper uses "ship" and "mandrill"; these are replaced by two images that
ship with scikit-image ("camera" and a grey "astronaut"). The simulation is
idealised as in the paper's own simulation: plane-wave illumination, no
background non-uniformity, known distances and displacements. Shot and read
noise are kept.
"""
import argparse
import os
import time
import numpy as np
from skimage import data, color

from common import (SimConfig, Simulator, acquisition_plan, to_recon_frames, reconstruct, metrics,
                    save_json, RESULTS, HEIGHTS_HSSA, HEIGHTS_MFAP, K, CANVAS_HR)
from hssa.simulate import phase_object
from hssa.recon import estimate_tilts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--max-phase", type=float, default=np.pi)
    args = ap.parse_args()
    cfg = SimConfig(spherical=False, background=False)
    specs, P = acquisition_plan()
    images = {"camera": data.camera(), "astronaut": color.rgb2gray(data.astronaut())}
    results, recons = {}, {}
    for name, img in images.items():
        obj, ph = phase_object(cfg, img, size_um=400.0, max_phase=args.max_phase)
        sim = Simulator(obj, cfg)
        t = time.time()
        frames = {k: sim.capture(s) for k, s in specs.items()}
        print(f"[{name}] simulated {len(frames)} frames in {time.time() - t:.0f}s", flush=True)
        ref = (HEIGHTS_HSSA[0], 0)
        keys = [ref] + [k for k in frames if k != ref]
        rf, _ = to_recon_frames([frames[k] for k in keys], cfg, use_true_geometry=True)
        RF = dict(zip(keys, rf))
        gt = metrics.gt_on_hr(obj, cfg.fine, K)
        W = cfg.sensor_px * K
        o = (CANVAS_HR - W) // 2
        gtw = gt[o:o + W, o:o + W]
        m = int(200.0 / (cfg.pixel / K))
        roi = (slice(W // 2 - m, W // 2 + m), slice(W // 2 - m, W // 2 + m))
        recons[f"{name}_gt"] = np.angle(gtw[roi]).astype(np.float32)
        sets = {
            "LISA_1h9a": ([RF[(HEIGHTS_HSSA[0], j)] for j in range(9)], "lisa"),
            "MFAP_9h1a": ([RF[(z, 0)] for z in HEIGHTS_MFAP], "mfap"),
            "HSSA_3h3a": ([RF[(z, j)] for z in HEIGHTS_HSSA for j in range(3)], "hssa"),
            "HSSA_tilt_kernel": (estimate_tilts([RF[(z, j)] for z in HEIGHTS_HSSA for j in range(3)],
                                                [i for i in range(3) for _ in range(3)], [0, 3, 6],
                                                cfg.pixel, cfg.wavelength), "hssa"),
        }
        for label, (rfs, method) in sets.items():
            t = time.time()
            kw = {"tilt": "kernel"} if label == "HSSA_tilt_kernel" else {}
            r = reconstruct(rfs, method=method, iters=args.iters, **kw)
            ev = metrics.evaluate_phase(r["x"][r["ref_window"]], gtw, roi)
            results[f"{name}/{label}"] = {"phase_rmse_rad": ev["phase_rmse"], "phase_nmse": ev["phase_nmse"],
                                          "phase_rmse_detail_rad": ev["phase_rmse_detail"],
                                          "align_shift_px": ev["align_shift"],
                                          "time_s": time.time() - t}
            recons[f"{name}_{label}"] = ev["phase"][roi].astype(np.float32)
            print(f"[{name}] {label:10s} phase RMSE={ev['phase_rmse']:.4f} rad  "
                  f"NMSE={ev['phase_nmse']:.5f} detail RMSE={ev['phase_rmse_detail']:.4f} "
                  f"shift={np.round(ev['align_shift'], 2)} ({time.time() - t:.0f}s)", flush=True)
            save_json(results, "phase_results.json")
    np.savez_compressed(os.path.join(RESULTS, "phase_recons.npz"), **recons)
    print("done")


if __name__ == "__main__":
    main()
