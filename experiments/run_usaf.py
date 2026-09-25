"""USAF 1951 experiments: Fig. 2 (heights x angles grid), Fig. 3 (LISA / MFAP /
HSSA with 9 frames each) and ablations.

Usage: python experiments/run_usaf.py [--iters 50] [--quick]
Writes results/usaf_*.json and results/usaf_recons.npz.
"""
import argparse
import os
import time
import numpy as np

from common import (SimConfig, Simulator, FrameSpec, usaf_object, make_layout, acquisition_plan,
                    to_recon_frames, reconstruct, metrics, hr_window_origin, save_json, RESULTS,
                    HEIGHTS_HSSA, HEIGHTS_MFAP, K, CANVAS_HR)
from hssa.recon import ReconFrame, estimate_tilts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ideal", action="store_true",
                    help="plane-wave illumination, no background non-uniformity")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    tag = "usaf_ideal" if args.ideal else "usaf"
    if args.seed != 1:
        tag += f"_s{args.seed}"

    cfg = SimConfig(seed=args.seed)
    if args.ideal:
        cfg.spherical = False
        cfg.background = False
    lay = make_layout()
    obj, origin = usaf_object(cfg, lay)
    specs, P = acquisition_plan(seed=args.seed)

    # oblique frames re-simulated as pure lateral shifts (paraxial equivalent)
    sim = Simulator(obj, cfg)
    shift_specs = {}
    for z in HEIGHTS_HSSA:
        for a in (1, 2):
            s = specs[(z, a)]
            cy, cx, cz = sim.chief_ray(s.pinhole)
            shift_specs[(z, a)] = FrameSpec(z, P[0], (s.jitter[0] + z * cy / cz, s.jitter[1] + z * cx / cz),
                                            s.label + "_shiftonly")

    t = time.time()
    frames = {k: sim.capture(s) for k, s in specs.items()}
    frames_shift = {k: sim.capture(s) for k, s in shift_specs.items()}
    print(f"simulated {len(frames) + len(frames_shift)} frames in {time.time() - t:.0f}s", flush=True)

    # --- preprocessing: one pass over every frame, reference = (975 um, on-axis)
    ref_key = (HEIGHTS_HSSA[0], 0)
    keys = [ref_key] + [k for k in frames if k != ref_key]
    t = time.time()
    rf_list, info = to_recon_frames([frames[k] for k in keys], cfg)
    RF = dict(zip(keys, rf_list))
    rf_s, info_s = to_recon_frames([frames[ref_key]] + [frames_shift[k] for k in frames_shift], cfg)
    RF_shift = dict(zip(frames_shift, rf_s[1:]))
    rf_true, _ = to_recon_frames([frames[k] for k in keys], cfg, use_true_geometry=True)
    RF_true = dict(zip(keys, rf_true))
    print(f"preprocessing done in {time.time() - t:.0f}s", flush=True)
    save_json({"frames": info, "shift_only_frames": info_s[1:],
               "pinholes_um": P,
               "angles_deg": [float(np.degrees(np.arctan(np.hypot(*p) / cfg.L))) for p in P]},
              f"{tag}_preprocessing.json")

    gt = metrics.gt_on_hr(obj, cfg.fine, K)
    o = (CANVAS_HR - cfg.sensor_px * K) // 2
    W = cfg.sensor_px * K
    gtw = gt[o:o + W, o:o + W]
    org = hr_window_origin(cfg, origin)
    dx = cfg.pixel / K

    results, recons = {}, {"gt": np.abs(gtw).astype(np.float32)}

    def run(name, rfs, method="hssa", **kw):
        t0 = time.time()
        r = reconstruct(rfs, method=method, iters=args.iters, **kw)
        xw = r["x"][r["ref_window"]]
        ev = metrics.evaluate_usaf(xw, gtw, dx, org, lay)
        results[name] = {"method": method, "n_frames": len(rfs), "mse": ev["mse"],
                         "finest": "-".join(map(str, ev["finest"])) if ev["finest"] else None,
                         "contrast": ev["contrast"], "final_err": float(r["err"][-1]),
                         "time_s": time.time() - t0, **{k: v for k, v in kw.items()}}
        recons[name] = ev["amp"].astype(np.float32)
        print(f"{name:34s} MSE={ev['mse']:.4f} finest={results[name]['finest']} "
              f"({time.time() - t0:.0f}s)", flush=True)
        save_json(results, f"{tag}_results.json")
        return ev

    # 0) the sensor-limited baseline: single hologram, up-sampled, back-propagated
    from hssa import optics
    z0 = RF[ref_key].z
    a0 = np.sqrt(np.clip(optics.upsample_fourier(frames[ref_key].I, K), 0, None)).astype(np.complex64)
    Hb = optics.asm_kernel(a0.shape, dx, cfg.wavelength, -z0)
    bp = optics.propagate(a0, Hb)
    ev = metrics.evaluate_usaf(bp, gtw, dx, org, lay)
    results["backprop_single"] = {"method": "backprop", "n_frames": 1, "mse": ev["mse"],
                                  "finest": "-".join(map(str, ev["finest"])) if ev["finest"] else None,
                                  "contrast": ev["contrast"]}
    recons["backprop_single"] = ev["amp"].astype(np.float32)
    print(f"{'backprop_single':34s} MSE={ev['mse']:.4f} finest={results['backprop_single']['finest']}")

    # 1) Fig. 2: HSSA with h heights x a angles
    for h in (1, 2, 3):
        for a in (1, 2, 3):
            rfs = [RF[(HEIGHTS_HSSA[i], j)] for i in range(h) for j in range(a)]
            run(f"grid_h{h}_a{a}", rfs)

    # 2) Fig. 3: nine frames each
    run("LISA_1h9a", [RF[(HEIGHTS_HSSA[0], j)] for j in range(9)], method="lisa")
    run("MFAP_9h1a", [RF[(z, 0)] for z in HEIGHTS_MFAP], method="mfap")
    hssa9 = [RF[(z, j)] for z in HEIGHTS_HSSA for j in range(3)]
    run("HSSA_3h3a", hssa9)

    # 3) ablations on the HSSA 3x3 data
    run("abl_HSSA_no_p_update", hssa9, method="mfap")
    run("abl_HSSA_literal_update", hssa9, update="literal")
    run("abl_HSSA_true_geometry", [RF_true[(z, j)] for z in HEIGHTS_HSSA for j in range(3)])
    shifted = [RF[(z, 0)] if j == 0 else RF_shift[(z, j)] for z in HEIGHTS_HSSA for j in range(3)]
    run("abl_HSSA_shift_instead_of_tilt", shifted)
    run("abl_MFAP_literal_update", [RF[(z, 0)] for z in HEIGHTS_MFAP], method="mfap", update="literal")
    run("abl_LISA_registered_no_angle", [RF[(HEIGHTS_HSSA[0], j)] for j in range(9)], method="mfap")

    # 4) extension: tilt estimated from the registered shifts, tilted kernel
    tilt9 = estimate_tilts(hssa9, [i for i in range(3) for _ in range(3)], [0, 3, 6],
                           cfg.pixel, cfg.wavelength)
    run("ext_HSSA_tilt_from_shift", tilt9, tilt="kernel")
    for h, a in ((3, 2), (2, 3)):
        sub = [hssa9[3 * i + j] for i in range(h) for j in range(a)]
        sub = estimate_tilts(sub, [i for i in range(h) for _ in range(a)], [a * i for i in range(h)],
                             cfg.pixel, cfg.wavelength)
        run(f"ext_grid_h{h}_a{a}_tilt_from_shift", sub, tilt="kernel")

    np.savez_compressed(os.path.join(RESULTS, f"{tag}_recons.npz"), **recons,
                        origin=np.array(org), dx=dx)
    print("done")


if __name__ == "__main__":
    main()
