"""Aggregate USAF results over seeds into results/summary.md."""
import glob
import json
import os
import re
import numpy as np

from common import RESULTS

ORDER = ["backprop_single"] + [f"grid_h{h}_a{a}" for h in (1, 2, 3) for a in (1, 2, 3)] + [
    "LISA_1h9a", "MFAP_9h1a", "HSSA_3h3a", "abl_HSSA_no_p_update", "abl_HSSA_literal_update",
    "abl_HSSA_true_geometry", "abl_HSSA_shift_instead_of_tilt", "abl_MFAP_literal_update",
    "abl_LISA_registered_no_angle", "ext_HSSA_tilt_from_shift", "ext_grid_h3_a2_tilt_from_shift",
    "ext_grid_h2_a3_tilt_from_shift"]


def elem_key(s):
    return tuple(map(int, s.split("-"))) if s else (0, 0)


def load(pattern):
    runs = {}
    for p in sorted(glob.glob(os.path.join(RESULTS, pattern))):
        tag = os.path.basename(p)[:-len("_results.json")]
        runs[tag] = json.load(open(p))
    return runs


def table(runs, title):
    tags = list(runs)
    lines = [f"### {title}", "", f"Seeds: {', '.join(tags)}", "",
             "| Configuration | Frames | MSE (mean ± sd) | Finest element per seed | "
             "Contrast 8-5 (mean) | Contrast 8-6 (mean) |", "|---|---|---|---|---|---|"]
    for name in ORDER:
        rs = [runs[t][name] for t in tags if name in runs[t]]
        if not rs:
            continue
        mse = np.array([r["mse"] for r in rs])
        fin = [r["finest"] or "none" for r in rs]
        c85 = np.mean([r["contrast"].get("8-5", np.nan) for r in rs])
        c86 = np.mean([r["contrast"].get("8-6", np.nan) for r in rs])
        sd = f" ± {mse.std(ddof=1):.4f}" if len(mse) > 1 else ""
        lines.append(f"| {name} | {rs[0]['n_frames']} | {mse.mean():.4f}{sd} | {', '.join(fin)} | "
                     f"{c85:.3f} | {c86:.3f} |")
    return "\n".join(lines) + "\n"


def main():
    out = ["# Result tables", "",
           "MSE: normalised amplitude vs. ground truth on the central region (groups 7-9).",
           "Finest element: finest USAF element in groups 7-9 such that it and every coarser one "
           "have gap-to-bar Michelson contrast >= 0.1 in both orientations.", ""]
    real = {k: v for k, v in load("usaf*_results.json").items() if not k.startswith("usaf_ideal")}
    if real:
        out.append(table(real, "Realistic simulation (spherical illumination, background, noise)"))
    ideal = load("usaf_ideal*_results.json")
    if ideal:
        out.append(table(ideal, "Idealised simulation (plane waves, no background; noise kept)"))
    ph = os.path.join(RESULTS, "phase_results.json")
    if os.path.exists(ph):
        r = json.load(open(ph))
        out += ["### Phase objects (0 to pi rad)", "",
                "| Object / method | Phase RMSE (rad) | NMSE | Detail RMSE, < 10 um (rad) |", "|---|---|---|---|"]
        out += [f"| {k} | {v['phase_rmse_rad']:.4f} | {v['phase_nmse']:.5f} | "
                f"{v.get('phase_rmse_detail_rad', float('nan')):.4f} |" for k, v in r.items()]
        out.append("")
    open(os.path.join(RESULTS, "summary.md"), "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
