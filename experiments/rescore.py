"""Re-score saved USAF reconstructions with the central-region metric.

Keeps the original whole-pattern MSE as `mse_fullpattern` when present.
Usage: python experiments/rescore.py usaf [usaf_ideal ...]
"""
import json
import os
import sys
import numpy as np

from common import RESULTS, make_layout, SimConfig, usaf_object, K, CANVAS_HR
from hssa import metrics


def rescore(tag, groups=(7, 8, 9)):
    d = np.load(os.path.join(RESULTS, f"{tag}_recons.npz"))
    path = os.path.join(RESULTS, f"{tag}_results.json")
    res = json.load(open(path))
    lay = make_layout()
    cfg = SimConfig()
    obj, _ = usaf_object(cfg, lay)
    gt = metrics.gt_on_hr(obj, cfg.fine, K)
    W = cfg.sensor_px * K
    o = (CANVAS_HR - W) // 2
    gtw = gt[o:o + W, o:o + W]
    origin, dx = tuple(d["origin"]), float(d["dx"])
    for name in res:
        if name not in d.files:
            continue
        sc = metrics.usaf_scores(d[name], gtw, dx, origin, lay, groups)
        if "mse_fullpattern" not in res[name]:
            res[name]["mse_fullpattern"] = res[name]["mse"]
        res[name]["mse"] = sc["mse"]
        res[name]["finest"] = "-".join(map(str, sc["finest"])) if sc["finest"] else None
        res[name]["contrast"] = sc["contrast"]
        print(f"{tag:12s} {name:34s} MSE={sc['mse']:.4f} finest={res[name]['finest']}")
    json.dump(res, open(path, "w"), indent=2)


if __name__ == "__main__":
    for t in sys.argv[1:]:
        rescore(t)
