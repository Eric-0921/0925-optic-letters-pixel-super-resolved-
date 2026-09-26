"""How far does HSSA's illumination variable p move from 1? (realistic, seed 1, 3x3)"""
import numpy as np
from common import (SimConfig, Simulator, usaf_object, make_layout, acquisition_plan, to_recon_frames,
                    reconstruct, save_json, HEIGHTS_HSSA)

cfg = SimConfig()
lay = make_layout()
obj, _ = usaf_object(cfg, lay)
specs, _ = acquisition_plan()
sim = Simulator(obj, cfg)
frames = [sim.capture(specs[(z, a)]) for z in HEIGHTS_HSSA for a in range(3)]
rf, _ = to_recon_frames(frames, cfg, use_true_geometry=True)
out = {}
for gamma in (0.05, 0.5):
    r = reconstruct(rf, gamma=gamma)
    w = r["ref_window"]
    dp = np.abs(r["p"][w] - 1)
    ph = np.angle(r["p"][w])
    out[f"gamma={gamma}"] = {"mean_abs_p_minus_1": float(dp.mean()), "max_abs_p_minus_1": float(dp.max()),
                             "p_phase_std_rad": float(ph.std())}
    print(gamma, out[f"gamma={gamma}"], flush=True)
save_json(out, "p_diagnostic.json")
