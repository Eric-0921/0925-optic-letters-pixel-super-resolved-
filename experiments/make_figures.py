"""Figures from results/*.npz and *.json (Fig. 2, Fig. 3, Fig. S1 analogues)."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import RESULTS, make_layout  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
SERIES = {"LISA": "#2a78d6", "MFAP": "#eb6834", "HSSA": "#1baf7a"}   # categorical slots 1-3
INK2 = "#52514e"


def zoom_slice(origin, dx, lay, groups=(7, 8, 9), margin=6.0):
    xs = [b for b in lay.sets if b.group in groups]
    x0 = min(b.x0 for b in xs) - margin
    x1 = max(b.x0 + 5 * b.w for b in xs) + margin
    y0 = min(b.y0 for b in xs) - margin
    y1 = max(b.y0 + 5 * b.w for b in xs) + margin
    return (slice(int((y0 - origin[0]) / dx), int((y1 - origin[0]) / dx)),
            slice(int((x0 - origin[1]) / dx), int((x1 - origin[1]) / dx)))


def group_profile(img, origin, dx, lay, group=7):
    """Concatenated profiles across the horizontal-bar sets of one group (along y)."""
    from scipy.ndimage import map_coordinates
    ys_all, vals_all, marks = [], [], []
    for b in sorted([b for b in lay.sets if b.group == group and b.orient == "h"], key=lambda b: b.y0):
        ys = np.linspace(b.y0 - b.w, b.y0 + 6 * b.w, 80)
        xs = b.x0 + b.w * np.linspace(1.0, 4.0, 9)
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        v = map_coordinates(img, [(Y - origin[0]) / dx - 0.5, (X - origin[1]) / dx - 0.5], order=3).mean(1)
        ys_all.append(ys); vals_all.append(v); marks.append((b.y0 + 2.5 * b.w, f"{group}-{b.element}"))
    return ys_all, vals_all, marks


def fig_grid(tag):
    d = np.load(os.path.join(RESULTS, f"{tag}_recons.npz"))
    res = json.load(open(os.path.join(RESULTS, f"{tag}_results.json")))
    lay = make_layout()
    origin, dx = tuple(d["origin"]), float(d["dx"])
    sl = zoom_slice(origin, dx, lay)
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.2))
    for i, h in enumerate((3, 2, 1)):
        for j, a in enumerate((1, 2, 3)):
            name = f"grid_h{h}_a{a}"
            ax = axes[i, j]
            ax.imshow(d[name][sl], cmap="gray", vmin=0, vmax=1.2, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            r = res[name]
            ax.set_title(f"{h} height x {a} angle\nMSE {r['mse']:.4f} | finest {r['finest']}",
                         fontsize=8, color="#0b0b0b")
    fig.suptitle("HSSA vs. number of heights and illumination angles (Fig. 2 analogue)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, f"fig2_{tag}_grid.png"), dpi=150)
    plt.close(fig)


def fig_compare(tag):
    d = np.load(os.path.join(RESULTS, f"{tag}_recons.npz"))
    res = json.load(open(os.path.join(RESULTS, f"{tag}_results.json")))
    lay = make_layout()
    origin, dx = tuple(d["origin"]), float(d["dx"])
    sl = zoom_slice(origin, dx, lay)
    names = [("backprop_single", "Back-propagation (1 frame)"), ("LISA_1h9a", "LISA (1 height x 9 angles)"),
             ("MFAP_9h1a", "MFAP (9 heights x 1 angle)"), ("HSSA_3h3a", "HSSA (3 heights x 3 angles)")]
    fig = plt.figure(figsize=(11, 6.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1])
    for j, (n, title) in enumerate(names):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(d[n][sl], cmap="gray", vmin=0, vmax=1.2, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title}\nMSE {res[n]['mse']:.4f} | finest {res[n]['finest']}", fontsize=8)
    ax = fig.add_subplot(gs[1, :])
    for n, key in (("LISA_1h9a", "LISA"), ("MFAP_9h1a", "MFAP"), ("HSSA_3h3a", "HSSA")):
        ys, vs, marks = group_profile(d[n], origin, dx, lay, 7)
        for k, (y, v) in enumerate(zip(ys, vs)):
            ax.plot(y - ys[0][0], v, color=SERIES[key], lw=1.5, label=key if k == 0 else None)
    ys, vs, marks = group_profile(d["gt"], origin, dx, lay, 7)
    for k, (y, v) in enumerate(zip(ys, vs)):
        ax.plot(y - ys[0][0], v, color=INK2, lw=1, ls=":", label="ground truth" if k == 0 else None)
    for yc, lab in marks:
        ax.text(yc - ys[0][0], 1.28, lab, ha="center", fontsize=8, color=INK2)
    ax.set_ylim(-0.05, 1.38)
    ax.set_xlabel("position along group 7 (um)")
    ax.set_ylabel("amplitude")
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    ax.legend(ncol=4, frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("Nine frames per method, USAF 1951 groups 7-9 (Fig. 3 analogue)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, f"fig3_{tag}_compare.png"), dpi=150)
    plt.close(fig)


def fig_phase():
    p = os.path.join(RESULTS, "phase_recons.npz")
    if not os.path.exists(p):
        return
    d = np.load(p)
    res = json.load(open(os.path.join(RESULTS, "phase_results.json")))
    imgs = ["camera", "astronaut"]
    cols = [("gt", "ground truth"), ("LISA_1h9a", "LISA"), ("MFAP_9h1a", "MFAP"), ("HSSA_3h3a", "HSSA")]
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.4))
    for i, im in enumerate(imgs):
        g = d[f"{im}_gt"]
        lo, hi = g.min(), g.max()
        for j, (k, title) in enumerate(cols):
            ax = axes[i, j]
            a = d[f"{im}_{k}"]
            ax.imshow(a, cmap="gray", vmin=lo, vmax=hi)
            ax.set_xticks([]); ax.set_yticks([])
            if k != "gt":
                r = res[f"{im}/{k}"]
                title = f"{title}\nRMSE {r['phase_rmse_rad']:.3f} rad | NMSE {r['phase_nmse']:.4f}"
            ax.set_title(title, fontsize=8)
    fig.suptitle("Pure-phase objects, 0 to pi rad (Supplement Fig. S1 analogue)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figS1_phase.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for tag in ("usaf", "usaf_ideal"):
        if os.path.exists(os.path.join(RESULTS, f"{tag}_recons.npz")):
            fig_grid(tag)
            fig_compare(tag)
    fig_phase()
    print("figures written")
