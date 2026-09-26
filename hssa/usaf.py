"""USAF 1951 resolution target (groups 5-9) with exact area-coverage rendering.

Dimensions follow the MIL-STD-150A definition: element (g, e) has
2**(g + (e-1)/6) line pairs per mm, bar width w = 1/(2 * lp/mm), bar length 5w,
three bars per orientation separated by w.
All lengths are in micrometres.
"""
from dataclasses import dataclass, field
import numpy as np


def linewidth_um(group, element):
    return 1e3 / (2 * 2 ** (group + (element - 1) / 6))


@dataclass
class BarSet:
    group: int
    element: int
    orient: str            # 'v': vertical bars (resolve along x), 'h': horizontal bars
    w: float               # bar width (um)
    x0: float              # top-left corner of the 5w x 5w block (um)
    y0: float

    def bar_centers(self):
        return [self.w * (0.5 + 2 * i) for i in range(3)]


@dataclass
class USAFLayout:
    sets: list = field(default_factory=list)
    rects: list = field(default_factory=list)   # (x0, y0, x1, y1) opaque rectangles
    bbox: tuple = (0, 0, 0, 0)                   # (x0, y0, x1, y1) of the pattern


def make_layout(groups=(5, 8, 9, 7, 6), gap_groups=18.0):
    """Place group columns side by side, each column centred vertically.

    The default order puts groups 8 and 9 near the middle of the pattern so
    that the finest elements sit near the optical axis.
    """
    cols = []
    for g in groups:
        y, col_w, items = 0.0, 0.0, []
        for e in range(1, 7):
            w = linewidth_um(g, e)
            items.append((e, w, y))
            col_w = max(col_w, 12 * w)
            y += 5 * w + max(2 * w, 2.0)
        cols.append((g, col_w, y - max(2 * items[-1][1], 2.0), items))
    height = max(c[2] for c in cols)
    lay = USAFLayout()
    x = 0.0
    for g, col_w, col_h, items in cols:
        yoff = (height - col_h) / 2
        for e, w, y in items:
            y = y + yoff
            vx, hx = x, x + 7 * w
            for i in range(3):
                lay.rects.append((vx + 2 * i * w, y, vx + 2 * i * w + w, y + 5 * w))
                lay.rects.append((hx, y + 2 * i * w, hx + 5 * w, y + 2 * i * w + w))
            lay.sets.append(BarSet(g, e, 'v', w, vx, y))
            lay.sets.append(BarSet(g, e, 'h', w, hx, y))
        x += col_w + gap_groups
    lay.bbox = (0.0, 0.0, x - gap_groups, height)
    return lay


def _coverage_1d(edges, a, b):
    """Fraction of each pixel [edges[i], edges[i+1]] covered by [a, b]."""
    lo = np.clip(edges[:-1], a, b)
    hi = np.clip(edges[1:], a, b)
    return (hi - lo) / (edges[1:] - edges[:-1])


def render(layout, shape, dx, origin, bar_t=0.0):
    """Amplitude transmittance on a grid of pitch dx (um).

    origin = (y, x) physical coordinate of the top-left pixel edge. Opaque bars
    have transmittance bar_t, the clear glass 1.
    """
    ye = origin[0] + dx * np.arange(shape[0] + 1)
    xe = origin[1] + dx * np.arange(shape[1] + 1)
    opaque = np.zeros(shape)
    for (x0, y0, x1, y1) in layout.rects:
        iy = np.nonzero((ye[1:] > y0) & (ye[:-1] < y1))[0]
        ix = np.nonzero((xe[1:] > x0) & (xe[:-1] < x1))[0]
        if len(iy) == 0 or len(ix) == 0:
            continue
        cy = _coverage_1d(ye[iy[0]:iy[-1] + 2], y0, y1)
        cx = _coverage_1d(xe[ix[0]:ix[-1] + 2], x0, x1)
        opaque[iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1] += np.outer(cy, cx)
    opaque = np.clip(opaque, 0, 1)
    return 1.0 - opaque * (1.0 - bar_t)


def groups_bbox(layout, groups, margin=6.0):
    """(x0, y0, x1, y1) of the bar sets of the given groups, with a margin (um)."""
    sets = [b for b in layout.sets if b.group in groups]
    return (min(b.x0 for b in sets) - margin, min(b.y0 for b in sets) - margin,
            max(b.x0 + 5 * b.w for b in sets) + margin, max(b.y0 + 5 * b.w for b in sets) + margin)


def element_contrast(img, dx, origin, bs, oversample=4):
    """Michelson contrast of the two gaps against their neighbouring bars.

    img: real amplitude image on a grid with pitch dx and top-left edge `origin`.
    Returns min over the two gaps of (I_gap - max(adjacent bars)) /
    (I_gap + max(adjacent bars)); negative means the gap is not visible.
    """
    from scipy.ndimage import map_coordinates
    w = bs.w
    n_along = 9
    centers = bs.bar_centers()
    gaps = [w * 2.0, w * 4.0]
    t_along = bs.x0 if bs.orient == 'h' else bs.y0
    along = t_along + w * np.linspace(1.0, 4.0, n_along)   # middle 60% of the bar length

    def sample(across_pos):
        vals = []
        for a in across_pos:
            if bs.orient == 'v':
                xs = np.full(n_along, bs.x0 + a)
                ys = along
            else:
                ys = np.full(n_along, bs.y0 + a)
                xs = along
            r = (ys - origin[0]) / dx - 0.5
            c = (xs - origin[1]) / dx - 0.5
            vals.append(map_coordinates(img, [r, c], order=3, mode='nearest').mean())
        return np.array(vals)

    b = sample(centers)
    gv = sample(gaps)
    c1 = (gv[0] - max(b[0], b[1])) / (gv[0] + max(b[0], b[1]) + 1e-12)
    c2 = (gv[1] - max(b[1], b[2])) / (gv[1] + max(b[1], b[2]) + 1e-12)
    return float(min(c1, c2))


def resolution_report(img, dx, origin, layout, thresh=0.1, groups=None):
    """Per-element contrast (min over both orientations) and finest resolved element.

    The finest resolved element is the last one, in order of increasing
    spatial frequency, for which this and every coarser element reach `thresh`.
    """
    table = {}
    for bs in layout.sets:
        if groups is not None and bs.group not in groups:
            continue
        c = element_contrast(img, dx, origin, bs)
        key = (bs.group, bs.element)
        table[key] = min(table.get(key, np.inf), c)
    finest = None
    for key in sorted(table):
        if table[key] >= thresh:
            finest = key
        else:
            break
    return table, finest
