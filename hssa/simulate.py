"""Forward simulation of the HSSA lensless on-chip microscope.

Units are micrometres. Parameters follow the paper and its Supplement 1:
lambda = 532 nm, pixel pitch 1.34 um, pinhole-to-sample distance L ~ 70 cm,
pinhole lateral offset up to D ~ 10 cm (illumination angle up to 8 deg),
nearest sample-to-sensor distance 0.975 mm.

The simulation is deliberately richer than the reconstruction model:
 * exact angular-spectrum propagation of a tilted (optionally spherical)
   illumination, so oblique frames carry the real non-paraxial terms;
 * a 4x finer grid than the sensor, with pixel integration (binning);
 * lateral stage jitter at each height, Poisson shot noise, read noise and
   ADC quantisation;
 * a smooth, lab-fixed illumination non-uniformity (the "background
   illumination wavefront" that HSSA's p variable is meant to absorb).
"""
from dataclasses import dataclass, field
import numpy as np

from . import optics


@dataclass
class SimConfig:
    wavelength: float = 0.532
    pixel: float = 1.34
    fine: int = 4                 # simulation grid = pixel / fine
    sensor_px: int = 512          # simulated sensor window (LR pixels per side)
    canvas_fine: int = 4096       # simulation canvas (fine pixels per side)
    L: float = 700e3              # pinhole to sample distance
    spherical: bool = True        # spherical wave from the pinhole, else plane wave
    background: bool = True       # smooth lab-fixed illumination non-uniformity
    bg_amp: float = 0.05          # relative amplitude modulation of the background
    bg_phase: float = 0.3         # rad
    bg_corr: float = 150.0        # correlation length (um)
    photons: float = 2000.0       # mean photo-electrons per pixel on clear background
    read_noise: float = 2.0       # e- rms
    bits: int = 10
    full_well: float = 6000.0
    taper: float = 0.2            # Tukey fraction applied at the canvas edge
    seed: int = 0

    @property
    def dxf(self):
        return self.pixel / self.fine


@dataclass
class FrameSpec:
    z: float                       # sample-to-sensor distance (um)
    pinhole: tuple = (0.0, 0.0)    # pinhole lateral offset (Dy, Dx) in um
    jitter: tuple = (0.0, 0.0)     # lateral sample displacement (dy, dx) in um
    label: str = ""


@dataclass
class Frame:
    I: np.ndarray                  # normalised LR intensity (clear background ~ 1)
    spec: FrameSpec
    true_shift_px: tuple = (0.0, 0.0)   # hologram displacement on the sensor (LR px)
    f0: tuple = (0.0, 0.0)              # chief-ray tilt as spatial frequency (cyc/um)


def _smooth_random_field(shape, dx, corr, rng):
    fy, fx = optics.freq_grid(shape, dx)
    filt = np.exp(-(np.pi * corr) ** 2 * (fx ** 2 + fy ** 2) / 2)
    n = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    f = optics.ifft2(optics.fft2(n) * filt)
    return f / np.std(f)


class Simulator:
    def __init__(self, obj, cfg: SimConfig):
        """obj: complex transmittance on the fine canvas (canvas_fine^2)."""
        self.cfg = cfg
        assert obj.shape == (cfg.canvas_fine, cfg.canvas_fine)
        self.obj = obj.astype(np.complex128)
        self.rng = np.random.default_rng(cfg.seed)
        n = cfg.canvas_fine
        c = (np.arange(n) - n / 2) * cfg.dxf
        self.y = c[:, None]
        self.x = c[None, :]
        self.win = optics.tukey2d((n, n), cfg.taper)
        if cfg.background:
            g = _smooth_random_field((n, n), cfg.dxf, cfg.bg_corr, self.rng)
            self.bg = (1 + cfg.bg_amp * g.real) * np.exp(1j * cfg.bg_phase * g.imag)
        else:
            self.bg = np.ones((n, n))

    def illumination(self, pinhole):
        cfg = self.cfg
        Dy, Dx = pinhole
        k = 2 * np.pi / cfg.wavelength
        if cfg.spherical:
            R = np.sqrt((self.x - Dx) ** 2 + (self.y - Dy) ** 2 + cfg.L ** 2)
            R0 = np.sqrt(Dx ** 2 + Dy ** 2 + cfg.L ** 2)
            u = np.exp(1j * k * (R - R0)) * (R0 / R)
        else:
            R0 = np.sqrt(Dx ** 2 + Dy ** 2 + cfg.L ** 2)
            u = np.exp(1j * k * (-Dx * self.x - Dy * self.y) / R0)
        return u

    def chief_ray(self, pinhole):
        """Direction cosines of the ray from the pinhole to the canvas centre."""
        Dy, Dx = pinhole
        R0 = np.sqrt(Dx ** 2 + Dy ** 2 + self.cfg.L ** 2)
        return (-Dy / R0, -Dx / R0, self.cfg.L / R0)

    def capture(self, spec: FrameSpec, noise=True):
        cfg = self.cfg
        n = cfg.canvas_fine
        obj = self.obj
        if spec.jitter != (0.0, 0.0):
            obj = 1 + optics.fourier_shift(obj - 1, (spec.jitter[0] / cfg.dxf, spec.jitter[1] / cfg.dxf))
        u = self.illumination(spec.pinhole) * self.bg * obj
        u = u * self.win
        H = optics.asm_kernel((n, n), cfg.dxf, cfg.wavelength, spec.z, dtype=np.complex128)
        U = optics.propagate(u, H)
        Ifine = np.abs(U) ** 2
        s = cfg.sensor_px * cfg.fine
        o = (n - s) // 2
        Ilr = optics.bin_mean(Ifine[o:o + s, o:o + s], cfg.fine)
        if noise:
            e = Ilr * cfg.photons
            e = self.rng.poisson(np.clip(e, 0, None)).astype(float)
            e = e + self.rng.normal(0, cfg.read_noise, e.shape)
            e = np.clip(e, 0, cfg.full_well)
            levels = 2 ** cfg.bits - 1
            e = np.round(e / cfg.full_well * levels) / levels * cfg.full_well
            Ilr = e / cfg.photons
        cy, cx, cz = self.chief_ray(spec.pinhole)
        # hologram displacement on the sensor: z * tan(theta) plus the sample jitter
        sy = spec.z * cy / cz + spec.jitter[0]
        sx = spec.z * cx / cz + spec.jitter[1]
        f0 = (cy / cfg.wavelength, cx / cfg.wavelength)
        return Frame(Ilr.astype(np.float32), spec, (sy / cfg.pixel, sx / cfg.pixel), f0)


def usaf_object(cfg: SimConfig, layout, bar_t=0.0):
    """USAF amplitude transmittance centred on the fine canvas.

    Returns (obj, origin) where origin is the physical coordinate (um) of the
    top-left pixel edge of the canvas in layout coordinates.
    """
    from .usaf import render
    n = cfg.canvas_fine
    cx = (layout.bbox[0] + layout.bbox[2]) / 2
    cy = (layout.bbox[1] + layout.bbox[3]) / 2
    origin = (cy - n / 2 * cfg.dxf, cx - n / 2 * cfg.dxf)
    t = render(layout, (n, n), cfg.dxf, origin, bar_t=bar_t)
    return t.astype(np.complex128), origin


def phase_object(cfg: SimConfig, image, size_um, max_phase):
    """Pure-phase object from a grey image, centred, with a smooth border taper."""
    from scipy.ndimage import zoom
    n = cfg.canvas_fine
    m = int(round(size_um / cfg.dxf))
    img = np.asarray(image, float)
    img = (img - img.min()) / (img.max() - img.min())
    img = zoom(img, (m / img.shape[0], m / img.shape[1]), order=3)[:m, :m]
    img = np.clip(img, 0, 1) * optics.tukey2d(img.shape, 0.1)
    ph = np.zeros((n, n))
    o = (n - img.shape[0]) // 2
    ph[o:o + img.shape[0], o:o + img.shape[1]] = img * max_phase
    return np.exp(1j * ph), ph
