"""
manim_stats/animations/clt_demo.py
====================================
Central Limit Theorem demonstration animations.

Pedagogical structure
---------------------
The CLT demo is organised as a sequence of self-contained *phases*,
each teaching one aspect of the theorem.  A full run proceeds:

    Phase 0 – Population
        Display the source distribution as a 3D histogram with PDF
        overlay.  Clearly label that it is NOT normal (Uniform,
        Exponential, Bimodal, etc.).

    Phase 1 – Single sample draw
        Animated particles fall from population bars into a sample
        collection zone.  Sample size n shown live.  x̄ computed and
        annotated as the particles merge to a single point.

    Phase 2 – Accumulate sample means
        Repeat Phase 1 silently k times, each trial adding one bar to a
        growing histogram of x̄ values.  Bars colour-coded by distance
        from the theoretical mean.

    Phase 3 – Normal convergence
        Once ≥ 30 trials are accumulated, a normal curve (μ, σ/√n)
        emerges over the x̄ histogram.  The fit improves visually.

    Phase 4 – n-sweep
        Repeat the full accumulation for n = 1, 2, 5, 10, 30, 100,
        showing the x̄ histogram narrowing as 1/√n.  σ/√n annotation
        updates live.

Phase 4 produces the most visually striking proof of the theorem.

Classes
-------
CLTConfig
PopulationDistribution3D
SampleMeanHistogram3D
NormalConvergenceOverlay3D
CLTDemo

Internals
---------
_SampleParticle3D
_VarianceAnnotation3D
_SampleDrawAnimation

Ready-to-render ThreeDScene subclasses
--------------------------------------
CLTUniformScene
CLTExponentialScene
CLTBimodalScene
CLTSweepScene
CLTComparisonScene

Usage
-----
    # Render from the command line:
    #   manim -pql clt_demo.py CLTUniformScene

    # Or embed phases into your own scene:
    from manim_stats.animations.clt_demo import CLTDemo, CLTConfig

    class MyScene(ThreeDScene):
        def construct(self):
            demo = CLTDemo(CLTConfig(
                source="exponential", n=10, n_trials=200,
            ))
            demo.phase_population(self)
            demo.phase_accumulate(self, n_visible=60)
            demo.phase_normal_overlay(self)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    List, Sequence, Tuple, Optional, Callable, Union, Dict
)
import numpy as np

from manim import (
    # Mobjects
    VGroup, VMobject, Polygon, Rectangle, Line, DashedLine,
    Dot, Dot3D, Text, MathTex, Ellipse, Arrow,
    # Scene
    ThreeDScene,
    # Animations
    AnimationGroup, LaggedStart, Succession,
    Create, FadeIn, FadeOut, Transform, MoveAlongPath,
    UpdateFromAlphaFunc, Flash, Write, Indicate,
    # Constants
    DEGREES, PI, TAU,
    RIGHT, LEFT, UP, DOWN, OUT, IN, ORIGIN,
    X_AXIS, Y_AXIS, Z_AXIS,
    WHITE, BLACK, GRAY, BLUE, GREEN, RED, YELLOW,
    # Colour
    ManimColor, color_to_rgb, rgba_to_color, color_to_rgba,
    # Utilities
    rate_functions, smooth,
    # Camera
    config as manim_config,
)

# ---------------------------------------------------------------------------
# Shared colour helpers
# ---------------------------------------------------------------------------

def _with_opacity(color: ManimColor, opacity: float) -> ManimColor:
    r, g, b = color_to_rgb(color)
    return rgba_to_color([r, g, b, max(0.0, min(1.0, opacity))])

def _darken(color: ManimColor, factor: float = 0.65) -> ManimColor:
    r, g, b = color_to_rgb(color)
    return ManimColor([r * factor, g * factor, b * factor])

def _lighten(color: ManimColor, factor: float = 1.35) -> ManimColor:
    r, g, b = color_to_rgb(color)
    return ManimColor([min(r * factor, 1.0), min(g * factor, 1.0), min(b * factor, 1.0)])

def _lerp_color(a: ManimColor, b: ManimColor, t: float) -> ManimColor:
    ra, ga, ba = color_to_rgb(a)
    rb, gb, bb = color_to_rgb(b)
    return ManimColor([ra + (rb - ra) * t, ga + (gb - ga) * t, ba + (bb - ba) * t])

def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(TAU))


# ---------------------------------------------------------------------------
# CLTConfig
# ---------------------------------------------------------------------------

@dataclass
class CLTConfig:
    """Complete configuration for a CLT demonstration.

    Source distribution
    ~~~~~~~~~~~~~~~~~~~
    ``source`` selects the population distribution:
    - ``"uniform"``     – U(lo, hi).
    - ``"exponential"`` – Exp(rate).
    - ``"bimodal"``     – mixture of two normals.
    - ``"bernoulli"``   – Bernoulli(p).  Extreme discrete skew.
    - ``"pareto"``      – Pareto(shape).  Heavy tail.

    Sampling
    ~~~~~~~~
    ``n``       : sample size per trial (the key CLT parameter).
    ``n_trials``: total number of x̄ trials to simulate.
    ``rng_seed``: reproducibility.

    Layout
    ~~~~~~
    ``pop_x``     : x-position of the population histogram.
    ``xbar_x``    : x-position of the x̄ histogram.
    ``hist_width``: scene-unit width of both histograms.
    ``hist_depth``: scene-unit depth (y) of bars.
    ``z_scale``   : value → scene-unit height multiplier.
    ``n_bins``    : number of histogram bins.

    Visual
    ~~~~~~
    ``pop_color``     : population histogram bar colour.
    ``xbar_color_lo`` : x̄ histogram colour for bars near the mean.
    ``xbar_color_hi`` : x̄ histogram colour for bars far from the mean.
    ``normal_color``  : convergence curve colour.
    ``particle_color``: colour of animated sample particles.
    ``glow_normal``   : whether to add a glow halo to the normal curve.

    Attributes
    ----------
    source : str
    n : int
    n_trials : int
    rng_seed : int
    pop_params : dict
        Distribution-specific parameters fed to the sampler.
    pop_x : float
    xbar_x : float
    hist_width : float
    hist_depth : float
    z_scale : float
    n_bins : int
    pop_color : ManimColor
    xbar_color_lo : ManimColor
    xbar_color_hi : ManimColor
    normal_color : ManimColor
    particle_color : ManimColor
    glow_normal : bool
    show_formula_panel : bool
    show_variance_annotation : bool
    show_source_label : bool
    """

    source: str = "uniform"
    n: int = 10
    n_trials: int = 300
    rng_seed: int = 42
    pop_params: Dict = field(default_factory=dict)

    pop_x: float = -2.5
    xbar_x: float = 2.5
    hist_width: float = 4.0
    hist_depth: float = 0.45
    z_scale: float = 0.9
    n_bins: int = 25

    pop_color: ManimColor = ManimColor("#4A90D9")
    xbar_color_lo: ManimColor = ManimColor("#2DAA6E")
    xbar_color_hi: ManimColor = ManimColor("#E0AA40")
    normal_color: ManimColor = ManimColor("#E8593C")
    particle_color: ManimColor = ManimColor("#FFD700")
    glow_normal: bool = True

    show_formula_panel: bool = True
    show_variance_annotation: bool = True
    show_source_label: bool = True


# ---------------------------------------------------------------------------
# Sampler helpers
# ---------------------------------------------------------------------------

def _make_sampler(
    source: str,
    params: Dict,
    rng: np.random.Generator,
) -> Callable[[int], np.ndarray]:
    """Return a callable ``sampler(n)`` for the given distribution.

    Parameters
    ----------
    source : str
        Distribution name.
    params : dict
        Distribution-specific keyword parameters.
    rng : np.random.Generator

    Returns
    -------
    Callable[[int], np.ndarray]
        Function that draws n samples and returns a 1-D array.
    """
    if source == "uniform":
        lo = params.get("lo", 0.0)
        hi = params.get("hi", 1.0)
        return lambda n: rng.uniform(lo, hi, n)

    elif source == "exponential":
        rate = params.get("rate", 1.0)
        return lambda n: rng.exponential(1.0 / rate, n)

    elif source == "bimodal":
        mu1 = params.get("mu1", -2.0)
        mu2 = params.get("mu2", 2.0)
        std1 = params.get("std1", 0.6)
        std2 = params.get("std2", 0.6)
        w = params.get("weight", 0.5)
        def _bimodal(n):
            mask = rng.random(n) < w
            a = rng.normal(mu1, std1, n)
            b = rng.normal(mu2, std2, n)
            return np.where(mask, a, b)
        return _bimodal

    elif source == "bernoulli":
        p = params.get("p", 0.2)
        return lambda n: rng.binomial(1, p, n).astype(float)

    elif source == "pareto":
        shape = params.get("shape", 1.5)
        # scipy-style: X = (np.random.pareto(shape) + 1)
        return lambda n: (rng.pareto(shape, n) + 1.0)

    else:
        raise ValueError(f"Unknown source distribution: {source!r}")


def _population_range(source: str, params: Dict) -> Tuple[float, float]:
    """Return a sensible (x_min, x_max) for the population histogram."""
    if source == "uniform":
        return params.get("lo", 0.0), params.get("hi", 1.0)
    elif source == "exponential":
        rate = params.get("rate", 1.0)
        return 0.0, 5.0 / rate
    elif source == "bimodal":
        mu1 = params.get("mu1", -2.0)
        mu2 = params.get("mu2", 2.0)
        std = max(params.get("std1", 0.6), params.get("std2", 0.6))
        return mu1 - 3 * std, mu2 + 3 * std
    elif source == "bernoulli":
        return -0.3, 1.3
    elif source == "pareto":
        return 1.0, 6.0
    else:
        return 0.0, 4.0


def _theoretical_mean_std(
    source: str,
    params: Dict,
    n: int,
) -> Tuple[float, float]:
    """Return (population_mean, population_std) for the source."""
    if source == "uniform":
        lo, hi = params.get("lo", 0.0), params.get("hi", 1.0)
        mu = (lo + hi) / 2
        sigma = (hi - lo) / np.sqrt(12)
    elif source == "exponential":
        rate = params.get("rate", 1.0)
        mu = 1.0 / rate
        sigma = 1.0 / rate
    elif source == "bimodal":
        mu1 = params.get("mu1", -2.0)
        mu2 = params.get("mu2", 2.0)
        w = params.get("weight", 0.5)
        mu = w * mu1 + (1 - w) * mu2
        # var = E[X²] − μ²
        std1 = params.get("std1", 0.6)
        std2 = params.get("std2", 0.6)
        ex2 = w * (mu1**2 + std1**2) + (1 - w) * (mu2**2 + std2**2)
        sigma = np.sqrt(ex2 - mu**2)
    elif source == "bernoulli":
        p = params.get("p", 0.2)
        mu = p
        sigma = np.sqrt(p * (1 - p))
    elif source == "pareto":
        shape = params.get("shape", 1.5)
        if shape > 1:
            mu = shape / (shape - 1)
        else:
            mu = float("inf")
        if shape > 2:
            sigma = np.sqrt(shape / ((shape - 1)**2 * (shape - 2)))
        else:
            sigma = float("inf")
    else:
        mu, sigma = 0.5, 1.0

    return float(mu), float(sigma)


# ---------------------------------------------------------------------------
# _SampleParticle3D  (internal)
# ---------------------------------------------------------------------------

class _SampleParticle3D(VGroup):
    """A single glowing dot representing one drawn observation.

    Used in ``_SampleDrawAnimation``.  Starts at a position inside
    the population bar for its value, moves to the sample collection
    zone, then merges with other particles to form x̄.

    Parameters
    ----------
    start_pos : np.ndarray
        Where the particle originates (centre of the source bar).
    end_pos : np.ndarray
        Where the particle lands in the sample zone.
    color : ManimColor
    radius : float
    """

    def __init__(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        color: ManimColor = ManimColor("#FFD700"),
        radius: float = 0.06,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.start_pos = np.array(start_pos, dtype=float)
        self.end_pos = np.array(end_pos, dtype=float)
        self.dot = Dot3D(point=start_pos, radius=radius, color=color)
        self.glow = Dot3D(
            point=start_pos,
            radius=radius * 2.5,
            color=_with_opacity(color, 0.15),
        )
        self.add(self.glow, self.dot)

    def animate_fall(self, run_time: float = 0.45) -> UpdateFromAlphaFunc:
        """Animate the particle moving from start to end position."""
        start = self.start_pos.copy()
        end = self.end_pos.copy()
        # Arc: rise slightly then fall toward end
        mid = (start + end) / 2 + np.array([0, 0, 0.6])

        def updater(mob: _SampleParticle3D, alpha: float) -> None:
            t = smooth(alpha)
            # Quadratic Bezier
            p = (1 - t)**2 * start + 2 * (1 - t) * t * mid + t**2 * end
            mob.dot.move_to(p)
            mob.glow.move_to(p)

        return UpdateFromAlphaFunc(self, updater, run_time=run_time)


# ---------------------------------------------------------------------------
# _VarianceAnnotation3D  (internal)
# ---------------------------------------------------------------------------

class _VarianceAnnotation3D(VGroup):
    """Floating annotation showing σ/√n and the current n value.

    Used in Phase 4 (n-sweep).  The annotation updates live as n changes.

    Parameters
    ----------
    mu : float
        Theoretical mean.
    sigma : float
        Population standard deviation.
    n : int
        Current sample size.
    position : np.ndarray
        3D scene position.
    scene : ThreeDScene or None
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        n: int,
        position: np.ndarray = ORIGIN,
        color: ManimColor = WHITE,
        font_size: int = 22,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._mu = mu
        self._sigma = sigma
        self._n = n
        self._pos = np.array(position, dtype=float)
        self._color = color
        self._font_size = font_size
        self._scene = scene
        self._labels: List[Text] = []
        self._build()

    def _se(self) -> float:
        return self._sigma / np.sqrt(max(self._n, 1))

    def _build(self) -> None:
        """Construct the text labels."""
        se = self._se()
        lines = [
            f"n = {self._n}",
            f"σ/√n = {se:.3f}",
            f"μ = {self._mu:.3f}",
        ]
        for i, txt in enumerate(lines):
            lbl = Text(txt, font_size=self._font_size, color=self._color)
            lbl.move_to(self._pos + np.array([0, 0, -i * 0.32]))
            self._labels.append(lbl)
            self.add(lbl)
        if self._scene is not None:
            for lbl in self._labels:
                self._scene.add_fixed_orientation_mobjects(lbl)

    def update_n(
        self,
        new_n: int,
        scene: ThreeDScene,
        run_time: float = 0.35,
    ) -> AnimationGroup:
        """Animate the annotation updating to a new n value.

        Returns an ``AnimationGroup`` that fades out the old labels
        and fades in new ones.
        """
        self._n = new_n
        old_labels = list(self._labels)

        # Remove old
        self._labels.clear()
        for lbl in old_labels:
            self.remove(lbl)

        # Build new
        se = self._se()
        new_lines = [
            f"n = {self._n}",
            f"σ/√n = {se:.3f}",
            f"μ = {self._mu:.3f}",
        ]
        new_label_mobs = []
        for i, txt in enumerate(new_lines):
            lbl = Text(txt, font_size=self._font_size, color=self._color)
            lbl.move_to(self._pos + np.array([0, 0, -i * 0.32]))
            new_label_mobs.append(lbl)
            self._labels.append(lbl)
            self.add(lbl)
        if self._scene is not None:
            for lbl in new_label_mobs:
                self._scene.add_fixed_orientation_mobjects(lbl)

        return AnimationGroup(
            *[FadeOut(lbl, run_time=run_time * 0.5) for lbl in old_labels],
            *[FadeIn(lbl, run_time=run_time) for lbl in new_label_mobs],
        )


# ---------------------------------------------------------------------------
# PopulationDistribution3D
# ---------------------------------------------------------------------------

class PopulationDistribution3D(VGroup):
    """3D histogram of the population distribution.

    Displays the source distribution as a bar histogram (not a PDF curve)
    so the student clearly sees its non-normal shape.  An optional PDF
    overlay shows the theoretical density.

    Visual layers
    ~~~~~~~~~~~~~
    - ``bars``    : VGroup of 3-faced prism bars (front/right/top shaded).
    - ``pdf_line``: VMobject PDF curve drawn over the bars (optional).
    - ``title``   : floating label naming the distribution.
    - ``x_axis``  : tick marks along the base.

    Parameters
    ----------
    cfg : CLTConfig
        The full demo config.  Uses ``cfg.source``, ``cfg.pop_params``,
        ``cfg.pop_x``, ``cfg.hist_width``, ``cfg.hist_depth``,
        ``cfg.z_scale``, ``cfg.n_bins``, ``cfg.pop_color``.
    n_population : int
        Number of samples to draw for the histogram (large → smooth).
    show_pdf : bool
        If True, draw a theoretical PDF curve over the bars.
    scene : ThreeDScene or None
    """

    # Source → human-readable name
    SOURCE_NAMES: Dict[str, str] = {
        "uniform":     "Uniform distribution",
        "exponential": "Exponential distribution",
        "bimodal":     "Bimodal mixture",
        "bernoulli":   "Bernoulli distribution",
        "pareto":      "Pareto distribution",
    }

    def __init__(
        self,
        cfg: CLTConfig,
        n_population: int = 5000,
        show_pdf: bool = True,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.cfg = cfg
        self._scene = scene

        rng = np.random.default_rng(cfg.rng_seed + 9999)
        sampler = _make_sampler(cfg.source, cfg.pop_params, rng)
        pop_data = sampler(n_population)

        x_min, x_max = _population_range(cfg.source, cfg.pop_params)
        # Clamp data to visible range
        pop_data = np.clip(pop_data, x_min, x_max)

        # Histogram
        counts, bin_edges = np.histogram(pop_data, bins=cfg.n_bins,
                                          range=(x_min, x_max))
        counts = counts / counts.max()   # normalise to [0,1]

        bar_w = (x_max - x_min) / cfg.n_bins
        scene_bar_w = cfg.hist_width / cfg.n_bins

        self.bars = VGroup()
        self._bin_centers_scene: List[float] = []
        self._bar_heights: List[float] = []

        col_base = cfg.pop_color
        col_right = _darken(col_base, 0.60)
        col_top   = _lighten(col_base, 1.30)

        for i, count in enumerate(counts):
            h = max(count * cfg.z_scale * 4.0, 0.01)
            bin_center_data = bin_edges[i] + bar_w / 2
            # Map data x to scene x
            t = (bin_center_data - x_min) / (x_max - x_min)
            scene_x = cfg.pop_x + (t - 0.5) * cfg.hist_width

            self._bin_centers_scene.append(scene_x)
            self._bar_heights.append(h)

            # Build 3-faced prism bar
            hw = scene_bar_w / 2 * 0.88   # small gap between bars
            hd = cfg.hist_depth / 2
            x, y = scene_x, 0.0
            z0, z1 = 0.0, h

            face_front = Polygon(
                np.array([x - hw, y - hd, z0]),
                np.array([x + hw, y - hd, z0]),
                np.array([x + hw, y - hd, z1]),
                np.array([x - hw, y - hd, z1]),
                fill_color=_with_opacity(col_base, 0.88),
                fill_opacity=1.0, stroke_width=0,
            )
            face_right = Polygon(
                np.array([x + hw, y - hd, z0]),
                np.array([x + hw, y + hd, z0]),
                np.array([x + hw, y + hd, z1]),
                np.array([x + hw, y - hd, z1]),
                fill_color=_with_opacity(col_right, 0.85),
                fill_opacity=1.0, stroke_width=0,
            )
            face_top = Polygon(
                np.array([x - hw, y - hd, z1]),
                np.array([x + hw, y - hd, z1]),
                np.array([x + hw, y + hd, z1]),
                np.array([x - hw, y + hd, z1]),
                fill_color=_with_opacity(col_top, 0.80),
                fill_opacity=1.0, stroke_width=0,
            )
            bar = VGroup(face_front, face_right, face_top)
            self.bars.add(bar)

        self.add(self.bars)

        # PDF overlay curve
        self.pdf_line = VGroup()
        if show_pdf:
            self.pdf_line = self._build_pdf_curve(x_min, x_max, pop_data)
            self.add(self.pdf_line)

        # Distribution title
        if cfg.show_source_label:
            self._build_title(scene)

        # X-axis tick marks
        self._build_x_axis(x_min, x_max, scene)

    # ------------------------------------------------------------------

    def _build_pdf_curve(
        self,
        x_min: float,
        x_max: float,
        pop_data: np.ndarray,
    ) -> VMobject:
        """Build a smoothed density curve matching the histogram scale."""
        xs_data = np.linspace(x_min, x_max, 200)
        # Kernel density estimate using histogram heights as proxy
        # (avoids scipy dependency)
        bandwidth = (x_max - x_min) / 20.0

        # Gaussian KDE manually
        kde_vals = np.zeros(len(xs_data))
        step = (x_max - x_min) / self.cfg.n_bins
        for i, (h, bc) in enumerate(zip(self._bar_heights, self._bin_centers_scene)):
            # Convert scene x back to data x for this bar
            t = (bc - self.cfg.pop_x) / self.cfg.hist_width + 0.5
            bc_data = x_min + t * (x_max - x_min)
            kde_vals += h * np.exp(-0.5 * ((xs_data - bc_data) / bandwidth) ** 2)

        kde_vals = kde_vals / kde_vals.max() * max(self._bar_heights) * 1.05

        pts_3d = []
        for xd, kv in zip(xs_data, kde_vals):
            t = (xd - x_min) / (x_max - x_min)
            sx = self.cfg.pop_x + (t - 0.5) * self.cfg.hist_width
            pts_3d.append(np.array([sx, -self.cfg.hist_depth / 2, max(kv, 0.0)]))

        curve = VMobject()
        curve.set_points_as_corners(pts_3d)
        curve.set_stroke(
            color=_with_opacity(_lighten(self.cfg.pop_color, 1.50), 0.85),
            width=2.2,
        )
        curve.set_fill(opacity=0)
        return curve

    def _build_title(self, scene: Optional[ThreeDScene]) -> None:
        name = self.SOURCE_NAMES.get(self.cfg.source, self.cfg.source)
        lbl = Text(name, font_size=20, color=_with_opacity(self.cfg.pop_color, 0.80))
        lbl.move_to(np.array([self.cfg.pop_x, 0, -0.45]))
        self.title = VGroup(lbl)
        if scene is not None:
            scene.add_fixed_orientation_mobjects(lbl)
        self.add(self.title)

    def _build_x_axis(
        self,
        x_min: float,
        x_max: float,
        scene: Optional[ThreeDScene],
    ) -> None:
        """Draw a few tick marks below the histogram baseline."""
        n_ticks = 5
        self.x_axis = VGroup()
        for i in range(n_ticks + 1):
            t = i / n_ticks
            val = x_min + t * (x_max - x_min)
            sx = self.cfg.pop_x + (t - 0.5) * self.cfg.hist_width
            y = -self.cfg.hist_depth / 2

            tick = Line(
                np.array([sx, y, -0.04]),
                np.array([sx, y,  0.04]),
                color=_with_opacity(WHITE, 0.40),
                stroke_width=1.0,
            )
            lbl = Text(f"{val:.1f}", font_size=14,
                       color=_with_opacity(WHITE, 0.45))
            lbl.move_to(np.array([sx, y, -0.22]))
            self.x_axis.add(tick)
            self.x_axis.add(lbl)
            if scene is not None:
                scene.add_fixed_orientation_mobjects(lbl)
        self.add(self.x_axis)

    # ------------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------------

    def animate_build(
        self,
        lag: float = 0.018,
        run_time_per_bar: float = 0.35,
    ) -> LaggedStart:
        """Grow bars left-to-right from the baseline."""
        anims = []
        for bar in self.bars:
            # Each bar is a VGroup(front, right, top)
            anims.append(FadeIn(bar, shift=UP * 0.0, run_time=run_time_per_bar))
        return LaggedStart(*anims, lag_ratio=lag)

    def get_bar_top_position(self, bin_index: int) -> np.ndarray:
        """Return the 3D top-centre of bar at *bin_index* (for particle origin)."""
        sx = self._bin_centers_scene[bin_index]
        h  = self._bar_heights[bin_index]
        return np.array([sx, -self.cfg.hist_depth / 2, h])


# ---------------------------------------------------------------------------
# SampleMeanHistogram3D
# ---------------------------------------------------------------------------

class SampleMeanHistogram3D(VGroup):
    """A live-growing histogram of accumulated sample means (x̄ values).

    Bars are built incrementally: each call to ``add_sample_mean``
    recomputes the bin and updates the affected bar in place.

    Bars are colour-coded by distance from the theoretical mean:
    bars close to the mean use ``cfg.xbar_color_lo``,
    bars far from the mean use ``cfg.xbar_color_hi``.

    Visual layers
    ~~~~~~~~~~~~~
    - ``bars``       : Dict[int, VGroup] — one prism per active bin.
    - ``normal_curve``: VMobject — appears after ``show_normal_overlay``.
    - ``title``      : "Distribution of x̄" label.

    Parameters
    ----------
    cfg : CLTConfig
    mu : float
        Theoretical population mean.
    sigma : float
        Theoretical population standard deviation.
    scene : ThreeDScene or None
    """

    def __init__(
        self,
        cfg: CLTConfig,
        mu: float,
        sigma: float,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.cfg = cfg
        self.mu = mu
        self.sigma = sigma
        self._scene = scene
        self._se = sigma / np.sqrt(max(cfg.n, 1))

        # Determine histogram range: μ ± 4 σ/√n  (covers the x̄ distribution)
        half_range = max(self._se * 4.5, sigma * 0.5)
        self._x_min = mu - half_range
        self._x_max = mu + half_range
        self._bin_width = (self._x_max - self._x_min) / cfg.n_bins
        self._scene_bar_w = cfg.hist_width / cfg.n_bins

        # Internal state
        self._counts: np.ndarray = np.zeros(cfg.n_bins, dtype=int)
        self._max_count: int = 1
        self._bar_mobs: Dict[int, VGroup] = {}   # bin_index → prism VGroup
        self._xbar_values: List[float] = []

        self.bars = VGroup()
        self.normal_curve = VGroup()
        self.title = VGroup()

        self.add(self.bars, self.normal_curve)
        self._build_title(scene)
        self._build_mean_line()

    # ------------------------------------------------------------------

    def _data_to_scene_x(self, val: float) -> float:
        t = (val - self._x_min) / (self._x_max - self._x_min)
        return self.cfg.xbar_x + (t - 0.5) * self.cfg.hist_width

    def _bin_index(self, val: float) -> Optional[int]:
        idx = int((val - self._x_min) / self._bin_width)
        if 0 <= idx < self.cfg.n_bins:
            return idx
        return None

    def _bar_color(self, bin_index: int) -> ManimColor:
        """Colour based on distance from mean bin."""
        mu_bin = (self.mu - self._x_min) / self._bin_width
        dist = abs(bin_index - mu_bin) / (self.cfg.n_bins / 2)
        dist = min(dist, 1.0)
        return _lerp_color(self.cfg.xbar_color_lo, self.cfg.xbar_color_hi, dist)

    def _build_bar_mob(self, bin_index: int, height: float) -> VGroup:
        """Create a prism bar for *bin_index* at *height* scene units."""
        sx = self.cfg.xbar_x + (
            (bin_index + 0.5) / self.cfg.n_bins - 0.5
        ) * self.cfg.hist_width
        hw = self._scene_bar_w / 2 * 0.88
        hd = self.cfg.hist_depth / 2
        x, y = sx, 0.0
        z0, z1 = 0.0, height

        col_base  = self._bar_color(bin_index)
        col_right = _darken(col_base, 0.60)
        col_top   = _lighten(col_base, 1.28)

        face_front = Polygon(
            np.array([x - hw, y - hd, z0]),
            np.array([x + hw, y - hd, z0]),
            np.array([x + hw, y - hd, z1]),
            np.array([x - hw, y - hd, z1]),
            fill_color=_with_opacity(col_base, 0.90),
            fill_opacity=1.0, stroke_width=0,
        )
        face_right = Polygon(
            np.array([x + hw, y - hd, z0]),
            np.array([x + hw, y + hd, z0]),
            np.array([x + hw, y + hd, z1]),
            np.array([x + hw, y - hd, z1]),
            fill_color=_with_opacity(col_right, 0.88),
            fill_opacity=1.0, stroke_width=0,
        )
        face_top = Polygon(
            np.array([x - hw, y - hd, z1]),
            np.array([x + hw, y - hd, z1]),
            np.array([x + hw, y + hd, z1]),
            np.array([x - hw, y + hd, z1]),
            fill_color=_with_opacity(col_top, 0.80),
            fill_opacity=1.0, stroke_width=0,
        )
        return VGroup(face_front, face_right, face_top)

    def _update_bar(self, bin_index: int) -> None:
        """Recompute bar height for *bin_index* and update its geometry."""
        count = self._counts[bin_index]
        height = (count / self._max_count) * self.cfg.z_scale * 4.0

        sx = self.cfg.xbar_x + (
            (bin_index + 0.5) / self.cfg.n_bins - 0.5
        ) * self.cfg.hist_width
        hw = self._scene_bar_w / 2 * 0.88
        hd = self.cfg.hist_depth / 2
        x, y = sx, 0.0
        z0, z1 = 0.0, max(height, 0.01)

        col_base  = self._bar_color(bin_index)
        col_right = _darken(col_base, 0.60)
        col_top   = _lighten(col_base, 1.28)

        bar = self._bar_mobs[bin_index]
        face_front, face_right, face_top = bar[0], bar[1], bar[2]

        face_front.set_points_as_corners([
            np.array([x - hw, y - hd, z0]),
            np.array([x + hw, y - hd, z0]),
            np.array([x + hw, y - hd, z1]),
            np.array([x - hw, y - hd, z1]),
            np.array([x - hw, y - hd, z0]),
        ])
        face_right.set_points_as_corners([
            np.array([x + hw, y - hd, z0]),
            np.array([x + hw, y + hd, z0]),
            np.array([x + hw, y + hd, z1]),
            np.array([x + hw, y - hd, z1]),
            np.array([x + hw, y - hd, z0]),
        ])
        face_top.set_points_as_corners([
            np.array([x - hw, y - hd, z1]),
            np.array([x + hw, y - hd, z1]),
            np.array([x + hw, y + hd, z1]),
            np.array([x - hw, y + hd, z1]),
            np.array([x - hw, y - hd, z1]),
        ])

    def _build_title(self, scene: Optional[ThreeDScene]) -> None:
        lbl = Text("Distribution of x̄", font_size=20,
                   color=_with_opacity(self.cfg.xbar_color_lo, 0.80))
        lbl.move_to(np.array([self.cfg.xbar_x, 0, -0.45]))
        self.title = VGroup(lbl)
        if scene is not None:
            scene.add_fixed_orientation_mobjects(lbl)
        self.add(self.title)

    def _build_mean_line(self) -> None:
        """Draw a faint vertical reference line at μ."""
        sx = self._data_to_scene_x(self.mu)
        hd = self.cfg.hist_depth / 2
        self.mean_line = DashedLine(
            np.array([sx, -hd, 0.0]),
            np.array([sx, -hd, self.cfg.z_scale * 5.0]),
            dash_length=0.07, dashed_ratio=0.4,
            color=_with_opacity(WHITE, 0.28),
            stroke_width=1.2,
        )
        self.add(self.mean_line)

    # ------------------------------------------------------------------
    # Data API
    # ------------------------------------------------------------------

    def add_sample_mean(
        self,
        xbar: float,
        scene: Optional[ThreeDScene] = None,
        animate: bool = False,
        run_time: float = 0.12,
    ) -> Optional[AnimationGroup]:
        """Record one x̄ value, updating the histogram.

        Parameters
        ----------
        xbar : float
            The sample mean to add.
        animate : bool
            If True, return an animation that grows/updates the bar.
            If False, update immediately (no animation).
        run_time : float
            Duration if animated.

        Returns
        -------
        AnimationGroup or None
        """
        self._xbar_values.append(xbar)
        idx = self._bin_index(xbar)
        if idx is None:
            return None

        self._counts[idx] += 1
        self._max_count = max(self._max_count, self._counts[idx])

        # Rescale all bars if max changed
        for i in self._bar_mobs:
            self._update_bar(i)

        if idx not in self._bar_mobs:
            # New bar — create it
            height = (self._counts[idx] / self._max_count) * self.cfg.z_scale * 4.0
            bar = self._build_bar_mob(idx, max(height, 0.01))
            self._bar_mobs[idx] = bar
            self.bars.add(bar)

            if animate:
                return FadeIn(bar, run_time=run_time)
        else:
            self._update_bar(idx)

        return None

    def reset(self) -> None:
        """Clear all accumulated data and bars."""
        self._counts[:] = 0
        self._max_count = 1
        self._xbar_values.clear()
        for bar in list(self._bar_mobs.values()):
            self.bars.remove(bar)
        self._bar_mobs.clear()
        if len(self.normal_curve) > 0:
            self.remove(self.normal_curve)
            self.normal_curve = VGroup()
            self.add(self.normal_curve)

    def build_normal_overlay(self) -> VMobject:
        """Build and return a normal PDF curve scaled to match current bars.

        Call after enough trials are accumulated (≥ 30 recommended).
        """
        xbar_se = self.sigma / np.sqrt(max(self.cfg.n, 1))
        xs = np.linspace(self._x_min, self._x_max, 300)
        ys = _normal_pdf(xs, self.mu, xbar_se)

        # Scale normal to match histogram max height
        max_bar_h = (self.cfg.z_scale * 4.0) if self._max_count < 1 else (
            self.cfg.z_scale * 4.0
        )
        ys_scaled = ys / ys.max() * max_bar_h if ys.max() > 0 else ys

        pts = []
        for xd, zv in zip(xs, ys_scaled):
            sx = self._data_to_scene_x(xd)
            pts.append(np.array([sx, -self.cfg.hist_depth / 2, max(zv, 0.0)]))

        curve = VMobject()
        curve.set_points_as_corners(pts)
        col = self.cfg.normal_color
        curve.set_stroke(color=_with_opacity(col, 0.92), width=2.8)
        curve.set_fill(opacity=0)

        if self.cfg.glow_normal:
            glow = VMobject()
            glow.set_points_as_corners(pts)
            glow.set_stroke(color=_with_opacity(col, 0.15), width=9.0)
            glow.set_fill(opacity=0)
            self.normal_curve.add(glow)

        self.normal_curve.add(curve)
        return curve

    @property
    def n_samples(self) -> int:
        return len(self._xbar_values)

    @property
    def empirical_mean(self) -> float:
        if not self._xbar_values:
            return self.mu
        return float(np.mean(self._xbar_values))

    @property
    def empirical_std(self) -> float:
        if len(self._xbar_values) < 2:
            return self._se
        return float(np.std(self._xbar_values))


# ---------------------------------------------------------------------------
# NormalConvergenceOverlay3D
# ---------------------------------------------------------------------------

class NormalConvergenceOverlay3D(VGroup):
    """Animated emergence of a normal curve over the x̄ histogram.

    As n increases, the curve narrows visibly.  This class handles
    a sequence of curves at different n values, morphing between them
    to show convergence.

    Parameters
    ----------
    cfg : CLTConfig
    mu : float
    sigma : float
    hist : SampleMeanHistogram3D
        The histogram to overlay (provides coordinate mapping).
    n_values : list[int]
        Sequence of n values to show (e.g. [1, 2, 5, 10, 30]).
    """

    def __init__(
        self,
        cfg: CLTConfig,
        mu: float,
        sigma: float,
        hist: SampleMeanHistogram3D,
        n_values: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.mu = mu
        self.sigma = sigma
        self._hist = hist
        self._n_values = n_values or [1, 2, 5, 10, 30, 100]
        self._current_curve: Optional[VMobject] = None

    def _make_curve_points(self, n: int) -> np.ndarray:
        se = self.sigma / np.sqrt(max(n, 1))
        half_range = se * 5.0
        x_min = self.mu - half_range
        x_max = self.mu + half_range
        xs = np.linspace(x_min, x_max, 300)
        ys = _normal_pdf(xs, self.mu, se)
        ys_scaled = ys / ys.max() * self.cfg.z_scale * 3.8 if ys.max() > 0 else ys

        pts = []
        for xd, zv in zip(xs, ys_scaled):
            sx = self._hist._data_to_scene_x(xd)
            pts.append(np.array([sx, -self.cfg.hist_depth / 2, max(float(zv), 0.0)]))
        return np.array(pts)

    def build_for_n(self, n: int) -> VMobject:
        """Build the normal curve for a given *n*."""
        pts = self._make_curve_points(n)
        col = self.cfg.normal_color
        curve = VMobject()
        curve.set_points_as_corners(pts)
        curve.set_stroke(color=_with_opacity(col, 0.90), width=2.8)
        curve.set_fill(opacity=0)
        return curve

    def animate_narrow(
        self,
        scene: ThreeDScene,
        n_sequence: Optional[List[int]] = None,
        run_time_each: float = 1.0,
        hold_time: float = 0.5,
    ) -> None:
        """Play the sequence of narrowing curves directly on *scene*.

        For each n in *n_sequence*, morphs the current curve to the
        new one.  Call ``scene.play`` and ``scene.wait`` internally.

        Parameters
        ----------
        n_sequence : list[int] or None
            Sequence of n values.  Defaults to ``self._n_values``.
        """
        seq = n_sequence or self._n_values
        first_n = seq[0]
        first_curve = self.build_for_n(first_n)
        self.add(first_curve)
        scene.play(Create(first_curve, run_time=run_time_each * 0.6))
        self._current_curve = first_curve

        for n in seq[1:]:
            new_curve = self.build_for_n(n)
            scene.play(
                Transform(self._current_curve, new_curve, run_time=run_time_each),
            )
            scene.wait(hold_time)


# ---------------------------------------------------------------------------
# CLTDemo  — orchestrator
# ---------------------------------------------------------------------------

class CLTDemo:
    """Orchestrator for a full Central Limit Theorem demonstration.

    ``CLTDemo`` owns the configuration and all visual objects.  Call
    ``phase_*`` methods in order to build the animation incrementally,
    or call ``run(scene)`` to execute all phases automatically.

    Parameters
    ----------
    cfg : CLTConfig
        Full configuration for the demo.

    Attributes
    ----------
    cfg : CLTConfig
    population : PopulationDistribution3D
    xbar_hist : SampleMeanHistogram3D
    variance_annotation : _VarianceAnnotation3D
    formula_panel : VGroup
    mu : float
        Theoretical population mean.
    sigma : float
        Theoretical population standard deviation.
    """

    def __init__(self, cfg: CLTConfig) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(cfg.rng_seed)
        self._sampler = _make_sampler(cfg.source, cfg.pop_params, rng)
        self.mu, self.sigma = _theoretical_mean_std(cfg.source, cfg.pop_params, cfg.n)
        self._rng = rng

        # Visual objects — not yet added to any scene
        self.population: Optional[PopulationDistribution3D] = None
        self.xbar_hist: Optional[SampleMeanHistogram3D] = None
        self.variance_annotation: Optional[_VarianceAnnotation3D] = None
        self.formula_panel: VGroup = VGroup()
        self._all_xbars: List[float] = []   # pre-simulated

        # Pre-simulate all trials for deterministic animation
        self._pre_simulate()

    # ------------------------------------------------------------------
    # Pre-simulation
    # ------------------------------------------------------------------

    def _pre_simulate(self) -> None:
        """Draw all n_trials × n samples upfront for reproducibility."""
        self._all_xbars = []
        for _ in range(self.cfg.n_trials):
            sample = self._sampler(self.cfg.n)
            self._all_xbars.append(float(sample.mean()))

    # ------------------------------------------------------------------
    # Phase 0: Population
    # ------------------------------------------------------------------

    def phase_population(
        self,
        scene: ThreeDScene,
        run_time_build: float = 1.5,
        show_pdf: bool = True,
    ) -> None:
        """Show the source distribution histogram and label it non-normal.

        Parameters
        ----------
        scene : ThreeDScene
        run_time_build : float
            Duration of the histogram build animation.
        show_pdf : bool
            Whether to overlay the theoretical PDF curve.
        """
        self.population = PopulationDistribution3D(
            cfg=self.cfg,
            n_population=8000,
            show_pdf=show_pdf,
            scene=scene,
        )
        scene.add(self.population)
        scene.play(self.population.animate_build(run_time_per_bar=0.28))
        if show_pdf and len(self.population.pdf_line) > 0:
            scene.play(Create(self.population.pdf_line, run_time=0.8))
        scene.wait(0.5)

    # ------------------------------------------------------------------
    # Phase 1: Single sample draw (particle animation)
    # ------------------------------------------------------------------

    def phase_single_draw(
        self,
        scene: ThreeDScene,
        n_particles: Optional[int] = None,
        run_time_per_particle: float = 0.40,
        show_xbar_label: bool = True,
    ) -> None:
        """Animate one sample of size n being drawn from the population.

        Particles fall from population bars to a collection zone,
        then merge into the sample mean x̄.

        Parameters
        ----------
        n_particles : int or None
            Number of particles to show (default = cfg.n, capped at 15
            for visual clarity).
        """
        if self.population is None:
            raise RuntimeError("Call phase_population first.")

        n_show = min(n_particles or self.cfg.n, 15)
        sample = self._sampler(self.cfg.n)
        xbar = float(sample.mean())

        x_min, x_max = _population_range(self.cfg.source, self.cfg.pop_params)

        # Collection zone: a line below the population histogram
        collect_y = -self.cfg.hist_depth / 2
        collect_z = -0.55
        particle_mobs: List[_SampleParticle3D] = []

        # Show n_show particles falling one by one
        for i, val in enumerate(sample[:n_show]):
            # Find which bin this value lands in
            t = np.clip((val - x_min) / (x_max - x_min), 0, 1)
            bin_idx = int(t * self.cfg.n_bins)
            bin_idx = min(bin_idx, self.cfg.n_bins - 1)

            start = self.population.get_bar_top_position(bin_idx)

            # Spread particles along x in collection zone
            spread_x = self.cfg.pop_x + (t - 0.5) * self.cfg.hist_width * 0.85
            end = np.array([spread_x, collect_y, collect_z])

            p = _SampleParticle3D(
                start_pos=start,
                end_pos=end,
                color=self.cfg.particle_color,
            )
            scene.add(p)
            particle_mobs.append(p)
            scene.play(p.animate_fall(run_time=run_time_per_particle), run_time=run_time_per_particle)

        scene.wait(0.3)

        # Merge all particles into xbar point
        xbar_pos = np.array([self.cfg.pop_x, collect_y, collect_z])
        merge_anims = []
        for p in particle_mobs:
            path = VMobject()
            path.set_points_as_corners([p.end_pos, xbar_pos])
            merge_anims.append(MoveAlongPath(p, path, run_time=0.35))
        scene.play(AnimationGroup(*merge_anims))

        # Flash at xbar and show label
        merged_dot = Dot3D(point=xbar_pos, radius=0.10, color=self.cfg.particle_color)
        scene.add(merged_dot)
        scene.play(Flash(merged_dot, color=self.cfg.particle_color, flash_radius=0.35))

        if show_xbar_label:
            lbl = Text(f"x̄ = {xbar:.3f}", font_size=22, color=self.cfg.particle_color)
            lbl.move_to(xbar_pos + np.array([-0.1, 0, 0.28]))
            scene.add_fixed_orientation_mobjects(lbl)
            scene.add(lbl)
            scene.play(FadeIn(lbl, run_time=0.35))
            scene.wait(1.0)
            scene.play(FadeOut(lbl, run_time=0.3))

        # Clean up
        scene.play(*[FadeOut(p, run_time=0.25) for p in particle_mobs],
                   FadeOut(merged_dot, run_time=0.25))

    # ------------------------------------------------------------------
    # Phase 2: Accumulate x̄ values
    # ------------------------------------------------------------------

    def phase_accumulate(
        self,
        scene: ThreeDScene,
        n_visible: int = 80,
        run_time_per_bar: float = 0.10,
        batch_silent: bool = True,
    ) -> None:
        """Add x̄ trials to the histogram one by one (first n_visible animated).

        Parameters
        ----------
        n_visible : int
            Number of trials to add with visible bar updates.
            Remaining trials (up to n_trials) are added silently.
        run_time_per_bar : float
            Animation time for each visible bar addition.
        batch_silent : bool
            If True, add remaining trials all at once after the visible batch.
        """
        if self.xbar_hist is None:
            self.xbar_hist = SampleMeanHistogram3D(
                cfg=self.cfg, mu=self.mu, sigma=self.sigma, scene=scene
            )
            scene.add(self.xbar_hist)
            scene.play(FadeIn(self.xbar_hist.title, run_time=0.4))
            scene.play(Create(self.xbar_hist.mean_line, run_time=0.5))

        # Animated batch
        for i, xbar in enumerate(self._all_xbars[:n_visible]):
            anim = self.xbar_hist.add_sample_mean(xbar, scene=scene, animate=True,
                                                    run_time=run_time_per_bar)
            if anim is not None:
                scene.play(anim)
            else:
                scene.wait(run_time_per_bar * 0.3)

        # Silent batch — add remaining immediately
        if batch_silent and len(self._all_xbars) > n_visible:
            for xbar in self._all_xbars[n_visible:]:
                self.xbar_hist.add_sample_mean(xbar, animate=False)
            scene.wait(0.4)

    # ------------------------------------------------------------------
    # Phase 3: Normal convergence overlay
    # ------------------------------------------------------------------

    def phase_normal_overlay(
        self,
        scene: ThreeDScene,
        run_time: float = 1.2,
        hold: float = 1.5,
    ) -> None:
        """Draw the theoretical normal curve over the x̄ histogram.

        Requires ``phase_accumulate`` to have been called first.
        """
        if self.xbar_hist is None:
            raise RuntimeError("Call phase_accumulate first.")

        curve = self.xbar_hist.build_normal_overlay()
        if not self.xbar_hist.normal_curve in scene.mobjects:
            scene.add(self.xbar_hist.normal_curve)

        scene.play(Create(curve, run_time=run_time))
        scene.wait(hold)

        # Show σ/√n annotation beside the curve
        if self.cfg.show_variance_annotation:
            se = self.sigma / np.sqrt(max(self.cfg.n, 1))
            se_lbl = Text(
                f"σ/√n = {se:.3f}",
                font_size=20,
                color=self.cfg.normal_color,
            )
            se_lbl.move_to(np.array([
                self.cfg.xbar_x + self.cfg.hist_width * 0.55,
                -self.cfg.hist_depth / 2,
                self.cfg.z_scale * 3.5,
            ]))
            scene.add_fixed_orientation_mobjects(se_lbl)
            scene.add(se_lbl)
            scene.play(FadeIn(se_lbl, run_time=0.5))
            scene.wait(hold)

    # ------------------------------------------------------------------
    # Phase 4: n-sweep
    # ------------------------------------------------------------------

    def phase_n_sweep(
        self,
        scene: ThreeDScene,
        n_values: Optional[List[int]] = None,
        n_trials_per: int = 200,
        n_visible_per: int = 30,
        run_time_transition: float = 0.9,
    ) -> None:
        """Repeat accumulation for multiple values of n, showing convergence.

        For each n in *n_values*:
        1. Reset the x̄ histogram.
        2. Update the variance annotation.
        3. Accumulate ``n_trials_per`` sample means.
        4. Draw the normal overlay.
        5. Pause to show the fit.

        Parameters
        ----------
        n_values : list[int] or None
            Sample sizes to demonstrate (default: [1, 2, 5, 10, 30, 100]).
        n_trials_per : int
            Number of x̄ trials per n value.
        n_visible_per : int
            Visibly animated trials per n value.
        run_time_transition : float
            Transition time between n values.
        """
        sweep_ns = n_values or [1, 2, 5, 10, 30, 100]

        # Initialise histogram and annotation if not present
        if self.xbar_hist is None:
            self.xbar_hist = SampleMeanHistogram3D(
                cfg=self.cfg, mu=self.mu, sigma=self.sigma, scene=scene
            )
            scene.add(self.xbar_hist)

        if self.variance_annotation is None and self.cfg.show_variance_annotation:
            self.variance_annotation = _VarianceAnnotation3D(
                mu=self.mu,
                sigma=self.sigma,
                n=sweep_ns[0],
                position=np.array([
                    self.cfg.xbar_x + self.cfg.hist_width * 0.65,
                    -self.cfg.hist_depth / 2,
                    self.cfg.z_scale * 3.0,
                ]),
                color=WHITE,
                font_size=20,
                scene=scene,
            )
            scene.add(self.variance_annotation)

        for n in sweep_ns:
            # 1 — Reset histogram and update config
            self.xbar_hist.reset()

            old_n = self.cfg.n
            self.cfg.n = n

            # Recompute SE for this n
            self.xbar_hist._se = self.sigma / np.sqrt(max(n, 1))
            half_range = max(self.xbar_hist._se * 4.5, self.sigma * 0.5)
            self.xbar_hist._x_min = self.mu - half_range
            self.xbar_hist._x_max = self.mu + half_range
            self.xbar_hist._bin_width = (
                self.xbar_hist._x_max - self.xbar_hist._x_min
            ) / self.cfg.n_bins

            # 2 — Update annotation
            if self.variance_annotation is not None:
                scene.play(
                    self.variance_annotation.update_n(n, scene, run_time=run_time_transition)
                )

            # 3 — Re-simulate and accumulate
            rng_n = np.random.default_rng(self.cfg.rng_seed + n * 17)
            sampler_n = _make_sampler(self.cfg.source, self.cfg.pop_params, rng_n)
            xbars_n = [float(sampler_n(n).mean()) for _ in range(n_trials_per)]

            for i, xbar in enumerate(xbars_n[:n_visible_per]):
                anim = self.xbar_hist.add_sample_mean(xbar, animate=True,
                                                       run_time=0.08)
                if anim is not None:
                    scene.play(anim)
            for xbar in xbars_n[n_visible_per:]:
                self.xbar_hist.add_sample_mean(xbar, animate=False)

            scene.wait(0.3)

            # 4 — Normal overlay
            self.xbar_hist.normal_curve = VGroup()
            self.xbar_hist.add(self.xbar_hist.normal_curve)
            curve = self.xbar_hist.build_normal_overlay()
            scene.play(Create(curve, run_time=0.8))

            # 5 — Hold
            scene.wait(1.2)

            # Clean up normal overlay before next iteration
            scene.play(FadeOut(self.xbar_hist.normal_curve, run_time=0.4))
            self.xbar_hist.normal_curve = VGroup()
            self.xbar_hist.add(self.xbar_hist.normal_curve)

        self.cfg.n = old_n  # restore

    # ------------------------------------------------------------------
    # Formula panel
    # ------------------------------------------------------------------

    def build_formula_panel(
        self,
        scene: ThreeDScene,
        position: Optional[np.ndarray] = None,
    ) -> VGroup:
        """Build a floating formula panel showing the CLT statement.

        Displays:
            x̄ → N(μ, σ²/n)  as  n → ∞

        Returns the ``VGroup`` for animation control.
        """
        if position is None:
            position = np.array([0.0, -self.cfg.hist_depth / 2, 5.2])

        lines = [
            ("CLT:", 22, WHITE),
            ("x̄  →  N(μ, σ²/n)", 20, ManimColor("#4A90D9")),
            (f"μ = {self.mu:.3f}", 18, ManimColor("#E0AA40")),
            (f"σ = {self.sigma:.3f}", 18, ManimColor("#E0AA40")),
            (f"n = {self.cfg.n}", 18, ManimColor("#2DAA6E")),
            (f"σ/√n = {self.sigma/np.sqrt(max(self.cfg.n,1)):.3f}", 18,
             ManimColor("#E8593C")),
        ]

        grp = VGroup()
        for i, (txt, fs, col) in enumerate(lines):
            lbl = Text(txt, font_size=fs, color=col)
            lbl.move_to(position + np.array([0, 0, -i * 0.35]))
            grp.add(lbl)
            scene.add_fixed_orientation_mobjects(lbl)

        self.formula_panel = grp
        scene.add(grp)
        return grp

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run(
        self,
        scene: ThreeDScene,
        phases: Optional[List[str]] = None,
    ) -> None:
        """Execute all (or selected) phases in order.

        Parameters
        ----------
        phases : list[str] or None
            Subset of ``["population", "single_draw", "accumulate",
            "normal", "formula"]``.
            If None, all phases run in order.
        """
        all_phases = ["population", "single_draw", "accumulate", "normal", "formula"]
        run_phases = phases if phases is not None else all_phases

        if "population" in run_phases:
            self.phase_population(scene)
        if "single_draw" in run_phases:
            self.phase_single_draw(scene)
        if "accumulate" in run_phases:
            self.phase_accumulate(scene, n_visible=60)
        if "normal" in run_phases:
            self.phase_normal_overlay(scene)
        if "formula" in run_phases and self.cfg.show_formula_panel:
            self.build_formula_panel(scene)
            scene.play(FadeIn(self.formula_panel, run_time=0.8))
            scene.wait(2.0)


# ---------------------------------------------------------------------------
# Ready-to-render ThreeDScene subclasses
# ---------------------------------------------------------------------------

class CLTUniformScene(ThreeDScene):
    """Full CLT demo with a Uniform(0, 1) population.

    Render:  manim -pql clt_demo.py CLTUniformScene
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.04)

        cfg = CLTConfig(
            source="uniform",
            pop_params={"lo": 0.0, "hi": 1.0},
            n=10,
            n_trials=300,
            pop_x=-2.8,
            xbar_x=2.8,
            hist_width=4.2,
            z_scale=0.85,
            n_bins=20,
        )
        demo = CLTDemo(cfg)
        demo.phase_population(self)
        demo.phase_single_draw(self)
        demo.phase_accumulate(self, n_visible=60)
        demo.phase_normal_overlay(self)
        demo.build_formula_panel(self)
        self.play(FadeIn(demo.formula_panel, run_time=0.7))
        self.wait(2)


class CLTExponentialScene(ThreeDScene):
    """Full CLT demo with an Exponential(rate=1) population.

    The source is strongly right-skewed — making the normal emergence
    particularly striking.

    Render:  manim -pql clt_demo.py CLTExponentialScene
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.035)

        cfg = CLTConfig(
            source="exponential",
            pop_params={"rate": 1.0},
            n=15,
            n_trials=400,
            pop_color=ManimColor("#E8593C"),
            xbar_color_lo=ManimColor("#1ABC9C"),
            xbar_color_hi=ManimColor("#E0AA40"),
            normal_color=ManimColor("#4A90D9"),
            pop_x=-2.8,
            xbar_x=2.8,
            hist_width=4.2,
            z_scale=0.85,
            n_bins=22,
        )
        demo = CLTDemo(cfg)
        demo.phase_population(self)
        demo.phase_single_draw(self, run_time_per_particle=0.35)
        demo.phase_accumulate(self, n_visible=70)
        demo.phase_normal_overlay(self)
        demo.build_formula_panel(self)
        self.play(FadeIn(demo.formula_panel, run_time=0.7))
        self.wait(2)


class CLTBimodalScene(ThreeDScene):
    """Full CLT demo with a bimodal mixture population.

    Two well-separated peaks converge to a single normal — the most
    pedagogically dramatic case.

    Render:  manim -pql clt_demo.py CLTBimodalScene
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.04)

        cfg = CLTConfig(
            source="bimodal",
            pop_params={"mu1": -2.0, "mu2": 2.0, "std1": 0.55, "std2": 0.55},
            n=12,
            n_trials=350,
            pop_color=ManimColor("#9B59B6"),
            xbar_color_lo=ManimColor("#2DAA6E"),
            xbar_color_hi=ManimColor("#E0AA40"),
            normal_color=ManimColor("#E8593C"),
            pop_x=-3.0,
            xbar_x=3.0,
            hist_width=4.5,
            z_scale=0.80,
            n_bins=24,
        )
        demo = CLTDemo(cfg)
        demo.phase_population(self)
        demo.phase_single_draw(self, run_time_per_particle=0.38)
        demo.phase_accumulate(self, n_visible=65)
        demo.phase_normal_overlay(self)
        demo.build_formula_panel(self)
        self.play(FadeIn(demo.formula_panel, run_time=0.7))
        self.wait(2)


class CLTSweepScene(ThreeDScene):
    """n-sweep demo: x̄ histogram narrows as n grows from 1 → 100.

    Shows σ/√n → 0 visually.  Most compelling proof of the theorem.

    Render:  manim -pql clt_demo.py CLTSweepScene
    """

    def construct(self):
        self.set_camera_orientation(phi=68 * DEGREES, theta=-48 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.03)

        cfg = CLTConfig(
            source="exponential",
            pop_params={"rate": 1.0},
            n=1,
            n_trials=200,
            pop_color=ManimColor("#E8593C"),
            xbar_color_lo=ManimColor("#1ABC9C"),
            xbar_color_hi=ManimColor("#E0AA40"),
            normal_color=ManimColor("#4A90D9"),
            pop_x=-3.0,
            xbar_x=3.0,
            hist_width=4.5,
            z_scale=0.80,
            n_bins=28,
            show_variance_annotation=True,
        )
        demo = CLTDemo(cfg)

        # Show population once
        demo.phase_population(self, run_time_build=1.2)
        self.wait(0.5)

        # Then sweep n
        demo.phase_n_sweep(
            self,
            n_values=[1, 2, 5, 10, 30, 100],
            n_trials_per=250,
            n_visible_per=35,
            run_time_transition=0.8,
        )
        self.wait(2)


class CLTComparisonScene(ThreeDScene):
    """Side-by-side comparison of three source distributions.

    Builds three ``CLTDemo`` objects with staggered x-positions,
    all accumulating simultaneously to show the CLT is distribution-agnostic.

    Render:  manim -pql clt_demo.py CLTComparisonScene
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)

        sources = [
            ("uniform",     {"lo": 0.0, "hi": 1.0},
             ManimColor("#4A90D9"), -5.2),
            ("exponential", {"rate": 1.0},
             ManimColor("#E8593C"),  0.0),
            ("bimodal",     {"mu1": -1.5, "mu2": 1.5, "std1": 0.45, "std2": 0.45},
             ManimColor("#9B59B6"),  5.2),
        ]

        demos: List[CLTDemo] = []
        hists: List[SampleMeanHistogram3D] = []

        for source, params, col, x_center in sources:
            cfg = CLTConfig(
                source=source,
                pop_params=params,
                n=15,
                n_trials=250,
                pop_color=col,
                xbar_color_lo=_lighten(col, 1.30),
                xbar_color_hi=_darken(col, 0.65),
                normal_color=_lighten(col, 1.60),
                pop_x=x_center - 1.4,
                xbar_x=x_center + 1.4,
                hist_width=2.2,
                hist_depth=0.35,
                z_scale=0.75,
                n_bins=16,
                show_source_label=True,
                show_formula_panel=False,
            )
            demo = CLTDemo(cfg)
            demos.append(demo)

        # Build all three populations simultaneously
        for demo in demos:
            demo.population = PopulationDistribution3D(
                cfg=demo.cfg, n_population=6000, show_pdf=True, scene=self
            )
            self.add(demo.population)

        self.play(AnimationGroup(*[
            demo.population.animate_build(run_time_per_bar=0.22)
            for demo in demos
        ]))
        self.wait(0.5)

        # Build all three x̄ histograms simultaneously
        for demo in demos:
            demo.xbar_hist = SampleMeanHistogram3D(
                cfg=demo.cfg, mu=demo.mu, sigma=demo.sigma, scene=self
            )
            self.add(demo.xbar_hist)

        # Add trials in batches — all three in parallel
        n_visible = 40
        for i in range(n_visible):
            batch_anims = []
            for demo in demos:
                xbar = demo._all_xbars[i]
                anim = demo.xbar_hist.add_sample_mean(xbar, animate=True, run_time=0.09)
                if anim is not None:
                    batch_anims.append(anim)
            if batch_anims:
                self.play(AnimationGroup(*batch_anims))

        # Add remaining silently
        for demo in demos:
            for xbar in demo._all_xbars[n_visible:]:
                demo.xbar_hist.add_sample_mean(xbar, animate=False)
        self.wait(0.3)

        # Normal overlays for all three
        curves = []
        for demo in demos:
            curve = demo.xbar_hist.build_normal_overlay()
            self.add(demo.xbar_hist.normal_curve)
            curves.append(curve)

        self.play(AnimationGroup(*[
            Create(c, run_time=1.0) for c in curves
        ]))
        self.wait(3)