"""
manim_stats/animations/sampling.py
=====================================
Animated sampling from distributions — visualising how samples are
drawn, how sampling methods differ, and how sample statistics vary.

Design philosophy
-----------------
Sampling is abstract.  These animations make it concrete by treating
each population member as a visible 3D dot (``PopulationCloud3D``)
and showing the selection process explicitly:  dots glow, float out
of the cloud, and regroup into a sample zone.

Four selection styles are supported for every sampler class:
    ``"highlight"``  – selected dots enlarge and glow in place.
    ``"extract"``    – selected dots float from the cloud to a
                       separate sample zone on the right.
    ``"fade"``       – unselected dots dim; selected stay bright.
    ``"draw"``       – an animated sweep line moves across the cloud,
                       picking dots as it passes (good for systematic).

Sampling methods
----------------
``SimpleRandomSampling3D``
    Uniform random sample without replacement.

``StratifiedSampling3D``
    Population split into visible coloured strata; proportional
    allocation within each stratum.

``ClusterSampling3D``
    Population grouped into spatial clusters; entire clusters selected.

``SystematicSampling3D``
    Every k-th member selected; sweep-line animation makes k visible.

``BootstrapSampling3D``
    Sample drawn with replacement; duplicate selections shown with
    concentric rings; bootstrap distribution accumulated.

``SamplingDistributionBuilder``
    Generic: run any sampler m times, compute a statistic each time,
    accumulate into a live 3D histogram to show the sampling distribution.

Classes
-------
SamplingConfig
PopulationCloud3D
SampleSelector
SimpleRandomSampling3D
StratifiedSampling3D
ClusterSampling3D
SystematicSampling3D
BootstrapSampling3D
SamplingDistributionBuilder

Helpers / internals
-------------------
_PopDot3D
_SampleZone3D
_SweepLine3D
_StrataRegion3D
_ClusterRegion3D
_SampleStatAnnotation3D

Ready-to-render scenes
----------------------
SRSScene
StratifiedScene
ClusterScene
SystematicScene
BootstrapScene

Usage
-----
    # Render a stratified sampling demo:
    #   manim -pql sampling.py StratifiedScene

    # Embed in your own scene:
    from manim_stats.animations.sampling import StratifiedSampling3D

    class MyScene(ThreeDScene):
        def construct(self):
            demo = StratifiedSampling3D(N=120, n=24, n_strata=3)
            demo.phase_population(self)
            demo.phase_sample(self)
            demo.phase_statistics(self)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple, Optional, Callable, Union, Dict, Set
import numpy as np

from manim import (
    VGroup, VMobject, Polygon, Rectangle, Line, DashedLine,
    Dot, Dot3D, Text, MathTex, Ellipse, Arrow, Circle,
    ThreeDScene,
    AnimationGroup, LaggedStart, Succession,
    Create, FadeIn, FadeOut, Transform, MoveAlongPath,
    UpdateFromAlphaFunc, Flash, Indicate, GrowFromCenter,
    DEGREES, PI, TAU,
    RIGHT, LEFT, UP, DOWN, OUT, IN, ORIGIN,
    X_AXIS, Y_AXIS, Z_AXIS,
    WHITE, BLACK, GRAY, BLUE, GREEN, RED, YELLOW,
    ManimColor, color_to_rgb, rgba_to_color, color_to_rgba,
    rate_functions, smooth,
)

# ---------------------------------------------------------------------------
# Colour helpers
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
# SamplingConfig
# ---------------------------------------------------------------------------

@dataclass
class SamplingConfig:
    """Complete visual configuration for all sampling animations.

    Population display
    ~~~~~~~~~~~~~~~~~~
    ``pop_dot_radius``    : size of each population dot.
    ``pop_dot_color``     : default colour before sampling.
    ``pop_dot_opacity``   : default opacity.
    ``pop_layout``        : ``"grid"`` | ``"random"`` | ``"hex"`` — how
                            dots are arranged in the cloud.
    ``pop_width``         : x-extent of the population cloud.
    ``pop_height``        : y-extent (depth) of the cloud.
    ``pop_center``        : 3D centre of the population cloud.

    Sampling display
    ~~~~~~~~~~~~~~~~
    ``selected_color``    : colour of selected dots.
    ``selected_radius_mult`` : scale factor applied to selected dots.
    ``unselected_opacity``: opacity of unselected dots after selection.
    ``glow_radius_mult``  : radius of the glow halo around selected dots.
    ``glow_opacity``      : opacity of glow halo.
    ``selection_style``   : ``"highlight"`` | ``"extract"`` | ``"fade"`` | ``"draw"``.

    Sample zone
    ~~~~~~~~~~~
    ``sample_zone_center``: where extracted dots land.
    ``sample_zone_width`` : width of the sample zone layout.
    ``sample_zone_height``: height of the sample zone layout.
    ``show_sample_zone_box`` : draw a bounding box around the sample zone.
    ``sample_zone_color`` : colour of sample zone box.

    Statistics annotation
    ~~~~~~~~~~~~~~~~~~~~~
    ``show_stat_annotation`` : show x̄, s, n annotation after selection.
    ``stat_font_size``       : font size for stat annotations.
    ``stat_color``           : colour of stat text.
    """

    # Population
    pop_dot_radius: float = 0.07
    pop_dot_color: ManimColor = ManimColor("#4A90D9")
    pop_dot_opacity: float = 0.70
    pop_layout: str = "random"       # grid | random | hex
    pop_width: float = 5.0
    pop_height: float = 4.0
    pop_center: np.ndarray = field(default_factory=lambda: np.array([-1.5, 0.0, 0.0]))

    # Selection
    selected_color: ManimColor = ManimColor("#FFD700")
    selected_radius_mult: float = 1.55
    unselected_opacity: float = 0.18
    glow_radius_mult: float = 2.8
    glow_opacity: float = 0.20
    selection_style: str = "extract"  # highlight | extract | fade | draw

    # Sample zone
    sample_zone_center: np.ndarray = field(default_factory=lambda: np.array([3.5, 0.0, 0.0]))
    sample_zone_width: float = 2.8
    sample_zone_height: float = 3.5
    show_sample_zone_box: bool = True
    sample_zone_color: ManimColor = ManimColor("#2DAA6E")

    # Stats
    show_stat_annotation: bool = True
    stat_font_size: int = 20
    stat_color: ManimColor = WHITE

    # Animation
    lag_between_dots: float = 0.04
    run_time_per_dot: float = 0.35
    run_time_fade: float = 0.5


# ── Presets ──────────────────────────────────────────────────────────────

CLEAN_SAMPLING = SamplingConfig(
    pop_dot_radius=0.065,
    glow_opacity=0.0,
    selection_style="highlight",
    show_sample_zone_box=False,
    show_stat_annotation=False,
)

DETAILED_SAMPLING = SamplingConfig(
    pop_dot_radius=0.075,
    selected_radius_mult=1.70,
    glow_opacity=0.25,
    selection_style="extract",
    show_sample_zone_box=True,
    show_stat_annotation=True,
)


# ---------------------------------------------------------------------------
# _PopDot3D  — internal
# ---------------------------------------------------------------------------

class _PopDot3D(VGroup):
    """A single population member dot with a glow halo.

    Parameters
    ----------
    position : np.ndarray
        3D position of the dot.
    value : float
        Associated data value (used for colouring and statistics).
    index : int
        Index in the population (for selection tracking).
    color : ManimColor
    radius : float
    glow_radius_mult : float
    glow_opacity : float
    stratum : int
        Stratum label (0 = ungrouped).
    cluster : int
        Cluster label (0 = ungrouped).
    """

    def __init__(
        self,
        position: np.ndarray,
        value: float = 0.0,
        index: int = 0,
        color: ManimColor = ManimColor("#4A90D9"),
        radius: float = 0.07,
        opacity: float = 0.70,
        glow_radius_mult: float = 2.8,
        glow_opacity: float = 0.18,
        stratum: int = 0,
        cluster: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dot_value = value
        self.dot_index = index
        self.dot_stratum = stratum
        self.dot_cluster = cluster
        self._radius = radius
        self._pos = np.array(position, dtype=float)

        self.glow = Dot3D(
            point=self._pos,
            radius=radius * glow_radius_mult,
            color=_with_opacity(color, glow_opacity),
        )
        self.dot = Dot3D(
            point=self._pos,
            radius=radius,
            color=_with_opacity(color, opacity),
        )
        self.add(self.glow, self.dot)

    def select(
        self,
        selected_color: ManimColor,
        radius_mult: float = 1.55,
        glow_mult: float = 2.8,
        glow_opacity: float = 0.25,
    ) -> AnimationGroup:
        """Return an animation that transforms this dot to its selected state."""
        new_dot_color = _with_opacity(selected_color, 0.95)
        new_glow_color = _with_opacity(selected_color, glow_opacity)
        return AnimationGroup(
            self.dot.animate.set_color(new_dot_color)
                .scale(radius_mult),
            self.glow.animate.set_color(new_glow_color)
                .scale(glow_mult / 2.8),
            run_time=0.25,
        )

    def deselect(
        self,
        original_color: ManimColor,
        unselected_opacity: float = 0.18,
    ) -> AnimationGroup:
        """Dim this dot as an unselected member."""
        return AnimationGroup(
            self.dot.animate.set_color(_with_opacity(original_color, unselected_opacity)),
            self.glow.animate.set_color(_with_opacity(original_color, 0.0)),
            run_time=0.20,
        )

    def move_to_sample_zone(
        self,
        target: np.ndarray,
        run_time: float = 0.45,
    ) -> UpdateFromAlphaFunc:
        """Arc the dot from its population position to the sample zone."""
        start = self._pos.copy()
        end = np.array(target, dtype=float)
        mid = (start + end) / 2 + np.array([0, 0, 0.8])

        def updater(mob: _PopDot3D, alpha: float) -> None:
            t = smooth(alpha)
            pos = (1 - t)**2 * start + 2 * (1 - t) * t * mid + t**2 * end
            mob.dot.move_to(pos)
            mob.glow.move_to(pos)

        return UpdateFromAlphaFunc(self, updater, run_time=run_time)


# ---------------------------------------------------------------------------
# PopulationCloud3D
# ---------------------------------------------------------------------------

class PopulationCloud3D(VGroup):
    """A cloud of N 3D dots representing a finite population.

    Each dot corresponds to one population member and stores a
    ``value`` attribute (the measurement of interest), a ``stratum``
    label, and a ``cluster`` label.

    Layout algorithms
    ~~~~~~~~~~~~~~~~~
    ``"grid"``    – evenly spaced rectangular grid; clearest spatial
                    structure, good for systematic sampling demo.
    ``"random"``  – uniform random scatter with a small z-jitter for
                    depth, giving a natural cloud appearance.
    ``"hex"``     – hexagonal close-pack; maximises visible dot density.

    Parameters
    ----------
    N : int
        Population size.
    values : np.ndarray or None
        Data values for each member.  If None, drawn from N(0, 1).
    strata : np.ndarray or None
        Integer stratum label for each member (0-based).
        Shape (N,).  If None all members are in stratum 0.
    clusters : np.ndarray or None
        Integer cluster label for each member (0-based).
        Shape (N,).  If None all members are in cluster 0.
    config : SamplingConfig
    rng_seed : int
    scene : ThreeDScene or None

    Attributes
    ----------
    dots : List[_PopDot3D]
        Direct access to each dot object.
    dot_group : VGroup
        All dots as a single group.
    N : int
    values : np.ndarray
    strata : np.ndarray
    clusters : np.ndarray
    """

    # Palette for stratum / cluster colouring
    _STRATUM_COLORS: List[ManimColor] = [
        ManimColor("#4A90D9"),  # blue
        ManimColor("#E8593C"),  # coral
        ManimColor("#2DAA6E"),  # emerald
        ManimColor("#E0AA40"),  # amber
        ManimColor("#9B59B6"),  # purple
        ManimColor("#1ABC9C"),  # teal
    ]

    def __init__(
        self,
        N: int = 100,
        values: Optional[np.ndarray] = None,
        strata: Optional[np.ndarray] = None,
        clusters: Optional[np.ndarray] = None,
        config: Optional[SamplingConfig] = None,
        rng_seed: int = 42,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.N = N
        self.cfg = config if config is not None else SamplingConfig()
        self._scene = scene
        rng = np.random.default_rng(rng_seed)

        # Values
        self.values = (
            values.copy() if values is not None
            else rng.normal(0.0, 1.0, N).astype(float)
        )

        # Stratum labels
        self.strata = (
            strata.copy().astype(int) if strata is not None
            else np.zeros(N, dtype=int)
        )
        self.n_strata = int(self.strata.max()) + 1

        # Cluster labels
        self.clusters = (
            clusters.copy().astype(int) if clusters is not None
            else np.zeros(N, dtype=int)
        )
        self.n_clusters = int(self.clusters.max()) + 1

        # Generate positions
        positions = self._generate_positions(rng)

        # Build dots
        self.dots: List[_PopDot3D] = []
        self.dot_group = VGroup()

        for i in range(N):
            s = int(self.strata[i])
            color = self._STRATUM_COLORS[s % len(self._STRATUM_COLORS)]

            dot = _PopDot3D(
                position=positions[i],
                value=float(self.values[i]),
                index=i,
                color=color,
                radius=self.cfg.pop_dot_radius,
                opacity=self.cfg.pop_dot_opacity,
                glow_radius_mult=self.cfg.glow_radius_mult,
                glow_opacity=self.cfg.glow_opacity,
                stratum=s,
                cluster=int(self.clusters[i]),
            )
            self.dots.append(dot)
            self.dot_group.add(dot)

        self.add(self.dot_group)

    # ------------------------------------------------------------------

    def _generate_positions(self, rng: np.random.Generator) -> np.ndarray:
        """Generate N 3D positions according to ``cfg.pop_layout``."""
        cfg = self.cfg
        cx, cy, cz = cfg.pop_center
        hw = cfg.pop_width / 2
        hh = cfg.pop_height / 2
        N = self.N

        if cfg.pop_layout == "grid":
            cols = int(np.ceil(np.sqrt(N * cfg.pop_width / cfg.pop_height)))
            rows = int(np.ceil(N / cols))
            xs = np.linspace(cx - hw, cx + hw, cols)
            ys = np.linspace(cy - hh, cy + hh, rows)
            pts = []
            for r in range(rows):
                for c in range(cols):
                    if len(pts) < N:
                        z = cz + rng.uniform(-0.05, 0.05)
                        pts.append([xs[c], ys[r], z])
            return np.array(pts[:N])

        elif cfg.pop_layout == "hex":
            # Hexagonal packing
            cols = int(np.ceil(np.sqrt(N * cfg.pop_width / cfg.pop_height)))
            rows = int(np.ceil(N / cols))
            dx = cfg.pop_width / cols
            dy = cfg.pop_height / rows
            pts = []
            for r in range(rows):
                for c in range(cols):
                    if len(pts) < N:
                        x_off = dx / 2 if r % 2 else 0.0
                        x = cx - hw + c * dx + x_off
                        y = cy - hh + r * dy
                        z = cz + rng.uniform(-0.04, 0.04)
                        pts.append([x, y, z])
            return np.array(pts[:N])

        else:  # "random" — default
            xs = rng.uniform(cx - hw, cx + hw, N)
            ys = rng.uniform(cy - hh, cy + hh, N)
            zs = rng.uniform(cz - 0.12, cz + 0.12, N)
            return np.column_stack([xs, ys, zs])

    # ------------------------------------------------------------------

    def animate_appear(
        self,
        lag: float = 0.008,
        run_time_per: float = 0.18,
    ) -> LaggedStart:
        """Fade all dots in with a stagger."""
        return LaggedStart(
            *[FadeIn(d, scale=0.3, run_time=run_time_per)
              for d in self.dots],
            lag_ratio=lag,
        )

    def color_by_value(
        self,
        lo_color: ManimColor = ManimColor("#0C4478"),
        hi_color: ManimColor = ManimColor("#A32D2D"),
    ) -> AnimationGroup:
        """Recolour all dots according to their value (low=lo_color, high=hi_color)."""
        lo, hi = float(self.values.min()), float(self.values.max())
        span = hi - lo if hi != lo else 1.0
        anims = []
        for dot in self.dots:
            t = (dot.dot_value - lo) / span
            col = _lerp_color(lo_color, hi_color, t)
            anims.append(
                dot.dot.animate.set_color(_with_opacity(col, self.cfg.pop_dot_opacity))
            )
        return AnimationGroup(*anims, run_time=0.6)

    def get_indices_by_stratum(self, s: int) -> List[int]:
        return [i for i in range(self.N) if self.strata[i] == s]

    def get_indices_by_cluster(self, c: int) -> List[int]:
        return [i for i in range(self.N) if self.clusters[i] == c]


# ---------------------------------------------------------------------------
# _SampleZone3D  — internal
# ---------------------------------------------------------------------------

class _SampleZone3D(VGroup):
    """The landing zone where extracted sample dots are displayed.

    Draws an optional bounding box and a "Sample" label.

    Parameters
    ----------
    center : np.ndarray
    width, height : float
    color : ManimColor
    label : str
    scene : ThreeDScene or None
    """

    def __init__(
        self,
        center: np.ndarray = ORIGIN,
        width: float = 2.8,
        height: float = 3.5,
        color: ManimColor = ManimColor("#2DAA6E"),
        label: str = "Sample",
        show_box: bool = True,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        cx, cy, cz = center
        hw, hh = width / 2, height / 2
        col = _with_opacity(color, 0.60)

        if show_box:
            # Draw a 3D bounding box (4 vertical edges + 4 base edges)
            corners_bot = [
                np.array([cx - hw, cy - hh, cz]),
                np.array([cx + hw, cy - hh, cz]),
                np.array([cx + hw, cy + hh, cz]),
                np.array([cx - hw, cy + hh, cz]),
            ]
            corners_top = [p + np.array([0, 0, height]) for p in corners_bot]

            edges = VGroup()
            for i in range(4):
                j = (i + 1) % 4
                edges.add(Line(corners_bot[i], corners_bot[j], color=col, stroke_width=1.0))
                edges.add(Line(corners_top[i], corners_top[j], color=col, stroke_width=1.0))
                edges.add(Line(corners_bot[i], corners_top[i], color=col, stroke_width=0.7))
            self.box = edges
            self.add(edges)

        # Label
        lbl = Text(label, font_size=20, color=color)
        lbl.move_to(center + np.array([0, -hh - 0.30, height / 2]))
        self._label = VGroup(lbl)
        if scene is not None:
            scene.add_fixed_orientation_mobjects(lbl)
        self.add(self._label)

    def grid_positions(
        self,
        n: int,
        width: float,
        height: float,
        center: np.ndarray,
    ) -> List[np.ndarray]:
        """Return n evenly-spaced positions inside the zone."""
        cx, cy, cz = center
        hw, hh = width / 2, height / 2
        cols = int(np.ceil(np.sqrt(n * width / height)))
        rows = int(np.ceil(n / cols))
        xs = np.linspace(cx - hw * 0.8, cx + hw * 0.8, max(cols, 1))
        ys = np.linspace(cy - hh * 0.8, cy + hh * 0.8, max(rows, 1))
        pts = []
        for r in range(rows):
            for c in range(cols):
                if len(pts) < n:
                    pts.append(np.array([xs[c], ys[r], cz + height * 0.2]))
        return pts


# ---------------------------------------------------------------------------
# _SweepLine3D  — internal
# ---------------------------------------------------------------------------

class _SweepLine3D(VGroup):
    """A vertical sweep line that travels across the population cloud.

    Used by ``SystematicSampling3D`` to visually show the selection
    interval k.

    Parameters
    ----------
    x_start, x_end : float
        Horizontal travel range.
    y_min, y_max : float
        Vertical extent of the line.
    z : float
        Z position.
    color : ManimColor
    """

    def __init__(
        self,
        x_start: float,
        x_end: float,
        y_min: float,
        y_max: float,
        z: float = 0.0,
        color: ManimColor = ManimColor("#E0AA40"),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.x_start = x_start
        self.x_end = x_end
        col = _with_opacity(color, 0.65)

        self.line = Line(
            np.array([x_start, y_min, z]),
            np.array([x_start, y_max, z]),
            color=col, stroke_width=2.0,
        )
        self.glow_line = Line(
            np.array([x_start, y_min, z]),
            np.array([x_start, y_max, z]),
            color=_with_opacity(color, 0.15), stroke_width=7.0,
        )
        self.add(self.glow_line, self.line)

    def animate_sweep(
        self,
        run_time: float = 2.0,
    ) -> UpdateFromAlphaFunc:
        """Animate the line sweeping from x_start to x_end."""
        x0, x1 = self.x_start, self.x_end

        def updater(mob: _SweepLine3D, alpha: float) -> None:
            x = x0 + (x1 - x0) * smooth(alpha)
            mob.line.put_start_and_end_on(
                mob.line.get_start() * 0 + np.array([x, mob.line.get_start()[1], mob.line.get_start()[2]]),
                mob.line.get_end()   * 0 + np.array([x, mob.line.get_end()[1],   mob.line.get_end()[2]]),
            )
            mob.glow_line.put_start_and_end_on(
                np.array([x, mob.glow_line.get_start()[1], mob.glow_line.get_start()[2]]),
                np.array([x, mob.glow_line.get_end()[1],   mob.glow_line.get_end()[2]]),
            )

        return UpdateFromAlphaFunc(self, updater, run_time=run_time)


# ---------------------------------------------------------------------------
# _StrataRegion3D  — internal
# ---------------------------------------------------------------------------

class _StrataRegion3D(VGroup):
    """A visual bounding box around one stratum in the population cloud.

    Parameters
    ----------
    dots : list of _PopDot3D
        Dots belonging to this stratum.
    color : ManimColor
    label : str
    padding : float
    scene : ThreeDScene or None
    """

    def __init__(
        self,
        dots: List[_PopDot3D],
        color: ManimColor = ManimColor("#4A90D9"),
        label: str = "",
        padding: float = 0.20,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not dots:
            return

        positions = np.array([d._pos for d in dots])
        x_min, y_min = positions[:, 0].min() - padding, positions[:, 1].min() - padding
        x_max, y_max = positions[:, 0].max() + padding, positions[:, 1].max() + padding
        z = float(positions[:, 2].mean())

        col = _with_opacity(color, 0.30)
        fill_col = _with_opacity(color, 0.05)

        # Flat rectangle in the xy-plane at z
        rect = Polygon(
            np.array([x_min, y_min, z]),
            np.array([x_max, y_min, z]),
            np.array([x_max, y_max, z]),
            np.array([x_min, y_max, z]),
            fill_color=fill_col,
            fill_opacity=1.0,
            stroke_color=col,
            stroke_width=1.4,
        )
        self.rect = rect
        self.add(rect)

        if label:
            lbl = Text(label, font_size=16, color=_with_opacity(color, 0.80))
            lbl.move_to(np.array([
                (x_min + x_max) / 2,
                y_min - 0.25,
                z,
            ]))
            self._label = VGroup(lbl)
            if scene is not None:
                scene.add_fixed_orientation_mobjects(lbl)
            self.add(self._label)


# ---------------------------------------------------------------------------
# _ClusterRegion3D  — internal
# ---------------------------------------------------------------------------

class _ClusterRegion3D(VGroup):
    """A visual bounding circle / box around one cluster.

    Parameters
    ----------
    dots : list of _PopDot3D
    color : ManimColor
    label : str
    scene : ThreeDScene or None
    """

    def __init__(
        self,
        dots: List[_PopDot3D],
        color: ManimColor = ManimColor("#4A90D9"),
        label: str = "",
        padding: float = 0.22,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not dots:
            return

        positions = np.array([d._pos for d in dots])
        centre = positions.mean(axis=0)
        r = float(np.max(np.linalg.norm(positions - centre, axis=1))) + padding
        z = float(centre[2])

        col = _with_opacity(color, 0.35)

        # Draw circle approximated as polygon in xy-plane
        n = 32
        angles = np.linspace(0, TAU, n, endpoint=False)
        pts = [centre + np.array([r * np.cos(a), r * np.sin(a), 0]) for a in angles]
        ring = Polygon(
            *pts,
            fill_color=_with_opacity(color, 0.06),
            fill_opacity=1.0,
            stroke_color=col,
            stroke_width=1.5,
        )
        self.ring = ring
        self.add(ring)

        if label:
            lbl = Text(label, font_size=15, color=_with_opacity(color, 0.75))
            lbl.move_to(centre + np.array([0, -r - 0.22, 0]))
            self._label = VGroup(lbl)
            if scene is not None:
                scene.add_fixed_orientation_mobjects(lbl)
            self.add(self._label)


# ---------------------------------------------------------------------------
# _SampleStatAnnotation3D  — internal
# ---------------------------------------------------------------------------

class _SampleStatAnnotation3D(VGroup):
    """Floating annotation showing sample statistics after selection.

    Displays: n, x̄ (mean), s (std dev), and optionally the population
    mean μ for comparison.

    Parameters
    ----------
    values : np.ndarray
        Values of the selected sample members.
    pop_mean : float or None
        True population mean for comparison annotation.
    position : np.ndarray
    config : SamplingConfig
    scene : ThreeDScene or None
    """

    def __init__(
        self,
        values: np.ndarray,
        pop_mean: Optional[float] = None,
        position: np.ndarray = ORIGIN,
        config: Optional[SamplingConfig] = None,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        cfg = config if config is not None else SamplingConfig()

        n = len(values)
        xbar = float(values.mean()) if n > 0 else 0.0
        s = float(values.std(ddof=1)) if n > 1 else 0.0

        lines = [
            (f"n = {n}",              ManimColor("#AABBCC")),
            (f"x̄ = {xbar:.3f}",     cfg.stat_color),
            (f"s = {s:.3f}",          cfg.stat_color),
        ]
        if pop_mean is not None:
            bias = xbar - pop_mean
            lines.append((f"μ = {pop_mean:.3f}", ManimColor("#E0AA40")))
            lines.append((f"bias = {bias:+.3f}", ManimColor("#E8593C") if abs(bias) > 0.1 else ManimColor("#2DAA6E")))

        pos = np.array(position, dtype=float)
        for i, (txt, col) in enumerate(lines):
            lbl = Text(txt, font_size=cfg.stat_font_size, color=col)
            lbl.move_to(pos + np.array([0, 0, -i * 0.32]))
            self.add(lbl)
            if scene is not None:
                scene.add_fixed_orientation_mobjects(lbl)


# ---------------------------------------------------------------------------
# SampleSelector  — core animation engine
# ---------------------------------------------------------------------------

class SampleSelector:
    """Animate the selection of a subset of dots from a population cloud.

    This is the engine used by all sampling method classes.  It handles
    the four selection styles and the extract-to-sample-zone logic.

    Parameters
    ----------
    population : PopulationCloud3D
    selected_indices : list of int
        Indices of dots to select.
    config : SamplingConfig
    scene : ThreeDScene or None
    """

    def __init__(
        self,
        population: PopulationCloud3D,
        selected_indices: List[int],
        config: Optional[SamplingConfig] = None,
        scene: Optional[ThreeDScene] = None,
    ):
        self.pop = population
        self.selected = list(selected_indices)
        self.cfg = config if config is not None else population.cfg
        self._scene = scene
        self._selected_set: Set[int] = set(selected_indices)

        # Sample zone (built lazily)
        self._zone: Optional[_SampleZone3D] = None
        self._zone_positions: Optional[List[np.ndarray]] = None

    # ------------------------------------------------------------------

    def build_sample_zone(
        self,
        scene: ThreeDScene,
        label: str = "Sample",
    ) -> _SampleZone3D:
        """Add the sample zone bounding box to the scene."""
        cfg = self.cfg
        zone = _SampleZone3D(
            center=cfg.sample_zone_center,
            width=cfg.sample_zone_width,
            height=cfg.sample_zone_height,
            color=cfg.sample_zone_color,
            label=label,
            show_box=cfg.show_sample_zone_box,
            scene=scene,
        )
        scene.add(zone)
        scene.play(FadeIn(zone, run_time=0.4))
        self._zone = zone

        # Pre-compute target positions in sample zone
        n = len(self.selected)
        dummy_zone = _SampleZone3D.__new__(_SampleZone3D)
        self._zone_positions = zone.grid_positions(
            n,
            cfg.sample_zone_width,
            cfg.sample_zone_height,
            cfg.sample_zone_center,
        )
        return zone

    def animate_highlight(
        self,
        scene: ThreeDScene,
        lag: float = 0.04,
    ) -> None:
        """Highlight selected dots in place; dim unselected."""
        cfg = self.cfg

        # Dim all first
        dim_anims = [
            self.pop.dots[i].deselect(
                self.pop.cfg.pop_dot_color,
                cfg.unselected_opacity,
            )
            for i in range(self.pop.N)
            if i not in self._selected_set
        ]
        if dim_anims:
            scene.play(AnimationGroup(*dim_anims, run_time=cfg.run_time_fade))

        # Highlight selected with stagger
        select_anims = [
            self.pop.dots[i].select(
                cfg.selected_color,
                cfg.selected_radius_mult,
                cfg.glow_radius_mult,
                cfg.glow_opacity * 1.5,
            )
            for i in self.selected
        ]
        scene.play(LaggedStart(*select_anims, lag_ratio=lag))

    def animate_extract(
        self,
        scene: ThreeDScene,
        lag: float = 0.05,
    ) -> None:
        """Extract selected dots, arc them to the sample zone."""
        cfg = self.cfg

        if self._zone_positions is None:
            self.build_sample_zone(scene)

        # Dim unselected first
        dim_anims = [
            self.pop.dots[i].deselect(
                self.pop.cfg.pop_dot_color,
                cfg.unselected_opacity,
            )
            for i in range(self.pop.N)
            if i not in self._selected_set
        ]
        if dim_anims:
            scene.play(AnimationGroup(*dim_anims, run_time=cfg.run_time_fade))

        # Glow-up selected, then arc to zone
        select_anims = [
            self.pop.dots[i].select(
                cfg.selected_color,
                cfg.selected_radius_mult,
                cfg.glow_radius_mult,
                cfg.glow_opacity * 1.4,
            )
            for i in self.selected
        ]
        scene.play(LaggedStart(*select_anims, lag_ratio=lag * 0.5))

        # Arc each selected dot to its target position
        arc_anims = []
        for j, idx in enumerate(self.selected):
            target = self._zone_positions[j] if self._zone_positions else cfg.sample_zone_center
            arc_anims.append(
                self.pop.dots[idx].move_to_sample_zone(
                    target, run_time=cfg.run_time_per_dot
                )
            )
        scene.play(LaggedStart(*arc_anims, lag_ratio=lag))

    def animate_fade(
        self,
        scene: ThreeDScene,
    ) -> None:
        """Fade unselected to near-invisible; selected stay bright."""
        cfg = self.cfg
        dim_anims = [
            self.pop.dots[i].deselect(
                self.pop.cfg.pop_dot_color,
                cfg.unselected_opacity,
            )
            for i in range(self.pop.N)
            if i not in self._selected_set
        ]
        bright_anims = [
            self.pop.dots[i].select(
                cfg.selected_color,
                cfg.selected_radius_mult,
            )
            for i in self._selected_set
        ]
        scene.play(
            AnimationGroup(
                AnimationGroup(*dim_anims),
                AnimationGroup(*bright_anims),
                run_time=cfg.run_time_fade,
            )
        )

    def animate(
        self,
        scene: ThreeDScene,
        style: Optional[str] = None,
        zone_label: str = "Sample",
    ) -> None:
        """Dispatch to the correct animation style.

        Parameters
        ----------
        style : str or None
            Override config style.  ``"highlight"``, ``"extract"``,
            ``"fade"``, ``"draw"``.
        """
        s = style or self.cfg.selection_style
        if s == "extract":
            self.build_sample_zone(scene, label=zone_label)
            self.animate_extract(scene)
        elif s == "highlight":
            self.animate_highlight(scene)
        else:
            self.animate_fade(scene)


# ---------------------------------------------------------------------------
# SimpleRandomSampling3D
# ---------------------------------------------------------------------------

class SimpleRandomSampling3D:
    """Animate simple random sampling without replacement.

    Phase sequence:
    1. ``phase_population`` — build and show the population cloud.
    2. ``phase_sample``     — select n dots uniformly at random.
    3. ``phase_statistics`` — annotate with x̄, s, n.

    Parameters
    ----------
    N : int
        Population size.
    n : int
        Sample size.
    values : np.ndarray or None
        Population values.  If None drawn from N(0,1).
    config : SamplingConfig
    rng_seed : int
    scene : ThreeDScene or None
    """

    def __init__(
        self,
        N: int = 100,
        n: int = 20,
        values: Optional[np.ndarray] = None,
        config: Optional[SamplingConfig] = None,
        rng_seed: int = 42,
        scene: Optional[ThreeDScene] = None,
    ):
        self.N = N
        self.n = n
        self.cfg = config if config is not None else DETAILED_SAMPLING
        self._scene = scene

        rng = np.random.default_rng(rng_seed)
        self.population = PopulationCloud3D(
            N=N, values=values, config=self.cfg,
            rng_seed=rng_seed, scene=scene,
        )
        self._sample_idx: List[int] = list(
            rng.choice(N, size=n, replace=False)
        )
        self._selector: Optional[SampleSelector] = None

    def phase_population(
        self,
        scene: ThreeDScene,
        run_time: float = 1.2,
        title: bool = True,
    ) -> None:
        """Add population cloud to scene and animate its appearance."""
        scene.add(self.population)
        scene.play(
            self.population.animate_appear(
                lag=0.008, run_time_per=0.15
            ),
            run_time=run_time,
        )
        if title:
            lbl = Text(
                f"Population  N = {self.N}",
                font_size=20,
                color=_with_opacity(self.cfg.pop_dot_color, 0.70),
            )
            lbl.move_to(self.cfg.pop_center + np.array([0, -self.cfg.pop_height / 2 - 0.35, 0]))
            scene.add_fixed_orientation_mobjects(lbl)
            scene.add(lbl)
            scene.play(FadeIn(lbl, run_time=0.3))
        scene.wait(0.4)

    def phase_sample(
        self,
        scene: ThreeDScene,
        style: Optional[str] = None,
    ) -> None:
        """Animate the random selection of n members."""
        self._selector = SampleSelector(
            self.population, self._sample_idx, self.cfg, scene
        )
        self._selector.animate(scene, style=style, zone_label=f"SRS  n = {self.n}")

    def phase_statistics(
        self,
        scene: ThreeDScene,
        position: Optional[np.ndarray] = None,
    ) -> None:
        """Annotate the sample with x̄, s, n."""
        if not self.cfg.show_stat_annotation:
            return
        sample_values = self.population.values[self._sample_idx]
        pos = position if position is not None else (
            self.cfg.sample_zone_center + np.array([0, 0, 2.0])
        )
        ann = _SampleStatAnnotation3D(
            values=sample_values,
            pop_mean=float(self.population.values.mean()),
            position=pos,
            config=self.cfg,
            scene=scene,
        )
        scene.add(ann)
        scene.play(FadeIn(ann, run_time=0.5))
        scene.wait(0.8)

    def run(self, scene: ThreeDScene, style: Optional[str] = None) -> None:
        self.phase_population(scene)
        self.phase_sample(scene, style=style)
        self.phase_statistics(scene)


# ---------------------------------------------------------------------------
# StratifiedSampling3D
# ---------------------------------------------------------------------------

class StratifiedSampling3D:
    """Animate proportional stratified sampling.

    Population is divided into ``n_strata`` visible strata (coloured
    bounding regions).  Samples are drawn proportionally from each
    stratum.  Shows clearly that each stratum contributes n × (N_h/N)
    members.

    Parameters
    ----------
    N : int
    n : int
        Total sample size.
    n_strata : int
        Number of strata (2–6).
    stratum_sizes : list of int or None
        Size of each stratum.  If None, N is divided equally.
    config : SamplingConfig
    rng_seed : int
    scene : ThreeDScene or None
    """

    def __init__(
        self,
        N: int = 120,
        n: int = 30,
        n_strata: int = 3,
        stratum_sizes: Optional[List[int]] = None,
        config: Optional[SamplingConfig] = None,
        rng_seed: int = 42,
        scene: Optional[ThreeDScene] = None,
    ):
        self.N = N
        self.n = n
        self.n_strata = n_strata
        self.cfg = config if config is not None else DETAILED_SAMPLING
        self._scene = scene

        rng = np.random.default_rng(rng_seed)

        # Build strata labels
        if stratum_sizes is not None:
            assert sum(stratum_sizes) == N
            sizes = stratum_sizes
        else:
            base = N // n_strata
            sizes = [base] * n_strata
            sizes[-1] += N - sum(sizes)

        strata_arr = np.concatenate([
            np.full(sz, s, dtype=int) for s, sz in enumerate(sizes)
        ])
        # Shuffle so strata are spatially mixed (then spatial layout will
        # separate them by y-band — override positions to be banded)
        rng.shuffle(strata_arr)

        # Values: each stratum has a different mean
        strata_means = np.linspace(-1.5, 1.5, n_strata)
        values = np.array([
            rng.normal(strata_means[s], 0.6)
            for s in strata_arr
        ])

        # Override layout to "grid" with y-bands per stratum
        cfg_copy = SamplingConfig(**self.cfg.__dict__)
        cfg_copy.pop_layout = "grid"

        self.population = PopulationCloud3D(
            N=N, values=values, strata=strata_arr,
            config=cfg_copy, rng_seed=rng_seed, scene=scene,
        )

        # Reposition dots into horizontal bands (one y-band per stratum)
        self._reposition_by_stratum(sizes)

        # Proportional allocation
        nh_list = [max(1, round(n * sz / N)) for sz in sizes]
        # Adjust last to hit total n exactly
        nh_list[-1] = n - sum(nh_list[:-1])

        # Draw proportional samples
        self._sample_idx: List[int] = []
        for s in range(n_strata):
            s_indices = self.population.get_indices_by_stratum(s)
            nh = nh_list[s]
            chosen = list(rng.choice(s_indices, size=min(nh, len(s_indices)), replace=False))
            self._sample_idx.extend(chosen)

        self._nh_list = nh_list
        self._sizes = sizes
        self._strata_regions: List[_StrataRegion3D] = []

    def _reposition_by_stratum(self, sizes: List[int]) -> None:
        """Move dots so each stratum occupies a horizontal y-band."""
        cfg = self.cfg
        cx, cy, cz = cfg.pop_center
        band_height = cfg.pop_height / self.n_strata
        for s in range(self.n_strata):
            indices = self.population.get_indices_by_stratum(s)
            y_center = cy - cfg.pop_height / 2 + (s + 0.5) * band_height
            for i, idx in enumerate(indices):
                dot = self.population.dots[idx]
                old_pos = dot._pos.copy()
                # Spread within band
                x_frac = (i % max(1, int(np.ceil(len(indices) ** 0.5)))) / max(1, int(np.ceil(len(indices) ** 0.5)) - 1 + 1e-9)
                new_x = cx - cfg.pop_width / 2 + x_frac * cfg.pop_width
                new_y = y_center + (i // max(1, int(np.ceil(len(indices) ** 0.5))) - 1) * 0.25
                new_pos = np.array([new_x, new_y, old_pos[2]])
                dot._pos = new_pos
                dot.dot.move_to(new_pos)
                dot.glow.move_to(new_pos)

    def phase_population(self, scene: ThreeDScene) -> None:
        scene.add(self.population)
        scene.play(self.population.animate_appear(lag=0.007, run_time_per=0.12))
        scene.wait(0.3)

    def phase_show_strata(
        self,
        scene: ThreeDScene,
        palette: Optional[List[ManimColor]] = None,
    ) -> None:
        """Draw bounding boxes around each stratum with labels."""
        colors = palette or PopulationCloud3D._STRATUM_COLORS
        self._strata_regions.clear()

        for s in range(self.n_strata):
            indices = self.population.get_indices_by_stratum(s)
            dots = [self.population.dots[i] for i in indices]
            col = colors[s % len(colors)]
            region = _StrataRegion3D(
                dots=dots, color=col,
                label=f"Stratum {s+1}  N={self._sizes[s]}",
                scene=scene,
            )
            self._strata_regions.append(region)
            scene.add(region)

        scene.play(LaggedStart(
            *[FadeIn(r, run_time=0.35) for r in self._strata_regions],
            lag_ratio=0.15,
        ))
        scene.wait(0.5)

    def phase_sample(self, scene: ThreeDScene) -> None:
        """Select proportional samples from each stratum, extract to zone."""
        selector = SampleSelector(
            self.population, self._sample_idx, self.cfg, scene
        )
        selector.build_sample_zone(
            scene, label=f"Stratified  n = {self.n}"
        )
        selector.animate_extract(scene, lag=0.04)

    def phase_statistics(self, scene: ThreeDScene) -> None:
        sample_values = self.population.values[np.array(self._sample_idx)]
        pos = self.cfg.sample_zone_center + np.array([0, 0, 2.2])
        ann = _SampleStatAnnotation3D(
            values=sample_values,
            pop_mean=float(self.population.values.mean()),
            position=pos, config=self.cfg, scene=scene,
        )
        scene.add(ann)
        scene.play(FadeIn(ann, run_time=0.5))
        scene.wait(0.8)

    def run(self, scene: ThreeDScene) -> None:
        self.phase_population(scene)
        self.phase_show_strata(scene)
        self.phase_sample(scene)
        self.phase_statistics(scene)


# ---------------------------------------------------------------------------
# ClusterSampling3D
# ---------------------------------------------------------------------------

class ClusterSampling3D:
    """Animate cluster sampling.

    Population grouped into ``n_clusters`` spatial clusters.  A random
    subset of ``n_selected_clusters`` clusters is chosen; all members of
    selected clusters form the sample.

    Parameters
    ----------
    N : int
    n_clusters : int
        Total number of clusters.
    n_selected_clusters : int
        Number of clusters to select.
    config : SamplingConfig
    rng_seed : int
    scene : ThreeDScene or None
    """

    def __init__(
        self,
        N: int = 120,
        n_clusters: int = 8,
        n_selected_clusters: int = 3,
        config: Optional[SamplingConfig] = None,
        rng_seed: int = 42,
        scene: Optional[ThreeDScene] = None,
    ):
        self.N = N
        self.n_clusters = n_clusters
        self.n_selected = n_selected_clusters
        self.cfg = config if config is not None else DETAILED_SAMPLING
        self._scene = scene

        rng = np.random.default_rng(rng_seed)

        # Assign cluster labels
        clusters_arr = np.repeat(np.arange(n_clusters), N // n_clusters)
        while len(clusters_arr) < N:
            clusters_arr = np.append(clusters_arr, n_clusters - 1)
        clusters_arr = clusters_arr[:N]
        rng.shuffle(clusters_arr)

        values = rng.normal(0, 1, N).astype(float)

        # Arrange clusters in a rough grid of circle-blobs
        cluster_centers = self._cluster_grid_centers(n_clusters)

        self.population = PopulationCloud3D(
            N=N, values=values, clusters=clusters_arr,
            config=self.cfg, rng_seed=rng_seed, scene=scene,
        )
        # Reposition dots near cluster centres
        self._reposition_by_cluster(clusters_arr, cluster_centers, rng)

        # Select clusters
        chosen_clusters = list(rng.choice(n_clusters, size=n_selected_clusters, replace=False))
        self._chosen_clusters: List[int] = chosen_clusters
        self._sample_idx: List[int] = [
            i for i in range(N) if int(clusters_arr[i]) in set(chosen_clusters)
        ]

        self._cluster_regions: List[_ClusterRegion3D] = []

    def _cluster_grid_centers(self, k: int) -> List[np.ndarray]:
        cfg = self.cfg
        cx, cy = cfg.pop_center[0], cfg.pop_center[1]
        cols = int(np.ceil(np.sqrt(k)))
        rows = int(np.ceil(k / cols))
        xs = np.linspace(cx - cfg.pop_width * 0.4, cx + cfg.pop_width * 0.4, cols)
        ys = np.linspace(cy - cfg.pop_height * 0.4, cy + cfg.pop_height * 0.4, rows)
        centres = []
        for r in range(rows):
            for c in range(cols):
                if len(centres) < k:
                    centres.append(np.array([xs[c], ys[r], cfg.pop_center[2]]))
        return centres

    def _reposition_by_cluster(
        self,
        clusters_arr: np.ndarray,
        centers: List[np.ndarray],
        rng: np.random.Generator,
    ) -> None:
        r_spread = 0.5
        for i, dot in enumerate(self.population.dots):
            c = int(clusters_arr[i])
            centre = centers[c]
            angle = rng.uniform(0, TAU)
            rad = rng.uniform(0, r_spread)
            new_pos = centre + np.array([rad * np.cos(angle), rad * np.sin(angle), 0])
            dot._pos = new_pos
            dot.dot.move_to(new_pos)
            dot.glow.move_to(new_pos)

    def phase_population(self, scene: ThreeDScene) -> None:
        scene.add(self.population)
        scene.play(self.population.animate_appear(lag=0.008, run_time_per=0.13))
        scene.wait(0.3)

    def phase_show_clusters(self, scene: ThreeDScene) -> None:
        colors = PopulationCloud3D._STRATUM_COLORS
        self._cluster_regions.clear()
        for c in range(self.n_clusters):
            indices = self.population.get_indices_by_cluster(c)
            dots = [self.population.dots[i] for i in indices]
            col = colors[c % len(colors)]
            region = _ClusterRegion3D(
                dots=dots, color=col,
                label=f"C{c+1}",
                scene=scene,
            )
            self._cluster_regions.append(region)
            scene.add(region)

        scene.play(LaggedStart(
            *[FadeIn(r, run_time=0.30) for r in self._cluster_regions],
            lag_ratio=0.10,
        ))
        scene.wait(0.5)

    def phase_select_clusters(self, scene: ThreeDScene) -> None:
        """Highlight selected clusters and dim rejected ones."""
        cfg = self.cfg
        colors = PopulationCloud3D._STRATUM_COLORS
        chosen_set = set(self._chosen_clusters)

        for c, region in enumerate(self._cluster_regions):
            if c not in chosen_set:
                scene.play(region.animate.set_opacity(0.10), run_time=0.3)
            else:
                col = colors[c % len(colors)]
                scene.play(
                    Flash(region, color=col, flash_radius=0.6, run_time=0.4),
                    region.animate.set_stroke(
                        color=_with_opacity(col, 0.90), width=2.5
                    ),
                )

    def phase_sample(self, scene: ThreeDScene) -> None:
        selector = SampleSelector(
            self.population, self._sample_idx, self.cfg, scene
        )
        n_sample = len(self._sample_idx)
        selector.build_sample_zone(
            scene, label=f"Cluster sample  n = {n_sample}"
        )
        selector.animate_extract(scene, lag=0.03)

    def run(self, scene: ThreeDScene) -> None:
        self.phase_population(scene)
        self.phase_show_clusters(scene)
        self.phase_select_clusters(scene)
        self.phase_sample(scene)


# ---------------------------------------------------------------------------
# SystematicSampling3D
# ---------------------------------------------------------------------------

class SystematicSampling3D:
    """Animate systematic (every k-th) sampling.

    Population is arranged in a ``"grid"`` layout so the selection
    interval k is visually obvious.  A random start r ∈ [0, k) is
    chosen; then every k-th dot from r is selected.

    The sweep-line animation ``phase_sweep`` shows the interval k
    explicitly before revealing the selected dots.

    Parameters
    ----------
    N : int
    n : int
        Sample size (k = N // n).
    config : SamplingConfig
    rng_seed : int
    """

    def __init__(
        self,
        N: int = 100,
        n: int = 20,
        config: Optional[SamplingConfig] = None,
        rng_seed: int = 42,
        scene: Optional[ThreeDScene] = None,
    ):
        self.N = N
        self.n = n
        self.k = max(1, N // n)
        self.cfg = config if config is not None else DETAILED_SAMPLING
        self._scene = scene

        rng = np.random.default_rng(rng_seed)
        self.r = int(rng.integers(0, self.k))   # random start
        self._sample_idx = list(range(self.r, N, self.k))[:n]

        values = rng.normal(0, 1, N)

        # Force grid layout
        cfg_grid = SamplingConfig(**self.cfg.__dict__)
        cfg_grid.pop_layout = "grid"

        self.population = PopulationCloud3D(
            N=N, values=values, config=cfg_grid,
            rng_seed=rng_seed, scene=scene,
        )

    def phase_population(self, scene: ThreeDScene) -> None:
        scene.add(self.population)
        scene.play(self.population.animate_appear(lag=0.006, run_time_per=0.12))

        # k annotation
        k_lbl = Text(f"k = N/n = {self.N}/{self.n} = {self.k}",
                     font_size=20, color=ManimColor("#E0AA40"))
        k_lbl.move_to(self.cfg.pop_center + np.array([0, -self.cfg.pop_height / 2 - 0.38, 0]))
        scene.add_fixed_orientation_mobjects(k_lbl)
        scene.add(k_lbl)
        scene.play(FadeIn(k_lbl, run_time=0.4))
        scene.wait(0.4)

    def phase_sweep(self, scene: ThreeDScene) -> None:
        """Animate a sweep line moving across the grid with step k."""
        cfg = self.cfg
        cx = cfg.pop_center[0]
        y_min = cfg.pop_center[1] - cfg.pop_height / 2
        y_max = cfg.pop_center[1] + cfg.pop_height / 2

        # Build k interval markers as vertical dashed lines
        cols = int(np.ceil(np.sqrt(self.N * cfg.pop_width / cfg.pop_height)))
        dx = cfg.pop_width / cols
        k_width = dx * self.k

        sweep = _SweepLine3D(
            x_start=cx - cfg.pop_width / 2,
            x_end=cx + cfg.pop_width / 2,
            y_min=y_min, y_max=y_max,
            z=cfg.pop_center[2],
            color=ManimColor("#E0AA40"),
        )
        scene.add(sweep)
        scene.play(FadeIn(sweep, run_time=0.3))
        scene.play(sweep.animate_sweep(run_time=1.8))
        scene.play(FadeOut(sweep, run_time=0.3))

    def phase_sample(self, scene: ThreeDScene) -> None:
        # Label random start
        r_lbl = Text(f"Random start r = {self.r}",
                     font_size=18, color=ManimColor("#E8593C"))
        r_lbl.move_to(self.cfg.pop_center + np.array([0, -self.cfg.pop_height / 2 - 0.70, 0]))
        scene.add_fixed_orientation_mobjects(r_lbl)
        scene.add(r_lbl)
        scene.play(FadeIn(r_lbl, run_time=0.3))

        selector = SampleSelector(
            self.population, self._sample_idx, self.cfg, scene
        )
        selector.animate(scene, style="highlight",
                         zone_label=f"Systematic n={self.n}")

    def run(self, scene: ThreeDScene) -> None:
        self.phase_population(scene)
        self.phase_sweep(scene)
        self.phase_sample(scene)


# ---------------------------------------------------------------------------
# BootstrapSampling3D
# ---------------------------------------------------------------------------

class BootstrapSampling3D:
    """Animate bootstrap resampling and build the bootstrap distribution.

    Phase sequence:
    1. ``phase_original_sample``  — show the n original sample dots.
    2. ``phase_bootstrap_draw``   — draw one bootstrap sample WITH
                                    replacement; duplicates shown with
                                    concentric rings.
    3. ``phase_accumulate``       — run B bootstrap trials, compute a
                                    statistic (default: mean), accumulate
                                    into a live histogram.
    4. ``phase_ci``               — shade the central 95% of the
                                    bootstrap distribution as the CI.

    Parameters
    ----------
    sample_values : np.ndarray
        The original sample (1-D array).
    B : int
        Number of bootstrap resamples.
    statistic : Callable[[np.ndarray], float]
        Statistic to compute on each resample.  Default: np.mean.
    config : SamplingConfig
    rng_seed : int
    scene : ThreeDScene or None
    """

    def __init__(
        self,
        sample_values: np.ndarray,
        B: int = 500,
        statistic: Optional[Callable[[np.ndarray], float]] = None,
        config: Optional[SamplingConfig] = None,
        rng_seed: int = 42,
        scene: Optional[ThreeDScene] = None,
    ):
        self.sample = np.asarray(sample_values, dtype=float)
        self.n = len(self.sample)
        self.B = B
        self.stat_fn = statistic if statistic is not None else np.mean
        self.cfg = config if config is not None else DETAILED_SAMPLING
        self._scene = scene

        rng = np.random.default_rng(rng_seed)

        # Pre-simulate all bootstrap statistics
        self._boot_stats: np.ndarray = np.array([
            self.stat_fn(rng.choice(self.sample, size=self.n, replace=True))
            for _ in range(B)
        ])

        # Original sample as a population cloud
        orig_cfg = SamplingConfig(**self.cfg.__dict__)
        orig_cfg.pop_layout = "hex"
        orig_cfg.pop_width = 3.5
        orig_cfg.pop_height = 2.5
        orig_cfg.pop_center = np.array([-2.5, 0.0, 0.0])
        orig_cfg.pop_dot_color = ManimColor("#2DAA6E")
        orig_cfg.pop_dot_opacity = 0.85

        self.sample_cloud = PopulationCloud3D(
            N=self.n,
            values=self.sample,
            config=orig_cfg,
            rng_seed=rng_seed,
            scene=scene,
        )

        # Bootstrap histogram state
        self._boot_hist_counts: np.ndarray = None
        self._boot_hist_bins: np.ndarray = None
        self._hist_bar_mobs: Dict[int, VGroup] = {}
        self._max_count: int = 1
        self._boot_hist_group = VGroup()

        self._init_bootstrap_hist()

    def _init_bootstrap_hist(self) -> None:
        lo = float(self._boot_stats.min()) - 0.1
        hi = float(self._boot_stats.max()) + 0.1
        n_bins = 20
        self._boot_bins = np.linspace(lo, hi, n_bins + 1)
        self._boot_hist_counts = np.zeros(n_bins, dtype=int)
        self._boot_bar_w = (hi - lo) / n_bins
        self._boot_chart_x = 2.5
        self._boot_chart_scene_w = 4.0
        self._boot_n_bins = n_bins
        self._boot_lo = lo
        self._boot_hi = hi

    def _boot_x_to_scene(self, val: float) -> float:
        t = (val - self._boot_lo) / (self._boot_hi - self._boot_lo)
        return self._boot_chart_x + (t - 0.5) * self._boot_chart_scene_w

    def _build_boot_bar(self, bin_idx: int, height: float) -> VGroup:
        scene_x = self._boot_x_to_scene(
            self._boot_bins[bin_idx] + self._boot_bar_w / 2
        )
        scene_bar_w = self._boot_chart_scene_w / self._boot_n_bins
        hw = scene_bar_w / 2 * 0.88
        hd = 0.35
        y, z0, z1 = 0.0, 0.0, max(height, 0.01)
        x = scene_x
        col = ManimColor("#4A90D9")
        col_r = _darken(col, 0.60)
        col_t = _lighten(col, 1.28)

        ff = Polygon(
            np.array([x-hw,y-hd,z0]), np.array([x+hw,y-hd,z0]),
            np.array([x+hw,y-hd,z1]), np.array([x-hw,y-hd,z1]),
            fill_color=_with_opacity(col, 0.88), fill_opacity=1.0, stroke_width=0,
        )
        fr = Polygon(
            np.array([x+hw,y-hd,z0]), np.array([x+hw,y+hd,z0]),
            np.array([x+hw,y+hd,z1]), np.array([x+hw,y-hd,z1]),
            fill_color=_with_opacity(col_r, 0.85), fill_opacity=1.0, stroke_width=0,
        )
        ft = Polygon(
            np.array([x-hw,y-hd,z1]), np.array([x+hw,y-hd,z1]),
            np.array([x+hw,y+hd,z1]), np.array([x-hw,y+hd,z1]),
            fill_color=_with_opacity(col_t, 0.78), fill_opacity=1.0, stroke_width=0,
        )
        return VGroup(ff, fr, ft)

    def _update_boot_bar(self, bin_idx: int) -> None:
        h = (self._boot_hist_counts[bin_idx] / max(self._max_count, 1)) * 4.0
        scene_x = self._boot_x_to_scene(
            self._boot_bins[bin_idx] + self._boot_bar_w / 2
        )
        scene_bar_w = self._boot_chart_scene_w / self._boot_n_bins
        hw = scene_bar_w / 2 * 0.88
        hd = 0.35
        x, y, z0, z1 = scene_x, 0.0, 0.0, max(h, 0.01)
        bar = self._hist_bar_mobs[bin_idx]
        for face, pts in [
            (bar[0], [[x-hw,y-hd,z0],[x+hw,y-hd,z0],[x+hw,y-hd,z1],[x-hw,y-hd,z1],[x-hw,y-hd,z0]]),
            (bar[1], [[x+hw,y-hd,z0],[x+hw,y+hd,z0],[x+hw,y+hd,z1],[x+hw,y-hd,z1],[x+hw,y-hd,z0]]),
            (bar[2], [[x-hw,y-hd,z1],[x+hw,y-hd,z1],[x+hw,y+hd,z1],[x-hw,y+hd,z1],[x-hw,y-hd,z1]]),
        ]:
            face.set_points_as_corners([np.array(p) for p in pts])

    def phase_original_sample(self, scene: ThreeDScene) -> None:
        """Show the original sample cloud."""
        scene.add(self.sample_cloud)
        scene.play(self.sample_cloud.animate_appear(lag=0.03, run_time_per=0.18))

        # x̄ annotation
        xbar = float(self.sample.mean())
        lbl = Text(f"Original sample\nn = {self.n},  x̄ = {xbar:.3f}",
                   font_size=18, color=ManimColor("#2DAA6E"))
        lbl.move_to(np.array([-2.5, 0, -1.6]))
        scene.add_fixed_orientation_mobjects(lbl)
        scene.add(lbl)
        scene.play(FadeIn(lbl, run_time=0.4))
        scene.wait(0.5)

    def phase_bootstrap_draw(
        self,
        scene: ThreeDScene,
        n_show: int = 1,
    ) -> None:
        """Animate one bootstrap draw showing replacement.

        Duplicate-selected members get concentric glow rings.
        """
        rng = np.random.default_rng(0)
        for _ in range(n_show):
            draw = rng.choice(self.n, size=self.n, replace=True)
            counts = np.bincount(draw, minlength=self.n)

            # Show multiplicity rings on duplicates
            ring_anims = []
            for idx, cnt in enumerate(counts):
                if cnt > 1:
                    dot = self.sample_cloud.dots[idx]
                    for ring_n in range(1, cnt):
                        ring_pos = dot._pos
                        r_ring = self.cfg.pop_dot_radius * (1.8 + ring_n * 0.9)
                        n_pts = 16
                        angles = np.linspace(0, TAU, n_pts, endpoint=False)
                        ring_pts = [
                            ring_pos + r_ring * np.array([np.cos(a), 0, np.sin(a)])
                            for a in angles
                        ]
                        ring = VMobject()
                        ring.set_points_as_corners(ring_pts + [ring_pts[0]])
                        ring.set_stroke(
                            color=_with_opacity(ManimColor("#FFD700"), 0.70),
                            width=1.5,
                        )
                        ring.set_fill(opacity=0)
                        scene.add(ring)
                        ring_anims.append(Create(ring, run_time=0.25))

            if ring_anims:
                scene.play(LaggedStart(*ring_anims, lag_ratio=0.05))
            scene.wait(0.5)

    def phase_accumulate(
        self,
        scene: ThreeDScene,
        n_visible: int = 50,
        run_time_per: float = 0.08,
    ) -> None:
        """Accumulate B bootstrap statistics into a histogram."""
        # Add histogram group
        scene.add(self._boot_hist_group)

        for i, stat_val in enumerate(self._boot_stats):
            idx = np.searchsorted(self._boot_bins[1:], stat_val)
            idx = min(idx, self._boot_n_bins - 1)
            self._boot_hist_counts[idx] += 1
            self._max_count = max(self._max_count, int(self._boot_hist_counts.max()))

            # Rescale existing bars
            for existing in self._hist_bar_mobs:
                self._update_boot_bar(existing)

            if idx not in self._hist_bar_mobs:
                h = (self._boot_hist_counts[idx] / self._max_count) * 4.0
                bar = self._build_boot_bar(idx, max(h, 0.01))
                self._hist_bar_mobs[idx] = bar
                self._boot_hist_group.add(bar)
                if i < n_visible:
                    scene.play(FadeIn(bar, run_time=run_time_per))
            elif i < n_visible:
                scene.wait(run_time_per * 0.3)

        scene.wait(0.4)

    def phase_ci(
        self,
        scene: ThreeDScene,
        alpha: float = 0.05,
    ) -> None:
        """Shade the (1-alpha) central bootstrap CI."""
        lo_ci = float(np.percentile(self._boot_stats, alpha / 2 * 100))
        hi_ci = float(np.percentile(self._boot_stats, (1 - alpha / 2) * 100))

        lo_x = self._boot_x_to_scene(lo_ci)
        hi_x = self._boot_x_to_scene(hi_ci)
        hd = 0.35

        ci_band = Polygon(
            np.array([lo_x, -hd, 0.0]),
            np.array([hi_x, -hd, 0.0]),
            np.array([hi_x, -hd, 4.2]),
            np.array([lo_x, -hd, 4.2]),
            fill_color=_with_opacity(ManimColor("#2DAA6E"), 0.18),
            fill_opacity=1.0, stroke_width=0,
        )
        ci_lo_line = DashedLine(
            np.array([lo_x, -hd, 0]), np.array([lo_x, -hd, 4.4]),
            dash_length=0.07, dashed_ratio=0.4,
            color=_with_opacity(ManimColor("#2DAA6E"), 0.75), stroke_width=1.8,
        )
        ci_hi_line = DashedLine(
            np.array([hi_x, -hd, 0]), np.array([hi_x, -hd, 4.4]),
            dash_length=0.07, dashed_ratio=0.4,
            color=_with_opacity(ManimColor("#2DAA6E"), 0.75), stroke_width=1.8,
        )

        ci_lbl = Text(
            f"{int((1-alpha)*100)}% CI\n[{lo_ci:.3f}, {hi_ci:.3f}]",
            font_size=18, color=ManimColor("#2DAA6E"),
        )
        ci_lbl.move_to(np.array([self._boot_chart_x, 0, 4.8]))
        scene.add_fixed_orientation_mobjects(ci_lbl)

        scene.add(ci_band, ci_lo_line, ci_hi_line, ci_lbl)
        scene.play(
            FadeIn(ci_band, run_time=0.5),
            Create(ci_lo_line, run_time=0.4),
            Create(ci_hi_line, run_time=0.4),
            FadeIn(ci_lbl, run_time=0.4),
        )
        scene.wait(1.0)

    def run(self, scene: ThreeDScene) -> None:
        self.phase_original_sample(scene)
        self.phase_bootstrap_draw(scene, n_show=1)
        self.phase_accumulate(scene, n_visible=50)
        self.phase_ci(scene)


# ---------------------------------------------------------------------------
# SamplingDistributionBuilder
# ---------------------------------------------------------------------------

class SamplingDistributionBuilder(VGroup):
    """Build and visualise the sampling distribution of any statistic.

    Runs a sampler function m times, computing a statistic each time,
    and accumulates results into a live 3D bar histogram.

    Parameters
    ----------
    sampler : Callable[[int], np.ndarray]
        Function that takes sample size n and returns a 1-D array.
    n : int
        Sample size passed to *sampler* each trial.
    m : int
        Number of sampling trials.
    statistic : Callable[[np.ndarray], float]
        Statistic to compute on each sample.
    true_value : float or None
        True population value (shown as reference line).
    chart_center : np.ndarray
    chart_width : float
    n_bins : int
    bar_color : ManimColor
    true_line_color : ManimColor
    rng_seed : int
    scene : ThreeDScene or None

    Attributes
    ----------
    stat_values : np.ndarray
        All m pre-simulated statistic values.
    """

    def __init__(
        self,
        sampler: Callable[[int], np.ndarray],
        n: int = 30,
        m: int = 500,
        statistic: Optional[Callable[[np.ndarray], float]] = None,
        true_value: Optional[float] = None,
        chart_center: np.ndarray = ORIGIN,
        chart_width: float = 5.0,
        n_bins: int = 25,
        z_scale: float = 0.80,
        bar_color: ManimColor = ManimColor("#4A90D9"),
        true_line_color: ManimColor = ManimColor("#E8593C"),
        rng_seed: int = 42,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n = n
        self.m = m
        self.true_value = true_value
        self.z_scale = z_scale
        self.bar_color = bar_color
        self.chart_center = np.array(chart_center, dtype=float)
        self.chart_width = chart_width
        self._scene = scene

        stat_fn = statistic if statistic is not None else np.mean

        rng = np.random.default_rng(rng_seed)
        self.stat_values = np.array([
            float(stat_fn(sampler(n))) for _ in range(m)
        ])

        # Histogram parameters
        lo = float(self.stat_values.min()) - 0.05
        hi = float(self.stat_values.max()) + 0.05
        self._lo, self._hi = lo, hi
        self._bins = np.linspace(lo, hi, n_bins + 1)
        self._n_bins = n_bins
        self._bin_w = (hi - lo) / n_bins
        self._scene_bar_w = chart_width / n_bins

        self._counts = np.zeros(n_bins, dtype=int)
        self._max_count = 1
        self._bar_mobs: Dict[int, VGroup] = {}

        self.bars = VGroup()
        self.add(self.bars)

        # Reference line at true value
        if true_value is not None:
            self._build_true_line(true_value, true_line_color, scene)

        # Axis
        self._build_axis(scene)

    # ------------------------------------------------------------------

    def _val_to_scene_x(self, val: float) -> float:
        t = (val - self._lo) / (self._hi - self._lo)
        return self.chart_center[0] + (t - 0.5) * self.chart_width

    def _build_true_line(
        self,
        val: float,
        color: ManimColor,
        scene: Optional[ThreeDScene],
    ) -> None:
        sx = self._val_to_scene_x(val)
        hd = 0.35
        y = self.chart_center[1]
        self.true_line = DashedLine(
            np.array([sx, y - hd, 0]),
            np.array([sx, y - hd, self.z_scale * 4.5]),
            dash_length=0.07, dashed_ratio=0.4,
            color=_with_opacity(color, 0.80), stroke_width=2.0,
        )
        lbl = Text(f"True value\n{val:.3f}", font_size=16, color=color)
        lbl.move_to(np.array([sx + 0.1, y, self.z_scale * 4.8]))
        self._true_lbl = VGroup(lbl)
        if scene is not None:
            scene.add_fixed_orientation_mobjects(lbl)
        self.add(self.true_line, self._true_lbl)

    def _build_axis(self, scene: Optional[ThreeDScene]) -> None:
        """Draw 5 tick labels along the base."""
        hd = 0.35
        y = self.chart_center[1]
        self._axis_labels = VGroup()
        for i in range(6):
            t = i / 5
            val = self._lo + t * (self._hi - self._lo)
            sx = self._val_to_scene_x(val)
            lbl = Text(f"{val:.2f}", font_size=13, color=_with_opacity(WHITE, 0.45))
            lbl.move_to(np.array([sx, y - hd, -0.24]))
            self._axis_labels.add(lbl)
            if scene is not None:
                scene.add_fixed_orientation_mobjects(lbl)
        self.add(self._axis_labels)

    def _build_bar(self, idx: int, height: float) -> VGroup:
        bin_centre = self._bins[idx] + self._bin_w / 2
        sx = self._val_to_scene_x(bin_centre)
        hw = self._scene_bar_w / 2 * 0.88
        hd = 0.35
        y = self.chart_center[1]
        x, z0, z1 = sx, 0.0, max(height, 0.01)
        col = self.bar_color
        col_r = _darken(col, 0.60)
        col_t = _lighten(col, 1.28)

        ff = Polygon(
            np.array([x-hw,y-hd,z0]), np.array([x+hw,y-hd,z0]),
            np.array([x+hw,y-hd,z1]), np.array([x-hw,y-hd,z1]),
            fill_color=_with_opacity(col, 0.90), fill_opacity=1.0, stroke_width=0,
        )
        fr = Polygon(
            np.array([x+hw,y-hd,z0]), np.array([x+hw,y+hd,z0]),
            np.array([x+hw,y+hd,z1]), np.array([x+hw,y-hd,z1]),
            fill_color=_with_opacity(col_r, 0.88), fill_opacity=1.0, stroke_width=0,
        )
        ft = Polygon(
            np.array([x-hw,y-hd,z1]), np.array([x+hw,y-hd,z1]),
            np.array([x+hw,y+hd,z1]), np.array([x-hw,y+hd,z1]),
            fill_color=_with_opacity(col_t, 0.80), fill_opacity=1.0, stroke_width=0,
        )
        return VGroup(ff, fr, ft)

    def _update_bar(self, idx: int) -> None:
        h = (self._counts[idx] / max(self._max_count, 1)) * self.z_scale * 4.0
        bin_centre = self._bins[idx] + self._bin_w / 2
        sx = self._val_to_scene_x(bin_centre)
        hw = self._scene_bar_w / 2 * 0.88
        hd = 0.35
        y = self.chart_center[1]
        x, z0, z1 = sx, 0.0, max(h, 0.01)
        bar = self._bar_mobs[idx]
        for face, pts in [
            (bar[0], [[x-hw,y-hd,z0],[x+hw,y-hd,z0],[x+hw,y-hd,z1],[x-hw,y-hd,z1],[x-hw,y-hd,z0]]),
            (bar[1], [[x+hw,y-hd,z0],[x+hw,y+hd,z0],[x+hw,y+hd,z1],[x+hw,y-hd,z1],[x+hw,y-hd,z0]]),
            (bar[2], [[x-hw,y-hd,z1],[x+hw,y-hd,z1],[x+hw,y+hd,z1],[x-hw,y+hd,z1],[x-hw,y-hd,z1]]),
        ]:
            face.set_points_as_corners([np.array(p) for p in pts])

    def add_trial(
        self,
        trial_index: int,
        animate: bool = True,
        run_time: float = 0.08,
    ) -> Optional[AnimationGroup]:
        val = float(self.stat_values[trial_index])
        idx = np.searchsorted(self._bins[1:], val)
        idx = min(int(idx), self._n_bins - 1)

        self._counts[idx] += 1
        self._max_count = max(self._max_count, int(self._counts.max()))

        for existing in self._bar_mobs:
            self._update_bar(existing)

        if idx not in self._bar_mobs:
            h = (self._counts[idx] / self._max_count) * self.z_scale * 4.0
            bar = self._build_bar(idx, max(h, 0.01))
            self._bar_mobs[idx] = bar
            self.bars.add(bar)
            if animate:
                return FadeIn(bar, run_time=run_time)
        return None

    def run_accumulation(
        self,
        scene: ThreeDScene,
        n_visible: int = 60,
        run_time_per: float = 0.08,
        show_normal_after: bool = True,
    ) -> None:
        """Animate the full accumulation directly on *scene*."""
        scene.play(FadeIn(self._axis_labels, run_time=0.35))
        if hasattr(self, "true_line"):
            scene.play(Create(self.true_line, run_time=0.5))
            scene.play(FadeIn(self._true_lbl, run_time=0.3))

        for i in range(min(n_visible, self.m)):
            anim = self.add_trial(i, animate=True, run_time=run_time_per)
            if anim is not None:
                scene.play(anim)
            else:
                scene.wait(run_time_per * 0.3)

        for i in range(n_visible, self.m):
            self.add_trial(i, animate=False)
        scene.wait(0.4)

        if show_normal_after:
            mu = float(self.stat_values.mean())
            sigma = float(self.stat_values.std())
            xs = np.linspace(self._lo, self._hi, 300)
            ys = _normal_pdf(xs, mu, sigma)
            ys_scaled = ys / ys.max() * self.z_scale * 4.0 if ys.max() > 0 else ys

            hd = 0.35
            y = self.chart_center[1]
            curve_pts = [
                np.array([self._val_to_scene_x(x), y - hd, max(float(z), 0.0)])
                for x, z in zip(xs, ys_scaled)
            ]
            curve = VMobject()
            curve.set_points_as_corners(curve_pts)
            col = ManimColor("#E8593C")
            curve.set_stroke(color=_with_opacity(col, 0.88), width=2.5)
            curve.set_fill(opacity=0)

            glow = VMobject()
            glow.set_points_as_corners(curve_pts)
            glow.set_stroke(color=_with_opacity(col, 0.13), width=8.0)
            glow.set_fill(opacity=0)

            scene.add(glow, curve)
            scene.play(Create(curve, run_time=1.0))
            scene.wait(0.8)


# ---------------------------------------------------------------------------
# Ready-to-render ThreeDScene subclasses
# ---------------------------------------------------------------------------

class SRSScene(ThreeDScene):
    """Simple random sampling demonstration.

    Render:  manim -pql sampling.py SRSScene
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.03)

        demo = SimpleRandomSampling3D(
            N=100, n=20,
            config=DETAILED_SAMPLING,
            rng_seed=42, scene=self,
        )
        demo.run(self)
        self.wait(2)


class StratifiedScene(ThreeDScene):
    """Proportional stratified sampling with 3 strata.

    Render:  manim -pql sampling.py StratifiedScene
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.025)

        demo = StratifiedSampling3D(
            N=120, n=30, n_strata=3,
            config=DETAILED_SAMPLING,
            rng_seed=42, scene=self,
        )
        demo.run(self)
        self.wait(2)


class ClusterScene(ThreeDScene):
    """Cluster sampling: 3 of 8 clusters selected.

    Render:  manim -pql sampling.py ClusterScene
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.03)

        demo = ClusterSampling3D(
            N=120, n_clusters=8, n_selected_clusters=3,
            config=DETAILED_SAMPLING,
            rng_seed=42, scene=self,
        )
        demo.run(self)
        self.wait(2)


class SystematicScene(ThreeDScene):
    """Systematic (every k-th) sampling with sweep-line animation.

    Render:  manim -pql sampling.py SystematicScene
    """

    def construct(self):
        self.set_camera_orientation(phi=62 * DEGREES, theta=-48 * DEGREES)

        demo = SystematicSampling3D(
            N=100, n=20,
            config=DETAILED_SAMPLING,
            rng_seed=42, scene=self,
        )
        demo.run(self)
        self.wait(2)


class BootstrapScene(ThreeDScene):
    """Bootstrap resampling with CI shading.

    Render:  manim -pql sampling.py BootstrapScene
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.025)

        rng = np.random.default_rng(7)
        sample = rng.normal(3.5, 1.2, 30)

        demo = BootstrapSampling3D(
            sample_values=sample,
            B=600,
            statistic=np.mean,
            config=DETAILED_SAMPLING,
            rng_seed=42, scene=self,
        )
        demo.run(self)
        self.wait(2)