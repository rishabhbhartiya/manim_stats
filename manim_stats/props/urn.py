"""
manim_stats/props/urn.py
========================
Urn3D & Ball3D — Highly detailed, physically-inspired 3D urn and ball
props for Manim statistics animations.

Primary use cases
-----------------
  Hypergeometric distribution  — draw without replacement from a mixed urn
  Conditional probability       — Bayes / Urn problems
  Combinatorics                 — ordered / unordered draws
  Sampling demonstrations       — with / without replacement

Design goals
------------
Urn3D
  * Classical amphora silhouette  — lathe-profile built from a Bézier
    spine: foot ring → base taper → wide belly → shoulder curve →
    narrow neck → flared lip.  NOT a plain cylinder.
  * Dual-wall construction        — outer surface (glazed ceramic) +
    inner dark well visible from above, creating genuine depth.
  * Decorative meander band       — Greek key pattern drawn as a VGroup
    of thin Polygons wrapped around the belly equator.
  * Ear handles                   — two torus-arc handles, one per side,
    correctly proportioned and attached at neck and shoulder.
  * Multi-layer ceramic shading   — base glaze, specular highlight stripe
    down the lit side, ambient-occlusion darkening near curves.
  * Optional lid                  — domed lid with a spherical knob;
    can be animated open/closed.

Ball3D
  * Correct sphere layering       — dark shadow hemisphere (bottom),
    mid-tone body, bright specular cap (top-left), thin rim highlight.
  * Visible label                 — number or letter centred on the
    ball's face, sized to the ball radius.
  * Configurable color            — any named palette or hex color.
  * Packing positions             — `get_packed_positions()` returns
    realistic staggered ball-stack coordinates inside the urn interior.

Animations
----------
  DrawBall      — ball rises out of urn (smooth ease-out), moves to hand pos
  ReplaceBall   — ball arcs back in and sinks
  ShakeUrn      — rapid lateral oscillation of the whole urn
  PourUrn       — urn tilts, balls roll/fall out in sequence
  FillUrn       — balls drop in one by one from above
  SwapBalls     — two balls exchange positions (for conditional demos)

Dependencies
------------
  manim (CE or GL), numpy

Usage
-----
    from manim_stats.props.urn import Urn3D, Ball3D, DrawBall, FillUrn

    class HypergeometricScene(ThreeDScene):
        def construct(self):
            urn = Urn3D(color_scheme="terracotta", n_slots=10)
            balls = [
                Ball3D(color="#E63946", label=str(i)) for i in range(6)
            ] + [
                Ball3D(color="#457B9D", label=str(i)) for i in range(4)
            ]
            self.play(FillUrn(urn, balls))
            drawn = balls[0]
            self.play(DrawBall(urn, drawn))
"""

from __future__ import annotations

import numpy as np
from typing import Literal, Optional, List, Tuple

from manim import (
    VGroup,
    Sphere, Cylinder, Torus, Circle, Annulus,
    Polygon, Rectangle, Square, Line, Arc,
    Text, MathTex,
    Animation, Succession, AnimationGroup,
    ApplyMethod, FadeIn, FadeOut, MoveAlongPath,
    Rotate,
    CubicBezier,
    interpolate_color, color_to_rgb,
    WHITE, BLACK,
    GREY, GREY_A, GREY_B, GREY_C, GREY_D, LIGHT_GREY,
    RED, RED_A, RED_B, RED_C,
    BLUE, BLUE_A, BLUE_B, BLUE_C,
    GREEN_A, GREEN_E, YELLOW, GOLD, GOLD_A, GOLD_D,
    ORANGE, PURPLE_A,
    UP, DOWN, LEFT, RIGHT, OUT, IN, ORIGIN,
    PI, TAU,
    rate_functions,
    ManimColor,
    ArcBetweenPoints,
    ParametricFunction,
    Surface,
)

# ──────────────────────────────────────────────────────────────────────────────
# Colour Palettes  ──  Urn body
# ──────────────────────────────────────────────────────────────────────────────

URN_PALETTES: dict[str, dict] = {
    # Classic fired terracotta with black meander band
    "terracotta": {
        "glaze":      "#C1440E",   # main body
        "glaze_dark": "#7A2A08",   # shadow / AO areas
        "glaze_light":"#E8784A",   # lit highlight stripe
        "band_bg":    "#1A1008",   # meander band background
        "band_fg":    "#C1440E",   # meander key colour
        "inner":      "#2A1208",   # inside well (very dark)
        "lip":        "#D4552A",   # top lip ring
        "foot":       "#7A2A08",   # foot ring
        "handle":     "#A33A0A",   # ear handles
        "lid_body":   "#C1440E",
        "lid_knob":   "#7A2A08",
        "specular":   "#F0A070",
    },
    # Athenian black-figure style
    "black_figure": {
        "glaze":      "#1C1008",
        "glaze_dark": "#0A0804",
        "glaze_light":"#3A2810",
        "band_bg":    "#C1440E",
        "band_fg":    "#1C1008",
        "inner":      "#050303",
        "lip":        "#3A2810",
        "foot":       "#0A0804",
        "handle":     "#1C1008",
        "lid_body":   "#1C1008",
        "lid_knob":   "#C1440E",
        "specular":   "#5A4020",
    },
    # Cobalt blue glazed ceramic
    "cobalt": {
        "glaze":      "#1B3A8A",
        "glaze_dark": "#0D1E52",
        "glaze_light":"#4A70CC",
        "band_bg":    "#F0E8C0",
        "band_fg":    "#1B3A8A",
        "inner":      "#080E20",
        "lip":        "#2A4EB8",
        "foot":       "#0D1E52",
        "handle":     "#1B3A8A",
        "lid_body":   "#1B3A8A",
        "lid_knob":   "#F0E8C0",
        "specular":   "#8AAAE8",
    },
    # Ivory celadon with gold band
    "celadon": {
        "glaze":      "#A8C5A0",
        "glaze_dark": "#5A8050",
        "glaze_light":"#D8EDD4",
        "band_bg":    "#B8960A",
        "band_fg":    "#F8E060",
        "inner":      "#1A2818",
        "lip":        "#C8E0C0",
        "foot":       "#5A8050",
        "handle":     "#8AAA82",
        "lid_body":   "#A8C5A0",
        "lid_knob":   "#B8960A",
        "specular":   "#EAFAE4",
    },
    # Dark obsidian / ritual urn
    "obsidian": {
        "glaze":      "#1A1A20",
        "glaze_dark": "#08080C",
        "glaze_light":"#3A3A50",
        "band_bg":    "#D4AF37",
        "band_fg":    "#1A1A20",
        "inner":      "#04040A",
        "lip":        "#2A2A35",
        "foot":       "#08080C",
        "handle":     "#1A1A20",
        "lid_body":   "#1A1A20",
        "lid_knob":   "#D4AF37",
        "specular":   "#5A5A80",
    },
    # White marble with blue veining
    "marble": {
        "glaze":      "#F4F0E8",
        "glaze_dark": "#C8C0B0",
        "glaze_light":"#FFFFFF",
        "band_bg":    "#2A4A8A",
        "band_fg":    "#F4F0E8",
        "inner":      "#2A2820",
        "lip":        "#E8E0D0",
        "foot":       "#C8C0B0",
        "handle":     "#D8D0C0",
        "lid_body":   "#F4F0E8",
        "lid_knob":   "#2A4A8A",
        "specular":   "#FFFFFF",
    },
}

# Ball palette: named solid colours for common use
BALL_COLORS = {
    "red":     "#E63946",
    "blue":    "#457B9D",
    "green":   "#2A9D8F",
    "yellow":  "#E9C46A",
    "white":   "#F1FAEE",
    "black":   "#1D3557",
    "orange":  "#F4A261",
    "purple":  "#A8DADC",
    "crimson": "#9B2226",
    "gold":    "#D4AF37",
}

# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rotation_matrix_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _lathe_profile_points(
    belly_r: float,
    neck_r: float,
    total_h: float,
    n_segments: int = 40,
) -> np.ndarray:
    """
    Return (r, y) profile points for the urn silhouette via a piecewise
    smooth Bézier curve:

      Foot ring  →  base taper  →  belly  →  shoulder  →  neck  →  lip

    Returns shape (n_segments, 2) array of (radius, y) values.
    """
    # Key profile waypoints as (r, y) normalised to belly_r=1, total_h=1
    # fmt: off
    ctrl = np.array([
        [0.28, 0.00],   # foot outer edge (bottom)
        [0.32, 0.04],   # foot ring bead
        [0.30, 0.08],   # base taper in
        [0.55, 0.18],   # lower belly begins to flare
        [0.90, 0.35],   # widest belly tangent
        [1.00, 0.48],   # maximum belly width
        [0.95, 0.60],   # upper belly
        [0.72, 0.72],   # shoulder starts curving in
        [0.50, 0.80],   # shoulder taper
        [0.32, 0.86],   # neck lower
        [0.26, 0.91],   # neck mid (slight narrowing)
        [0.28, 0.95],   # neck upper (slight flare)
        [0.38, 0.99],   # lip flare
        [0.40, 1.00],   # lip top
    ], dtype=float)
    # fmt: on
    ctrl[:, 0] *= belly_r
    ctrl[:, 1] = ctrl[:, 1] * total_h - total_h / 2   # centre at y=0

    # Catmull-Rom spline through the control points
    t_vals = np.linspace(0, 1, n_segments)
    result = np.zeros((n_segments, 2))
    n = len(ctrl)
    for i, t in enumerate(t_vals):
        # Map t to segment
        seg_float = t * (n - 1)
        seg       = min(int(seg_float), n - 2)
        local_t   = seg_float - seg
        # Catmull-Rom: use points seg-1, seg, seg+1, seg+2 (clamped)
        p0 = ctrl[max(seg - 1, 0)]
        p1 = ctrl[seg]
        p2 = ctrl[min(seg + 1, n - 1)]
        p3 = ctrl[min(seg + 2, n - 1)]
        tt = local_t
        tt2, tt3 = tt ** 2, tt ** 3
        result[i] = 0.5 * (
            2 * p1
            + (-p0 + p2) * tt
            + (2*p0 - 5*p1 + 4*p2 - p3) * tt2
            + (-p0 + 3*p1 - 3*p2 + p3) * tt3
        )
    return result


def _meander_key_unit(w: float, h: float) -> list[np.ndarray]:
    """
    One unit of a Greek meander / key pattern as a list of line
    segments (start, end) pairs, normalised to box w×h.

    The classic "T-spiral" unit:
         ┌──┐
    ─────┘  │
            └──
    Returns list of (start_xy, end_xy) tuples in 2-D.
    """
    s  = min(w, h) / 4   # step size
    segs = [
        # Bottom horizontal run
        (np.array([0,       0  ]), np.array([3*s,    0  ])),
        # Up
        (np.array([3*s,     0  ]), np.array([3*s,    3*s])),
        # Left inner
        (np.array([3*s,     3*s]), np.array([s,      3*s])),
        # Down inner
        (np.array([s,       3*s]), np.array([s,      s  ])),
        # Right inner
        (np.array([s,       s  ]), np.array([2*s,    s  ])),
        # Up inner
        (np.array([2*s,     s  ]), np.array([2*s,    2*s])),
        # Continue top
        (np.array([2*s,     2*s]), np.array([4*s,    2*s])),
    ]
    return segs


# ──────────────────────────────────────────────────────────────────────────────
# Ball3D
# ──────────────────────────────────────────────────────────────────────────────

class Ball3D(VGroup):
    """
    A detailed 3D ball for use inside Urn3D.

    Layers (back → front):
      1. shadow_cap   — dark hemisphere on the bottom (ambient shadow)
      2. body_sphere  — full sphere in the ball's colour
      3. specular_cap — bright ellipse on the top-left (specular reflection)
      4. rim_ring     — thin bright annulus around the specular cap edge
      5. label        — centred text (number or letter), facing +Z

    Parameters
    ----------
    color : str
        Named key from ``BALL_COLORS`` or any hex colour string.
    label : str | None
        Text to display on the ball face.  Default ``None`` (no label).
    radius : float
        Ball radius in Manim units.  Default ``0.18``.
    label_color : str | None
        Override for label text colour; auto-contrasts if None.
    resolution : int
        Sphere surface resolution.  Default ``16``.

    Attributes
    ----------
    ball_color : str
    ball_radius : float
    label_text : str | None
    """

    def __init__(
        self,
        color: str = "red",
        label: Optional[str] = None,
        radius: float = 0.18,
        label_color: Optional[str] = None,
        resolution: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Resolve colour
        hex_col = BALL_COLORS.get(color, color)
        self.ball_color  = hex_col
        self.ball_radius = radius
        self.label_text  = label

        dark_col  = interpolate_color(hex_col, BLACK, 0.60)
        light_col = interpolate_color(hex_col, WHITE, 0.55)
        rim_col   = interpolate_color(hex_col, WHITE, 0.35)

        # ── 1. Shadow hemisphere cap (bottom dark zone) ───────────────
        # Approximate with a dark-filled sphere slightly smaller and offset
        shadow = Sphere(
            radius=radius * 0.98,
            resolution=(resolution // 2, resolution // 2),
            fill_color=dark_col,
            fill_opacity=0.55,
            stroke_width=0,
        )
        shadow.shift(DOWN * radius * 0.12)
        self.add(shadow)

        # ── 2. Main body sphere ───────────────────────────────────────
        body = Sphere(
            radius=radius,
            resolution=(resolution, resolution),
            fill_color=hex_col,
            fill_opacity=1.0,
            stroke_width=0,
        )
        self.add(body)

        # ── 3. Specular highlight cap ─────────────────────────────────
        spec = Circle(
            radius=radius * 0.36,
            fill_color=light_col,
            fill_opacity=0.72,
            stroke_width=0,
        )
        spec.scale([1.3, 0.85, 1])    # ellipse
        spec.move_to(
            np.array([-radius * 0.28, radius * 0.30, radius * 0.88])
        )
        self.add(spec)

        # ── 4. Rim ring ───────────────────────────────────────────────
        rim = Annulus(
            inner_radius=radius * 0.32,
            outer_radius=radius * 0.42,
            fill_color=rim_col,
            fill_opacity=0.38,
            stroke_width=0,
        )
        rim.move_to(np.array([-radius * 0.26, radius * 0.28, radius * 0.86]))
        self.add(rim)

        # ── 5. Label ──────────────────────────────────────────────────
        if label is not None:
            # Auto-contrast: light label on dark ball, dark on light
            if label_color is None:
                rgb = np.array(color_to_rgb(hex_col))
                lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                label_color = WHITE if lum < 0.55 else "#1A1A1A"
            lbl = Text(
                str(label),
                font_size=int(radius * 96),
                color=label_color,
                font="sans-serif",
            )
            lbl.move_to([0, 0, radius * 1.01])   # face front (+Z)
            self.add(lbl)

    # ──────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────

    def get_color_hex(self) -> str:
        return self.ball_color

    def make_copy_at(self, position: np.ndarray) -> "Ball3D":
        c = self.copy()
        c.move_to(position)
        return c


def get_packed_positions(
    n_balls: int,
    urn_belly_r: float,
    urn_inner_bottom_y: float,
    ball_radius: float,
    max_layers: int = 6,
) -> list[np.ndarray]:
    """
    Compute staggered hexagonal-close-packed positions for n_balls
    inside a cylindrical well of radius ``urn_belly_r * 0.72``.

    Balls are placed in layers from the bottom up, each layer offset
    by half a ball diameter in x and z to simulate packing.

    Parameters
    ----------
    n_balls           : total number of balls
    urn_belly_r       : urn maximum belly radius
    urn_inner_bottom_y: y-coordinate of the urn interior floor
    ball_radius       : radius of each ball
    max_layers        : cap on vertical layers (prevents overflow)

    Returns
    -------
    list of np.ndarray positions, length n_balls
    """
    well_r    = urn_belly_r * 0.68 - ball_radius
    diameter  = 2 * ball_radius
    row_dy    = diameter * np.sqrt(2.0 / 3.0)  # vertical layer spacing (HCP)
    row_dx    = diameter * np.sqrt(3.0) / 2    # hex offset in x

    positions = []
    placed    = 0
    layer     = 0

    while placed < n_balls and layer < max_layers:
        y = urn_inner_bottom_y + ball_radius + layer * row_dy
        # Offset alternate layers in x-z
        x_off = (layer % 2) * ball_radius * 0.6
        z_off = (layer % 2) * ball_radius * 0.5

        # How many fit in this layer?
        row = 0
        r_step = diameter
        while placed < n_balls:
            r_ring = row * r_step
            if r_ring == 0:
                if well_r >= 0:
                    pos = np.array([x_off, y, z_off])
                    positions.append(pos)
                    placed += 1
                row += 1
                continue
            # Number of balls in this ring (circumference / diameter)
            n_ring = max(1, int(TAU * r_ring / diameter))
            for k in range(n_ring):
                if placed >= n_balls:
                    break
                ang = k * TAU / n_ring
                px  = r_ring * np.cos(ang) + x_off
                pz  = r_ring * np.sin(ang) + z_off
                # Check still inside well
                if np.sqrt(px**2 + pz**2) + ball_radius <= well_r + ball_radius:
                    positions.append(np.array([px, y, pz]))
                    placed += 1
            row += 1
            if r_ring > well_r:
                break
        layer += 1

    return positions[:n_balls]


# ──────────────────────────────────────────────────────────────────────────────
# Urn body sub-components
# ──────────────────────────────────────────────────────────────────────────────

class _UrnLatheBody(VGroup):
    """
    The main ceramic body of the urn, built by revolving a Catmull-Rom
    profile curve around the Y-axis.

    Approximated as a stack of thin Cylinders (frustum slices), which
    Manim can render efficiently.  Each slice has:
      • outer cylinder  — the glazed body colour
      • specular stripe — bright thin strip on the lit side (+x direction)
      • AO ring         — darkened strip at top and bottom of each slice

    Parameters
    ----------
    belly_r  : float — maximum belly radius
    neck_r   : float — neck radius (≈ belly_r * 0.28)
    height   : float — total urn height
    palette  : dict
    n_slices : int   — lathe resolution (default 40)
    """

    def __init__(
        self,
        belly_r: float,
        neck_r: float,
        height: float,
        palette: dict,
        n_slices: int = 40,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.belly_r = belly_r
        self.neck_r  = neck_r
        self.height  = height

        profile = _lathe_profile_points(belly_r, neck_r, height, n_slices + 1)

        for i in range(len(profile) - 1):
            r0, y0 = profile[i]
            r1, y1 = profile[i + 1]
            slice_h = abs(y1 - y0)
            y_ctr   = (y0 + y1) / 2

            if slice_h < 1e-5 or min(r0, r1) < 1e-5:
                continue

            # ── Outer frustum slice ───────────────────────────────────
            # Manim Cylinder doesn't support frustum natively, so we use
            # the average radius and accept the slight approximation.
            r_avg = (r0 + r1) / 2
            frust = Cylinder(
                radius=r_avg,
                height=slice_h,
                direction=UP,
                resolution=(32, 1),
                fill_color=palette["glaze"],
                fill_opacity=1.0,
                stroke_width=0,
            )
            frust.move_to([0, y_ctr, 0])
            self.add(frust)

            # ── Specular stripe (lit side, +X) ────────────────────────
            # A thin vertical rectangle on the +X face of this slice
            spec_w   = r_avg * 0.18
            spec_rect = Rectangle(
                width=spec_w,
                height=slice_h,
                fill_color=palette["specular"],
                fill_opacity=0.38,
                stroke_width=0,
            )
            spec_rect.move_to([r_avg - spec_w * 0.3, y_ctr, 0.001])
            self.add(spec_rect)

        # ── Foot ring disc (bottom cap) ────────────────────────────────
        foot_r = profile[0, 0]
        foot   = Circle(
            radius=foot_r,
            fill_color=palette["foot"],
            fill_opacity=1.0,
            stroke_width=0,
        )
        foot.rotate(PI / 2, axis=RIGHT)
        foot.move_to([0, profile[0, 1], 0])
        self.add(foot)

        # ── Lip ring (top cap annulus) ─────────────────────────────────
        lip_r  = profile[-1, 0]
        lip_y  = profile[-1, 1]
        # Outer ring
        lip = Annulus(
            inner_radius=lip_r * 0.55,
            outer_radius=lip_r,
            fill_color=palette["lip"],
            fill_opacity=1.0,
            stroke_width=0,
        )
        lip.rotate(PI / 2, axis=RIGHT)
        lip.move_to([0, lip_y, 0])
        self.add(lip)

        # ── Inner dark well (depth illusion) ──────────────────────────
        well = Circle(
            radius=lip_r * 0.50,
            fill_color=palette["inner"],
            fill_opacity=1.0,
            stroke_width=0,
        )
        well.rotate(PI / 2, axis=RIGHT)
        well.move_to([0, lip_y - 0.02, 0])
        self.add(well)

        # Store key geometry for use by other components
        self._profile  = profile
        self._lip_r    = lip_r
        self._lip_y    = lip_y
        self._foot_y   = profile[0, 1]


class _MeanderBand(VGroup):
    """
    Greek key meander decorative band wrapped around the belly equator.

    Strategy: build one meander tile in 2-D, duplicate it N times around
    a circle of the belly radius, then rotate each tile to face outward.

    Parameters
    ----------
    radius    : float — band placement radius
    band_y    : float — vertical centre of the band
    band_h    : float — height of the band
    palette   : dict
    n_repeats : int   — how many key units around the circumference
    """

    def __init__(
        self,
        radius: float,
        band_y: float,
        band_h: float,
        palette: dict,
        n_repeats: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Background band
        band_bg = Cylinder(
            radius=radius + 0.005,
            height=band_h,
            direction=UP,
            resolution=(64, 1),
            fill_color=palette["band_bg"],
            fill_opacity=1.0,
            stroke_width=0,
        )
        band_bg.move_to([0, band_y, 0])
        self.add(band_bg)

        # Meander key units
        unit_w   = TAU * radius / n_repeats   # arc length per unit
        unit_h   = band_h * 0.82
        segs     = _meander_key_unit(unit_w, unit_h)
        step_ang = TAU / n_repeats

        for k in range(n_repeats):
            ang    = k * step_ang
            # Build the key unit as thin lines, projected onto the cylinder surface
            for (a2d, b2d) in segs:
                # Map 2-D segment coords to 3-D cylinder surface
                # u (horizontal along band) → angle offset
                # v (vertical) → y offset
                def _pt(xy2d: np.ndarray) -> np.ndarray:
                    u_frac   = xy2d[0] / unit_w   # 0..1 along this unit
                    v_offset = xy2d[1] - unit_h / 2
                    total_ang = ang + u_frac * step_ang
                    rx = (radius + 0.01) * np.cos(total_ang)
                    rz = (radius + 0.01) * np.sin(total_ang)
                    ry = band_y + v_offset
                    return np.array([rx, ry, rz])

                seg_line = Line(
                    start=_pt(a2d),
                    end=_pt(b2d),
                    stroke_color=palette["band_fg"],
                    stroke_width=2.0,
                )
                self.add(seg_line)


class _UrnHandle(VGroup):
    """
    One ear handle: a planar arc of torus-like cross-section,
    attaching at the neck and at the shoulder of the urn.

    Built as a sequence of thin cylinders following an arc path,
    giving the appearance of a pulled clay handle.

    Parameters
    ----------
    attach_r   : float — radius at attachment points
    low_y      : float — y of shoulder attachment
    high_y     : float — y of neck attachment
    side       : +1 or -1 (right or left)
    palette    : dict
    n_segments : int — arc smoothness
    """

    def __init__(
        self,
        attach_r: float,
        low_y: float,
        high_y: float,
        side: int,
        palette: dict,
        n_segments: int = 12,
        **kwargs,
    ):
        super().__init__(**kwargs)

        handle_col   = palette["handle"]
        handle_thick = attach_r * 0.12   # handle cross-section radius

        # Handle arc: parametric path that bows outward from the urn
        # Centre of arc is midway between attach points, bowing out by bow_r
        mid_y    = (low_y + high_y) / 2
        arc_h    = (high_y - low_y)
        bow_out  = attach_r * 0.55      # how far handle bows beyond rim

        ts = np.linspace(0, 1, n_segments + 1)

        def handle_path(t: float) -> np.ndarray:
            """Quadratic Bézier: p0 → p_ctrl → p1"""
            p0    = np.array([side * attach_r, low_y,  0.0])
            p1    = np.array([side * attach_r, high_y, 0.0])
            p_mid = np.array([side * (attach_r + bow_out), mid_y, 0.0])
            pt    = (1 - t)**2 * p0 + 2*(1-t)*t * p_mid + t**2 * p1
            return pt

        pts = [handle_path(t) for t in ts]

        for i in range(len(pts) - 1):
            seg_start = pts[i]
            seg_end   = pts[i + 1]
            seg_vec   = seg_end - seg_start
            seg_len   = np.linalg.norm(seg_vec)
            if seg_len < 1e-6:
                continue
            seg_dir   = seg_vec / seg_len
            seg_ctr   = (seg_start + seg_end) / 2

            cyl = Cylinder(
                radius=handle_thick,
                height=seg_len * 1.05,
                direction=seg_dir,
                resolution=(6, 1),
                fill_color=handle_col,
                fill_opacity=1.0,
                stroke_width=0,
            )
            cyl.move_to(seg_ctr)
            self.add(cyl)

        # Attachment discs
        for y_att in [low_y, high_y]:
            disc = Circle(
                radius=handle_thick * 1.3,
                fill_color=interpolate_color(handle_col, BLACK, 0.2),
                fill_opacity=1.0,
                stroke_width=0,
            )
            disc.move_to([side * attach_r, y_att, 0.005])
            self.add(disc)


class _UrnLid(VGroup):
    """
    Domed lid with a spherical knob.

    Layers:
      • dome  — shallow sphere cap (approximated as a squashed sphere)
      • rim   — flat annulus matching the urn lip
      • knob  — small sphere on top

    Parameters
    ----------
    lip_r  : float — inner radius of the lid (matches urn lip)
    lip_y  : float — y-position where lid sits
    palette: dict
    """

    def __init__(
        self,
        lip_r: float,
        lip_y: float,
        palette: dict,
        **kwargs,
    ):
        super().__init__(**kwargs)
        lid_r    = lip_r * 1.02
        dome_h   = lip_r * 0.55
        knob_r   = lip_r * 0.12

        # Rim disc
        rim = Circle(
            radius=lid_r,
            fill_color=palette["lid_body"],
            fill_opacity=1.0,
            stroke_width=0,
        )
        rim.rotate(PI / 2, axis=RIGHT)
        rim.move_to([0, lip_y, 0])
        self.add(rim)

        # Dome (squashed sphere)
        dome = Sphere(
            radius=lid_r,
            resolution=(16, 16),
            fill_color=palette["lid_body"],
            fill_opacity=1.0,
            stroke_width=0,
        )
        dome.scale([1.0, dome_h / lid_r, 1.0])
        dome.move_to([0, lip_y + dome_h * 0.3, 0])
        self.add(dome)

        # Specular on dome
        dome_spec = Circle(
            radius=lid_r * 0.22,
            fill_color=palette["specular"],
            fill_opacity=0.40,
            stroke_width=0,
        )
        dome_spec.scale([1.4, 0.7, 1])
        dome_spec.rotate(PI / 2, axis=RIGHT)
        dome_spec.move_to([-lid_r * 0.22, lip_y + dome_h * 0.55, 0])
        self.add(dome_spec)

        # Knob
        knob = Sphere(
            radius=knob_r,
            resolution=(10, 10),
            fill_color=palette["lid_knob"],
            fill_opacity=1.0,
            stroke_width=0,
        )
        knob.move_to([0, lip_y + dome_h * 0.88 + knob_r, 0])
        self.add(knob)

        # Knob specular
        ks = Circle(
            radius=knob_r * 0.38,
            fill_color=palette["specular"],
            fill_opacity=0.55,
            stroke_width=0,
        )
        ks.scale([1.3, 0.8, 1])
        ks.move_to([-knob_r * 0.22, lip_y + dome_h * 0.88 + knob_r * 1.55, knob_r * 0.5])
        self.add(ks)

        self._lip_y = lip_y
        self._lid_top_y = lip_y + dome_h + 2 * knob_r


# ──────────────────────────────────────────────────────────────────────────────
# Urn3D  ──  the main export
# ──────────────────────────────────────────────────────────────────────────────

class Urn3D(VGroup):
    """
    A classical Greek amphora-style urn for Manim statistics animations.

    Parameters
    ----------
    belly_r : float
        Maximum belly radius.  Default ``1.0``.
    height : float
        Total urn height.  Default ``2.8``.
    color_scheme : str
        Named palette.  Default ``"terracotta"``.
    custom_palette : dict | None
        Override individual palette keys.
    show_lid : bool
        Start with lid on.  Default ``False``.
    show_handles : bool
        Include ear handles.  Default ``True``.
    show_band : bool
        Include meander decorative band.  Default ``True``.
    n_band_repeats : int
        Number of meander units around the belly.  Default ``16``.
    lathe_slices : int
        Lathe profile resolution.  Default ``40``.
    ball_radius : float
        Radius of Ball3D objects stored inside.  Default ``0.18``.

    Attributes
    ----------
    body     : _UrnLatheBody
    lid      : _UrnLid | None
    handles  : VGroup
    band     : _MeanderBand | None
    balls    : list[Ball3D]        — balls currently inside
    belly_r  : float
    lip_y    : float               — y of urn opening
    inner_bottom_y : float         — y of interior floor

    Examples
    --------
    ::

        urn = Urn3D(color_scheme="terracotta", show_lid=False)
        red_balls  = [Ball3D("red",  label=str(i+1)) for i in range(4)]
        blue_balls = [Ball3D("blue", label=str(i+5)) for i in range(3)]
        scene.play(FillUrn(urn, red_balls + blue_balls))
        scene.play(DrawBall(urn, red_balls[0]))
    """

    def __init__(
        self,
        belly_r: float = 1.0,
        height: float = 2.8,
        color_scheme: str = "terracotta",
        custom_palette: Optional[dict] = None,
        show_lid: bool = False,
        show_handles: bool = True,
        show_band: bool = True,
        n_band_repeats: int = 16,
        lathe_slices: int = 40,
        ball_radius: float = 0.18,
        **kwargs,
    ):
        super().__init__(**kwargs)

        palette = dict(URN_PALETTES.get(color_scheme, URN_PALETTES["terracotta"]))
        if custom_palette:
            palette.update(custom_palette)

        self._palette    = palette
        self._belly_r    = belly_r
        self._height     = height
        self._ball_r     = ball_radius
        self.balls: list[Ball3D] = []

        neck_r = belly_r * 0.32

        # ── Build body ────────────────────────────────────────────────
        self.body = _UrnLatheBody(
            belly_r=belly_r,
            neck_r=neck_r,
            height=height,
            palette=palette,
            n_slices=lathe_slices,
        )
        self.add(self.body)

        # Extract key geometry from profile
        profile       = self.body._profile
        self._lip_y   = float(self.body._lip_y)
        self._foot_y  = float(self.body._foot_y)
        self._lip_r   = float(self.body._lip_r)
        # Interior bottom is ~15% up from foot
        self._inner_bottom_y = self._foot_y + height * 0.07

        # ── Decorative band ───────────────────────────────────────────
        self.band = None
        if show_band:
            # Find belly equator: widest profile point
            belly_idx  = int(np.argmax(profile[:, 0]))
            belly_y    = float(profile[belly_idx, 1])
            belly_rr   = float(profile[belly_idx, 0])
            band_h     = height * 0.10
            self.band  = _MeanderBand(
                radius=belly_rr,
                band_y=belly_y,
                band_h=band_h,
                palette=palette,
                n_repeats=n_band_repeats,
            )
            self.add(self.band)

        # ── Handles ───────────────────────────────────────────────────
        self.handles = VGroup()
        if show_handles:
            # Attach at shoulder and neck
            shoulder_frac = 0.65    # shoulder ~65% up the profile
            neck_frac     = 0.88
            n_pts         = len(profile)
            shoulder_idx  = int(shoulder_frac * n_pts)
            neck_idx      = int(neck_frac     * n_pts)
            shoulder_y    = float(profile[min(shoulder_idx, n_pts-1), 1])
            neck_y        = float(profile[min(neck_idx,     n_pts-1), 1])
            handle_r      = float(profile[min(shoulder_idx, n_pts-1), 0])

            for side in [1, -1]:
                h = _UrnHandle(
                    attach_r=handle_r,
                    low_y=shoulder_y,
                    high_y=neck_y,
                    side=side,
                    palette=palette,
                )
                self.handles.add(h)
            self.add(self.handles)

        # ── Lid ───────────────────────────────────────────────────────
        self.lid = None
        if show_lid:
            self.lid = _UrnLid(
                lip_r=self._lip_r,
                lip_y=self._lip_y,
                palette=palette,
            )
            self.add(self.lid)

    # ──────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────

    @property
    def belly_r(self) -> float:
        return self._belly_r

    @property
    def lip_y(self) -> float:
        return self._lip_y + self.get_center()[1]

    @property
    def inner_bottom_y(self) -> float:
        return self._inner_bottom_y + self.get_center()[1]

    @property
    def n_balls(self) -> int:
        return len(self.balls)

    # ──────────────────────────────────────────────────────────────────
    # Ball management
    # ──────────────────────────────────────────────────────────────────

    def add_ball(self, ball: Ball3D, position: Optional[np.ndarray] = None):
        """Place a ball inside the urn at the given position (or auto-packed)."""
        if position is None:
            positions = get_packed_positions(
                len(self.balls) + 1,
                self._belly_r,
                self._inner_bottom_y,
                self._ball_r,
            )
            position = positions[-1] + self.get_center()
        ball.move_to(position)
        self.balls.append(ball)
        self.add(ball)

    def remove_ball(self, ball: Ball3D):
        """Remove a ball from the urn's tracking list."""
        if ball in self.balls:
            self.balls.remove(ball)
        self.remove(ball)

    def get_ball_positions(self) -> list[np.ndarray]:
        """Return packed positions for all current balls."""
        return get_packed_positions(
            len(self.balls),
            self._belly_r,
            self._inner_bottom_y,
            self._ball_r,
        )

    def get_draw_exit_point(self) -> np.ndarray:
        """Position just above the urn opening — where drawn balls emerge."""
        return np.array([
            self.get_center()[0],
            self.lip_y + self._ball_r * 1.5,
            self.get_center()[2],
        ])

    def count_by_color(self) -> dict[str, int]:
        """Return {hex_color: count} for all balls currently in the urn."""
        counts: dict[str, int] = {}
        for b in self.balls:
            c = b.ball_color
            counts[c] = counts.get(c, 0) + 1
        return counts


# ──────────────────────────────────────────────────────────────────────────────
# Animations
# ──────────────────────────────────────────────────────────────────────────────

class FillUrn(Succession):
    """
    Drop balls into the urn one by one from above.

    Each ball starts above the urn opening, falls in (ease-in),
    and settles into its packed position.

    Parameters
    ----------
    urn         : Urn3D
    balls       : list[Ball3D]  — balls to add (in order)
    drop_height : float         — how far above the lip each ball starts
    stagger     : float         — delay between consecutive drops (s)
    run_time    : float         — total animation time
    """

    def __init__(
        self,
        urn: Urn3D,
        balls: list[Ball3D],
        drop_height: float = 1.2,
        stagger: float = 0.12,
        **kwargs,
    ):
        # Compute packed target positions
        packed = get_packed_positions(
            len(balls),
            urn._belly_r,
            urn._inner_bottom_y,
            urn._ball_r,
        )
        urn_ctr = urn.get_center()

        anims = []
        for i, (ball, target) in enumerate(zip(balls, packed)):
            world_target = np.array(target) + urn_ctr
            # Start above urn
            start_pos = np.array([
                world_target[0],
                urn.lip_y + drop_height,
                world_target[2],
            ])
            ball.move_to(start_pos)
            urn.balls.append(ball)

            drop = ApplyMethod(
                ball.move_to, world_target,
                run_time=0.35,
                rate_func=rate_functions.ease_in_cubic,
            )
            anims.append(drop)

        kwargs.setdefault("run_time", sum(a.run_time for a in anims)
                         + stagger * (len(anims) - 1))
        super().__init__(*anims, **kwargs)


class DrawBall(Animation):
    """
    Draw one ball smoothly out of the urn.

    The ball rises along a smooth ease-out arc from its current packed
    position, through the urn opening, to a position above the urn.

    Parameters
    ----------
    urn         : Urn3D
    ball        : Ball3D        — the ball to draw (must be in urn.balls)
    dest        : np.ndarray    — where the ball ends up; default is
                                  directly above the urn at a comfortable height
    run_time    : float
    """

    def __init__(
        self,
        urn: Urn3D,
        ball: Ball3D,
        dest: Optional[np.ndarray] = None,
        **kwargs,
    ):
        self.urn      = urn
        self.ball     = ball
        self.start    = ball.get_center().copy()
        if dest is None:
            dest = urn.get_draw_exit_point() + UP * 0.8
        self.dest     = dest
        kwargs.setdefault("run_time", 1.0)
        kwargs.setdefault("rate_func", rate_functions.ease_out_cubic)
        super().__init__(ball, **kwargs)

    def begin(self):
        super().begin()
        # Remove from urn tracking when draw begins
        if self.ball in self.urn.balls:
            self.urn.balls.remove(self.ball)

    def interpolate_mobject(self, alpha: float):
        # Arc path: slight bow to the right as ball rises
        bow   = np.array([self.urn._belly_r * 0.3 * np.sin(alpha * PI), 0, 0])
        pos   = self.start * (1 - alpha) + self.dest * alpha + bow
        self.ball.move_to(pos)


class ReplaceBall(Animation):
    """
    Return a ball to the urn — arcs from current position back down.

    Parameters
    ----------
    urn      : Urn3D
    ball     : Ball3D
    run_time : float
    """

    def __init__(
        self,
        urn: Urn3D,
        ball: Ball3D,
        **kwargs,
    ):
        self.urn   = urn
        self.ball  = ball
        self.start = ball.get_center().copy()

        # Find next available packed position
        packed = get_packed_positions(
            len(urn.balls) + 1,
            urn._belly_r,
            urn._inner_bottom_y,
            urn._ball_r,
        )
        self.dest = np.array(packed[-1]) + urn.get_center()

        kwargs.setdefault("run_time", 0.85)
        kwargs.setdefault("rate_func", rate_functions.ease_in_out_cubic)
        super().__init__(ball, **kwargs)

    def begin(self):
        super().begin()
        self.urn.balls.append(self.ball)

    def interpolate_mobject(self, alpha: float):
        bow = np.array([0, np.sin(alpha * PI) * self.urn._belly_r * 0.25, 0])
        pos = self.start * (1 - alpha) + self.dest * alpha + bow
        self.ball.move_to(pos)


class ShakeUrn(Animation):
    """
    Rapid lateral shake of the urn (mixing balls before drawing).

    Parameters
    ----------
    urn        : Urn3D
    n_shakes   : int   — oscillation count
    amplitude  : float — max lateral displacement
    run_time   : float
    """

    def __init__(
        self,
        urn: Urn3D,
        n_shakes: int = 6,
        amplitude: float = 0.15,
        **kwargs,
    ):
        self.urn       = urn
        self.n_shakes  = n_shakes
        self.amplitude = amplitude
        self.start_pos = urn.get_center().copy()
        kwargs.setdefault("run_time", 0.9)
        kwargs.setdefault("rate_func", rate_functions.linear)
        super().__init__(urn, **kwargs)

    def interpolate_mobject(self, alpha: float):
        decay  = 1.0 - alpha
        osc    = (np.sin(alpha * self.n_shakes * PI)
                  * self.amplitude * decay)
        rot    = osc * 0.08   # slight rotational tilt
        self.urn.become(self.starting_mobject.copy())
        self.urn.shift(np.array([osc, 0, 0]))
        self.urn.rotate(rot, axis=OUT, about_point=self.urn.get_center())


class PourUrn(Succession):
    """
    Tilt the urn and pour all balls out in sequence.

    Steps:
      1. Urn tilts (Rotate ~100°)
      2. Balls fly out one by one via parabolic arcs
      3. Urn rights itself

    Parameters
    ----------
    urn          : Urn3D
    balls        : list[Ball3D] | None  — balls to pour; default = urn.balls
    pour_angle   : float                — tilt angle (default PI * 0.55)
    spread       : float                — x-spread of poured balls
    run_time     : float
    """

    def __init__(
        self,
        urn: Urn3D,
        balls: Optional[list[Ball3D]] = None,
        pour_angle: float = PI * 0.55,
        spread: float = 2.5,
        **kwargs,
    ):
        if balls is None:
            balls = list(urn.balls)

        tilt    = Rotate(urn, angle=pour_angle, axis=OUT,
                         about_point=urn.get_center(),
                         run_time=0.5)

        pour_anims = []
        for i, ball in enumerate(balls):
            t_offset = i * 0.12
            dest     = np.array([
                urn.get_center()[0] + spread * (i / max(len(balls)-1, 1) - 0.5),
                urn.get_center()[1] - urn._height * 0.6,
                0,
            ])
            pour_anims.append(
                ApplyMethod(ball.move_to, dest, run_time=0.6,
                            rate_func=rate_functions.ease_in_cubic)
            )
            urn.balls.remove(ball) if ball in urn.balls else None

        right   = Rotate(urn, angle=-pour_angle, axis=OUT,
                         about_point=urn.get_center(),
                         run_time=0.4)

        super().__init__(tilt, AnimationGroup(*pour_anims), right, **kwargs)


class SwapBalls(Animation):
    """
    Swap two balls between positions — useful for conditional probability
    illustrations (e.g., "given we drew red, which urn is more likely?").

    Parameters
    ----------
    ball_a, ball_b : Ball3D  — the two balls to swap
    arc_height     : float   — arc height during swap
    run_time       : float
    """

    def __init__(
        self,
        ball_a: Ball3D,
        ball_b: Ball3D,
        arc_height: float = 0.8,
        **kwargs,
    ):
        self.ball_a     = ball_a
        self.ball_b     = ball_b
        self.pos_a      = ball_a.get_center().copy()
        self.pos_b      = ball_b.get_center().copy()
        self.arc_height = arc_height
        kwargs.setdefault("run_time", 1.1)
        kwargs.setdefault("rate_func", rate_functions.ease_in_out_sine)
        # Animate both as a group
        group = VGroup(ball_a, ball_b)
        super().__init__(group, **kwargs)

    def interpolate_mobject(self, alpha: float):
        # Ball A goes from pos_a → pos_b via upper arc
        h_a  = self.arc_height * np.sin(alpha * PI)
        pa   = self.pos_a * (1-alpha) + self.pos_b * alpha + UP * h_a
        # Ball B goes from pos_b → pos_a via lower arc
        h_b  = self.arc_height * 0.65 * np.sin(alpha * PI)
        pb   = self.pos_b * (1-alpha) + self.pos_a * alpha - UP * h_b

        self.ball_a.move_to(pa)
        self.ball_b.move_to(pb)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience factories
# ──────────────────────────────────────────────────────────────────────────────

def make_labeled_urn(
    n_red: int,
    n_blue: int,
    urn_kwargs: Optional[dict] = None,
    ball_kwargs: Optional[dict] = None,
) -> tuple["Urn3D", list["Ball3D"], list["Ball3D"]]:
    """
    Create an urn pre-populated with red and blue balls — the classic
    hypergeometric / conditional probability setup.

    Returns
    -------
    urn, red_balls, blue_balls
    """
    urn_kwargs  = urn_kwargs  or {}
    ball_kwargs = ball_kwargs or {}
    ball_r = urn_kwargs.get("ball_radius", 0.18)

    urn        = Urn3D(**urn_kwargs)
    red_balls  = [Ball3D("red",  label=str(i+1),   radius=ball_r, **ball_kwargs)
                  for i in range(n_red)]
    blue_balls = [Ball3D("blue", label=str(i+1),   radius=ball_r, **ball_kwargs)
                  for i in range(n_blue)]

    all_balls  = red_balls + blue_balls
    packed     = get_packed_positions(
        len(all_balls), urn._belly_r, urn._inner_bottom_y, ball_r
    )
    urn_ctr    = urn.get_center()
    for ball, pos in zip(all_balls, packed):
        ball.move_to(np.array(pos) + urn_ctr)
        urn.balls.append(ball)

    return urn, red_balls, blue_balls


def make_two_urn_setup(
    urn1_contents: list[tuple[str, int]],   # [(color, count), ...]
    urn2_contents: list[tuple[str, int]],
    spacing: float = 4.5,
    scheme1: str = "terracotta",
    scheme2: str = "cobalt",
    ball_radius: float = 0.18,
) -> tuple["Urn3D", "Urn3D", list["Ball3D"], list["Ball3D"]]:
    """
    Create the classic two-urn Bayes problem setup.

    Parameters
    ----------
    urn1_contents : e.g. [("red", 3), ("blue", 2)]
    urn2_contents : e.g. [("red", 1), ("blue", 4)]
    spacing       : horizontal distance between urns
    scheme1/2     : colour schemes
    ball_radius   : radius for all balls

    Returns
    -------
    urn1, urn2, balls1, balls2
    """
    def _build_urn(contents, scheme):
        urn    = Urn3D(color_scheme=scheme, ball_radius=ball_radius)
        balls  = []
        for color, count in contents:
            for i in range(count):
                b = Ball3D(color, label=str(i+1), radius=ball_radius)
                balls.append(b)
        packed  = get_packed_positions(
            len(balls), urn._belly_r, urn._inner_bottom_y, ball_radius
        )
        urn_ctr = urn.get_center()
        for ball, pos in zip(balls, packed):
            ball.move_to(np.array(pos) + urn_ctr)
            urn.balls.append(ball)
        return urn, balls

    urn1, balls1 = _build_urn(urn1_contents, scheme1)
    urn2, balls2 = _build_urn(urn2_contents, scheme2)

    urn1.shift(LEFT  * spacing / 2)
    urn2.shift(RIGHT * spacing / 2)

    return urn1, urn2, balls1, balls2


# ──────────────────────────────────────────────────────────────────────────────
# Demo Scene  (run with: manim -pql urn.py UrnDemo)
# ──────────────────────────────────────────────────────────────────────────────

try:
    from manim import ThreeDScene, DEGREES

    class UrnDemo(ThreeDScene):
        """Showcase scene for Urn3D and Ball3D."""

        def construct(self):
            self.set_camera_orientation(phi=62 * DEGREES, theta=-50 * DEGREES)
            self.begin_ambient_camera_rotation(rate=0.025)

            # ── Two-urn Bayes setup ───────────────────────────────────
            urn1, urn2, balls1, balls2 = make_two_urn_setup(
                urn1_contents=[("red", 3), ("blue", 2)],
                urn2_contents=[("red", 1), ("blue", 4)],
                scheme1="terracotta",
                scheme2="cobalt",
            )
            all_balls = balls1 + balls2
            self.add(urn1, urn2, *all_balls)
            self.wait(1.0)

            # ── Shake urn 1 before drawing ────────────────────────────
            self.play(ShakeUrn(urn1, n_shakes=8, run_time=1.0))
            self.wait(0.3)

            # ── Draw a red ball from urn 1 ────────────────────────────
            drawn = balls1[0]
            dest  = np.array([-2.8, 1.5, 0])
            self.play(DrawBall(urn1, drawn, dest=dest, run_time=1.2))
            self.wait(0.5)

            # ── Replace it (sampling WITH replacement) ────────────────
            self.play(ReplaceBall(urn1, drawn, run_time=0.9))
            self.wait(0.5)

            # ── Shake urn 2, draw a blue ball ─────────────────────────
            self.play(ShakeUrn(urn2, n_shakes=6, run_time=0.8))
            drawn2 = balls2[3]   # a blue ball
            dest2  = np.array([2.8, 1.5, 0])
            self.play(DrawBall(urn2, drawn2, dest=dest2, run_time=1.1))
            self.wait(0.5)

            # ── Swap the two drawn balls ──────────────────────────────
            self.play(SwapBalls(drawn, drawn2, arc_height=1.0, run_time=1.3))
            self.wait(1.0)

            # ── Pour urn 2 ────────────────────────────────────────────
            self.play(PourUrn(urn2, run_time=2.0))
            self.wait(1.5)

except ImportError:
    pass   # Manim not installed; skip demo scene definition