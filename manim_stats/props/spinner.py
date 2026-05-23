"""
manim_stats/props/spinner.py
============================
Spinner3D — A highly detailed, physically-inspired probability spinner
for Manim statistics animations.

Primary use cases
-----------------
  Uniform discrete / continuous probability  — equal-sector spinners
  Non-uniform probability                    — custom sector weights
  Experimental vs theoretical probability    — repeated spin demos
  Geometric distribution                     — spin until outcome X
  Markov chains                              — state-transition spinner

Design goals
------------
Board
  * Circular wooden/acrylic base with a raised rim.
  * Wood-grain texture approximated by thin concentric and radial
    Line strokes at low opacity.
  * Four countersunk screw holes at cardinal points on the rim.
  * Raised outer rim ring with evenly-spaced tick marks (major ticks
    at sector boundaries, minor ticks between them — like a protractor).

Sectors
  * Each sector is a filled pie wedge (Polygon approximating an arc).
  * Three visual layers per sector:
      – base fill  (sector colour)
      – inner shading gradient  (radial dark→light approximated with
        a slightly brighter inner annulus wedge)
      – outer arc label zone  (slightly darkened annular strip for text)
  * Thin bright divider lines between sectors (raised border effect).
  * Sector label: probability fraction + percentage, font-size scaled
    to wedge arc length so it always fits.

Hub
  * Outer brass ring (Annulus).
  * Inner bearing disc (Circle, slightly recessed colour).
  * Center pin (small Circle, very dark).
  * Specular dot highlight offset toward top-left.

Needle
  * True elongated diamond / rhombus shape — wide at the pivot,
    tapering to a sharp point at the tip and a rounded counterweight
    tail at the base.
  * Specular stripe down the lit side.
  * Drop shadow (slightly dark ellipse beneath the needle on the board).
  * Pivot hole circle at center.

Animations
----------
  SpinToOutcome   — exponential angular deceleration, lands on target
                    sector with a realistic overshoot-and-settle wobble.
  FreeSpinDecay   — pure physics spin with no pre-determined outcome;
                    needle decelerates and stops wherever it lands.
  FlickSpin       — short sharp impulse, needle travels only 1–3 sectors.
  RicochetSpin    — needle "catches" on a sector boundary and ticks
                    backward slightly before settling (ratchet sound feel).
  SpinSequence    — animate N successive spins, yielding outcome list.

Dependencies
------------
  manim (CE or GL), numpy, fractions

Usage
-----
    from manim_stats.props.spinner import Spinner3D, SpinToOutcome

    class GeometricScene(ThreeDScene):
        def construct(self):
            spinner = Spinner3D.uniform(n=4, radius=2.0)
            self.add(spinner)
            for _ in range(6):
                self.play(SpinToOutcome(spinner, outcome=0,
                                        n_full_rotations=3))
                self.wait(0.4)
"""

from __future__ import annotations

import numpy as np
from fractions import Fraction
from typing import Optional, Sequence, Literal, List, Tuple

from manim import (
    VGroup,
    Circle, Annulus, Arc, Line, Polygon, Rectangle, Dot,
    Text, MathTex, Tex,
    Animation, Succession, AnimationGroup,
    ApplyMethod, Rotate, FadeIn, FadeOut,
    interpolate_color, color_to_rgb,
    WHITE, BLACK,
    GREY, GREY_A, GREY_B, GREY_C, GREY_D, LIGHT_GREY,
    RED,    RED_A,    RED_B,    RED_C,    RED_D,
    BLUE,   BLUE_A,   BLUE_B,   BLUE_C,   BLUE_D,   BLUE_E,
    GREEN,  GREEN_A,  GREEN_B,  GREEN_C,  GREEN_D,  GREEN_E,
    YELLOW, YELLOW_A, YELLOW_B, YELLOW_E,
    ORANGE, PURPLE_A, PURPLE_B, TEAL, TEAL_A, TEAL_B,
    GOLD,   GOLD_A,   GOLD_B,   GOLD_C,   GOLD_D,   GOLD_E,
    MAROON, PINK,
    UP, DOWN, LEFT, RIGHT, OUT, IN, ORIGIN,
    PI, TAU,
    rate_functions,
    ManimColor,
)

# ──────────────────────────────────────────────────────────────────────────────
# Default sector colour cycle
# ──────────────────────────────────────────────────────────────────────────────

SECTOR_COLORS: list[str] = [
    "#E63946",   # vivid red
    "#457B9D",   # steel blue
    "#2A9D8F",   # teal
    "#E9C46A",   # warm yellow
    "#F4A261",   # orange
    "#A8DADC",   # light teal
    "#9B2226",   # crimson
    "#D4AF37",   # gold
    "#6A4C93",   # purple
    "#1B998B",   # seafoam
    "#FFBC42",   # amber
    "#3A86FF",   # bright blue
    "#8AC926",   # lime green
    "#FF595E",   # coral
    "#6D6875",   # dusty purple
    "#B5838D",   # rose
]

# Board palette options
BOARD_PALETTES: dict[str, dict] = {
    "wood": {
        "base":       "#C8A96E",   # warm maple
        "grain":      "#B8955A",   # grain lines
        "rim":        "#8B6340",   # darker rim
        "tick_major": "#3A2510",
        "tick_minor": "#6A4820",
        "screw":      "#706050",
        "screw_head": "#D4B896",
    },
    "acrylic_white": {
        "base":       "#F4F0E8",
        "grain":      "#E0DCD0",
        "rim":        "#C8C0B0",
        "tick_major": "#1A1A1A",
        "tick_minor": "#7A7A7A",
        "screw":      "#8A8A8A",
        "screw_head": "#E8E8E8",
    },
    "acrylic_dark": {
        "base":       "#1C1C1E",
        "grain":      "#2A2A2C",
        "rim":        "#0A0A0C",
        "tick_major": "#D4AF37",
        "tick_minor": "#605840",
        "screw":      "#3A3A3A",
        "screw_head": "#5A5A5A",
    },
    "felt_green": {
        "base":       "#2D5A27",
        "grain":      "#265220",
        "rim":        "#1A3A16",
        "tick_major": "#F0E8C0",
        "tick_minor": "#A0C878",
        "screw":      "#3A5030",
        "screw_head": "#608050",
    },
    "casino_red": {
        "base":       "#7A1018",
        "grain":      "#6A0C14",
        "rim":        "#4A0810",
        "tick_major": "#F0E8C0",
        "tick_minor": "#C08840",
        "screw":      "#5A2820",
        "screw_head": "#C08878",
    },
}

HUB_PALETTES: dict[str, dict] = {
    "brass": {
        "outer":    "#B8860B",
        "outer_hl": "#DAA520",
        "bearing":  "#8B6914",
        "pin":      "#2A1A08",
        "spec":     "#FFD700",
    },
    "chrome": {
        "outer":    "#B0B8C0",
        "outer_hl": "#E8F0F8",
        "bearing":  "#808898",
        "pin":      "#1A1A2A",
        "spec":     "#FFFFFF",
    },
    "black_oxide": {
        "outer":    "#2A2A30",
        "outer_hl": "#4A4A58",
        "bearing":  "#1A1A20",
        "pin":      "#080810",
        "spec":     "#6A6A80",
    },
}

NEEDLE_PALETTES: dict[str, dict] = {
    "red_chrome": {
        "body":   "#CC2222",
        "dark":   "#6A0808",
        "spec":   "#FF8080",
        "shadow": "#1A0808",
        "tail":   "#881818",
    },
    "black_chrome": {
        "body":   "#1A1A20",
        "dark":   "#080810",
        "spec":   "#5A5A70",
        "shadow": "#050508",
        "tail":   "#2A2A30",
    },
    "gold": {
        "body":   "#D4AF37",
        "dark":   "#8B7010",
        "spec":   "#FFE066",
        "shadow": "#2A1A08",
        "tail":   "#B89028",
    },
    "white": {
        "body":   "#F0EEE8",
        "dark":   "#A0A098",
        "spec":   "#FFFFFF",
        "shadow": "#202020",
        "tail":   "#C8C8C0",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _wedge_polygon(
    r_inner: float,
    r_outer: float,
    start_angle: float,
    end_angle: float,
    n_arc_pts: int = 32,
    z: float = 0.0,
) -> list[list[float]]:
    """
    Vertices for a filled annular wedge (sector ring).
    Returns a flat list suitable for Polygon(*verts).
    """
    angles_fwd = np.linspace(start_angle, end_angle,  n_arc_pts)
    angles_rev = np.linspace(end_angle,   start_angle, n_arc_pts)

    outer_pts = [[r_outer * np.cos(a), r_outer * np.sin(a), z]
                 for a in angles_fwd]
    inner_pts = [[r_inner * np.cos(a), r_inner * np.sin(a), z]
                 for a in angles_rev]
    return outer_pts + inner_pts


def _pie_polygon(
    radius: float,
    start_angle: float,
    end_angle: float,
    n_arc_pts: int = 48,
    z: float = 0.0,
) -> list[list[float]]:
    """Vertices for a filled pie-slice (sector from centre)."""
    angles = np.linspace(start_angle, end_angle, n_arc_pts)
    arc_pts = [[radius * np.cos(a), radius * np.sin(a), z]
               for a in angles]
    return [[0, 0, z]] + arc_pts


def _sector_label_position(
    r_mid: float,
    mid_angle: float,
) -> np.ndarray:
    """World position for a label at the centre of a sector."""
    return np.array([r_mid * np.cos(mid_angle),
                     r_mid * np.sin(mid_angle),
                     0.0])


def _arc_text_centered(
    text: str,
    radius: float,
    mid_angle: float,
    font_size: float,
    color: str,
    font: str = "sans-serif",
) -> Text:
    """Single text mob centred at the sector midpoint."""
    t = Text(text, font_size=font_size, color=color, font=font)
    pos = _sector_label_position(radius, mid_angle)
    t.move_to(pos)
    # Rotate to face outward from centre
    t.rotate(mid_angle - PI / 2, about_point=pos)
    return t


# ──────────────────────────────────────────────────────────────────────────────
# Sub-components
# ──────────────────────────────────────────────────────────────────────────────

class _SpinnerBoard(VGroup):
    """
    The circular board of the spinner.

    Layers (back → front):
      1. base_disc        — filled circle, board body colour
      2. grain_lines      — concentric and radial lines at low opacity
                            (wood/felt texture approximation)
      3. rim_ring         — Annulus for the raised outer rim
      4. tick_marks       — minor + major ticks around the rim
      5. screw_holes      — 4 decorative countersunk screw holes

    Parameters
    ----------
    radius       : float — board radius
    board_palette: dict
    tick_minor_n : int   — minor ticks between sector boundaries (default 4)
    n_sectors    : int   — number of sectors (drives major tick placement)
    """

    def __init__(
        self,
        radius: float,
        board_palette: dict,
        n_sectors: int = 4,
        tick_minor_n: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        bp   = board_palette
        rim_w = radius * 0.055

        # ── 1. Base disc ──────────────────────────────────────────────
        base = Circle(
            radius=radius,
            fill_color=bp["base"],
            fill_opacity=1.0,
            stroke_width=0,
        )
        self.add(base)

        # ── 2. Wood / felt grain texture ──────────────────────────────
        grain = VGroup()
        # Concentric rings
        n_rings = 8
        for i in range(1, n_rings + 1):
            r_g = radius * 0.92 * (i / n_rings)
            ring = Circle(
                radius=r_g,
                stroke_color=bp["grain"],
                stroke_width=0.5,
                stroke_opacity=0.30,
                fill_opacity=0,
            )
            grain.add(ring)
        # Radial grain lines (sparse)
        n_grain_lines = 24
        for i in range(n_grain_lines):
            ang = i * TAU / n_grain_lines
            grain.add(Line(
                start=[0, 0, 0.001],
                end=[(radius * 0.90) * np.cos(ang),
                     (radius * 0.90) * np.sin(ang), 0.001],
                stroke_color=bp["grain"],
                stroke_width=0.4,
                stroke_opacity=0.18,
            ))
        self.add(grain)

        # ── 3. Rim ring ───────────────────────────────────────────────
        rim = Annulus(
            inner_radius=radius - rim_w,
            outer_radius=radius,
            fill_color=bp["rim"],
            fill_opacity=1.0,
            stroke_width=0,
        )
        rim.move_to([0, 0, 0.002])
        self.add(rim)

        # ── 4. Tick marks on the rim ──────────────────────────────────
        ticks = VGroup()
        # Total ticks: n_sectors major + n_sectors*(tick_minor_n) minor
        total_ticks = n_sectors * (tick_minor_n + 1)
        for i in range(total_ticks):
            ang     = i * TAU / total_ticks
            is_major = (i % (tick_minor_n + 1) == 0)
            t_len    = rim_w * (0.85 if is_major else 0.45)
            t_width  = 1.6    if is_major else 0.8
            t_color  = bp["tick_major"] if is_major else bp["tick_minor"]
            r_outer  = radius - rim_w * 0.08
            r_inner  = r_outer - t_len
            ticks.add(Line(
                start=[r_outer * np.cos(ang), r_outer * np.sin(ang), 0.003],
                end  =[r_inner * np.cos(ang), r_inner * np.sin(ang), 0.003],
                stroke_color=t_color,
                stroke_width=t_width,
            ))
        self.add(ticks)

        # ── 5. Screw holes (4 cardinal points) ───────────────────────
        screws = VGroup()
        screw_r = rim_w * 0.38
        for ang in [0, PI/2, PI, 3*PI/2]:
            r_pos  = radius - rim_w * 0.50
            cx, cy = r_pos * np.cos(ang), r_pos * np.sin(ang)
            # Countersink shadow
            sink = Circle(
                radius=screw_r * 1.35,
                fill_color=bp["screw"],
                fill_opacity=0.80,
                stroke_width=0,
            )
            sink.move_to([cx, cy, 0.003])
            # Screw head
            head = Circle(
                radius=screw_r,
                fill_color=bp["screw_head"],
                fill_opacity=1.0,
                stroke_width=0,
            )
            head.move_to([cx, cy, 0.004])
            # Cross slot lines
            for slot_ang in [0, PI/2]:
                screws.add(Line(
                    start=[cx + screw_r*0.70*np.cos(slot_ang),
                           cy + screw_r*0.70*np.sin(slot_ang), 0.005],
                    end  =[cx - screw_r*0.70*np.cos(slot_ang),
                           cy - screw_r*0.70*np.sin(slot_ang), 0.005],
                    stroke_color=bp["screw"],
                    stroke_width=0.8,
                ))
            screws.add(sink, head)
        self.add(screws)


class _SpinnerSectors(VGroup):
    """
    All pie sectors of the spinner, with labels.

    Each sector has:
      1. base wedge          — filled pie slice
      2. light inner wedge   — brighter annular arc near the centre
                               (radial highlight, simulates convex sheen)
      3. dark outer strip    — slightly darker arc near the rim (label bg)
      4. divider line        — bright thin line at each sector boundary
      5. label text          — probability string, rotated to face outward

    Parameters
    ----------
    weights      : sequence[float] — unnormalised sector weights
    labels       : sequence[str] | None — override labels per sector
    colors       : sequence[str] | None — override colours per sector
    radius       : float — usable board radius (inside rim)
    start_angle  : float — angle of first sector's leading edge (default PI/2)
    show_labels  : bool
    label_style  : "fraction" | "percent" | "both" | "name"
    """

    def __init__(
        self,
        weights: Sequence[float],
        radius: float,
        labels: Optional[Sequence[str]] = None,
        colors: Optional[Sequence[str]] = None,
        start_angle: float = PI / 2,
        show_labels: bool = True,
        label_style: Literal["fraction", "percent", "both", "name"] = "both",
        **kwargs,
    ):
        super().__init__(**kwargs)

        total    = sum(weights)
        n        = len(weights)
        r_board  = radius * 0.95   # slightly inside rim
        r_inner  = r_board * 0.12  # inner dead zone (hub will cover)
        r_label  = r_board * 0.65  # label radial position

        # Cumulative angles
        cum_angles = [start_angle]
        for w in weights:
            cum_angles.append(cum_angles[-1] - (w / total) * TAU)

        self._sector_angles = []   # (mid_angle, start, end) for each sector
        self._sector_colors = []

        for i in range(n):
            a_start  = cum_angles[i]
            a_end    = cum_angles[i + 1]
            mid_ang  = (a_start + a_end) / 2
            span     = abs(a_end - a_start)

            col      = (colors[i % len(colors)] if colors
                        else SECTOR_COLORS[i % len(SECTOR_COLORS)])
            dark_col = interpolate_color(col, BLACK, 0.32)
            lite_col = interpolate_color(col, WHITE, 0.28)

            self._sector_angles.append((mid_ang, a_start, a_end))
            self._sector_colors.append(col)

            # ── 1. Base pie slice ─────────────────────────────────────
            verts = _pie_polygon(r_board, a_end, a_start, n_arc_pts=48,
                                 z=0.001)
            base  = Polygon(
                *verts,
                fill_color=col,
                fill_opacity=1.0,
                stroke_width=0,
            )
            self.add(base)

            # ── 2. Inner bright arc (convex highlight near centre) ────
            if span > 0.08:
                hl_verts = _wedge_polygon(
                    r_inner, r_board * 0.42,
                    a_end, a_start, n_arc_pts=24, z=0.002,
                )
                hl_wedge = Polygon(
                    *hl_verts,
                    fill_color=lite_col,
                    fill_opacity=0.28,
                    stroke_width=0,
                )
                self.add(hl_wedge)

            # ── 3. Outer dark strip (label background) ────────────────
            if span > 0.08:
                outer_verts = _wedge_polygon(
                    r_board * 0.80, r_board,
                    a_end, a_start, n_arc_pts=24, z=0.002,
                )
                outer_strip = Polygon(
                    *outer_verts,
                    fill_color=dark_col,
                    fill_opacity=0.30,
                    stroke_width=0,
                )
                self.add(outer_strip)

            # ── 4. Labels ─────────────────────────────────────────────
            if show_labels and span > 0.15:
                label_str = self._make_label_str(
                    weights[i], total, labels, i, label_style
                )
                # Scale font to arc length
                arc_len   = r_label * span
                font_size = max(8, min(28, arc_len * 18))

                lbl_color = (WHITE
                             if _perceived_luminance(col) < 0.50
                             else "#1A1A1A")
                lbl = Text(
                    label_str,
                    font_size=font_size,
                    color=lbl_color,
                    font="sans-serif",
                )
                lbl_pos = _sector_label_position(r_label, mid_ang)
                lbl.move_to(lbl_pos + np.array([0, 0, 0.012]))
                lbl.rotate(mid_ang - PI / 2, about_point=lbl_pos)
                self.add(lbl)

        # ── 5. Sector divider lines ───────────────────────────────────
        for ang in cum_angles:
            divider = Line(
                start=[r_inner * np.cos(ang),  r_inner * np.sin(ang),  0.006],
                end  =[r_board * np.cos(ang),  r_board * np.sin(ang),  0.006],
                stroke_color=WHITE,
                stroke_width=1.6,
                stroke_opacity=0.70,
            )
            self.add(divider)

    @staticmethod
    def _make_label_str(
        w: float,
        total: float,
        labels: Optional[Sequence[str]],
        i: int,
        style: str,
    ) -> str:
        if style == "name" and labels is not None:
            return str(labels[i])
        pct   = w / total * 100
        try:
            frac = Fraction(w / total).limit_denominator(20)
            frac_str = f"{frac.numerator}/{frac.denominator}"
        except Exception:
            frac_str = f"{pct:.0f}%"

        if style == "fraction":
            return frac_str
        elif style == "percent":
            return f"{pct:.1f}%"
        elif style == "both":
            return f"{frac_str}\n{pct:.0f}%"
        elif style == "name" and labels is not None:
            return str(labels[i])
        return f"{pct:.0f}%"


def _perceived_luminance(hex_color: str) -> float:
    """Relative luminance of a hex colour (0=black, 1=white)."""
    try:
        rgb = np.array(color_to_rgb(hex_color), dtype=float)
        return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    except Exception:
        return 0.5


class _SpinnerHub(VGroup):
    """
    Multi-layer center hub: brass ring → bearing disc → pin → specular.

    Parameters
    ----------
    radius      : float — hub outer radius
    hub_palette : dict
    """

    def __init__(
        self,
        radius: float,
        hub_palette: dict,
        **kwargs,
    ):
        super().__init__(**kwargs)
        hp = hub_palette
        z0 = 0.020

        # Outer brass ring
        outer = Circle(
            radius=radius,
            fill_color=hp["outer"],
            fill_opacity=1.0,
            stroke_width=0,
        )
        outer.move_to([0, 0, z0])
        self.add(outer)

        # Highlight ring (top-left arc brightening)
        hl_ring = Annulus(
            inner_radius=radius * 0.72,
            outer_radius=radius,
            fill_color=hp["outer_hl"],
            fill_opacity=0.35,
            stroke_width=0,
        )
        hl_ring.move_to([-radius * 0.12, radius * 0.12, z0 + 0.001])
        self.add(hl_ring)

        # Bearing disc
        bearing = Circle(
            radius=radius * 0.65,
            fill_color=hp["bearing"],
            fill_opacity=1.0,
            stroke_width=0,
        )
        bearing.move_to([0, 0, z0 + 0.002])
        self.add(bearing)

        # Knurling lines on bearing (decorative)
        n_knurl = 12
        for i in range(n_knurl):
            ang = i * TAU / n_knurl
            r0  = radius * 0.40
            r1  = radius * 0.62
            self.add(Line(
                start=[r0 * np.cos(ang), r0 * np.sin(ang), z0 + 0.003],
                end  =[r1 * np.cos(ang), r1 * np.sin(ang), z0 + 0.003],
                stroke_color=interpolate_color(hp["bearing"], BLACK, 0.25),
                stroke_width=0.6,
                stroke_opacity=0.60,
            ))

        # Center pin
        pin = Circle(
            radius=radius * 0.22,
            fill_color=hp["pin"],
            fill_opacity=1.0,
            stroke_width=0,
        )
        pin.move_to([0, 0, z0 + 0.004])
        self.add(pin)

        # Specular highlight dot
        spec = Circle(
            radius=radius * 0.10,
            fill_color=hp["spec"],
            fill_opacity=0.80,
            stroke_width=0,
        )
        spec.move_to([-radius * 0.08, radius * 0.09, z0 + 0.005])
        self.add(spec)

        self._hub_radius = radius


class _SpinnerNeedle(VGroup):
    """
    The spinner needle — an elongated diamond / rhombus shape with:
      * Tapered tip (sharp point in the +Y direction)
      * Counterweight tail (shorter blunt lobe in the -Y direction)
      * Specular stripe along the lit (+X) face
      * Drop shadow ellipse underneath
      * Pivot hole circle at the rotation centre

    The needle is built in its "pointing up" orientation (tip = +Y).
    Rotation is applied by `SpinToOutcome` / `FreeSpinDecay`.

    Parameters
    ----------
    length         : float — tip-to-tail total length
    width          : float — max width at the pivot (default length * 0.09)
    needle_palette : dict
    """

    def __init__(
        self,
        length: float,
        width: Optional[float] = None,
        needle_palette: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if needle_palette is None:
            needle_palette = NEEDLE_PALETTES["red_chrome"]
        if width is None:
            width = length * 0.085

        np_   = needle_palette
        z0    = 0.025

        tip_y   =  length * 0.62   # tip above pivot
        tail_y  = -length * 0.38   # tail below pivot
        hw      = width / 2        # half-width at pivot

        # ── Drop shadow (dark ellipse on board) ──────────────────────
        shadow = Circle(
            radius=length * 0.48,
            fill_color=np_["shadow"],
            fill_opacity=0.18,
            stroke_width=0,
        )
        shadow.scale([0.18, 1.0, 1.0])
        shadow.move_to([length * 0.04, 0, z0 - 0.002])
        self.add(shadow)

        # ── Main needle body (diamond polygon) ───────────────────────
        # 4 vertices: tip, right-shoulder, tail, left-shoulder
        shoulder_y  =  length * 0.05   # shoulders are slightly above pivot
        tip_v       = [0,     tip_y,       z0]
        right_v     = [ hw,   shoulder_y,  z0]
        tail_v      = [0,     tail_y,      z0]
        left_v      = [-hw,   shoulder_y,  z0]

        body = Polygon(
            tip_v, right_v, tail_v, left_v,
            fill_color=np_["body"],
            fill_opacity=1.0,
            stroke_width=0,
        )
        self.add(body)

        # ── Dark underside (right half slightly darker) ───────────────
        dark_right = Polygon(
            tip_v, right_v, tail_v,
            fill_color=np_["dark"],
            fill_opacity=0.40,
            stroke_width=0,
        )
        dark_right.move_to(dark_right.get_center() + np.array([0, 0, 0.001]))
        self.add(dark_right)

        # ── Specular stripe (left face) ───────────────────────────────
        # A thin tapered polygon along the left edge of the needle
        spec_w  = hw * 0.35
        spec = Polygon(
            [0,        tip_y * 0.92,  z0 + 0.002],
            [-spec_w,  shoulder_y,    z0 + 0.002],
            [0,        tail_y * 0.65, z0 + 0.002],
            fill_color=np_["spec"],
            fill_opacity=0.55,
            stroke_width=0,
        )
        self.add(spec)

        # ── Tail cap (blunt rounded lobe) ─────────────────────────────
        tail_cap = Circle(
            radius=hw * 1.10,
            fill_color=np_["tail"],
            fill_opacity=1.0,
            stroke_width=0,
        )
        tail_cap.move_to([0, tail_y + hw * 0.80, z0 + 0.001])
        self.add(tail_cap)

        # ── Tip cap ───────────────────────────────────────────────────
        tip_cap = Circle(
            radius=hw * 0.22,
            fill_color=interpolate_color(np_["body"], WHITE, 0.45),
            fill_opacity=1.0,
            stroke_width=0,
        )
        tip_cap.move_to([0, tip_y - hw * 0.10, z0 + 0.002])
        self.add(tip_cap)

        # ── Pivot hole ────────────────────────────────────────────────
        pivot_hole = Circle(
            radius=hw * 0.42,
            fill_color=np_["dark"],
            fill_opacity=1.0,
            stroke_width=0,
        )
        pivot_hole.move_to([0, 0, z0 + 0.005])
        self.add(pivot_hole)

        self._tip_y  = tip_y
        self._tail_y = tail_y
        self._length = length


# ──────────────────────────────────────────────────────────────────────────────
# Spinner3D  ──  the main export
# ──────────────────────────────────────────────────────────────────────────────

class Spinner3D(VGroup):
    """
    A detailed circular probability spinner.

    Parameters
    ----------
    weights : sequence[float]
        Unnormalised sector weights.  E.g. ``[1, 1, 1, 1]`` for uniform
        4-sector; ``[3, 1, 2]`` for 3 unequal sectors.
    radius : float
        Board radius in Manim units.  Default ``2.0``.
    labels : sequence[str] | None
        Override label text per sector (used with ``label_style="name"``).
        If ``None``, labels are derived from the weights.
    colors : sequence[str] | None
        Override sector colours.  Cycles through ``SECTOR_COLORS`` if None.
    sector_names : sequence[str] | None
        Display names shown on sectors (label_style="name").
    board_palette : str
        Named board palette key.  Default ``"wood"``.
    hub_palette : str
        Named hub palette key.  Default ``"brass"``.
    needle_palette : str
        Named needle palette key.  Default ``"red_chrome"``.
    show_labels : bool
        Show probability labels on sectors.  Default ``True``.
    label_style : "fraction" | "percent" | "both" | "name"
        Label format.  Default ``"both"``.
    start_angle : float
        Angle of first sector's leading edge.  Default ``PI/2`` (12 o'clock).
    needle_angle : float
        Initial needle pointing angle.  Default ``PI/2`` (pointing up).
    tick_minor_n : int
        Minor tick marks per sector.  Default ``4``.
    lathe_slices : int
        Unused (kept for API consistency).

    Attributes
    ----------
    board       : _SpinnerBoard
    sectors     : _SpinnerSectors
    hub         : _SpinnerHub
    needle      : _SpinnerNeedle
    weights     : list[float]
    n_sectors   : int
    needle_angle: float   — current needle pointing angle (radians)
    sector_mid_angles : list[float]   — midpoint angle of each sector

    Class methods
    -------------
    Spinner3D.uniform(n, **kw)     — equal-weight n-sector spinner
    Spinner3D.from_probs(p, **kw)  — from probability list summing to 1
    Spinner3D.named(items, **kw)   — from [(name, weight), ...] list

    Examples
    --------
    ::

        # 4-sector uniform spinner
        s = Spinner3D.uniform(4, radius=2.0, board_palette="wood")

        # Custom weights spinner
        s = Spinner3D([3, 1, 2], radius=2.5, label_style="percent")

        # Named outcomes
        s = Spinner3D.named([
            ("Red",    3),
            ("Blue",   2),
            ("Green",  1),
        ], label_style="name")

        # Animate
        scene.play(SpinToOutcome(s, outcome=0, n_full_rotations=4))
    """

    def __init__(
        self,
        weights: Sequence[float],
        radius: float = 2.0,
        labels: Optional[Sequence[str]] = None,
        colors: Optional[Sequence[str]] = None,
        sector_names: Optional[Sequence[str]] = None,
        board_palette: str = "wood",
        hub_palette: str = "brass",
        needle_palette: str = "red_chrome",
        show_labels: bool = True,
        label_style: Literal[
            "fraction", "percent", "both", "name"
        ] = "both",
        start_angle: float = PI / 2,
        needle_angle: float = PI / 2,
        tick_minor_n: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        weights = list(weights)
        n       = len(weights)
        total   = sum(weights)

        self._weights     = weights
        self._total       = total
        self._n_sectors   = n
        self._radius      = radius
        self._start_angle = start_angle
        self._needle_angle = needle_angle

        # Resolve palettes
        bp  = BOARD_PALETTES.get(board_palette,  BOARD_PALETTES["wood"])
        hp  = HUB_PALETTES.get(hub_palette,      HUB_PALETTES["brass"])
        np_ = NEEDLE_PALETTES.get(needle_palette, NEEDLE_PALETTES["red_chrome"])

        # ── Board ─────────────────────────────────────────────────────
        self.board = _SpinnerBoard(
            radius=radius,
            board_palette=bp,
            n_sectors=n,
            tick_minor_n=tick_minor_n,
        )
        self.add(self.board)

        # ── Sectors ───────────────────────────────────────────────────
        display_labels = sector_names if (label_style == "name"
                                          and sector_names) else labels
        self.sectors = _SpinnerSectors(
            weights=weights,
            radius=radius,
            labels=display_labels,
            colors=colors,
            start_angle=start_angle,
            show_labels=show_labels,
            label_style=label_style,
        )
        self.add(self.sectors)

        # Precompute sector midpoint angles for outcome lookup
        cum = [start_angle]
        for w in weights:
            cum.append(cum[-1] - (w / total) * TAU)
        self._sector_mid_angles = [
            (cum[i] + cum[i+1]) / 2 for i in range(n)
        ]
        self._sector_start_angles = cum[:-1]
        self._sector_end_angles   = cum[1:]

        # ── Hub ───────────────────────────────────────────────────────
        hub_r = radius * 0.085
        self.hub = _SpinnerHub(radius=hub_r, hub_palette=hp)
        self.add(self.hub)

        # ── Needle ────────────────────────────────────────────────────
        needle_len = radius * 0.82
        self.needle = _SpinnerNeedle(
            length=needle_len,
            needle_palette=np_,
        )
        # Rotate to initial angle
        self.needle.rotate(
            needle_angle - PI / 2,   # offset: needle built pointing +Y = PI/2
            axis=OUT,
            about_point=ORIGIN,
        )
        self.add(self.needle)

        # Track current angle
        self._current_angle = needle_angle

    # ──────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────

    @property
    def weights(self) -> list[float]:
        return self._weights

    @property
    def n_sectors(self) -> int:
        return self._n_sectors

    @property
    def needle_angle(self) -> float:
        return self._current_angle

    @needle_angle.setter
    def needle_angle(self, angle: float):
        delta = angle - self._current_angle
        self.needle.rotate(delta, axis=OUT,
                           about_point=self.get_center())
        self._current_angle = angle

    @property
    def sector_mid_angles(self) -> list[float]:
        return self._sector_mid_angles

    # ──────────────────────────────────────────────────────────────────
    # Outcome lookup
    # ──────────────────────────────────────────────────────────────────

    def angle_to_sector(self, angle: float) -> int:
        """
        Return the sector index (0-based) that contains ``angle``.
        ``angle`` is in radians; needle typically points in +Y direction
        initially.
        """
        # Normalise angle to [0, TAU)
        a = angle % TAU
        for i, (sa, ea) in enumerate(
            zip(self._sector_start_angles, self._sector_end_angles)
        ):
            # Sectors go clockwise (end < start), normalise
            sa_n = sa % TAU
            ea_n = ea % TAU
            if ea_n <= sa_n:   # wraps around
                if a >= ea_n and a <= sa_n:
                    return i
            else:
                if ea_n <= a <= sa_n:
                    return i
        return 0   # fallback

    def sector_target_angle(self, sector: int) -> float:
        """Mid-angle of sector ``sector`` (for landing the needle)."""
        return self._sector_mid_angles[sector % self._n_sectors]

    def random_spin_outcome(self) -> int:
        """Sample a sector index according to the weight distribution."""
        rng = np.random.default_rng()
        probs = np.array(self._weights) / self._total
        return int(rng.choice(self._n_sectors, p=probs))

    # ──────────────────────────────────────────────────────────────────
    # Class-method constructors
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def uniform(cls, n: int = 4, **kwargs) -> "Spinner3D":
        """
        Equal-weight spinner with ``n`` sectors.

        Example::

            s = Spinner3D.uniform(6, radius=2.5, board_palette="acrylic_dark")
        """
        return cls(weights=[1] * n, **kwargs)

    @classmethod
    def from_probs(
        cls,
        probs: Sequence[float],
        **kwargs,
    ) -> "Spinner3D":
        """
        Create spinner from a probability list (need not sum to exactly 1;
        will be normalised).

        Example::

            s = Spinner3D.from_probs([0.5, 0.3, 0.2],
                                     label_style="percent")
        """
        return cls(weights=list(probs), **kwargs)

    @classmethod
    def named(
        cls,
        items: Sequence[tuple[str, float]],
        **kwargs,
    ) -> "Spinner3D":
        """
        Create spinner from ``[(name, weight), ...]``.

        Example::

            s = Spinner3D.named([
                ("Heads", 1),
                ("Tails", 1),
            ], radius=1.8, label_style="name")
        """
        names, weights = zip(*items)
        return cls(
            weights=list(weights),
            sector_names=list(names),
            label_style="name",
            **kwargs,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Physics helpers
# ──────────────────────────────────────────────────────────────────────────────

def _exponential_decay_angle(
    alpha: float,
    start_angle: float,
    total_extra_angle: float,
    decay_k: float = 4.0,
) -> float:
    """
    Angular position following exponential deceleration.

    θ(α) = start + total_extra * (1 - e^(-k·α)) / (1 - e^(-k))

    At α=0: θ = start
    At α=1: θ = start + total_extra
    """
    norm = (1 - np.exp(-decay_k * alpha)) / (1 - np.exp(-decay_k) + 1e-12)
    return start_angle + total_extra_angle * norm


def _wobble_offset(alpha: float, amplitude: float, n_wobbles: int = 3) -> float:
    """
    Small overshoot wobble near the end of the spin (α → 1).

    Returns an angular offset that decays quickly to 0.
    Used to simulate a needle "settling" into its sector.
    """
    # Only active in the last 15% of the animation
    if alpha < 0.85:
        return 0.0
    local = (alpha - 0.85) / 0.15   # 0 → 1 in the last 15%
    decay = np.exp(-local * 6)
    return amplitude * decay * np.sin(local * n_wobbles * PI)


# ──────────────────────────────────────────────────────────────────────────────
# Animations
# ──────────────────────────────────────────────────────────────────────────────

class SpinToOutcome(Animation):
    """
    Spin the needle to land precisely on a target sector, using
    exponential angular deceleration with a settling wobble.

    The needle completes ``n_full_rotations`` complete turns plus enough
    extra angle to land on the midpoint of ``outcome``.

    Parameters
    ----------
    spinner         : Spinner3D
    outcome         : int       — target sector index (0-based)
    n_full_rotations: int       — full revolutions before landing (default 4)
    wobble_amplitude: float     — settling overshoot angle (radians, default 0.06)
    decay_k         : float     — exponential decay rate (default 4.5)
    run_time        : float
    """

    def __init__(
        self,
        spinner: Spinner3D,
        outcome: int,
        n_full_rotations: int = 4,
        wobble_amplitude: float = 0.06,
        decay_k: float = 4.5,
        **kwargs,
    ):
        self.spinner          = spinner
        self.outcome          = outcome % spinner.n_sectors
        self.wobble_amplitude = wobble_amplitude
        self.decay_k          = decay_k

        start_angle  = spinner._current_angle
        target_angle = spinner.sector_target_angle(outcome)

        # We always spin clockwise (decreasing angle).
        # Total angle = full rotations + delta to target
        delta = (start_angle - target_angle) % TAU
        self.total_sweep   = n_full_rotations * TAU + delta
        self.start_angle   = start_angle

        kwargs.setdefault("run_time", 3.0)
        kwargs.setdefault("rate_func", rate_functions.linear)   # custom below
        super().__init__(spinner.needle, **kwargs)

    def interpolate_mobject(self, alpha: float):
        # Exponential deceleration
        swept = _exponential_decay_angle(
            alpha, 0, self.total_sweep, self.decay_k
        )
        # Settling wobble in the final phase
        wobble = _wobble_offset(alpha, self.wobble_amplitude)

        new_angle = self.start_angle - swept + wobble

        # Apply rotation
        self.spinner.needle.become(self.starting_mobject.copy())
        delta = new_angle - self.start_angle
        self.spinner.needle.rotate(
            delta, axis=OUT,
            about_point=self.spinner.get_center(),
        )

        if alpha >= 1.0:
            self.spinner._current_angle = (
                self.start_angle - self.total_sweep
            ) % TAU


class FreeSpinDecay(Animation):
    """
    Pure physics spin with no predetermined outcome.
    The needle starts at ``angular_velocity`` rad/s and decelerates
    via exponential drag, stopping wherever it lands.

    After the animation, ``spinner.needle_angle`` holds the final angle
    and ``spinner.angle_to_sector(spinner.needle_angle)`` gives the outcome.

    Parameters
    ----------
    spinner          : Spinner3D
    angular_velocity : float   — initial angular speed (rad/s, default 8π)
    drag_k           : float   — drag coefficient (default 1.5)
    run_time         : float
    """

    def __init__(
        self,
        spinner: Spinner3D,
        angular_velocity: float = 8 * PI,
        drag_k: float = 1.5,
        **kwargs,
    ):
        self.spinner    = spinner
        self.omega_0    = angular_velocity
        self.drag_k     = drag_k
        self.start_angle = spinner._current_angle

        # Total angle swept: ∫₀^T ω₀·e^(−k·t) dt = ω₀/k·(1 − e^(−k·T))
        # With normalised time (T=1): total = ω₀/k · (1 − e^(−k))
        run_time_ = kwargs.get("run_time", 3.5)
        self.total_swept = (
            angular_velocity / drag_k * (1 - np.exp(-drag_k * run_time_))
        )

        kwargs.setdefault("run_time", 3.5)
        kwargs.setdefault("rate_func", rate_functions.linear)
        super().__init__(spinner.needle, **kwargs)

    def interpolate_mobject(self, alpha: float):
        # ω(t) = ω₀·e^(−k·t) → θ(t) = ω₀/k·(1−e^(−k·t))
        swept = (self.omega_0 / self.drag_k
                 * (1 - np.exp(-self.drag_k * alpha * self.run_time)))
        new_angle = self.start_angle - swept

        self.spinner.needle.become(self.starting_mobject.copy())
        delta = new_angle - self.start_angle
        self.spinner.needle.rotate(
            delta, axis=OUT,
            about_point=self.spinner.get_center(),
        )

        if alpha >= 1.0:
            self.spinner._current_angle = new_angle % TAU


class FlickSpin(Animation):
    """
    Short sharp impulse — needle travels only 1–3 sector widths.
    Fast ease-in, slow ease-out.  Good for illustrating a single trial.

    Parameters
    ----------
    spinner  : Spinner3D
    outcome  : int     — target sector index
    run_time : float
    """

    def __init__(
        self,
        spinner: Spinner3D,
        outcome: int,
        **kwargs,
    ):
        self.spinner     = spinner
        self.outcome     = outcome % spinner.n_sectors
        self.start_angle = spinner._current_angle
        target           = spinner.sector_target_angle(outcome)
        delta            = (self.start_angle - target) % TAU
        # At most 1.5 rotations
        self.total_sweep = delta % TAU + TAU * 0.5

        kwargs.setdefault("run_time", 0.9)
        kwargs.setdefault("rate_func", rate_functions.ease_in_out_expo)
        super().__init__(spinner.needle, **kwargs)

    def interpolate_mobject(self, alpha: float):
        swept     = alpha * self.total_sweep
        new_angle = self.start_angle - swept

        self.spinner.needle.become(self.starting_mobject.copy())
        self.spinner.needle.rotate(
            new_angle - self.start_angle,
            axis=OUT,
            about_point=self.spinner.get_center(),
        )

        if alpha >= 1.0:
            self.spinner._current_angle = new_angle % TAU


class RicochetSpin(Succession):
    """
    Needle lands near the boundary of a sector, ticks slightly past it,
    then bounces back to settle in the correct sector.

    Simulates the "click-stop" ratchet feel of a physical spinner.

    Parameters
    ----------
    spinner  : Spinner3D
    outcome  : int   — final target sector
    overshoot_frac : float — fraction of sector width to overshoot (0.2–0.4)
    run_time : float
    """

    def __init__(
        self,
        spinner: Spinner3D,
        outcome: int,
        overshoot_frac: float = 0.28,
        **kwargs,
    ):
        n        = spinner.n_sectors
        outcome  = outcome % n
        total_w  = sum(spinner.weights)
        # Sector angular width
        sec_w    = (spinner.weights[outcome] / total_w) * TAU
        overshoot = sec_w * overshoot_frac

        # Phase 1: main spin, lands just past the sector boundary
        boundary_angle = spinner._sector_end_angles[outcome]
        phase1 = SpinToOutcome(
            spinner, outcome,
            n_full_rotations=3,
            wobble_amplitude=0.0,
            run_time=2.0,
        )
        # Override target to boundary
        phase1.total_sweep += overshoot

        # Phase 2: bounce back
        def _bounce(sp=spinner, ov=overshoot):
            return ApplyMethod(
                sp.needle.rotate,
                ov * 0.80,   # bounce back ~80% of overshoot
                about_point=sp.get_center(),
                run_time=0.20,
                rate_func=rate_functions.ease_out_bounce,
            )

        # Phase 3: settle
        def _settle(sp=spinner, ov=overshoot):
            return ApplyMethod(
                sp.needle.rotate,
                -ov * 0.80,
                about_point=sp.get_center(),
                run_time=0.12,
                rate_func=rate_functions.ease_in_out_sine,
            )

        kwargs.setdefault("run_time", 2.32)
        super().__init__(phase1, _bounce(), _settle(), **kwargs)


class SpinSequence(Succession):
    """
    Animate ``n_trials`` successive spins of the spinner, yielding
    a list of outcomes.

    Parameters
    ----------
    spinner   : Spinner3D
    n_trials  : int
    outcomes  : list[int] | None  — specify outcomes; random if None
    spin_cls  : Animation class   — which spin animation to use
    spin_kwargs : dict             — forwarded to each spin animation
    pause     : float             — wait time between spins
    """

    def __init__(
        self,
        spinner: Spinner3D,
        n_trials: int = 5,
        outcomes: Optional[list[int]] = None,
        spin_cls=None,
        spin_kwargs: Optional[dict] = None,
        pause: float = 0.3,
        **kwargs,
    ):
        if spin_cls is None:
            spin_cls = SpinToOutcome
        if spin_kwargs is None:
            spin_kwargs = {"n_full_rotations": 3, "run_time": 2.0}
        if outcomes is None:
            outcomes = [spinner.random_spin_outcome()
                        for _ in range(n_trials)]

        self.outcomes = outcomes[:n_trials]
        anims = []
        for outcome in self.outcomes:
            anims.append(spin_cls(spinner, outcome, **spin_kwargs))

        kwargs.setdefault("run_time",
                          sum(a.run_time for a in anims) + pause * len(anims))
        super().__init__(*anims, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience factories
# ──────────────────────────────────────────────────────────────────────────────

def make_bernoulli_spinner(
    p: float = 0.5,
    labels: tuple[str, str] = ("Success", "Failure"),
    colors: tuple[str, str] = ("#2A9D8F", "#E63946"),
    **spinner_kwargs,
) -> Spinner3D:
    """
    Two-sector spinner for Bernoulli / coin-flip demonstrations.

    Parameters
    ----------
    p      : probability of success (sector 0)
    labels : names for the two sectors
    colors : colours for the two sectors
    """
    return Spinner3D(
        weights=[p, 1 - p],
        sector_names=list(labels),
        colors=list(colors),
        label_style="name",
        **spinner_kwargs,
    )


def make_die_spinner(
    n: int = 6,
    **spinner_kwargs,
) -> Spinner3D:
    """
    Uniform n-sector spinner mimicking a fair die.

    Parameters
    ----------
    n : number of faces (2–20)
    """
    spinner_kwargs.setdefault("label_style", "name")
    return Spinner3D(
        weights=[1] * n,
        sector_names=[str(i + 1) for i in range(n)],
        **spinner_kwargs,
    )


def make_markov_spinner(
    state_names: Sequence[str],
    transition_row: Sequence[float],
    current_state: int = 0,
    **spinner_kwargs,
) -> Spinner3D:
    """
    Spinner representing one row of a Markov transition matrix.
    Used to visualise the probability of transitioning from
    ``current_state`` to each other state.

    Parameters
    ----------
    state_names      : names of all states
    transition_row   : transition probabilities from the current state
    current_state    : index of the "from" state (used only for display)
    """
    spinner_kwargs.setdefault("label_style", "name")
    return Spinner3D(
        weights=list(transition_row),
        sector_names=list(state_names),
        **spinner_kwargs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Demo Scene  (run with: manim -pql spinner.py SpinnerDemo)
# ──────────────────────────────────────────────────────────────────────────────

try:
    from manim import ThreeDScene, DEGREES, Wait

    class SpinnerDemo(ThreeDScene):
        """Showcase scene for Spinner3D and all animations."""

        def construct(self):
            self.set_camera_orientation(phi=72 * DEGREES, theta=-45 * DEGREES)
            self.begin_ambient_camera_rotation(rate=0.02)

            # ── Four-sector uniform spinner (wood board) ──────────────
            s4 = Spinner3D.uniform(
                n=4,
                radius=2.0,
                board_palette="wood",
                hub_palette="brass",
                needle_palette="red_chrome",
                label_style="both",
            )
            s4.shift(LEFT * 3.5)
            self.add(s4)

            # ── Named Bernoulli spinner (acrylic dark) ────────────────
            bern = make_bernoulli_spinner(
                p=0.65,
                labels=("Win", "Lose"),
                colors=("#2A9D8F", "#E63946"),
                radius=1.6,
                board_palette="acrylic_dark",
                hub_palette="chrome",
                needle_palette="gold",
            )
            bern.shift(RIGHT * 2.5)
            self.add(bern)
            self.wait(0.8)

            # ── Spin the 4-sector spinner to sector 2 ─────────────────
            self.play(
                SpinToOutcome(s4, outcome=2, n_full_rotations=5,
                              wobble_amplitude=0.07, run_time=3.2),
                run_time=3.2,
            )
            self.wait(0.5)

            # ── Flick the Bernoulli spinner ───────────────────────────
            self.play(FlickSpin(bern, outcome=0, run_time=1.1))
            self.wait(0.4)

            # ── Ricochet on the 4-sector spinner ─────────────────────
            self.play(
                RicochetSpin(s4, outcome=3, overshoot_frac=0.30,
                             run_time=2.4),
            )
            self.wait(0.5)

            # ── 6-sector die spinner, run 4 trials ───────────────────
            d6_spin = make_die_spinner(
                n=6,
                radius=1.8,
                board_palette="casino_red",
                hub_palette="brass",
                needle_palette="white",
                label_style="name",
            )
            d6_spin.shift(DOWN * 0.5 + LEFT * 0.5)
            self.play(FadeIn(d6_spin, shift=UP * 0.3), run_time=0.6)

            self.play(SpinSequence(
                d6_spin,
                n_trials=4,
                outcomes=[5, 2, 5, 1],
                spin_kwargs={"n_full_rotations": 3, "run_time": 1.8},
                pause=0.25,
            ))
            self.wait(2.0)

except ImportError:
    pass   # Manim not installed; skip demo scene definition