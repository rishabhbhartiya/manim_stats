"""
manim_stats/props/coin.py
=========================
Coin3D — A highly detailed, physically-realistic 3D coin for use in
Manim statistics animations (Bernoulli trials, Binomial distributions,
probability demonstrations, etc.)

Design goals
------------
* Reeded / milled edge  — serrated ring of thin rectangular teeth around
  the cylindrical rim, just like a real coin.
* Raised relief faces   — both obverse and reverse have a recessed field,
  a raised inner disc, embossed star ring, and a text arc.
* Multi-layer shading   — gold base + darker groove layer + bright
  highlight cap, giving the illusion of depth without ray-tracing.
* Rich animation suite  — FlipCoin (clean single flip), TumbleCoin
  (tumble with random wobble axis), LandCoin (landing bounce with squash),
  SpinCoin (upright edge-spin like a dropped coin slowing to a stop),
  RollCoin (rolling along a surface).
* Full configurability  — radius, thickness, reed_count, colors, outcome.

Dependencies
------------
  manim (CE or GL), numpy

Usage
-----
    from manim_stats.props.coin import Coin3D, FlipCoin, TumbleCoin

    class BernoulliScene(ThreeDScene):
        def construct(self):
            coin = Coin3D(outcome="heads", color_scheme="gold")
            self.add(coin)
            self.play(FlipCoin(coin, outcome="tails"))
"""

from __future__ import annotations

import numpy as np
from manim import (
    # Core
    VGroup, Group,
    # 3-D primitives
    Cylinder, Annulus, Circle, Arc, Dot,
    # Geometry helpers
    Line, Polygon, RegularPolygon,
    # Text / TeX
    Text, MathTex,
    # Animation base-classes
    Animation, Succession, AnimationGroup,
    # Transform family
    Rotate, ApplyMethod, MoveAlongPath,
    # Updaters / value trackers
    ValueTracker,
    # Colour helpers
    color_to_rgba, interpolate_color,
    # Common colours
    WHITE, BLACK, GOLD, GOLD_A, GOLD_B, GOLD_C, GOLD_D, GOLD_E,
    GREY, GREY_A, GREY_B, GREY_C, LIGHT_GREY,
    BLUE_A, BLUE_E,
    # Math / positioning constants
    UP, DOWN, LEFT, RIGHT, OUT, IN,
    ORIGIN, PI, TAU,
    # Rate functions
    rate_functions,
    # Typing
    ManimColor,
)
from manim.utils.color import ManimColor, color_to_rgb
from typing import Literal, Optional
import warnings

# ---------------------------------------------------------------------------
# Colour Palettes
# ---------------------------------------------------------------------------

COIN_PALETTES: dict[str, dict] = {
    "gold": {
        "edge":      GOLD_D,       # rim / reeds
        "body":      GOLD_C,       # main face field
        "relief":    GOLD_A,       # raised disc / stars
        "groove":    GOLD_E,       # recessed grooves (darker)
        "highlight": "#FFE97A",    # bright specular highlight cap
        "text":      GOLD_D,
    },
    "silver": {
        "edge":      GREY_B,
        "body":      LIGHT_GREY,
        "relief":    WHITE,
        "groove":    GREY_C,
        "highlight": "#F8F8FF",
        "text":      GREY_B,
    },
    "copper": {
        "edge":      "#8B4513",
        "body":      "#B5651D",
        "relief":    "#CD853F",
        "groove":    "#6B3410",
        "highlight": "#DEB887",
        "text":      "#8B4513",
    },
    "dark": {
        "edge":      GREY_C,
        "body":      GREY,
        "relief":    GREY_A,
        "groove":    BLACK,
        "highlight": GREY_A,
        "text":      GREY_A,
    },
}

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _arc_text(
    text: str,
    radius: float,
    start_angle: float,
    span_angle: float,
    font_size: float = 14,
    color: ManimColor = WHITE,
    font: str = "serif",
) -> VGroup:
    """Place individual characters along a circular arc."""
    group = VGroup()
    n = len(text)
    if n == 0:
        return group
    delta = span_angle / max(n - 1, 1)
    for i, ch in enumerate(text):
        angle = start_angle + i * delta
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        char = Text(ch, font_size=font_size, color=color, font=font)
        char.move_to([x, y, 0])
        # Rotate each character so it faces outward
        char.rotate(angle - PI / 2, about_point=char.get_center())
        group.add(char)
    return group


def _star_polygon(
    n_points: int = 5,
    outer_r: float = 0.1,
    inner_r: float = 0.04,
    color: ManimColor = WHITE,
    fill_opacity: float = 1.0,
) -> Polygon:
    """Build a star polygon as a Manim Polygon."""
    verts = []
    for k in range(2 * n_points):
        r = outer_r if k % 2 == 0 else inner_r
        angle = PI / 2 + k * PI / n_points
        verts.append([r * np.cos(angle), r * np.sin(angle), 0])
    star = Polygon(*verts, color=color, fill_color=color,
                   fill_opacity=fill_opacity, stroke_width=0)
    return star


# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------

class _CoinEdge(VGroup):
    """
    Reeded (milled) coin edge — a ring of thin rectangular teeth around
    the cylindrical rim, simulating the knurled edge of a real coin.

    Implementation strategy
    -----------------------
    Manim's Cylinder is rendered as a parametric surface; adding
    individual groove geometry on top of it is expensive.  Instead we
    approximate the reeded edge with two concentric thin cylinders:
      • outer_ring  — full cylinder at coin radius (the base shape)
      • reeds       — N thin VGroup rectangles (Polygons) extruded along
                      the edge, slightly darker, creating the visual rhythm
                      of milled serrations when viewed from any angle.
    """

    def __init__(
        self,
        radius: float,
        thickness: float,
        reed_count: int = 80,
        color: ManimColor = GOLD_D,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.coin_radius = radius
        self.coin_thickness = thickness

        # ── Base cylindrical band ──────────────────────────────────────
        edge_cyl = Cylinder(
            radius=radius,
            height=thickness,
            direction=UP,
            resolution=(reed_count * 2, 2),
            fill_opacity=1,
            stroke_width=0,
            fill_color=color,
        )
        self.add(edge_cyl)

        # ── Reed grooves (dark thin strips) ───────────────────────────
        groove_color = interpolate_color(color, BLACK, 0.35)
        groove_w = TAU * radius / reed_count * 0.38   # ~38 % of pitch = dark
        reeds = VGroup()
        for i in range(reed_count):
            angle = i * TAU / reed_count
            # A thin rectangular polygon in 3-D, standing vertically
            x0 = radius * np.cos(angle)
            y0 = radius * np.sin(angle)
            # Offset slightly outward so it sits on the surface
            r_out = radius + 0.002
            x1 = r_out * np.cos(angle + groove_w / radius)
            y1 = r_out * np.sin(angle + groove_w / radius)
            bottom_z = -thickness / 2
            top_z    =  thickness / 2
            rect = Polygon(
                [x0, y0, bottom_z],
                [x1, y1, bottom_z],
                [x1, y1, top_z],
                [x0, y0, top_z],
                fill_color=groove_color,
                fill_opacity=1,
                stroke_width=0,
            )
            reeds.add(rect)
        self.add(reeds)


class _CoinFace(VGroup):
    """
    One face of a coin (obverse = heads, reverse = tails).

    Layers (back → front, all at z = face_z):
      1. base_disc     – solid filled circle (the field)
      2. rim_ring      – thin annulus at the edge, slightly raised color
      3. inner_disc    – raised central relief disc
      4. engraving_lines – radial fine lines from inner disc to rim (decorative)
      5. star_ring     – evenly-spaced 5-pointed stars between disc and rim
      6. arc_text      – curved text along the inner rim
      7. center_symbol – large central emblem (crown for heads, cross for tails)
      8. highlight_cap – bright elliptical gradient blob simulating specular light
    """

    def __init__(
        self,
        radius: float,
        face_z: float,                          # +thickness/2 or -thickness/2
        side: Literal["heads", "tails"] = "heads",
        palette: dict = None,
        font: str = "serif",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if palette is None:
            palette = COIN_PALETTES["gold"]
        self.face_radius = radius
        self.face_z = face_z

        # ── 1. Base disc (field) ──────────────────────────────────────
        base = Circle(
            radius=radius,
            fill_color=palette["body"],
            fill_opacity=1,
            stroke_width=0,
        )
        base.move_to([0, 0, face_z])
        self.add(base)

        # ── 2. Outer rim ring ─────────────────────────────────────────
        rim = Annulus(
            inner_radius=radius * 0.88,
            outer_radius=radius,
            fill_color=palette["edge"],
            fill_opacity=1,
            stroke_width=0,
        )
        rim.move_to([0, 0, face_z + 0.001])
        self.add(rim)

        # ── 3. Inner relief disc ──────────────────────────────────────
        relief = Circle(
            radius=radius * 0.72,
            fill_color=palette["relief"],
            fill_opacity=1,
            stroke_width=0,
        )
        relief.move_to([0, 0, face_z + 0.002])
        self.add(relief)

        # ── 4. Radial engraving lines ─────────────────────────────────
        n_lines = 36
        groove_col = interpolate_color(palette["body"], BLACK, 0.22)
        engravings = VGroup()
        for i in range(n_lines):
            ang = i * TAU / n_lines
            r_in  = radius * 0.73
            r_out = radius * 0.87
            line = Line(
                start=[r_in  * np.cos(ang), r_in  * np.sin(ang), face_z + 0.0015],
                end  =[r_out * np.cos(ang), r_out * np.sin(ang), face_z + 0.0015],
                stroke_color=groove_col,
                stroke_width=0.6,
            )
            engravings.add(line)
        self.add(engravings)

        # ── 5. Star ring ──────────────────────────────────────────────
        n_stars   = 12
        star_r    = radius * 0.825
        star_size = radius * 0.055
        stars = VGroup()
        for i in range(n_stars):
            ang = i * TAU / n_stars
            s = _star_polygon(
                n_points=5,
                outer_r=star_size,
                inner_r=star_size * 0.42,
                color=palette["edge"],
            )
            s.move_to([star_r * np.cos(ang), star_r * np.sin(ang), face_z + 0.003])
            stars.add(s)
        self.add(stars)

        # ── 6. Arc text ───────────────────────────────────────────────
        if side == "heads":
            arc_str   = "· HEADS ·"
            arc_start = PI * 1.15
            arc_span  = -PI * 1.3
        else:
            arc_str   = "· TAILS ·"
            arc_start = PI * 0.1
            arc_span  =  PI * 1.3
        arc_grp = _arc_text(
            arc_str,
            radius=radius * 0.60,
            start_angle=arc_start,
            span_angle=arc_span,
            font_size=radius * 28,     # scale font with coin size
            color=palette["text"],
            font=font,
        )
        arc_grp.shift([0, 0, face_z + 0.004])
        self.add(arc_grp)

        # ── 7. Centre symbol ──────────────────────────────────────────
        sym_scale = radius * 0.85
        if side == "heads":
            # Crown: three arcs on top of a rectangle base
            crown = self._build_crown(sym_scale * 0.38, palette["groove"])
        else:
            # Ornate cross / shield emblem
            crown = self._build_shield(sym_scale * 0.38, palette["groove"])
        crown.move_to([0, 0, face_z + 0.005])
        self.add(crown)

        # ── 8. Specular highlight cap ─────────────────────────────────
        # Small bright ellipse offset toward top-left — simulates a
        # point light source above-left of the coin.
        highlight = Circle(
            radius=radius * 0.28,
            fill_color=palette["highlight"],
            fill_opacity=0.38,
            stroke_width=0,
        )
        highlight.scale([1.4, 0.65, 1])          # squash into ellipse
        highlight.move_to([
            -radius * 0.22,
             radius * 0.28,
            face_z + 0.006,
        ])
        self.add(highlight)

    # ------------------------------------------------------------------
    # Symbol builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_crown(size: float, color: ManimColor) -> VGroup:
        """Stylised 3-point crown from simple polygons."""
        g = VGroup()
        # Base band
        base = Polygon(
            [-size,       -size * 0.15, 0],
            [ size,       -size * 0.15, 0],
            [ size,        size * 0.15, 0],
            [-size,        size * 0.15, 0],
            fill_color=color, fill_opacity=1, stroke_width=0,
        )
        g.add(base)
        # Left tooth
        lt = Polygon(
            [-size,        size * 0.15, 0],
            [-size * 0.65, size * 0.8,  0],
            [-size * 0.3,  size * 0.15, 0],
            fill_color=color, fill_opacity=1, stroke_width=0,
        )
        g.add(lt)
        # Centre tooth (tallest)
        ct = Polygon(
            [-size * 0.2,  size * 0.15, 0],
            [ 0,           size,         0],
            [ size * 0.2,  size * 0.15, 0],
            fill_color=color, fill_opacity=1, stroke_width=0,
        )
        g.add(ct)
        # Right tooth
        rt = Polygon(
            [ size * 0.3,  size * 0.15, 0],
            [ size * 0.65, size * 0.8,  0],
            [ size,        size * 0.15, 0],
            fill_color=color, fill_opacity=1, stroke_width=0,
        )
        g.add(rt)
        # Jewel dots on each tooth tip
        for pos in [
            [-size * 0.65 + size * 0.0, size * 0.78,  0],
            [0,                          size * 0.98,   0],
            [ size * 0.65,              size * 0.78,   0],
        ]:
            dot = Circle(
                radius=size * 0.08,
                fill_color=interpolate_color(color, WHITE, 0.5),
                fill_opacity=1,
                stroke_width=0,
            )
            dot.move_to(pos)
            g.add(dot)
        return g

    @staticmethod
    def _build_shield(size: float, color: ManimColor) -> VGroup:
        """Simple heraldic shield outline + inner cross."""
        g = VGroup()
        # Outer shield
        shield = Polygon(
            [-size * 0.85,  size * 0.6,   0],
            [ size * 0.85,  size * 0.6,   0],
            [ size * 0.85, -size * 0.1,   0],
            [ 0,           -size,          0],
            [-size * 0.85, -size * 0.1,   0],
            fill_color=color, fill_opacity=1, stroke_width=0,
        )
        g.add(shield)
        # Inner cross (lighter)
        cross_col = interpolate_color(color, WHITE, 0.45)
        h_bar = Polygon(
            [-size * 0.7,  size * 0.15, 0],
            [ size * 0.7,  size * 0.15, 0],
            [ size * 0.7, -size * 0.15, 0],
            [-size * 0.7, -size * 0.15, 0],
            fill_color=cross_col, fill_opacity=1, stroke_width=0,
        )
        v_bar = Polygon(
            [-size * 0.15,  size * 0.55, 0],
            [ size * 0.15,  size * 0.55, 0],
            [ size * 0.15, -size * 0.75, 0],
            [-size * 0.15, -size * 0.75, 0],
            fill_color=cross_col, fill_opacity=1, stroke_width=0,
        )
        g.add(h_bar, v_bar)
        return g


# ---------------------------------------------------------------------------
# Coin3D  ──  the main export
# ---------------------------------------------------------------------------

class Coin3D(VGroup):
    """
    A detailed, physically-inspired 3D coin for Manim statistics scenes.

    Parameters
    ----------
    radius : float
        Coin radius in Manim units.  Default ``1.0``.
    thickness : float
        Coin thickness.  Default ``0.12``.
    reed_count : int
        Number of milled serrations on the edge.  Default ``80``.
    outcome : "heads" | "tails" | None
        Which face is currently up (+Z).  Controls initial orientation.
        ``None`` → stands on edge (upright).
    color_scheme : "gold" | "silver" | "copper" | "dark"
        Named colour palette.  Default ``"gold"``.
    custom_palette : dict | None
        Override individual colour keys (merged with named palette).
    font : str
        Font family for arc text.  Default ``"serif"``.

    Attributes
    ----------
    heads_face : _CoinFace   (the obverse)
    tails_face : _CoinFace   (the reverse — opposite normal direction)
    edge       : _CoinEdge
    outcome    : str  ("heads" | "tails")

    Examples
    --------
    Basic usage::

        coin = Coin3D(outcome="heads")
        scene.add(coin)

    Custom copper coin::

        coin = Coin3D(
            radius=1.2,
            thickness=0.15,
            reed_count=100,
            outcome="tails",
            color_scheme="copper",
        )

    Flip animation::

        scene.play(FlipCoin(coin, outcome="tails", run_time=1.5))
    """

    def __init__(
        self,
        radius: float = 1.0,
        thickness: float = 0.12,
        reed_count: int = 80,
        outcome: Literal["heads", "tails"] = "heads",
        color_scheme: Literal["gold", "silver", "copper", "dark"] = "gold",
        custom_palette: Optional[dict] = None,
        font: str = "serif",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # ── resolve palette ───────────────────────────────────────────
        palette = dict(COIN_PALETTES.get(color_scheme, COIN_PALETTES["gold"]))
        if custom_palette:
            palette.update(custom_palette)

        self._radius    = radius
        self._thickness = thickness
        self._outcome   = outcome
        self._palette   = palette

        # ── build components ──────────────────────────────────────────
        self.edge = _CoinEdge(
            radius=radius,
            thickness=thickness,
            reed_count=reed_count,
            color=palette["edge"],
        )

        # Heads face sits at +z = thickness/2 (face normal = +Z)
        self.heads_face = _CoinFace(
            radius=radius,
            face_z=thickness / 2 + 0.001,
            side="heads",
            palette=palette,
            font=font,
        )

        # Tails face sits at -z = -thickness/2 (face normal = -Z)
        # We build it at +z then flip it below
        self.tails_face = _CoinFace(
            radius=radius,
            face_z=-(thickness / 2 + 0.001),
            side="tails",
            palette=palette,
            font=font,
        )
        # Mirror the tails face so its details face outward (-Z direction)
        self.tails_face.scale([-1, 1, 1])   # reflect across YZ plane

        self.add(self.edge, self.heads_face, self.tails_face)

        # ── orient according to outcome ───────────────────────────────
        self._apply_outcome(outcome)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def outcome(self) -> str:
        return self._outcome

    @outcome.setter
    def outcome(self, value: Literal["heads", "tails"]):
        self._outcome = value

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def thickness(self) -> float:
        return self._thickness

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_outcome(self, outcome: str):
        """Rotate coin so the correct face points upward (+Y in 3-D scene)."""
        if outcome == "heads":
            # Heads face (+Z local) should point up (+Y world).
            # Rotate -90° around X-axis: Z → Y
            self.rotate(-PI / 2, axis=RIGHT, about_point=ORIGIN)
        elif outcome == "tails":
            # Tails face (-Z local) should point up.
            # Rotate +90° around X-axis: -Z → Y
            self.rotate(PI / 2, axis=RIGHT, about_point=ORIGIN)
        # else: leave flat / on edge as-is

    def show_heads_up(self):
        """Instantly orient so heads faces up (no animation)."""
        self._outcome = "heads"
        angle_to_flat = self.get_angle_from_axis()  # custom if needed
        self.rotate(-PI / 2, axis=RIGHT, about_point=self.get_center())

    def show_tails_up(self):
        """Instantly orient so tails faces up (no animation)."""
        self._outcome = "tails"
        self.rotate(PI / 2, axis=RIGHT, about_point=self.get_center())

    def set_outcome(self, outcome: Literal["heads", "tails"]):
        """Snap to outcome with no animation."""
        self._outcome = outcome

    def get_face(self, side: Literal["heads", "tails"]) -> _CoinFace:
        return self.heads_face if side == "heads" else self.tails_face

    def copy_with_outcome(self, outcome: Literal["heads", "tails"]) -> "Coin3D":
        """Return a fresh Coin3D with given outcome, same styling."""
        return Coin3D(
            radius=self._radius,
            thickness=self._thickness,
            outcome=outcome,
            color_scheme="gold",      # palette already set
            custom_palette=self._palette,
        )


# ---------------------------------------------------------------------------
# Animations
# ---------------------------------------------------------------------------

class FlipCoin(Animation):
    """
    Smooth single-axis flip of a ``Coin3D``.

    The coin rotates ``flip_rotations * PI`` radians around its
    horizontal axis, landing on the specified ``outcome``.

    Parameters
    ----------
    coin          : Coin3D
    outcome       : "heads" | "tails"  — which face lands up
    flip_rotations: int (default 2)    — full rotations before landing
    axis          : np.ndarray         — rotation axis (default RIGHT)
    run_time      : float
    rate_func     : callable
    """

    def __init__(
        self,
        coin: Coin3D,
        outcome: Literal["heads", "tails"] = "tails",
        flip_rotations: int = 2,
        axis: np.ndarray = None,
        **kwargs,
    ):
        self.coin     = coin
        self.outcome  = outcome
        self.axis     = axis if axis is not None else RIGHT
        # Total angle: full rotations + half turn to land correctly
        extra = 0 if outcome == coin.outcome else PI
        self.total_angle = flip_rotations * TAU + extra
        kwargs.setdefault("run_time", 1.2)
        kwargs.setdefault("rate_func", rate_functions.ease_in_out_cubic)
        super().__init__(coin, **kwargs)

    def interpolate_mobject(self, alpha: float):
        angle = alpha * self.total_angle
        self.coin.become(self.starting_mobject.copy())
        self.coin.rotate(angle, axis=self.axis, about_point=self.coin.get_center())
        # Update logical outcome at the end
        if alpha >= 1.0:
            self.coin.set_outcome(self.outcome)


class TumbleCoin(Animation):
    """
    Physics-inspired tumble: the coin rotates around a slightly
    randomised axis (not perfectly horizontal), giving a realistic
    tumbling-through-air feel.

    Parameters
    ----------
    coin          : Coin3D
    outcome       : "heads" | "tails"
    tumble_count  : int   — number of full tumble rotations (default 3)
    wobble_angle  : float — off-axis wobble in radians (default 0.25)
    arc_height    : float — coin rises and falls along a parabolic arc
    run_time      : float
    """

    def __init__(
        self,
        coin: Coin3D,
        outcome: Literal["heads", "tails"] = "tails",
        tumble_count: int = 3,
        wobble_angle: float = 0.25,
        arc_height: float = 1.5,
        **kwargs,
    ):
        self.coin        = coin
        self.outcome     = outcome
        self.arc_height  = arc_height

        # Random wobble axis (reproducible via seed if needed)
        base_axis = RIGHT
        wobble_dir = np.array([
            np.cos(wobble_angle) * base_axis[0],
            np.sin(wobble_angle),
            np.cos(wobble_angle) * base_axis[2],
        ])
        self.axis = wobble_dir / np.linalg.norm(wobble_dir)

        extra = 0 if outcome == coin.outcome else PI
        self.total_angle = tumble_count * TAU + extra

        self.start_pos = coin.get_center().copy()

        kwargs.setdefault("run_time", 2.0)
        kwargs.setdefault("rate_func", rate_functions.linear)
        super().__init__(coin, **kwargs)

    def interpolate_mobject(self, alpha: float):
        # Parabolic arc: up then down
        height_offset = 4 * self.arc_height * alpha * (1 - alpha)
        new_pos = self.start_pos + height_offset * UP

        self.coin.become(self.starting_mobject.copy())
        angle = alpha * self.total_angle
        self.coin.rotate(angle, axis=self.axis, about_point=self.coin.get_center())
        self.coin.move_to(new_pos)

        if alpha >= 1.0:
            self.coin.set_outcome(self.outcome)


class LandCoin(Succession):
    """
    Landing animation: coin falls onto a surface with a squash-and-stretch
    bounce, then settles flat showing the final outcome.

    Built as a ``Succession`` of:
      1. TumbleCoin  (through air)
      2. Brief squash bounce  (ApplyMethod scale)
      3. Settle (rotate to perfectly flat)

    Parameters
    ----------
    coin        : Coin3D
    outcome     : "heads" | "tails"
    drop_height : float  — starting height above landing surface
    """

    def __init__(
        self,
        coin: Coin3D,
        outcome: Literal["heads", "tails"] = "tails",
        drop_height: float = 3.0,
        **kwargs,
    ):
        tumble = TumbleCoin(
            coin, outcome=outcome,
            tumble_count=2,
            arc_height=drop_height * 0.4,
            run_time=1.4,
        )
        # Tiny squash on impact
        squash  = ApplyMethod(coin.scale, [1.08, 0.92, 1.08], run_time=0.08,
                              rate_func=rate_functions.ease_out_expo)
        restore = ApplyMethod(coin.scale, [1 / 1.08, 1 / 0.92, 1 / 1.08],
                              run_time=0.12,
                              rate_func=rate_functions.ease_in_out_sine)

        kwargs.setdefault("run_time", tumble.run_time + 0.20)
        super().__init__(tumble, squash, restore, **kwargs)


class SpinCoin(Animation):
    """
    Upright edge-spin: the coin stands on its edge and spins with
    decreasing frequency — like a coin slowly coming to rest on a table.

    The axis of rotation is the coin's own axis (perpendicular to the faces,
    i.e. UP after the coin stands upright), so this must be called *before*
    applying an outcome orientation, or the coin should be placed upright.

    Parameters
    ----------
    coin          : Coin3D
    n_spins       : float   — rotations at the start (spins slow → 0)
    tilt_angle    : float   — slight lean forward (radians, default 0.12)
    run_time      : float
    """

    def __init__(
        self,
        coin: Coin3D,
        n_spins: float = 8.0,
        tilt_angle: float = 0.12,
        **kwargs,
    ):
        self.coin       = coin
        self.n_spins    = n_spins
        self.tilt_angle = tilt_angle
        kwargs.setdefault("run_time", 3.5)
        kwargs.setdefault("rate_func", rate_functions.ease_in_cubic)
        super().__init__(coin, **kwargs)

    def interpolate_mobject(self, alpha: float):
        # Spin rate decreases quadratically → total angle = n_spins * TAU
        # Instantaneous phase = n_spins * TAU * (1 - (1-alpha)^2)
        phase = self.n_spins * TAU * (1 - (1 - alpha) ** 2)
        self.coin.become(self.starting_mobject.copy())
        # Tilt slightly forward (along X-axis)
        self.coin.rotate(self.tilt_angle * (1 - alpha), axis=RIGHT,
                         about_point=self.coin.get_center())
        # Spin around Y-axis (vertical)
        self.coin.rotate(phase, axis=UP, about_point=self.coin.get_center())


class RollCoin(Animation):
    """
    Roll the coin along a straight path on a surface.

    The coin rolls upright, rotating around its edge-contact line so
    circumference and travel distance stay consistent.

    Parameters
    ----------
    coin        : Coin3D
    distance    : float       — how far to roll (in Manim units)
    direction   : np.ndarray  — roll direction vector (default RIGHT)
    run_time    : float
    """

    def __init__(
        self,
        coin: Coin3D,
        distance: float = 5.0,
        direction: np.ndarray = None,
        **kwargs,
    ):
        self.coin      = coin
        self.distance  = distance
        self.direction = (direction / np.linalg.norm(direction)
                          if direction is not None else RIGHT)
        # Rotation axis is perpendicular to direction and UP (i.e. into screen)
        self.rot_axis  = np.cross(self.direction, UP)
        # Total rotation angle = distance / radius
        self.total_rotation = distance / coin.radius
        self.start_pos = coin.get_center().copy()

        kwargs.setdefault("run_time", 2.0)
        kwargs.setdefault("rate_func", rate_functions.linear)
        super().__init__(coin, **kwargs)

    def interpolate_mobject(self, alpha: float):
        self.coin.become(self.starting_mobject.copy())
        travelled = alpha * self.distance
        new_center = self.start_pos + travelled * self.direction
        self.coin.move_to(new_center)
        angle = alpha * self.total_rotation
        self.coin.rotate(angle, axis=self.rot_axis,
                         about_point=self.coin.get_center())


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------

def make_coin_row(
    n: int,
    outcomes: list[Literal["heads", "tails"]] = None,
    spacing: float = 0.3,
    **coin_kwargs,
) -> VGroup:
    """
    Create a horizontal row of ``n`` coins.

    Parameters
    ----------
    n        : number of coins
    outcomes : list of "heads"/"tails" per coin;
               if shorter than n, cycles through the list.
               Default: alternating heads/tails.
    spacing  : gap between adjacent coins
    **coin_kwargs : forwarded to ``Coin3D``

    Returns
    -------
    VGroup of Coin3D objects, centred at the origin.
    """
    if outcomes is None:
        outcomes = ["heads", "tails"]
    coins = VGroup()
    r = coin_kwargs.get("radius", 1.0)
    step = 2 * r + spacing
    for i in range(n):
        outcome = outcomes[i % len(outcomes)]
        c = Coin3D(outcome=outcome, **coin_kwargs)
        c.shift(RIGHT * step * i)
        coins.add(c)
    coins.center()
    return coins


def make_coin_grid(
    rows: int,
    cols: int,
    outcomes: list[list[Literal["heads", "tails"]]] = None,
    h_spacing: float = 0.25,
    v_spacing: float = 0.25,
    **coin_kwargs,
) -> VGroup:
    """
    Create a grid of coins (e.g. for visualising a sample of Bernoulli trials).

    Parameters
    ----------
    rows, cols : grid dimensions
    outcomes   : 2-D list of outcomes; falls back to random if None
    h_spacing  : horizontal gap
    v_spacing  : vertical gap
    """
    if outcomes is None:
        rng = np.random.default_rng(42)
        outcomes = [
            ["heads" if rng.random() > 0.5 else "tails" for _ in range(cols)]
            for _ in range(rows)
        ]
    grid = VGroup()
    r = coin_kwargs.get("radius", 1.0)
    h_step = 2 * r + h_spacing
    v_step = 2 * r + v_spacing
    for ri in range(rows):
        for ci in range(cols):
            outcome = outcomes[ri][ci]
            c = Coin3D(outcome=outcome, **coin_kwargs)
            c.shift(RIGHT * h_step * ci + DOWN * v_step * ri)
            grid.add(c)
    grid.center()
    return grid


# ---------------------------------------------------------------------------
# Demo Scene  (run with: manim -pql coin.py CoinDemo)
# ---------------------------------------------------------------------------

try:
    from manim import ThreeDScene, DEGREES
    from manim.camera.camera import Camera

    class CoinDemo(ThreeDScene):
        """Quick smoke-test / showcase scene for Coin3D."""

        def construct(self):
            self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
            self.begin_ambient_camera_rotation(rate=0.04)

            # ── Single coin, gold heads-up ────────────────────────────
            coin = Coin3D(radius=1.0, outcome="heads", color_scheme="gold")
            self.play(coin.animate.shift(LEFT * 2), run_time=0.5)
            self.wait(0.5)

            # ── Flip to tails ─────────────────────────────────────────
            self.play(FlipCoin(coin, outcome="tails", flip_rotations=3,
                               run_time=1.8))
            self.wait(0.5)

            # ── Silver coin on the right, spinning on edge ────────────
            silver = Coin3D(radius=0.9, outcome="heads", color_scheme="silver")
            silver.shift(RIGHT * 2.5)
            # Stand upright first
            silver.rotate(PI / 2, axis=OUT)
            self.add(silver)
            self.play(SpinCoin(silver, n_spins=6, run_time=4.0))
            self.wait(0.5)

            # ── Coin row (copper, small) ──────────────────────────────
            row = make_coin_row(
                5,
                outcomes=["heads", "tails", "heads", "heads", "tails"],
                radius=0.45,
                thickness=0.07,
                color_scheme="copper",
            )
            row.shift(DOWN * 2.5)
            self.play(*[c.animate.move_to(c.get_center()) for c in row],
                      run_time=0.8)
            self.add(row)
            self.wait(2)

except ImportError:
    pass   # Manim not installed; skip demo scene definition