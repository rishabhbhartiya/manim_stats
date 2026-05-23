"""
manim_stats/animations/flip_roll.py
=====================================
Coin flip and die roll animations for probability demonstrations.

Design philosophy
-----------------
Physical props (coins, dice) are the most concrete way to introduce
probability.  These animations are built at two levels:

    Geometry level   – ``Coin3D`` and ``Die3D`` are cylindrical /
                       polyhedral VGroup objects with correctly labelled
                       faces, independent shading, and gloss highlights.
                       They stand in for the future ``props/coin.py`` and
                       ``props/die.py`` — that module will simply export
                       the same class names.

    Animation level  – ``CoinFlipAnimation``, ``DieRollAnimation``, and
                       the sequence / accumulator classes consume those
                       props and add motion: arc trajectories, spin rates,
                       bounce physics, and result annotation.

Physical realism comes from four sub-animations composed together:
    1. Lift     – prop rises from rest position to arc apex.
    2. Spin     – prop rotates at a rate proportional to arc speed.
    3. Descent  – prop follows the other half of the arc.
    4. Settle   – one or two damped bounces then comes to rest, face up.

Teaching animations
-------------------
``BinomialAccumulator``
    Flip n coins, collect heads counts into a growing bar chart,
    watch Binomial(n, p) emerge.

``LLNAccumulator``
    Flip a single coin repeatedly, plot the running relative frequency
    of heads, watch it converge to p.  Demonstrates the Law of Large
    Numbers.

``DiceSumAccumulator``
    Roll two (or more) dice, record the sum, accumulate into a bar
    chart.  The triangular / normal-ish distribution of sums emerges.

Classes
-------
Coin3D
Die3D
CoinFlipAnimation
DieRollAnimation
BinomialAccumulator
LLNAccumulator
DiceSumAccumulator

Helpers / internals
-------------------
FlipConfig
RollConfig
_ArcPath
_BounceUpdater
_ResultLabel3D

Ready-to-render scenes
----------------------
CoinFlipScene
DieRollScene
BinomialScene
LLNScene
DiceSumScene

Usage
-----
    # Render one coin flip:
    #   manim -pql flip_roll.py CoinFlipScene

    # Embed in your own scene:
    from manim_stats.animations.flip_roll import CoinFlipAnimation, Coin3D

    class MyScene(ThreeDScene):
        def construct(self):
            coin = Coin3D(radius=0.5, thickness=0.12)
            self.add(coin)
            flip = CoinFlipAnimation(coin, outcome="heads", arc_height=2.0)
            self.play(flip.animate(self))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple, Optional, Callable, Union, Dict
import numpy as np

from manim import (
    VGroup, VMobject, Polygon, Line, DashedLine, Dot3D,
    Ellipse, Text, MathTex, Arc, Circle,
    Cylinder, ThreeDScene,
    AnimationGroup, LaggedStart, Succession,
    Create, FadeIn, FadeOut, Transform, MoveAlongPath,
    Rotate, UpdateFromAlphaFunc, Flash, Indicate,
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
    return ManimColor([ra+(rb-ra)*t, ga+(gb-ga)*t, ba+(bb-ba)*t])


# ---------------------------------------------------------------------------
# Coin3D
# ---------------------------------------------------------------------------

class Coin3D(VGroup):
    """A 3D coin prop: a flat cylinder with distinct faces.

    Geometry
    --------
    The coin is built from three layers of polygons:

    - ``face_heads``  – top circular face (n_segments-gon) in gold/silver.
    - ``face_tails``  – bottom circular face, slightly darker.
    - ``rim``         – a band of thin quadrilaterals around the edge,
                        producing visible depth and shading.
    - ``heads_label`` – "H" text on the heads face.
    - ``tails_label`` – "T" text on the tails face.
    - ``gloss``       – a small semi-transparent arc across the top face.

    Parameters
    ----------
    radius : float
        Coin radius in scene units.
    thickness : float
        Coin thickness (distance between faces).
    heads_color : ManimColor
        Main face colour (heads side).
    tails_color : ManimColor
        Tails face colour.
    rim_color : ManimColor
        Edge band colour.
    n_segments : int
        Polygon resolution for circular faces (higher = smoother).
    show_label : bool
        Whether to add "H"/"T" text labels to faces.
    position : np.ndarray
        Initial 3D position of the coin centre.
    """

    def __init__(
        self,
        radius: float = 0.45,
        thickness: float = 0.10,
        heads_color: ManimColor = ManimColor("#D4A820"),   # gold
        tails_color: ManimColor = ManimColor("#B8922A"),   # darker gold
        rim_color: ManimColor = ManimColor("#8A6A18"),     # deep gold
        n_segments: int = 48,
        show_label: bool = True,
        position: np.ndarray = ORIGIN,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.coin_radius = radius
        self.thickness = thickness
        self.heads_color = heads_color
        self.tails_color = tails_color
        self.rim_color = rim_color
        self._n_seg = n_segments
        self._pos = np.array(position, dtype=float)

        # Unit circle angles
        angles = np.linspace(0, TAU, n_segments, endpoint=False)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)

        half_t = thickness / 2

        # Heads face (z = +half_t)
        heads_pts = [
            self._pos + np.array([radius * cos_a[i], radius * sin_a[i], half_t])
            for i in range(n_segments)
        ]
        self.face_heads = Polygon(
            *heads_pts,
            fill_color=_with_opacity(heads_color, 0.96),
            fill_opacity=1.0,
            stroke_color=_darken(heads_color, 0.70),
            stroke_width=0.6,
        )

        # Tails face (z = -half_t)
        tails_pts = [
            self._pos + np.array([radius * cos_a[i], radius * sin_a[i], -half_t])
            for i in range(n_segments)
        ]
        # Reverse winding so normal faces -z
        self.face_tails = Polygon(
            *reversed(tails_pts),
            fill_color=_with_opacity(tails_color, 0.92),
            fill_opacity=1.0,
            stroke_color=_darken(tails_color, 0.65),
            stroke_width=0.5,
        )

        # Rim: n_segments quads around the edge
        rim_quads = VGroup()
        rim_col_outer = _with_opacity(rim_color, 0.90)
        rim_col_inner = _with_opacity(_darken(rim_color, 0.78), 0.88)
        for i in range(n_segments):
            j = (i + 1) % n_segments
            # Two triangles per quad (Manim Polygon works fine for quads)
            quad = Polygon(
                self._pos + np.array([radius * cos_a[i], radius * sin_a[i],  half_t]),
                self._pos + np.array([radius * cos_a[j], radius * sin_a[j],  half_t]),
                self._pos + np.array([radius * cos_a[j], radius * sin_a[j], -half_t]),
                self._pos + np.array([radius * cos_a[i], radius * sin_a[i], -half_t]),
                fill_color=_lerp_color(rim_col_outer, rim_col_inner,
                                       0.5 + 0.5 * cos_a[i]),  # shading varies around rim
                fill_opacity=1.0,
                stroke_width=0,
            )
            rim_quads.add(quad)
        self.rim = rim_quads

        # Gloss arc on heads face
        n_gloss = n_segments // 3
        gloss_pts = [
            self._pos + np.array([
                radius * 0.82 * np.cos(a),
                radius * 0.82 * np.sin(a),
                half_t + 0.001,
            ])
            for a in np.linspace(PI * 0.15, PI * 0.65, n_gloss)
        ]
        # Close the gloss arc into a thin crescent
        gloss_inner_pts = [
            self._pos + np.array([
                radius * 0.55 * np.cos(a),
                radius * 0.55 * np.sin(a),
                half_t + 0.001,
            ])
            for a in np.linspace(PI * 0.65, PI * 0.15, n_gloss)
        ]
        self.gloss = Polygon(
            *(gloss_pts + gloss_inner_pts),
            fill_color=_with_opacity(WHITE, 0.18),
            fill_opacity=1.0,
            stroke_width=0,
        )

        # Labels
        self.heads_label = VGroup()
        self.tails_label = VGroup()
        if show_label:
            h_lbl = Text("H", font_size=int(radius * 80), color=_darken(heads_color, 0.55))
            h_lbl.move_to(self._pos + np.array([0, 0, half_t + 0.005]))
            t_lbl = Text("T", font_size=int(radius * 80), color=_darken(tails_color, 0.50))
            t_lbl.move_to(self._pos + np.array([0, 0, -half_t - 0.005]))
            self.heads_label.add(h_lbl)
            self.tails_label.add(t_lbl)

        self.add(
            self.face_tails, self.rim, self.face_heads,
            self.gloss, self.heads_label, self.tails_label,
        )

    # ------------------------------------------------------------------

    def get_position(self) -> np.ndarray:
        return self.get_center()

    def show_heads(self) -> "Coin3D":
        """Orient the coin so heads (H) faces upward (+z)."""
        # Ensure face_heads is at top — default orientation is already heads-up
        return self

    def show_tails(self) -> "Coin3D":
        """Orient the coin so tails (T) faces upward (+z)."""
        self.rotate(PI, axis=RIGHT)
        return self


# ---------------------------------------------------------------------------
# Die3D
# ---------------------------------------------------------------------------

class Die3D(VGroup):
    """A 3D six-sided die (D6) with pip dots on each face.

    Geometry
    --------
    Six square faces as Polygon objects, each with independent shading
    (front-facing face is brightest, side/top are progressively darker).
    Pip dots are ``Dot3D`` spheres inset slightly above each face.

    Standard die orientation (Western convention):
        1 opposite 6,  2 opposite 5,  3 opposite 4.
    When face *k* is on top (+z), the die reads correctly.

    Parameters
    ----------
    size : float
        Half-edge length of the die (full edge = 2 × size).
    face_color : ManimColor
        Base colour for all faces (shaded per face).
    pip_color : ManimColor
        Dot colour.
    edge_color : ManimColor
        Edge stroke colour.
    corner_radius_frac : float
        Fraction of size used to round corners (visual only —
        approximated by slightly inset face polygons).
    position : np.ndarray
        Initial centre of the die.
    """

    # Face normals for a unit cube (which face points in which direction)
    _FACE_NORMALS: List[Tuple[np.ndarray, int]] = [
        (np.array([ 0,  0,  1]),  1),   # +z → face 1
        (np.array([ 0,  0, -1]),  6),   # -z → face 6
        (np.array([ 1,  0,  0]),  2),   # +x → face 2
        (np.array([-1,  0,  0]),  5),   # -x → face 5
        (np.array([ 0,  1,  0]),  3),   # +y → face 3
        (np.array([ 0, -1,  0]),  4),   # -y → face 4
    ]

    # Pip positions for faces 1–6 in a local (u,v) ∈ [-1,1]² coordinate
    _PIP_LAYOUTS: Dict[int, List[Tuple[float, float]]] = {
        1: [(0.0, 0.0)],
        2: [(-0.45, -0.45), (0.45, 0.45)],
        3: [(-0.45, -0.45), (0.0, 0.0), (0.45, 0.45)],
        4: [(-0.45, -0.45), (0.45, -0.45), (-0.45, 0.45), (0.45, 0.45)],
        5: [(-0.45, -0.45), (0.45, -0.45), (0.0, 0.0),
            (-0.45, 0.45), (0.45, 0.45)],
        6: [(-0.45, -0.60), (0.45, -0.60),
            (-0.45,  0.00), (0.45,  0.00),
            (-0.45,  0.60), (0.45,  0.60)],
    }

    def __init__(
        self,
        size: float = 0.40,
        face_color: ManimColor = ManimColor("#F0EAD6"),   # ivory
        pip_color: ManimColor = ManimColor("#1A1A2E"),    # near-black
        edge_color: ManimColor = ManimColor("#A09070"),
        corner_radius_frac: float = 0.12,
        position: np.ndarray = ORIGIN,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.die_size = size
        self.face_color = face_color
        self.pip_color = pip_color
        self._pos = np.array(position, dtype=float)

        s = size
        cr = corner_radius_frac * s   # corner inset for visual rounding
        ec = _with_opacity(edge_color, 0.55)

        self.faces = VGroup()
        self.pips = VGroup()

        # Build six faces
        for normal, face_num in self._FACE_NORMALS:
            shade = self._face_shade(normal)
            col = _with_opacity(_lerp_color(_darken(face_color, 0.80),
                                             face_color, shade), 0.97)

            # Build square face perpendicular to normal
            face_pts = self._face_corners(normal, s, cr)
            face_poly = Polygon(
                *face_pts,
                fill_color=col,
                fill_opacity=1.0,
                stroke_color=ec,
                stroke_width=0.8,
            )
            self.faces.add(face_poly)

            # Build pips on this face
            for (pu, pv) in self._PIP_LAYOUTS[face_num]:
                pip_pos = self._pip_position(normal, s, pu, pv)
                pip = Dot3D(
                    point=pip_pos,
                    radius=s * 0.085,
                    color=_with_opacity(pip_color, 0.95),
                )
                self.pips.add(pip)

        self.add(self.faces, self.pips)
        if not np.allclose(position, ORIGIN):
            self.move_to(position)

    # ------------------------------------------------------------------

    @staticmethod
    def _face_shade(normal: np.ndarray) -> float:
        """Shading factor 0–1 based on face orientation.
        Top (+z) = brightest, sides middling, bottom = darkest.
        """
        key_light = np.array([0.3, -0.5, 1.0])
        key_light = key_light / np.linalg.norm(key_light)
        dot = float(np.dot(normal, key_light))
        return 0.55 + 0.45 * max(dot, 0.0)

    def _face_corners(
        self,
        normal: np.ndarray,
        s: float,
        cr: float,
    ) -> List[np.ndarray]:
        """Compute the four corners of a face with its normal."""
        pos = self._pos + normal * s

        # Build two tangent vectors perpendicular to normal
        if abs(normal[2]) < 0.9:
            up = np.array([0, 0, 1], dtype=float)
        else:
            up = np.array([0, 1, 0], dtype=float)

        right_vec = np.cross(normal, up)
        right_vec /= np.linalg.norm(right_vec)
        up_vec = np.cross(right_vec, normal)
        up_vec /= np.linalg.norm(up_vec)

        inset = s - cr
        corners = [
            pos + inset * (-right_vec - up_vec),
            pos + inset * ( right_vec - up_vec),
            pos + inset * ( right_vec + up_vec),
            pos + inset * (-right_vec + up_vec),
        ]
        return corners

    def _pip_position(
        self,
        normal: np.ndarray,
        s: float,
        pu: float,
        pv: float,
    ) -> np.ndarray:
        """Compute a pip's 3D position on the face."""
        pos = self._pos + normal * s
        inset_offset = 0.008   # pip floats slightly above face

        if abs(normal[2]) < 0.9:
            up = np.array([0, 0, 1], dtype=float)
        else:
            up = np.array([0, 1, 0], dtype=float)

        right_vec = np.cross(normal, up)
        right_vec /= np.linalg.norm(right_vec)
        up_vec = np.cross(right_vec, normal)
        up_vec /= np.linalg.norm(up_vec)

        return pos + pu * s * right_vec + pv * s * up_vec + normal * inset_offset

    def get_top_face(self) -> int:
        """Return which face number (1–6) is currently on top (+z)."""
        best_face = 1
        best_dot = -2.0
        for normal, face_num in self._FACE_NORMALS:
            # Rotate normal by die's current transform (approximate: use center shift)
            d = float(np.dot(normal, np.array([0, 0, 1])))
            if d > best_dot:
                best_dot = d
                best_face = face_num
        return best_face


# ---------------------------------------------------------------------------
# FlipConfig
# ---------------------------------------------------------------------------

@dataclass
class FlipConfig:
    """Visual and physics parameters for a coin flip animation.

    Flip styles
    ~~~~~~~~~~~
    ``"tumble"``
        Coin tumbles end-over-end (rotates around a diameter axis).
        Most realistic.  The number of full rotations is ``n_spins``.
    ``"rotate_y"``
        Coin spins around its central axis (like a top).  Shows edge
        prominently.  Good for emphasising the coin's cylindrical shape.
    ``"arc"``
        Parabolic arc with tumble.  The coin moves laterally while
        flipping — useful when showing multiple coins side by side.

    Settle styles
    ~~~~~~~~~~~~~
    ``"instant"``
        No bouncing — coin appears in final position immediately.
    ``"bounce"``
        One soft bounce with damping before coming to rest.
    ``"wobble"``
        The coin lands edge-on, rocks left-right twice, then falls flat.

    Attributes
    ----------
    flip_style : str
        ``"tumble"`` | ``"rotate_y"`` | ``"arc"``.
    settle_style : str
        ``"instant"`` | ``"bounce"`` | ``"wobble"``.
    arc_height : float
        Maximum height of the flip arc in scene units.
    arc_lateral : float
        Lateral (x) displacement during an ``"arc"`` flip.
    n_spins : float
        Number of full rotations during the flip.
    spin_axis : np.ndarray
        Axis of rotation (normalised).  Default RIGHT for tumble.
    run_time : float
        Total animation duration in seconds.
    bounce_height : float
        Maximum height of the settle bounce (fraction of arc_height).
    bounce_damping : float
        Damping factor per bounce (0 < damping < 1).
    show_result_label : bool
        Whether to show "H" / "T" label after settling.
    result_label_font_size : int
    result_label_color : ManimColor
    result_label_offset : float
        Z offset for the result label above the coin.
    trail_opacity : float
        Opacity of motion trail dots (0 = no trail).
    trail_n_dots : int
        Number of trail dots to draw.
    """

    flip_style: str = "tumble"
    settle_style: str = "bounce"
    arc_height: float = 2.2
    arc_lateral: float = 0.0
    n_spins: float = 3.5
    spin_axis: np.ndarray = field(default_factory=lambda: RIGHT.copy())
    run_time: float = 1.4
    bounce_height: float = 0.22
    bounce_damping: float = 0.42
    show_result_label: bool = True
    result_label_font_size: int = 22
    result_label_color: ManimColor = WHITE
    result_label_offset: float = 0.55
    trail_opacity: float = 0.0
    trail_n_dots: int = 8


# ---------------------------------------------------------------------------
# RollConfig
# ---------------------------------------------------------------------------

@dataclass
class RollConfig:
    """Visual and physics parameters for a die roll animation.

    Roll styles
    ~~~~~~~~~~~
    ``"toss"``
        Die arcs through the air while tumbling on two axes.
        Lands, bounces once, settles face-up.
    ``"roll"``
        Die rolls across a flat surface (rotates along one axis while
        translating).  Stays on the floor throughout.
    ``"drop"``
        Die falls from above, hits the floor, bounces 1–2 times.

    Attributes
    ----------
    roll_style : str
        ``"toss"`` | ``"roll"`` | ``"drop"``.
    arc_height : float
        Peak height (for ``"toss"`` and ``"drop"``).
    arc_lateral : float
        Lateral travel distance (for ``"toss"`` and ``"roll"``).
    n_tumbles : float
        Number of full tumble rotations during flight.
    tumble_axis : np.ndarray
        Primary tumble axis.
    secondary_spin : float
        Additional spin around the vertical axis (adds randomness).
    run_time : float
        Total animation duration.
    n_bounces : int
        Number of bounces after landing (0–3).
    bounce_decay : float
        Height multiplier per bounce (0 < decay < 1).
    show_result_label : bool
    result_label_font_size : int
    result_label_color : ManimColor
    """

    roll_style: str = "toss"
    arc_height: float = 1.8
    arc_lateral: float = 0.0
    n_tumbles: float = 2.5
    tumble_axis: np.ndarray = field(default_factory=lambda: RIGHT.copy())
    secondary_spin: float = 0.5
    run_time: float = 1.6
    n_bounces: int = 1
    bounce_decay: float = 0.38
    show_result_label: bool = True
    result_label_font_size: int = 22
    result_label_color: ManimColor = WHITE


# ---------------------------------------------------------------------------
# _ArcPath  (internal)
# ---------------------------------------------------------------------------

def _arc_trajectory(
    start: np.ndarray,
    end: np.ndarray,
    apex_height: float,
    t: float,
) -> np.ndarray:
    """Evaluate a parabolic arc from *start* to *end* at parameter *t* ∈ [0,1].

    The arc peaks at ``apex_height`` above the midpoint's z.

    Parameters
    ----------
    start, end : np.ndarray
        Start and end positions.
    apex_height : float
        Maximum height above the start/end z level.
    t : float
        Arc parameter 0 → 1.

    Returns
    -------
    np.ndarray
        3D position at parameter *t*.
    """
    # Quadratic Bézier with control point at arc apex
    mid = (start + end) / 2 + np.array([0, 0, apex_height])
    return (1 - t)**2 * start + 2 * (1 - t) * t * mid + t**2 * end


def _bounce_trajectory(
    land: np.ndarray,
    t: float,
    bounce_height: float,
    bounce_damping: float,
    n_bounces: int = 1,
) -> np.ndarray:
    """Position during a damped bounce sequence starting at *land*.

    The settle occupies t ∈ [0, 1].  Each bounce takes equal time.

    Parameters
    ----------
    land : np.ndarray
        Landing position (z = floor level).
    t : float
        Settle parameter 0 → 1.
    bounce_height : float
        Height of the first bounce.
    bounce_damping : float
        Height multiplier per subsequent bounce.
    n_bounces : int
        Number of bounces.
    """
    if n_bounces == 0 or bounce_height < 0.01:
        return land.copy()

    # Divide [0, 1] into n_bounces equal segments
    seg = 1.0 / n_bounces
    seg_idx = min(int(t / seg), n_bounces - 1)
    t_local = (t - seg_idx * seg) / seg   # ∈ [0, 1] within this bounce

    h = bounce_height * (bounce_damping ** seg_idx)
    z_offset = 4 * h * t_local * (1 - t_local)   # parabolic mini-arc

    pos = land.copy()
    pos[2] += z_offset
    return pos


# ---------------------------------------------------------------------------
# _ResultLabel3D  (internal)
# ---------------------------------------------------------------------------

class _ResultLabel3D(VGroup):
    """Floating label shown after a coin flip or die roll settles.

    For coins: "H" / "T".
    For dice: the face value as a digit.

    Parameters
    ----------
    text : str
    position : np.ndarray
    color : ManimColor
    font_size : int
    scene : ThreeDScene or None
    """

    def __init__(
        self,
        text: str,
        position: np.ndarray,
        color: ManimColor = WHITE,
        font_size: int = 22,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        lbl = Text(text, font_size=font_size, color=color)
        lbl.move_to(np.array(position, dtype=float))
        self.add(lbl)
        if scene is not None:
            scene.add_fixed_orientation_mobjects(lbl)


# ---------------------------------------------------------------------------
# CoinFlipAnimation
# ---------------------------------------------------------------------------

class CoinFlipAnimation:
    """Physics-based coin flip animation.

    Usage::

        coin = Coin3D(position=np.array([0, 0, 0]))
        flip = CoinFlipAnimation(coin, outcome="heads")
        # inside a ThreeDScene.construct:
        self.play(*flip.build(self))

    Parameters
    ----------
    coin : Coin3D
        The coin to animate.  It is modified in place.
    outcome : str
        ``"heads"`` or ``"tails"``.
    config : FlipConfig
    rest_position : np.ndarray or None
        Where the coin settles.  Defaults to its current position.
    scene : ThreeDScene or None
        Provide to register result labels for fixed orientation.
    """

    def __init__(
        self,
        coin: Coin3D,
        outcome: str = "heads",
        config: Optional[FlipConfig] = None,
        rest_position: Optional[np.ndarray] = None,
        scene: Optional[ThreeDScene] = None,
    ):
        self.coin = coin
        self.outcome = outcome.lower()
        self.cfg = config if config is not None else FlipConfig()
        self._rest = (
            rest_position.copy() if rest_position is not None
            else coin.get_center().copy()
        )
        self._scene = scene

    # ------------------------------------------------------------------

    def build(
        self,
        scene: ThreeDScene,
    ) -> List:
        """Return a list of Manim animations to ``scene.play(*result)``.

        Builds the full flip as a ``Succession`` of three phases:
        flight arc + spin, then settle bounce.
        """
        scene_ref = scene if self._scene is None else self._scene
        coin = self.coin
        cfg = self.cfg
        start = coin.get_center().copy()
        end = self._rest.copy()

        if cfg.arc_lateral != 0.0:
            end = end + np.array([cfg.arc_lateral, 0, 0])

        # Phase 1+2: arc flight + spin combined via UpdateFromAlphaFunc
        flip_run = cfg.run_time * 0.70
        settle_run = cfg.run_time * 0.30

        # Total spin angle
        total_spin_angle = cfg.n_spins * TAU
        # Add half turn if tails (so the right face ends up on top)
        if self.outcome == "tails":
            total_spin_angle += PI

        # Spin axis
        spin_ax = np.array(cfg.spin_axis, dtype=float)
        spin_ax /= np.linalg.norm(spin_ax)

        # For "rotate_y" style, spin around Z axis instead
        if cfg.flip_style == "rotate_y":
            spin_ax = np.array([0, 0, 1], dtype=float)

        def flight_updater(mob: VGroup, alpha: float) -> None:
            t = rate_functions.ease_in_out_sine(alpha)
            pos = _arc_trajectory(start, end, cfg.arc_height, t)
            mob.move_to(pos)
            # Cumulative rotation
            angle = total_spin_angle * t
            mob.rotate(
                angle - getattr(mob, "_last_flip_angle", 0.0),
                axis=spin_ax,
                about_point=mob.get_center(),
            )
            mob._last_flip_angle = angle

        # Initialise tracking attribute
        coin._last_flip_angle = 0.0

        flight_anim = UpdateFromAlphaFunc(
            coin, flight_updater, run_time=flip_run
        )

        # Phase 3: settle bounce
        if cfg.settle_style == "instant" or cfg.bounce_height < 0.01:
            settle_anim = UpdateFromAlphaFunc(
                coin,
                lambda mob, a: mob.move_to(end),
                run_time=0.01,
            )
        else:
            n_b = 1 if cfg.settle_style == "bounce" else 2

            def settle_updater(mob: VGroup, alpha: float) -> None:
                t = smooth(alpha)
                pos = _bounce_trajectory(
                    end, t, cfg.bounce_height, cfg.bounce_damping, n_b
                )
                mob.move_to(pos)

            settle_anim = UpdateFromAlphaFunc(
                coin, settle_updater, run_time=settle_run
            )

        anims: List = [Succession(flight_anim, settle_anim)]

        # Trail dots
        if cfg.trail_opacity > 0 and cfg.trail_n_dots > 0:
            trail = self._build_trail(start, end, cfg)
            anims.append(FadeIn(trail, run_time=flip_run * 0.3))
            anims.append(FadeOut(trail, run_time=flip_run * 0.3))

        # Result label
        if cfg.show_result_label:
            label_text = "H" if self.outcome == "heads" else "T"
            label_pos = end + np.array([0, 0, cfg.result_label_offset])
            lbl = _ResultLabel3D(
                label_text, label_pos,
                color=cfg.result_label_color,
                font_size=cfg.result_label_font_size,
                scene=scene_ref,
            )
            scene_ref.add(lbl)
            anims.append(FadeIn(lbl, scale=0.5, run_time=0.3))

        return anims

    def _build_trail(
        self,
        start: np.ndarray,
        end: np.ndarray,
        cfg: FlipConfig,
    ) -> VGroup:
        """Build ghost-dot trail along the arc."""
        trail = VGroup()
        for i in range(1, cfg.trail_n_dots + 1):
            t = i / (cfg.trail_n_dots + 1)
            pos = _arc_trajectory(start, end, cfg.arc_height, t)
            fade = cfg.trail_opacity * (1.0 - t)
            d = Dot3D(point=pos, radius=0.04,
                      color=_with_opacity(WHITE, fade))
            trail.add(d)
        return trail


# ---------------------------------------------------------------------------
# DieRollAnimation
# ---------------------------------------------------------------------------

class DieRollAnimation:
    """Physics-based die roll animation.

    Usage::

        die = Die3D(position=np.array([0, 0, 0.4]))
        roll = DieRollAnimation(die, outcome=4, config=RollConfig(roll_style="toss"))
        self.play(*roll.build(self))

    Parameters
    ----------
    die : Die3D
        The die to animate.
    outcome : int
        The face value (1–6) that lands face-up.
    config : RollConfig
    rest_position : np.ndarray or None
    scene : ThreeDScene or None
    """

    # Rotation needed to bring each face to +z (face-up) from rest
    # Rest orientation: face 1 on +z.  Rotations are (axis, angle).
    _FACE_TO_TOP: Dict[int, Tuple[np.ndarray, float]] = {
        1: (RIGHT,  0.0),           # already face-up
        6: (RIGHT,  PI),            # flip 180° around x
        2: (UP,     PI / 2),        # rotate 90° around z
        5: (UP,    -PI / 2),        # rotate -90° around z
        3: (RIGHT, -PI / 2),        # rotate -90° around x
        4: (RIGHT,  PI / 2),        # rotate +90° around x
    }

    def __init__(
        self,
        die: Die3D,
        outcome: int = 1,
        config: Optional[RollConfig] = None,
        rest_position: Optional[np.ndarray] = None,
        scene: Optional[ThreeDScene] = None,
    ):
        if outcome not in range(1, 7):
            raise ValueError(f"outcome must be 1–6, got {outcome}")
        self.die = die
        self.outcome = outcome
        self.cfg = config if config is not None else RollConfig()
        self._rest = (
            rest_position.copy() if rest_position is not None
            else die.get_center().copy()
        )
        self._scene = scene

    # ------------------------------------------------------------------

    def build(self, scene: ThreeDScene) -> List:
        """Return a list of Manim animations."""
        scene_ref = scene if self._scene is None else self._scene
        die = self.die
        cfg = self.cfg
        start = die.get_center().copy()
        end = self._rest.copy()

        # Compute total rotation to show correct face on top
        settle_ax, settle_angle = self._FACE_TO_TOP[self.outcome]

        # Total tumble + settle
        tumble_angle = cfg.n_tumbles * TAU + settle_angle
        tumble_ax = np.array(cfg.tumble_axis, dtype=float)
        tumble_ax /= np.linalg.norm(tumble_ax)

        # Secondary spin (around Z, adds visual randomness)
        secondary_total = cfg.secondary_spin * TAU

        roll_run = cfg.run_time * 0.65
        settle_run = cfg.run_time * 0.35

        if cfg.roll_style == "roll":
            # Roll along surface: lateral motion + rotation around X
            def roll_updater(mob: VGroup, alpha: float) -> None:
                t = rate_functions.ease_out_cubic(alpha)
                # Stay at floor z = rest z, only move laterally
                pos = start + (end - start) * t
                mob.move_to(pos)
                dangle = tumble_angle * alpha - getattr(mob, "_last_roll", 0.0)
                mob.rotate(dangle, axis=tumble_ax, about_point=mob.get_center())
                mob._last_roll = tumble_angle * alpha

            die._last_roll = 0.0
            flight_anim = UpdateFromAlphaFunc(die, roll_updater, run_time=roll_run)

        elif cfg.roll_style == "drop":
            # Drop from directly above
            drop_start = end + np.array([0, 0, cfg.arc_height])
            die.move_to(drop_start)

            def drop_updater(mob: VGroup, alpha: float) -> None:
                t = rate_functions.ease_in_cubic(alpha)
                pos = drop_start + (end - drop_start) * t
                mob.move_to(pos)
                dangle = tumble_angle * alpha - getattr(mob, "_last_drop", 0.0)
                mob.rotate(dangle, axis=tumble_ax, about_point=mob.get_center())
                mob._last_drop = tumble_angle * alpha

            die._last_drop = 0.0
            flight_anim = UpdateFromAlphaFunc(die, drop_updater, run_time=roll_run)

        else:
            # Default: toss (arc trajectory + tumble)
            def toss_updater(mob: VGroup, alpha: float) -> None:
                t = smooth(alpha)
                pos = _arc_trajectory(start, end, cfg.arc_height, t)
                mob.move_to(pos)
                dangle = tumble_angle * t - getattr(mob, "_last_toss", 0.0)
                mob.rotate(dangle, axis=tumble_ax, about_point=mob.get_center())
                if cfg.secondary_spin > 0:
                    d2 = secondary_total * t - getattr(mob, "_last_spin2", 0.0)
                    mob.rotate(d2, axis=Z_AXIS, about_point=mob.get_center())
                    mob._last_spin2 = secondary_total * t
                mob._last_toss = tumble_angle * t

            die._last_toss = 0.0
            die._last_spin2 = 0.0
            flight_anim = UpdateFromAlphaFunc(die, toss_updater, run_time=roll_run)

        # Settle with bounces
        if cfg.n_bounces == 0:
            settle_anim = UpdateFromAlphaFunc(
                die, lambda mob, a: mob.move_to(end), run_time=0.01
            )
        else:
            def settle_updater(mob: VGroup, alpha: float) -> None:
                pos = _bounce_trajectory(
                    end, smooth(alpha),
                    end[2] * cfg.bounce_decay,
                    cfg.bounce_decay,
                    cfg.n_bounces,
                )
                mob.move_to(pos)

            settle_anim = UpdateFromAlphaFunc(die, settle_updater, run_time=settle_run)

        anims: List = [Succession(flight_anim, settle_anim)]

        # Result label
        if cfg.show_result_label:
            label_pos = end + np.array([0, 0, die.die_size * 2.0])
            lbl = _ResultLabel3D(
                str(self.outcome), label_pos,
                color=cfg.result_label_color,
                font_size=cfg.result_label_font_size,
                scene=scene_ref,
            )
            scene_ref.add(lbl)
            anims.append(FadeIn(lbl, scale=0.5, run_time=0.3))

        return anims


# ---------------------------------------------------------------------------
# BinomialAccumulator
# ---------------------------------------------------------------------------

class BinomialAccumulator(VGroup):
    """Flip n coins, accumulate heads counts into a Binomial histogram.

    Each trial flips ``n_coins`` coins (using ``CoinFlipAnimation`` for the
    first ``n_visible_trials`` trials) and records the number of heads.
    The result is added to a live bar chart.

    After enough trials, the Binomial(n_coins, p) PMF is overlaid as
    a step plot for comparison.

    Parameters
    ----------
    n_coins : int
        Number of coins per trial (n parameter of Binomial).
    p : float
        Probability of heads (bias).
    n_trials : int
        Total number of trials to simulate.
    rng_seed : int
    chart_x : float
        X position of the bar chart in the scene.
    chart_z_scale : float
        Height scaling for bars.
    bar_color_lo : ManimColor
        Bars near the expected value.
    bar_color_hi : ManimColor
        Bars in the tails.
    pmf_color : ManimColor
        Colour of the theoretical PMF overlay.
    scene : ThreeDScene or None
    """

    def __init__(
        self,
        n_coins: int = 10,
        p: float = 0.5,
        n_trials: int = 500,
        rng_seed: int = 42,
        chart_x: float = 0.0,
        chart_z_scale: float = 0.70,
        bar_color_lo: ManimColor = ManimColor("#2DAA6E"),
        bar_color_hi: ManimColor = ManimColor("#E0AA40"),
        pmf_color: ManimColor = ManimColor("#E8593C"),
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_coins = n_coins
        self.p = p
        self.n_trials = n_trials
        self.chart_x = chart_x
        self.z_scale = chart_z_scale
        self.bar_color_lo = bar_color_lo
        self.bar_color_hi = bar_color_hi
        self.pmf_color = pmf_color
        self._scene = scene

        rng = np.random.default_rng(rng_seed)
        # Pre-simulate
        self._results: List[int] = [
            int(rng.binomial(n_coins, p)) for _ in range(n_trials)
        ]

        # Counts array indexed 0..n_coins
        self._counts: np.ndarray = np.zeros(n_coins + 1, dtype=int)
        self._max_count: int = 1
        self._bar_mobs: Dict[int, VGroup] = {}

        # Bar geometry params
        self._bar_w = min(0.55, 4.0 / (n_coins + 1))
        self._bar_depth = 0.35
        self._spacing = min(0.70, 5.0 / (n_coins + 1))
        x_total = n_coins * self._spacing
        self._x0 = chart_x - x_total / 2

        self.bars = VGroup()
        self.pmf_overlay = VGroup()
        self.add(self.bars, self.pmf_overlay)

        # Labels
        self._build_axis_labels(scene)

    # ------------------------------------------------------------------

    def _bar_x(self, k: int) -> float:
        return self._x0 + k * self._spacing

    def _bar_color(self, k: int) -> ManimColor:
        mu = self.n_coins * self.p
        dist = abs(k - mu) / max(self.n_coins * self.p * (1 - self.p), 0.5)
        t = min(dist / 3.0, 1.0)
        return _lerp_color(self.bar_color_lo, self.bar_color_hi, t)

    def _build_bar(self, k: int, height: float) -> VGroup:
        x = self._bar_x(k)
        hw = self._bar_w / 2
        hd = self._bar_depth / 2
        y, z0, z1 = 0.0, 0.0, max(height, 0.01)
        col = self._bar_color(k)
        col_r = _darken(col, 0.60)
        col_t = _lighten(col, 1.28)

        ff = Polygon(
            np.array([x-hw, y-hd, z0]), np.array([x+hw, y-hd, z0]),
            np.array([x+hw, y-hd, z1]), np.array([x-hw, y-hd, z1]),
            fill_color=_with_opacity(col, 0.90), fill_opacity=1.0, stroke_width=0,
        )
        fr = Polygon(
            np.array([x+hw, y-hd, z0]), np.array([x+hw, y+hd, z0]),
            np.array([x+hw, y+hd, z1]), np.array([x+hw, y-hd, z1]),
            fill_color=_with_opacity(col_r, 0.88), fill_opacity=1.0, stroke_width=0,
        )
        ft = Polygon(
            np.array([x-hw, y-hd, z1]), np.array([x+hw, y-hd, z1]),
            np.array([x+hw, y+hd, z1]), np.array([x-hw, y+hd, z1]),
            fill_color=_with_opacity(col_t, 0.80), fill_opacity=1.0, stroke_width=0,
        )
        return VGroup(ff, fr, ft)

    def _update_bar(self, k: int) -> None:
        count = self._counts[k]
        height = (count / self._max_count) * self.z_scale * 4.0
        x = self._bar_x(k)
        hw = self._bar_w / 2
        hd = self._bar_depth / 2
        y, z0, z1 = 0.0, 0.0, max(height, 0.01)
        bar = self._bar_mobs[k]
        ff, fr, ft = bar[0], bar[1], bar[2]

        for face, pts in [
            (ff, [[x-hw,y-hd,z0],[x+hw,y-hd,z0],[x+hw,y-hd,z1],[x-hw,y-hd,z1],[x-hw,y-hd,z0]]),
            (fr, [[x+hw,y-hd,z0],[x+hw,y+hd,z0],[x+hw,y+hd,z1],[x+hw,y-hd,z1],[x+hw,y-hd,z0]]),
            (ft, [[x-hw,y-hd,z1],[x+hw,y-hd,z1],[x+hw,y+hd,z1],[x-hw,y+hd,z1],[x-hw,y-hd,z1]]),
        ]:
            face.set_points_as_corners([np.array(p) for p in pts])

    def _build_axis_labels(self, scene: Optional[ThreeDScene]) -> None:
        self.axis_labels = VGroup()
        for k in range(self.n_coins + 1):
            x = self._bar_x(k)
            lbl = Text(str(k), font_size=14, color=_with_opacity(WHITE, 0.50))
            lbl.move_to(np.array([x, -self._bar_depth / 2, -0.25]))
            self.axis_labels.add(lbl)
            if scene is not None:
                scene.add_fixed_orientation_mobjects(lbl)
        self.add(self.axis_labels)

    # ------------------------------------------------------------------

    def add_trial(
        self,
        trial_index: int,
        animate: bool = True,
        run_time: float = 0.10,
    ) -> Optional[AnimationGroup]:
        """Record the result for *trial_index* and update the histogram."""
        k = self._results[trial_index]
        self._counts[k] += 1
        self._max_count = max(self._max_count, int(self._counts.max()))

        # Rescale all bars
        for i in self._bar_mobs:
            self._update_bar(i)

        if k not in self._bar_mobs:
            h = (self._counts[k] / self._max_count) * self.z_scale * 4.0
            bar = self._build_bar(k, max(h, 0.01))
            self._bar_mobs[k] = bar
            self.bars.add(bar)
            if animate:
                return FadeIn(bar, run_time=run_time)
        return None

    def build_pmf_overlay(self) -> VGroup:
        """Build the theoretical Binomial(n, p) PMF as a step-line overlay.

        Returns the VGroup so the caller can animate it with ``Create``.
        """
        from math import comb as _comb
        self.pmf_overlay = VGroup()
        self.add(self.pmf_overlay)

        max_bar_h = self.z_scale * 4.0

        prev_prob = 0.0
        pts: List[np.ndarray] = []
        for k in range(self.n_coins + 1):
            prob = _comb(self.n_coins, k) * (self.p ** k) * ((1 - self.p) ** (self.n_coins - k))
            x = self._bar_x(k)
            hd = self._bar_depth / 2
            z = prob * max_bar_h / max(
                _comb(self.n_coins, int(self.n_coins * self.p)) *
                (self.p ** int(self.n_coins * self.p)) *
                ((1 - self.p) ** (self.n_coins - int(self.n_coins * self.p))),
                1e-9,
            )
            pts.append(np.array([x, -hd, z]))

        curve = VMobject()
        curve.set_points_as_corners(pts)
        curve.set_stroke(color=_with_opacity(self.pmf_color, 0.88), width=2.5)
        curve.set_fill(opacity=0)

        if True:  # glow
            glow = VMobject()
            glow.set_points_as_corners(pts)
            glow.set_stroke(color=_with_opacity(self.pmf_color, 0.14), width=8.0)
            glow.set_fill(opacity=0)
            self.pmf_overlay.add(glow)

        self.pmf_overlay.add(curve)

        # Dot at each k
        for k, pt in enumerate(pts):
            d = Dot3D(point=pt, radius=0.06,
                      color=_with_opacity(self.pmf_color, 0.85))
            self.pmf_overlay.add(d)

        return self.pmf_overlay

    def run_accumulation(
        self,
        scene: ThreeDScene,
        n_visible: int = 50,
        run_time_per: float = 0.09,
        show_pmf_after: bool = True,
    ) -> None:
        """Animate the full accumulation directly on *scene*.

        Parameters
        ----------
        n_visible : int
            Number of trials animated individually.
        run_time_per : float
            Duration per animated trial.
        show_pmf_after : bool
            If True, draw the Binomial PMF overlay after accumulation.
        """
        scene.play(FadeIn(self.axis_labels, run_time=0.4))

        for i in range(min(n_visible, self.n_trials)):
            anim = self.add_trial(i, animate=True, run_time=run_time_per)
            if anim is not None:
                scene.play(anim)
            else:
                scene.wait(run_time_per * 0.3)

        # Silent batch
        for i in range(n_visible, self.n_trials):
            self.add_trial(i, animate=False)
        scene.wait(0.3)

        if show_pmf_after:
            pmf = self.build_pmf_overlay()
            scene.play(Create(pmf, run_time=1.2))
            scene.wait(1.0)


# ---------------------------------------------------------------------------
# LLNAccumulator
# ---------------------------------------------------------------------------

class LLNAccumulator(VGroup):
    """Flip a biased coin repeatedly and track the running relative frequency.

    Plots the running proportion of heads as a ``LineSeries3D``-style
    VMobject that extends right with each new flip.  Shows convergence
    to the true probability p (Law of Large Numbers).

    Parameters
    ----------
    p : float
        True probability of heads.
    n_flips : int
        Total number of flips to simulate.
    rng_seed : int
    chart_x_start : float
        Left edge of the line plot.
    chart_width : float
        Total width of the plot in scene units.
    chart_z_scale : float
        Total height of the plot (z-axis scale).
    line_color : ManimColor
        Running frequency line colour.
    true_p_color : ManimColor
        Horizontal reference line at z = p.
    scene : ThreeDScene or None
    """

    def __init__(
        self,
        p: float = 0.5,
        n_flips: int = 300,
        rng_seed: int = 42,
        chart_x_start: float = -3.0,
        chart_width: float = 6.0,
        chart_z_scale: float = 3.5,
        line_color: ManimColor = ManimColor("#4A90D9"),
        true_p_color: ManimColor = ManimColor("#E8593C"),
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.p = p
        self.n_flips = n_flips
        self._scene = scene
        self.x_start = chart_x_start
        self.width = chart_width
        self.z_scale = chart_z_scale
        self.line_color = line_color
        self.true_p_color = true_p_color

        rng = np.random.default_rng(rng_seed)
        self._flips: np.ndarray = rng.binomial(1, p, n_flips).astype(float)

        # Running mean after each flip
        self._running: np.ndarray = np.cumsum(self._flips) / np.arange(1, n_flips + 1)

        # x positions for each flip
        self._xs: np.ndarray = np.linspace(chart_x_start,
                                            chart_x_start + chart_width, n_flips)

        # Visual objects
        self.freq_line = VMobject()
        self.freq_line.set_stroke(color=_with_opacity(line_color, 0.90), width=2.2)
        self.freq_line.set_fill(opacity=0)
        self._pts_built: int = 0

        # Glow
        self.freq_glow = VMobject()
        self.freq_glow.set_stroke(color=_with_opacity(line_color, 0.12), width=7.0)
        self.freq_glow.set_fill(opacity=0)

        # True p reference line
        true_z = p * chart_z_scale
        self.true_p_line = DashedLine(
            np.array([chart_x_start, 0, true_z]),
            np.array([chart_x_start + chart_width, 0, true_z]),
            dash_length=0.08, dashed_ratio=0.4,
            color=_with_opacity(true_p_color, 0.75),
            stroke_width=1.8,
        )

        # p label
        p_lbl = Text(f"p = {p}", font_size=18, color=true_p_color)
        p_lbl.move_to(np.array([chart_x_start + chart_width + 0.35, 0, true_z]))
        self._p_label = VGroup(p_lbl)
        if scene is not None:
            scene.add_fixed_orientation_mobjects(p_lbl)

        # n label (updates live)
        self._n_label_pos = np.array([chart_x_start + chart_width * 0.5, 0, -0.35])

        self.add(self.true_p_line, self._p_label, self.freq_glow, self.freq_line)

    # ------------------------------------------------------------------

    def _flip_to_point(self, i: int) -> np.ndarray:
        return np.array([self._xs[i], 0.0, self._running[i] * self.z_scale])

    def extend_to(self, n: int) -> None:
        """Extend the frequency line to *n* flips (immediate, no animation)."""
        n = min(n, self.n_flips)
        if n <= self._pts_built:
            return
        pts = [self._flip_to_point(i) for i in range(self._pts_built, n)]
        all_pts = (
            [self._flip_to_point(i) for i in range(self._pts_built)] + pts
            if self._pts_built > 0 else pts
        )
        if len(all_pts) > 1:
            self.freq_line.set_points_as_corners(all_pts)
            self.freq_glow.set_points_as_corners(all_pts)
        self._pts_built = n

    def animate_draw(
        self,
        scene: ThreeDScene,
        batch_size: int = 5,
        run_time_per_batch: float = 0.06,
        final_hold: float = 1.5,
    ) -> None:
        """Animate the line extending flip-by-flip directly on *scene*.

        Parameters
        ----------
        batch_size : int
            Number of flips added per animation frame.
        run_time_per_batch : float
            Duration per batch animation.
        """
        scene.play(Create(self.true_p_line, run_time=0.6))
        scene.play(FadeIn(self._p_label, run_time=0.3))

        all_pts: List[np.ndarray] = []

        def make_extend_updater(target_n: int):
            start_pts = list(all_pts)

            def updater(mob: VMobject, alpha: float) -> None:
                new_pts = [self._flip_to_point(i)
                           for i in range(len(start_pts),
                                          len(start_pts) + batch_size)]
                combined = start_pts + new_pts[:max(1, int(alpha * batch_size))]
                if len(combined) > 1:
                    mob.set_points_as_corners(combined)
            return updater

        for start in range(0, self.n_flips, batch_size):
            end = min(start + batch_size, self.n_flips)
            new_pts = [self._flip_to_point(i) for i in range(start, end)]
            all_pts.extend(new_pts)

            if len(all_pts) > 1:
                self.freq_line.set_points_as_corners(all_pts)
                self.freq_glow.set_points_as_corners(all_pts)
            scene.wait(run_time_per_batch)

        scene.wait(final_hold)

    def get_final_frequency(self) -> float:
        return float(self._running[-1])


# ---------------------------------------------------------------------------
# DiceSumAccumulator
# ---------------------------------------------------------------------------

class DiceSumAccumulator(VGroup):
    """Roll multiple dice, accumulate sum distribution as a bar chart.

    With 2 dice the sums 2–12 form a triangular distribution.
    With more dice the CLT kicks in and the distribution normalises.

    Parameters
    ----------
    n_dice : int
        Number of dice per roll (2 = classic, 3+ shows CLT).
    n_faces : int
        Faces per die (default 6 for standard D6).
    n_trials : int
        Number of rolls to simulate.
    rng_seed : int
    chart_x : float
        X centre of the bar chart.
    z_scale : float
    bar_color : ManimColor
        Base colour for all bars.
    normal_color : ManimColor
        Colour of the CLT normal overlay (shown when n_dice ≥ 3).
    scene : ThreeDScene or None
    """

    def __init__(
        self,
        n_dice: int = 2,
        n_faces: int = 6,
        n_trials: int = 500,
        rng_seed: int = 42,
        chart_x: float = 0.0,
        z_scale: float = 0.75,
        bar_color: ManimColor = ManimColor("#E8593C"),
        normal_color: ManimColor = ManimColor("#4A90D9"),
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_dice = n_dice
        self.n_faces = n_faces
        self.n_trials = n_trials
        self.z_scale = z_scale
        self.bar_color = bar_color
        self.normal_color = normal_color
        self._scene = scene
        self.chart_x = chart_x

        rng = np.random.default_rng(rng_seed)
        rolls = rng.integers(1, n_faces + 1, size=(n_trials, n_dice))
        self._sums: np.ndarray = rolls.sum(axis=1)

        # Possible sums: n_dice … n_dice * n_faces
        self._min_sum = n_dice
        self._max_sum = n_dice * n_faces
        n_buckets = self._max_sum - self._min_sum + 1
        self._counts: np.ndarray = np.zeros(n_buckets, dtype=int)
        self._max_count: int = 1
        self._bar_mobs: Dict[int, VGroup] = {}

        self._bar_w = min(0.50, 5.0 / n_buckets)
        self._bar_depth = 0.35
        self._spacing = min(0.65, 6.0 / n_buckets)
        x_total = (n_buckets - 1) * self._spacing
        self._x0 = chart_x - x_total / 2

        self.bars = VGroup()
        self.normal_overlay = VGroup()
        self.add(self.bars, self.normal_overlay)
        self._build_axis_labels(scene)

    # ------------------------------------------------------------------

    def _sum_x(self, s: int) -> float:
        return self._x0 + (s - self._min_sum) * self._spacing

    def _build_bar(self, s: int, height: float) -> VGroup:
        x = self._sum_x(s)
        hw = self._bar_w / 2
        hd = self._bar_depth / 2
        y, z0, z1 = 0.0, 0.0, max(height, 0.01)
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

    def _update_bar(self, s: int) -> None:
        count = self._counts[s - self._min_sum]
        height = (count / self._max_count) * self.z_scale * 4.0
        x = self._sum_x(s)
        hw = self._bar_w / 2
        hd = self._bar_depth / 2
        y, z0, z1 = 0.0, 0.0, max(height, 0.01)
        bar = self._bar_mobs[s]
        for face, pts in [
            (bar[0], [[x-hw,y-hd,z0],[x+hw,y-hd,z0],[x+hw,y-hd,z1],[x-hw,y-hd,z1],[x-hw,y-hd,z0]]),
            (bar[1], [[x+hw,y-hd,z0],[x+hw,y+hd,z0],[x+hw,y+hd,z1],[x+hw,y-hd,z1],[x+hw,y-hd,z0]]),
            (bar[2], [[x-hw,y-hd,z1],[x+hw,y-hd,z1],[x+hw,y+hd,z1],[x-hw,y+hd,z1],[x-hw,y-hd,z1]]),
        ]:
            face.set_points_as_corners([np.array(p) for p in pts])

    def _build_axis_labels(self, scene: Optional[ThreeDScene]) -> None:
        self.axis_labels = VGroup()
        step = max(1, (self._max_sum - self._min_sum) // 10)
        for s in range(self._min_sum, self._max_sum + 1, step):
            lbl = Text(str(s), font_size=14, color=_with_opacity(WHITE, 0.50))
            lbl.move_to(np.array([self._sum_x(s), -self._bar_depth / 2, -0.25]))
            self.axis_labels.add(lbl)
            if scene is not None:
                scene.add_fixed_orientation_mobjects(lbl)
        self.add(self.axis_labels)

    def add_trial(self, trial_index: int, animate: bool = True,
                  run_time: float = 0.09) -> Optional[AnimationGroup]:
        s = int(self._sums[trial_index])
        idx = s - self._min_sum
        self._counts[idx] += 1
        self._max_count = max(self._max_count, int(self._counts.max()))

        for existing_s in self._bar_mobs:
            self._update_bar(existing_s)

        if s not in self._bar_mobs:
            h = (self._counts[idx] / self._max_count) * self.z_scale * 4.0
            bar = self._build_bar(s, max(h, 0.01))
            self._bar_mobs[s] = bar
            self.bars.add(bar)
            if animate:
                return FadeIn(bar, run_time=run_time)
        return None

    def build_normal_overlay(self) -> VGroup:
        """Build a normal approximation overlay (useful for n_dice ≥ 3)."""
        import numpy as np as _np
        mu_sum = self.n_dice * (self.n_faces + 1) / 2
        var_one = (self.n_faces ** 2 - 1) / 12
        sigma_sum = np.sqrt(self.n_dice * var_one)

        xs_data = np.linspace(self._min_sum, self._max_sum, 300)
        ys = np.exp(-0.5 * ((xs_data - mu_sum) / sigma_sum) ** 2)
        ys = ys / ys.max() * self.z_scale * 4.0

        pts = [
            np.array([self._sum_x(int(round(xd))), -self._bar_depth / 2, max(float(zv), 0.0)])
            for xd, zv in zip(xs_data, ys)
        ]
        curve = VMobject()
        curve.set_points_as_corners(pts)
        col = self.normal_color
        curve.set_stroke(color=_with_opacity(col, 0.90), width=2.5)
        curve.set_fill(opacity=0)

        glow = VMobject()
        glow.set_points_as_corners(pts)
        glow.set_stroke(color=_with_opacity(col, 0.13), width=8.0)
        glow.set_fill(opacity=0)

        self.normal_overlay.add(glow, curve)
        return self.normal_overlay

    def run_accumulation(
        self,
        scene: ThreeDScene,
        n_visible: int = 60,
        run_time_per: float = 0.09,
        show_normal_after: bool = True,
    ) -> None:
        scene.play(FadeIn(self.axis_labels, run_time=0.4))
        for i in range(min(n_visible, self.n_trials)):
            anim = self.add_trial(i, animate=True, run_time=run_time_per)
            if anim is not None:
                scene.play(anim)
            else:
                scene.wait(run_time_per * 0.3)
        for i in range(n_visible, self.n_trials):
            self.add_trial(i, animate=False)
        scene.wait(0.4)
        if show_normal_after and self.n_dice >= 3:
            ov = self.build_normal_overlay()
            scene.play(Create(ov, run_time=1.1))
            scene.wait(1.0)


# ---------------------------------------------------------------------------
# Ready-to-render ThreeDScene subclasses
# ---------------------------------------------------------------------------

class CoinFlipScene(ThreeDScene):
    """Demonstrate a single biased coin flip with trail and result label.

    Render:  manim -pql flip_roll.py CoinFlipScene
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.05)

        coin = Coin3D(radius=0.48, thickness=0.11)
        self.add(coin)
        self.wait(0.3)

        for outcome, x_pos in [("heads", -1.2), ("tails", 1.2)]:
            cfg = FlipConfig(
                flip_style="tumble",
                settle_style="bounce",
                arc_height=2.4,
                n_spins=3.5,
                run_time=1.5,
                bounce_height=0.25,
                trail_opacity=0.30,
                trail_n_dots=10,
            )
            c = Coin3D(radius=0.48, thickness=0.11,
                       position=np.array([x_pos, 0, 0]))
            self.add(c)
            flip = CoinFlipAnimation(c, outcome=outcome, config=cfg, scene=self)
            self.play(*flip.build(self))
            self.wait(0.4)

        self.wait(1.5)


class DieRollScene(ThreeDScene):
    """Roll a die with each of the three roll styles back to back.

    Render:  manim -pql flip_roll.py DieRollScene
    """

    def construct(self):
        self.set_camera_orientation(phi=68 * DEGREES, theta=-50 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.04)

        for outcome, x, style in [(3, -2.5, "toss"), (5, 0.0, "roll"), (1, 2.5, "drop")]:
            die = Die3D(size=0.38, position=np.array([x, 0.0, 0.38]))
            self.add(die)

            cfg = RollConfig(
                roll_style=style,
                arc_height=2.0,
                n_tumbles=2.5,
                run_time=1.8,
                n_bounces=1,
                bounce_decay=0.35,
            )
            roll = DieRollAnimation(
                die, outcome=outcome, config=cfg,
                rest_position=np.array([x, 0.0, die.die_size]),
                scene=self,
            )
            self.play(*roll.build(self))
            self.wait(0.5)

        self.wait(1.5)


class BinomialScene(ThreeDScene):
    """Accumulate coin flips into a Binomial(10, 0.5) histogram.

    Render:  manim -pql flip_roll.py BinomialScene
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES)

        accum = BinomialAccumulator(
            n_coins=10, p=0.5, n_trials=600,
            chart_x=0.0, chart_z_scale=0.80,
            scene=self,
        )
        self.add(accum)
        accum.run_accumulation(self, n_visible=60,
                               run_time_per=0.09, show_pmf_after=True)
        self.wait(2)


class LLNScene(ThreeDScene):
    """Running frequency converges to p.  Law of Large Numbers.

    Uses a biased coin (p=0.3) so convergence is visually obvious.

    Render:  manim -pql flip_roll.py LLNScene
    """

    def construct(self):
        self.set_camera_orientation(phi=62 * DEGREES, theta=-48 * DEGREES)

        accum = LLNAccumulator(
            p=0.30, n_flips=400, chart_x_start=-3.5,
            chart_width=7.0, chart_z_scale=3.5,
            scene=self,
        )
        self.add(accum)
        accum.animate_draw(self, batch_size=4, run_time_per_batch=0.05)
        self.wait(2)


class DiceSumScene(ThreeDScene):
    """Two-dice sum distribution → triangular shape.

    Render:  manim -pql flip_roll.py DiceSumScene
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES)

        # Two dice
        accum2 = DiceSumAccumulator(
            n_dice=2, n_faces=6, n_trials=500,
            chart_x=-2.5, z_scale=0.75,
            bar_color=ManimColor("#4A90D9"),
            scene=self,
        )
        self.add(accum2)
        accum2.run_accumulation(self, n_visible=55, run_time_per=0.09,
                                show_normal_after=False)

        title2 = Text("2 dice: triangular", font_size=20,
                      color=ManimColor("#4A90D9"))
        title2.move_to(np.array([-2.5, 0, -0.55]))
        self.add_fixed_orientation_mobjects(title2)
        self.add(title2)
        self.wait(1.0)

        # Three dice — shows CLT beginning
        accum3 = DiceSumAccumulator(
            n_dice=3, n_faces=6, n_trials=500,
            chart_x=2.5, z_scale=0.75,
            bar_color=ManimColor("#2DAA6E"),
            normal_color=ManimColor("#E8593C"),
            scene=self,
        )
        self.add(accum3)
        accum3.run_accumulation(self, n_visible=55, run_time_per=0.09,
                                show_normal_after=True)

        title3 = Text("3 dice: approaching normal", font_size=20,
                      color=ManimColor("#2DAA6E"))
        title3.move_to(np.array([2.5, 0, -0.55]))
        self.add_fixed_orientation_mobjects(title3)
        self.add(title3)
        self.wait(2)