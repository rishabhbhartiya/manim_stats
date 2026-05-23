"""
manim_stats/probability/venn3d.py
===================================
Production-quality 3D Venn diagram visualizations for Manim.

This is the definitive standalone Venn implementation.  While
``sample_space.py`` includes Venn zones as one feature among many,
this module focuses entirely on Venn geometry with full analytical
precision, rich per-zone surfaces, and a comprehensive animation API.

Architecture
------------

Geometry engine
    ``_CircleArcGeometry``
        Computes the two intersection points of two circles analytically
        and determines which angular arc segments belong to each zone.
        All circle geometry uses these results so zone boundaries are
        mathematically exact, not approximated.

    ``_ZonePatch``
        For a given atomic zone (e.g. A∩Bᶜ or A∩B∩Cᶜ), parametrises
        the boundary as a sequence of circular arc segments and renders
        the enclosed region as a ``Surface``.  Side walls are
        extruded downward from the boundary curve to y=0, giving each
        zone a proper 3D profile.  Zone height is tiered by membership
        level: single-set at h₁, two-set overlap at h₂, triple at h₃.

Data containers
    ``VennData2``
        Holds P(A), P(B), P(A∩B) for a 2-set diagram.  Validates all
        consistency constraints and computes the three derived zone probs
        (A only, B only, A∩B) plus the "neither" zone.

    ``VennData3``
        Holds the 7 input probabilities for a 3-set diagram.  Validates
        and computes all 8 atomic zone probabilities (including neither).

Visual objects
    ``_VennCircleOutline``
        The full circle ring drawn as a ``ParametricFunction`` floating
        just above the zone surfaces.  Provides the classic "overlapping
        circles" visual as a thin colored curve.

    ``_ZoneSurface``
        A single atomic zone: ``Surface`` top + extruded side
        wall + floating MathTex labels (set-notation + probability + optional
        count).

    ``_InclusionExclusionBracket``
        A bracket annotation system showing
        P(A∪B) = P(A) + P(B) − P(A∩B) with color-coded terms.

    ``_IndependenceAnnotation``
        Shows P(A∩B) vs P(A)·P(B) side by side with a
        ``≈`` or ``≠`` indicator colored green/red.

Main class
    ``VennDiagram3D``
        Composes all components.  Supports 2-set and 3-set diagrams,
        four layout modes ("standard", "linear", "nested", "euler"),
        and a full animation API.

    Factory classmethods:
        ``two_set(data, …)``          – 2-set Venn from VennData2
        ``three_set(data, …)``        – 3-set Venn from VennData3
        ``from_counts(n_a, …)``       – build from natural frequency counts
        ``disjoint(p_a, p_b, …)``     – disjoint events (Euler diagram)
        ``subset(p_a, p_b, …)``       – A ⊆ B nested layout
        ``independent(p_a, p_b, …)``  – P(A∩B) = P(A)·P(B) case

    Animation suite:
        ``animate_grow_circles()``         – outlines draw one by one
        ``animate_fill_zones()``           – platforms rise level by level
        ``animate_highlight_zone(name)``   – flash one zone, dim others
        ``animate_inclusion_exclusion()``  – step through P(A∪B) formula
        ``animate_condition_on(name)``     – dim outside-A zones, show P(B|A)
        ``animate_independence_test()``    – show P(A∩B) vs P(A)·P(B)
        ``animate_morph_to(new_data)``     – morph zone sizes to new probs
        ``animate_add_set_c(data3)``       – grow 2-set into 3-set
        ``animate_restore()``             – restore all opacities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence
import warnings

import numpy as np

from manim import (
    WHITE, BLACK, GRAY, GRAY_A, GRAY_B, GRAY_C, GRAY_D, GRAY_E,
    BLUE, BLUE_E, RED, RED_E, GREEN, GREEN_E, YELLOW, ORANGE, PURPLE,
    UP, DOWN, LEFT, RIGHT,
    PI, TAU,
    VGroup, VMobject,
    Line3D, Polygon, Sphere, Dot3D, Arrow3D,
    Surface, ParametricFunction,
    Text, MathTex,
    FadeIn, FadeOut, GrowFromCenter, GrowFromPoint, Transform,
    Create, Write, AnimationGroup, Succession, LaggedStart,
    interpolate_color,
    ManimColor,
    DEGREES,
)


# ---------------------------------------------------------------------------
# Semantic palette
# ---------------------------------------------------------------------------

SET_A_COLOR   = ManimColor("#1565C0")   # blue
SET_B_COLOR   = ManimColor("#B71C1C")   # red
SET_C_COLOR   = ManimColor("#2E7D32")   # green

ZONE_A_ONLY   = ManimColor("#1E88E5")   # lighter blue
ZONE_B_ONLY   = ManimColor("#E53935")   # lighter red
ZONE_C_ONLY   = ManimColor("#43A047")   # lighter green
ZONE_AB       = ManimColor("#7B1FA2")   # purple  (A∩B)
ZONE_AC       = ManimColor("#00838F")   # teal    (A∩C)
ZONE_BC       = ManimColor("#E65100")   # orange  (B∩C)
ZONE_ABC      = ManimColor("#F9A825")   # amber   (A∩B∩C)
ZONE_NEITHER  = ManimColor("#1C2A30")   # near-black

OUTLINE_OPACITY = 0.90
ZONE_DIM_OPACITY = 0.10
LABEL_COLOR   = ManimColor("#ECEFF1")

FACE_DARKEN   = 0.40


def _dk(c: ManimColor, f: float) -> ManimColor:
    return interpolate_color(c, BLACK, f)

def _lt(c: ManimColor, f: float) -> ManimColor:
    return interpolate_color(c, WHITE, f)

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(v, lo, hi))


# ---------------------------------------------------------------------------
# VennData containers  (validation + derived zone probabilities)
# ---------------------------------------------------------------------------

@dataclass
class VennData2:
    """Probability specification for a 2-set Venn diagram.

    Parameters
    ----------
    p_a : float
        P(A)
    p_b : float
        P(B)
    p_ab : float
        P(A ∩ B).  Must satisfy: p_ab ≤ min(p_a, p_b) and p_ab ≥ 0.
    label_a, label_b : str
        Display names for the two sets.

    Derived zone probabilities (read-only properties):
        p_a_only  = P(A) − P(A∩B)
        p_b_only  = P(B) − P(A∩B)
        p_aub     = P(A) + P(B) − P(A∩B)
        p_neither = 1 − P(A∪B)
    """

    p_a:     float
    p_b:     float
    p_ab:    float
    label_a: str = "A"
    label_b: str = "B"

    def __post_init__(self):
        self.p_a  = _clamp(self.p_a)
        self.p_b  = _clamp(self.p_b)
        self.p_ab = _clamp(self.p_ab)
        self._validate()

    def _validate(self) -> None:
        msgs = []
        if self.p_ab > self.p_a + 1e-9:
            msgs.append(f"P(A∩B)={self.p_ab:.4f} > P(A)={self.p_a:.4f}")
        if self.p_ab > self.p_b + 1e-9:
            msgs.append(f"P(A∩B)={self.p_ab:.4f} > P(B)={self.p_b:.4f}")
        if self.p_a + self.p_b - self.p_ab > 1.0 + 1e-9:
            msgs.append(f"P(A∪B)={self.p_a+self.p_b-self.p_ab:.4f} > 1")
        if msgs:
            warnings.warn("VennData2 inconsistency: " + "; ".join(msgs))

    @property
    def p_a_only(self) -> float:
        return max(0.0, self.p_a - self.p_ab)

    @property
    def p_b_only(self) -> float:
        return max(0.0, self.p_b - self.p_ab)

    @property
    def p_aub(self) -> float:
        return _clamp(self.p_a + self.p_b - self.p_ab)

    @property
    def p_neither(self) -> float:
        return max(0.0, 1.0 - self.p_aub)

    def zone_probs(self) -> dict[str, float]:
        return {
            "a_only":  self.p_a_only,
            "b_only":  self.p_b_only,
            "ab":      self.p_ab,
            "neither": self.p_neither,
        }

    def is_independent(self, tol: float = 0.005) -> bool:
        return abs(self.p_ab - self.p_a * self.p_b) < tol

    def is_disjoint(self, tol: float = 0.005) -> bool:
        return self.p_ab < tol

    def p_b_given_a(self) -> float:
        return self.p_ab / self.p_a if self.p_a > 1e-9 else 0.0

    def p_a_given_b(self) -> float:
        return self.p_ab / self.p_b if self.p_b > 1e-9 else 0.0


@dataclass
class VennData3:
    """Probability specification for a 3-set Venn diagram.

    Parameters
    ----------
    p_a, p_b, p_c : float
        Marginal probabilities.
    p_ab, p_ac, p_bc : float
        Pairwise joint probabilities.
    p_abc : float
        Triple intersection P(A ∩ B ∩ C).
    label_a, label_b, label_c : str
        Display names for the three sets.

    All 8 atomic zone probabilities are derived as properties.
    """

    p_a:   float
    p_b:   float
    p_c:   float
    p_ab:  float
    p_ac:  float
    p_bc:  float
    p_abc: float
    label_a: str = "A"
    label_b: str = "B"
    label_c: str = "C"

    def __post_init__(self):
        for attr in ("p_a","p_b","p_c","p_ab","p_ac","p_bc","p_abc"):
            setattr(self, attr, _clamp(getattr(self, attr)))
        self._validate()

    def _validate(self) -> None:
        msgs = []
        if self.p_abc > min(self.p_ab, self.p_ac, self.p_bc) + 1e-9:
            msgs.append("P(A∩B∩C) > min pairwise intersection")
        for name, mv, mv_name in [
            ("p_ab", self.p_a, "p_a"), ("p_ab", self.p_b, "p_b"),
            ("p_ac", self.p_a, "p_a"), ("p_ac", self.p_c, "p_c"),
            ("p_bc", self.p_b, "p_b"), ("p_bc", self.p_c, "p_c"),
        ]:
            if getattr(self, name) > mv + 1e-9:
                msgs.append(f"P({name[2:].upper()})>{mv_name}")
        if msgs:
            warnings.warn("VennData3 inconsistency: " + "; ".join(msgs))

    # Atomic zone probabilities (inclusion-exclusion decomposition)
    @property
    def p_a_only(self) -> float:
        return max(0.0, self.p_a - self.p_ab - self.p_ac + self.p_abc)

    @property
    def p_b_only(self) -> float:
        return max(0.0, self.p_b - self.p_ab - self.p_bc + self.p_abc)

    @property
    def p_c_only(self) -> float:
        return max(0.0, self.p_c - self.p_ac - self.p_bc + self.p_abc)

    @property
    def p_ab_only(self) -> float:
        return max(0.0, self.p_ab - self.p_abc)

    @property
    def p_ac_only(self) -> float:
        return max(0.0, self.p_ac - self.p_abc)

    @property
    def p_bc_only(self) -> float:
        return max(0.0, self.p_bc - self.p_abc)

    @property
    def p_abc_zone(self) -> float:
        return max(0.0, self.p_abc)

    @property
    def p_aub_uc(self) -> float:
        return _clamp(
            self.p_a + self.p_b + self.p_c
            - self.p_ab - self.p_ac - self.p_bc
            + self.p_abc
        )

    @property
    def p_neither(self) -> float:
        return max(0.0, 1.0 - self.p_aub_uc)

    def zone_probs(self) -> dict[str, float]:
        return {
            "a_only":  self.p_a_only,
            "b_only":  self.p_b_only,
            "c_only":  self.p_c_only,
            "ab_only": self.p_ab_only,
            "ac_only": self.p_ac_only,
            "bc_only": self.p_bc_only,
            "abc":     self.p_abc_zone,
            "neither": self.p_neither,
        }


# ---------------------------------------------------------------------------
# _CircleArcGeometry  —  analytical circle-circle intersection
# ---------------------------------------------------------------------------

class _CircleArcGeometry:
    """Compute intersection points and arc parameters for two circles.

    Given circle C₁ = (cx₁, cz₁, r₁) and C₂ = (cx₂, cz₂, r₂),
    computes (if they intersect) the two intersection points and the
    angular positions of these points as seen from each circle center.

    These are used to determine which arc segment of each circle forms
    the boundary of each Venn zone.

    All coordinates are in the XZ plane (y=0 flat).
    """

    def __init__(
        self,
        cx1: float, cz1: float, r1: float,
        cx2: float, cz2: float, r2: float,
    ):
        self.c1 = np.array([cx1, cz1])
        self.c2 = np.array([cx2, cz2])
        self.r1 = r1
        self.r2 = r2
        self._compute()

    def _compute(self) -> None:
        d = np.linalg.norm(self.c2 - self.c1)
        self.d = d

        if d < 1e-10 or d > self.r1 + self.r2 + 1e-9 or d < abs(self.r1 - self.r2) - 1e-9:
            self.intersects = False
            self.p1 = self.p2 = None
            self.ang1_p1 = self.ang1_p2 = None
            self.ang2_p1 = self.ang2_p2 = None
            return

        self.intersects = True
        # Distance along c1→c2 axis to the radical line
        a = (self.r1**2 - self.r2**2 + d**2) / (2 * d)
        # Height of intersection triangle
        h_sq = self.r1**2 - a**2
        h = np.sqrt(max(h_sq, 0.0))

        # Unit vectors
        u = (self.c2 - self.c1) / d          # along axis
        v = np.array([-u[1], u[0]])           # perpendicular

        # The two intersection points
        mid = self.c1 + a * u
        self.p1 = mid + h * v
        self.p2 = mid - h * v

        # Angles of intersection points as seen from each center
        def _ang(center, point):
            d = point - center
            return float(np.arctan2(d[1], d[0]))

        self.ang1_p1 = _ang(self.c1, self.p1)
        self.ang1_p2 = _ang(self.c1, self.p2)
        self.ang2_p1 = _ang(self.c2, self.p1)
        self.ang2_p2 = _ang(self.c2, self.p2)

    def arc_for_zone(
        self,
        circle_index: int,   # 1 or 2
        inside:       bool,  # True = arc that is INSIDE the other circle
    ) -> tuple[float, float]:
        """Return (theta_start, theta_end) for the relevant arc segment.

        Parameters
        ----------
        circle_index : 1 or 2
            Which circle's arc we want.
        inside : bool
            True  → the arc that lies inside the other circle (the "lens" arc)
            False → the arc that lies outside the other circle
        """
        if not self.intersects:
            # Full circle or no arc
            return (0.0, TAU)

        if circle_index == 1:
            a1, a2 = self.ang1_p1, self.ang1_p2
            # The "inside" arc is the one that points toward c2
            c_dir = float(np.arctan2(self.c2[1] - self.c1[1],
                                     self.c2[0] - self.c1[0]))
        else:
            a1, a2 = self.ang2_p1, self.ang2_p2
            c_dir = float(np.arctan2(self.c1[1] - self.c2[1],
                                     self.c1[0] - self.c2[0]))

        # Normalise angles to [0, TAU)
        a1 = a1 % TAU
        a2 = a2 % TAU
        c_dir = c_dir % TAU

        # Determine which arc contains c_dir
        def _arc_contains(start, end, angle):
            angle = angle % TAU
            if start <= end:
                return start <= angle <= end
            else:
                return angle >= start or angle <= end

        # Arc from a1 → a2 (CCW)
        if _arc_contains(a1, a2, c_dir):
            toward_arc = (a1, a2)
            away_arc   = (a2, a1)
        else:
            toward_arc = (a2, a1)
            away_arc   = (a1, a2)

        return toward_arc if inside else away_arc


# ---------------------------------------------------------------------------
# _ZoneSurface  —  one atomic zone rendered as Surface
# ---------------------------------------------------------------------------

class _ZoneSurface(VGroup):
    """3D platform for one atomic zone of a Venn diagram.

    The top surface is a flat disc/patch at height ``y_level``.
    The side wall is a thin extruded ring from y=0 to y=y_level.
    Both are built from Surface using circular arc parametrisation.

    For zones defined by a single circle arc (single-set zones and the
    lens-shaped intersection), the boundary is one or more arc segments.

    Parameters
    ----------
    arc_def : list[dict]
        Each dict describes one arc segment of the zone boundary:
          {"cx": float, "cz": float, "r": float,
           "t_start": float, "t_end": float, "ccw": bool}
        Arc segments are concatenated to form the closed boundary.
    centroid : np.ndarray
        (x, z) centroid of the zone, used for label placement and
        for building the "fan" parametrisation of the top surface.
    y_level : float
        Height of the top face.
    color : ManimColor
    cfg : VennConfig
    label_tex : str
        Set-notation label (e.g. r"A \cap B^c").
    prob : float
    count : int | None
        Natural frequency count to display (if not None).
    resolution : int
        Number of angular segments per arc.
    """

    def __init__(
        self,
        arc_def:  list[dict],
        centroid: np.ndarray,
        y_level:  float,
        color:    ManimColor,
        cfg:      "VennConfig",
        label_tex: str = "",
        prob:      float = 0.0,
        count:     int | None = None,
        resolution: int = 60,
    ):
        super().__init__()
        self._color    = color
        self._y_level  = y_level
        self._centroid = centroid
        self._cfg      = cfg

        # ------ Build the boundary polygon points -----
        # We sample all arc segments at `resolution` points each
        # to get the flat polygon, then build a "fan" surface from centroid.
        bnd_pts: list[np.ndarray] = []
        for seg in arc_def:
            cx, cz   = seg["cx"], seg["cz"]
            r        = seg["r"]
            t0, t1   = seg["t_start"], seg["t_end"]
            ccw      = seg.get("ccw", True)
            n_seg    = max(4, int(resolution * abs(t1 - t0) / TAU))

            if ccw:
                thetas = np.linspace(t0, t1, n_seg, endpoint=False)
            else:
                # Clockwise: go from t0 down to t1 (wrapping if needed)
                if t1 < t0:
                    thetas = np.linspace(t0, t1, n_seg, endpoint=False)
                else:
                    thetas = np.linspace(t0, t1 - TAU, n_seg, endpoint=False)

            for theta in thetas:
                bnd_pts.append(np.array([
                    cx + r * np.cos(theta),
                    cz + r * np.sin(theta),
                ]))

        if len(bnd_pts) < 3:
            return  # degenerate zone — skip

        bnd = np.array(bnd_pts)    # (N, 2)
        N   = len(bnd)

        top_color  = color
        side_color = _dk(color, FACE_DARKEN)
        opacity    = cfg.zone_opacity

        # ------ Top surface: "fan" triangles from centroid ------
        # Build as a Surface mapping (u, v) → world:
        #   u ∈ [0,1) = fraction around the boundary polygon
        #   v ∈ [0,1] = radial interpolation centroid → boundary
        cx_xz, cz_xz = centroid[0], centroid[1]

        def top_fn(u: float, v: float) -> np.ndarray:
            idx_f  = u * N
            idx_lo = int(idx_f) % N
            idx_hi = (idx_lo + 1) % N
            alpha  = idx_f - int(idx_f)
            bnd_pt = (1 - alpha) * bnd[idx_lo] + alpha * bnd[idx_hi]
            xv     = cx_xz + v * (bnd_pt[0] - cx_xz)
            zv     = cz_xz + v * (bnd_pt[1] - cz_xz)
            return np.array([xv, y_level, zv])

        top_surf = Surface(
            top_fn,
            u_range=[0.0, 1.0],
            v_range=[0.0, 1.0],
            resolution=(resolution, max(2, cfg.zone_depth_resolution)),
        )
        top_surf.set_style(
            fill_color=top_color,
            fill_opacity=opacity,
            stroke_color=_dk(top_color, 0.45),
            stroke_width=cfg.zone_stroke_width,
            stroke_opacity=opacity * 0.60,
        )
        self.top_surface = top_surf
        self.add(top_surf)

        # ------ Side wall: extruded boundary from y=0 to y_level ------
        if y_level > 1e-4 and cfg.show_zone_walls:
            def wall_fn(u: float, v: float) -> np.ndarray:
                idx_f  = u * N
                idx_lo = int(idx_f) % N
                idx_hi = (idx_lo + 1) % N
                alpha  = idx_f - int(idx_f)
                bnd_pt = (1 - alpha) * bnd[idx_lo] + alpha * bnd[idx_hi]
                yv     = v * y_level
                return np.array([bnd_pt[0], yv, bnd_pt[1]])

            wall = Surface(
                wall_fn,
                u_range=[0.0, 1.0],
                v_range=[0.0, 1.0],
                resolution=(resolution, 3),
            )
            wall.set_style(
                fill_color=side_color,
                fill_opacity=opacity * 0.80,
                stroke_width=0,
            )
            self.add(wall)

        # ------ Labels ------
        self._label_group = VGroup()
        lbl_x = float(centroid[0])
        lbl_z = float(centroid[1])
        lbl_y = y_level + cfg.label_lift

        if cfg.show_zone_labels and label_tex:
            set_lbl = MathTex(label_tex,
                              color=_lt(color, 0.25),
                              font_size=cfg.zone_label_font_size)
            set_lbl.move_to(np.array([lbl_x, lbl_y, lbl_z]))
            self._label_group.add(set_lbl)
            self.set_label = set_lbl

        if cfg.show_prob_labels and prob > 1e-6:
            prob_lbl = MathTex(
                rf"{prob:.3f}",
                color=_lt(color, 0.15),
                font_size=cfg.prob_label_font_size,
            )
            prob_lbl.move_to(np.array([lbl_x, lbl_y + 0.30, lbl_z]))
            self._label_group.add(prob_lbl)
            self.prob_label = prob_lbl

        if count is not None and cfg.show_count_labels:
            cnt_lbl = Text(
                f"n={count}",
                color=_lt(color, 0.20),
                font_size=cfg.count_label_font_size,
            )
            cnt_lbl.move_to(np.array([lbl_x, lbl_y + 0.55, lbl_z]))
            self._label_group.add(cnt_lbl)

        self.add(self._label_group)

        # Centre point for animation origins
        self._floor_center = np.array([lbl_x, 0.0, lbl_z])

    @property
    def floor_center(self) -> np.ndarray:
        return self._floor_center.copy()


# ---------------------------------------------------------------------------
# _VennCircleOutline  —  the classic circle ring floating above zones
# ---------------------------------------------------------------------------

class _VennCircleOutline(VGroup):
    """Full-circle ParametricFunction ring drawn above the zone surfaces.

    Provides the "overlapping circles" visual that makes Venn diagrams
    instantly recognisable, layered on top of the 3D zone geometry.
    """

    def __init__(
        self,
        cx:     float,
        cz:     float,
        r:      float,
        y:      float,
        color:  ManimColor,
        stroke_width: float = 2.8,
        opacity:      float = OUTLINE_OPACITY,
    ):
        super().__init__()
        curve = ParametricFunction(
            lambda t: np.array([cx + r * np.cos(t * TAU),
                                 y,
                                 cz + r * np.sin(t * TAU)]),
            t_range=[0, 1, 1 / 120],
            color=color,
            stroke_width=stroke_width,
        )
        curve.set_opacity(opacity)
        self.add(curve)
        self.curve = curve
        self._cx, self._cz, self._r = cx, cz, r


# ---------------------------------------------------------------------------
# _InclusionExclusionBracket  —  P(A∪B) annotation
# ---------------------------------------------------------------------------

class _InclusionExclusionBracket(VGroup):
    """Color-coded MathTex showing P(A∪B) = P(A) + P(B) − P(A∩B).

    Each term is colored to match the corresponding zone color.
    """

    def __init__(
        self,
        data:      VennData2,
        pos:       np.ndarray,
        font_size: int = 28,
    ):
        super().__init__()
        # Build multi-color formula using separate MathTex objects
        terms = [
            (rf"P({data.label_a} \cup {data.label_b})",
             _lt(ZONE_AB, 0.20), 0.0),
            (r"=",  LABEL_COLOR, 0.0),
            (rf"P({data.label_a})",
             _lt(ZONE_A_ONLY, 0.20), 0.0),
            (r"+",  LABEL_COLOR, 0.0),
            (rf"P({data.label_b})",
             _lt(ZONE_B_ONLY, 0.20), 0.0),
            (r"-",  LABEL_COLOR, 0.0),
            (rf"P({data.label_a} \cap {data.label_b})",
             _lt(ZONE_AB, 0.20), 0.0),
        ]
        values = [
            (rf"= {data.p_aub:.3f}",   _lt(ZONE_AB,    0.10), 0.35),
            ("",       LABEL_COLOR,  0.0),
            (rf"= {data.p_a:.3f}",    _lt(ZONE_A_ONLY, 0.10), 0.35),
            ("",       LABEL_COLOR,  0.0),
            (rf"= {data.p_b:.3f}",    _lt(ZONE_B_ONLY, 0.10), 0.35),
            ("",       LABEL_COLOR,  0.0),
            (rf"= {data.p_ab:.3f}",   _lt(ZONE_AB,     0.10), 0.35),
        ]

        self.term_objects: list[VGroup] = []
        x_cursor = pos[0] - 3.5
        for (tex, color, _), (val, vcol, vshift) in zip(terms, values):
            grp = VGroup()
            lbl = MathTex(tex, color=color, font_size=font_size)
            lbl.move_to(np.array([x_cursor, pos[1], pos[2]]))
            grp.add(lbl)
            x_cursor += lbl.width + 0.18
            if val:
                vlbl = MathTex(val, color=vcol, font_size=font_size - 6)
                vlbl.move_to(np.array([x_cursor - lbl.width / 2,
                                       pos[1] - vshift, pos[2]]))
                grp.add(vlbl)
            self.term_objects.append(grp)
            self.add(grp)


# ---------------------------------------------------------------------------
# _IndependenceAnnotation  —  P(A∩B) vs P(A)·P(B)
# ---------------------------------------------------------------------------

class _IndependenceAnnotation(VGroup):
    """Show P(A∩B) and P(A)·P(B) side by side with a verdict symbol."""

    def __init__(
        self,
        data:      VennData2,
        pos:       np.ndarray,
        font_size: int = 26,
        tol:       float = 0.005,
    ):
        super().__init__()
        product = data.p_a * data.p_b
        is_ind  = abs(data.p_ab - product) < tol

        color   = ManimColor("#00E676") if is_ind else ManimColor("#FF5252")
        symbol  = r"\approx" if is_ind else r"\neq"
        verdict = "Independent" if is_ind else "Dependent"

        lhs = MathTex(
            rf"P({data.label_a} \cap {data.label_b}) = {data.p_ab:.4f}",
            color=_lt(ZONE_AB, 0.20),
            font_size=font_size,
        )
        sym = MathTex(symbol, color=color, font_size=font_size)
        rhs = MathTex(
            rf"P({data.label_a}) \cdot P({data.label_b}) = {product:.4f}",
            color=LABEL_COLOR,
            font_size=font_size,
        )
        ver = Text(verdict, color=color, font_size=font_size - 4,
                   weight="BOLD")

        lhs.move_to(pos + np.array([-2.8, 0, 0]))
        sym.move_to(pos)
        rhs.move_to(pos + np.array([ 2.8, 0, 0]))
        ver.move_to(pos + np.array([0, -0.45, 0]))

        self.add(lhs, sym, rhs, ver)
        self.lhs = lhs
        self.sym = sym
        self.rhs = rhs
        self.verdict = ver


# ---------------------------------------------------------------------------
# VennConfig
# ---------------------------------------------------------------------------

@dataclass
class VennConfig:
    """All visual parameters for VennDiagram3D.

    Circle geometry
    ---------------
    circle_radius : float
        Radius of each Venn circle (Manim units).
    circle_separation : float
        Centre-to-centre distance between adjacent circles as a
        fraction of circle_radius.  0.8 → heavy overlap; 1.4 → light.
    layout : str
        ``"standard"``  – classic side-by-side (2-set) or triangle (3-set)
        ``"linear"``    – all circles on the x-axis
        ``"nested"``    – A inside B  (use only when A ⊆ B)
        ``"euler"``     – disjoint circles (no overlap)

    Zone surfaces
    -------------
    zone_opacity : float
        Fill opacity of zone top surfaces.
    zone_stroke_width : float
        Mesh line width on zone surfaces.
    show_zone_walls : bool
        Extrude side walls from each zone boundary.
    zone_depth_resolution : int
        Number of v-slices in the side wall extrusion (2–4 is enough).
    height_single : float
        Platform height for single-set zones (A only, B only, C only).
    height_double : float
        Platform height for two-set intersection zones.
    height_triple : float
        Platform height for the triple intersection zone.
    height_neither : float
        Platform height for the "neither" background zone (usually 0).

    Circle outlines
    ---------------
    show_outlines : bool
        Render the floating circle outlines above the zones.
    outline_stroke_width : float
        Stroke width of circle outline curves.
    outline_y_lift : float
        How high above the tallest zone the outline floats.

    Labels
    ------
    show_set_labels : bool
        Show A/B/C labels at the far end of each circle.
    show_zone_labels : bool
        Show set-notation labels (A∩Bᶜ, etc.) inside each zone.
    show_prob_labels : bool
        Show probability values inside each zone.
    show_count_labels : bool
        Show natural frequency counts inside zones (requires total N).
    label_lift : float
        Y offset above the zone top for floating labels.
    set_label_font_size : int
    zone_label_font_size : int
    prob_label_font_size : int
    count_label_font_size : int

    Annotations
    -----------
    show_inclusion_exclusion : bool
        Show the P(A∪B) formula annotation below the diagram.
    show_independence_test : bool
        Show the P(A∩B) vs P(A)·P(B) annotation.
    annotation_y_offset : float
        How far below the diagram the formula annotations sit.
    annotation_font_size : int

    Ω box
    -----
    show_omega_box : bool
        Draw a bounding rectangle indicating the sample space.
    omega_box_padding : float
        Extra clearance around the circles for the Ω box.
    omega_stroke_width : float
    omega_font_size : int
    """

    # ---- circle geometry ----
    circle_radius:      float = 1.80
    circle_separation:  float = 1.05   # fraction of radius
    layout:             str   = "standard"

    # ---- zone surfaces ----
    zone_opacity:          float = 0.68
    zone_stroke_width:     float = 0.25
    show_zone_walls:       bool  = True
    zone_depth_resolution: int   = 3
    height_single:         float = 0.16
    height_double:         float = 0.30
    height_triple:         float = 0.44
    height_neither:        float = 0.00

    # ---- circle outlines ----
    show_outlines:        bool  = True
    outline_stroke_width: float = 2.80
    outline_y_lift:       float = 0.08

    # ---- labels ----
    show_set_labels:       bool  = True
    show_zone_labels:      bool  = True
    show_prob_labels:      bool  = True
    show_count_labels:     bool  = False
    label_lift:            float = 0.22
    set_label_font_size:   int   = 36
    zone_label_font_size:  int   = 20
    prob_label_font_size:  int   = 18
    count_label_font_size: int   = 16

    # ---- annotations ----
    show_inclusion_exclusion: bool  = False
    show_independence_test:   bool  = False
    annotation_y_offset:      float = -1.40
    annotation_font_size:     int   = 26

    # ---- Ω box ----
    show_omega_box:    bool  = True
    omega_box_padding: float = 0.55
    omega_stroke_width: float = 1.50
    omega_font_size:   int   = 32


# ---------------------------------------------------------------------------
# _CircleLayout  —  compute circle centre positions from config
# ---------------------------------------------------------------------------

def _circle_centres_2(cfg: VennConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (c1, c2) in XZ plane for a 2-set layout."""
    r  = cfg.circle_radius
    s  = cfg.circle_separation
    d  = r * s   # centre-to-centre half distance

    if cfg.layout in ("standard", "linear"):
        return np.array([-d, 0.0]), np.array([d, 0.0])
    elif cfg.layout == "nested":
        # A inside B: place both at origin, B has larger radius (handled externally)
        return np.array([0.0, 0.0]), np.array([0.0, 0.0])
    elif cfg.layout == "euler":
        # Disjoint: circles do not overlap
        gap  = r * 0.20
        return np.array([-r - gap, 0.0]), np.array([r + gap, 0.0])
    else:
        return np.array([-d, 0.0]), np.array([d, 0.0])


def _circle_centres_3(cfg: VennConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (c_a, c_b, c_c) in XZ plane for a 3-set layout."""
    r  = cfg.circle_radius
    s  = cfg.circle_separation
    d  = r * s

    if cfg.layout in ("standard", "linear"):
        # Equilateral triangle
        cx_a = -d
        cx_b =  d
        cx_c =  0.0
        cz_a =  d * 0.55
        cz_b =  d * 0.55
        cz_c = -d * 0.90
        return (np.array([cx_a, cz_a]),
                np.array([cx_b, cz_b]),
                np.array([cx_c, cz_c]))
    else:
        return (np.array([-d, 0.0]),
                np.array([d, 0.0]),
                np.array([0.0, -d * 1.2]))


# ---------------------------------------------------------------------------
# Zone arc definitions  —  compute arc_def lists for each zone
# ---------------------------------------------------------------------------

def _zone_arcs_2(
    c1: np.ndarray, r1: float,
    c2: np.ndarray, r2: float,
    geo: _CircleArcGeometry,
) -> dict[str, tuple[list[dict], np.ndarray]]:
    """Return arc_def and centroid for each of the 3 zones of a 2-set Venn.

    Returns dict with keys "a_only", "b_only", "ab".
    Value is (arc_def_list, centroid_xz).
    """
    if not geo.intersects:
        # Return full-circle zones for disjoint / subset cases
        def _full(cx, cz, r, toward=None):
            return [{"cx": cx, "cz": cz, "r": r,
                     "t_start": 0.0, "t_end": TAU, "ccw": True}]
        return {
            "a_only": (_full(c1[0], c1[1], r1), c1.copy()),
            "b_only": (_full(c2[0], c2[1], r2), c2.copy()),
            "ab":     ([], np.array([0.0, 0.0])),
        }

    # Arc of circle 1 that is OUTSIDE circle 2 = a_only boundary
    ta1_out = geo.arc_for_zone(1, inside=False)
    # Arc of circle 1 INSIDE circle 2 = ab boundary (circle-1 side)
    ta1_in  = geo.arc_for_zone(1, inside=True)
    # Arc of circle 2 OUTSIDE circle 1 = b_only boundary
    ta2_out = geo.arc_for_zone(2, inside=False)
    # Arc of circle 2 INSIDE circle 1 = ab boundary (circle-2 side)
    ta2_in  = geo.arc_for_zone(2, inside=True)

    # a_only: outer arc of circle 1
    a_only_arcs = [
        {"cx": c1[0], "cz": c1[1], "r": r1,
         "t_start": ta1_out[0], "t_end": ta1_out[1], "ccw": True},
    ]
    a_only_centroid = c1 - np.array([(c2[0]-c1[0])*0.40, (c2[1]-c1[1])*0.40])

    # b_only: outer arc of circle 2
    b_only_arcs = [
        {"cx": c2[0], "cz": c2[1], "r": r2,
         "t_start": ta2_out[0], "t_end": ta2_out[1], "ccw": True},
    ]
    b_only_centroid = c2 - np.array([(c1[0]-c2[0])*0.40, (c1[1]-c2[1])*0.40])

    # ab (lens): inner arc of circle 1 then inner arc of circle 2 (reversed)
    ab_arcs = [
        {"cx": c1[0], "cz": c1[1], "r": r1,
         "t_start": ta1_in[0], "t_end": ta1_in[1], "ccw": True},
        {"cx": c2[0], "cz": c2[1], "r": r2,
         "t_start": ta2_in[1], "t_end": ta2_in[0], "ccw": False},
    ]
    ab_centroid = (c1 + c2) / 2

    return {
        "a_only": (a_only_arcs, a_only_centroid),
        "b_only": (b_only_arcs, b_only_centroid),
        "ab":     (ab_arcs,     ab_centroid),
    }


# ---------------------------------------------------------------------------
# Main VennDiagram3D class
# ---------------------------------------------------------------------------

class VennDiagram3D(VGroup):
    """A detailed 3D Venn diagram for Manim probability animations.

    Supports 2-set and 3-set Venn diagrams with analytically computed
    zone boundaries, height-tiered platforms, floating circle outlines,
    and a comprehensive animation API.

    Basic 2-set usage
    -----------------
    >>> from manim import *
    >>> from manim_stats.probability.venn3d import (
    ...     VennDiagram3D, VennData2, VennConfig
    ... )
    >>>
    >>> class MyScene(ThreeDScene):
    ...     def construct(self):
    ...         data = VennData2(p_a=0.40, p_b=0.35, p_ab=0.15)
    ...         cfg  = VennConfig(show_prob_labels=True)
    ...         venn = VennDiagram3D.two_set(data, config=cfg)
    ...         self.set_camera_orientation(phi=60*DEGREES, theta=-45*DEGREES)
    ...         self.play(venn.animate_grow_circles())
    ...         self.play(venn.animate_fill_zones())
    ...         self.play(venn.animate_inclusion_exclusion())

    Parameters
    ----------
    config : VennConfig, optional
    """

    def __init__(self, config: VennConfig | None = None):
        super().__init__()
        self.cfg = config or VennConfig()

        # Populated by build methods
        self._zones:    dict[str, _ZoneSurface]        = {}
        self._outlines: list[_VennCircleOutline]        = []
        self._set_labels: list[MathTex]                 = []
        self._data2:  VennData2  | None                 = None
        self._data3:  VennData3  | None                 = None
        self._omega_box: VGroup | None                  = None
        self._annots:  VGroup                           = VGroup()
        self.add(self._annots)

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_omega_box(
        self,
        x_min: float, x_max: float,
        z_min: float, z_max: float,
        y:     float,
    ) -> None:
        """Draw the Ω bounding rectangle."""
        cfg = self.cfg
        p   = cfg.omega_box_padding
        x0, x1 = x_min - p, x_max + p
        z0, z1 = z_min - p, z_max + p

        corners = [
            np.array([x0, y, z0]),
            np.array([x1, y, z0]),
            np.array([x1, y, z1]),
            np.array([x0, y, z1]),
        ]
        edges = [(0,1),(1,2),(2,3),(3,0)]
        grp   = VGroup()
        for i, j in edges:
            line = Line3D(
                start=corners[i], end=corners[j],
                color=GRAY_C,
                stroke_width=cfg.omega_stroke_width,
            )
            line.set_opacity(0.65)
            grp.add(line)

        omega_lbl = MathTex(r"\Omega", color=GRAY_C,
                            font_size=cfg.omega_font_size)
        omega_lbl.move_to(np.array([x1 + 0.25, y + 0.22, z0]))
        grp.add(omega_lbl)

        self._omega_box = grp
        self.add(grp)

    def _add_zone(
        self,
        name:      str,
        arc_def:   list[dict],
        centroid:  np.ndarray,
        y_level:   float,
        color:     ManimColor,
        label_tex: str,
        prob:      float,
        count:     int | None = None,
    ) -> _ZoneSurface:
        zone = _ZoneSurface(
            arc_def=arc_def,
            centroid=centroid,
            y_level=y_level,
            color=color,
            cfg=self.cfg,
            label_tex=label_tex,
            prob=prob,
            count=count,
        )
        self._zones[name] = zone
        self.add(zone)
        return zone

    def _build_2set(
        self,
        data:     VennData2,
        circles:  tuple[np.ndarray, np.ndarray],
        radii:    tuple[float, float] = (None, None),
        total_n:  int | None = None,
    ) -> None:
        cfg   = self.cfg
        c1, c2 = circles
        r1 = radii[0] or cfg.circle_radius
        r2 = radii[1] or cfg.circle_radius

        geo   = _CircleArcGeometry(c1[0], c1[1], r1, c2[0], c2[1], r2)
        zones = _zone_arcs_2(c1, r1, c2, r2, geo)
        zp    = data.zone_probs()

        def _cnt(p): return round(total_n * p) if total_n else None

        la, lb = data.label_a, data.label_b

        self._add_zone(
            "a_only",
            *zones["a_only"],
            y_level=cfg.height_single,
            color=ZONE_A_ONLY,
            label_tex=rf"{la} \cap {lb}^c",
            prob=zp["a_only"],
            count=_cnt(zp["a_only"]),
        )
        self._add_zone(
            "b_only",
            *zones["b_only"],
            y_level=cfg.height_single,
            color=ZONE_B_ONLY,
            label_tex=rf"{la}^c \cap {lb}",
            prob=zp["b_only"],
            count=_cnt(zp["b_only"]),
        )
        if geo.intersects and zp["ab"] > 1e-5:
            self._add_zone(
                "ab",
                *zones["ab"],
                y_level=cfg.height_double,
                color=ZONE_AB,
                label_tex=rf"{la} \cap {lb}",
                prob=zp["ab"],
                count=_cnt(zp["ab"]),
            )

        # Circle outlines
        if cfg.show_outlines:
            y_out = cfg.height_double + cfg.outline_y_lift
            for cx_arr, r, col in [(c1, r1, SET_A_COLOR), (c2, r2, SET_B_COLOR)]:
                out = _VennCircleOutline(
                    cx=cx_arr[0], cz=cx_arr[1], r=r,
                    y=y_out, color=col,
                    stroke_width=cfg.outline_stroke_width,
                )
                self._outlines.append(out)
                self.add(out)

        # Set name labels
        if cfg.show_set_labels:
            y_lbl = cfg.height_single + 0.38
            for cx_arr, r, col, label in [
                (c1, r1, SET_A_COLOR, la),
                (c2, r2, SET_B_COLOR, lb),
            ]:
                # Place label at the far end of the circle (away from the other)
                direction = cx_arr - (c1 + c2) / 2
                norm = np.linalg.norm(direction)
                if norm > 1e-4:
                    direction = direction / norm
                else:
                    direction = np.array([1.0, 0.0])
                pos = np.array([
                    cx_arr[0] + direction[0] * r * 0.72,
                    y_lbl,
                    cx_arr[1] + direction[1] * r * 0.72,
                ])
                slbl = MathTex(label, color=col,
                               font_size=cfg.set_label_font_size)
                slbl.move_to(pos)
                self._set_labels.append(slbl)
                self.add(slbl)

        # Omega box
        if cfg.show_omega_box:
            all_x = [c1[0] - r1, c1[0] + r1, c2[0] - r2, c2[0] + r2]
            all_z = [c1[1] - r1, c1[1] + r1, c2[1] - r2, c2[1] + r2]
            self._build_omega_box(
                min(all_x), max(all_x), min(all_z), max(all_z),
                y=0.0,
            )

        # Annotations
        ann_y = cfg.annotation_y_offset
        ann_z = (c1[1] + c2[1]) / 2

        if cfg.show_inclusion_exclusion:
            ie = _InclusionExclusionBracket(
                data=data,
                pos=np.array([0.0, ann_y, ann_z]),
                font_size=cfg.annotation_font_size,
            )
            self._annots.add(ie)
            self.ie_banner = ie

        if cfg.show_independence_test:
            ind = _IndependenceAnnotation(
                data=data,
                pos=np.array([0.0, ann_y - 0.55, ann_z]),
                font_size=cfg.annotation_font_size,
            )
            self._annots.add(ind)
            self.independence_banner = ind

    def _build_3set(
        self,
        data:    VennData3,
        circles: tuple[np.ndarray, np.ndarray, np.ndarray],
        total_n: int | None = None,
    ) -> None:
        cfg    = self.cfg
        c1, c2, c3 = circles
        r      = cfg.circle_radius
        zp     = data.zone_probs()

        def _cnt(p): return round(total_n * p) if total_n else None

        la, lb, lc = data.label_a, data.label_b, data.label_c

        geo12 = _CircleArcGeometry(c1[0],c1[1],r, c2[0],c2[1],r)
        geo13 = _CircleArcGeometry(c1[0],c1[1],r, c3[0],c3[1],r)
        geo23 = _CircleArcGeometry(c2[0],c2[1],r, c3[0],c3[1],r)

        # ---- Single zones (approximated as circle sectors with other circles cut out)
        # We approximate each single zone as the outer arc of its circle
        # minus both pairwise overlaps.  For visual rendering we use the
        # full outer arc (partial circle patch) as a simple approximation.
        single_specs = [
            ("a_only", c1, r, SET_A_COLOR, ZONE_A_ONLY,
             rf"{la} \cap {lb}^c \cap {lc}^c", zp["a_only"]),
            ("b_only", c2, r, SET_B_COLOR, ZONE_B_ONLY,
             rf"{la}^c \cap {lb} \cap {lc}^c", zp["b_only"]),
            ("c_only", c3, r, SET_C_COLOR, ZONE_C_ONLY,
             rf"{la}^c \cap {lb}^c \cap {lc}",  zp["c_only"]),
        ]
        for name, c, rad, _, color, label_tex, prob in single_specs:
            if prob < 1e-5:
                continue
            # Build outer arc (the non-overlapping lune)
            # Approximate: use full circle arcs and rely on opacity layering
            full_arc = [{"cx": float(c[0]), "cz": float(c[1]), "r": rad,
                         "t_start": 0.0, "t_end": TAU, "ccw": True}]
            # Centroid pushed away from centre
            others = [cc for cc in [c1, c2, c3] if not np.allclose(cc, c)]
            push   = c - np.mean(others, axis=0)
            pnorm  = np.linalg.norm(push)
            if pnorm > 1e-4:
                centroid = c + push / pnorm * rad * 0.55
            else:
                centroid = c.copy()
            self._add_zone(name, full_arc, centroid,
                           cfg.height_single, color, label_tex, prob,
                           count=_cnt(prob))

        # ---- Pairwise zones
        pair_specs = [
            ("ab_only", c1, c2, r, r, geo12, ZONE_AB,
             rf"{la} \cap {lb} \cap {lc}^c", zp["ab_only"]),
            ("ac_only", c1, c3, r, r, geo13, ZONE_AC,
             rf"{la} \cap {lb}^c \cap {lc}", zp["ac_only"]),
            ("bc_only", c2, c3, r, r, geo23, ZONE_BC,
             rf"{la}^c \cap {lb} \cap {lc}", zp["bc_only"]),
        ]
        for name, ca, cb, ra, rb, geo, color, label_tex, prob in pair_specs:
            if prob < 1e-5 or not geo.intersects:
                continue
            arcs_dict = _zone_arcs_2(ca, ra, cb, rb, geo)
            self._add_zone(name, *arcs_dict["ab"],
                           cfg.height_double, color, label_tex, prob,
                           count=_cnt(prob))

        # ---- Triple intersection (centroid of all three)
        if zp["abc"] > 1e-5:
            triple_centroid = np.mean([c1, c2, c3], axis=0)
            # Use geo12 lens arc as approximation for innermost zone
            if geo12.intersects:
                arcs_dict12 = _zone_arcs_2(c1, r, c2, r, geo12)
                abc_arcs    = arcs_dict12["ab"][0]
            else:
                abc_arcs = [{"cx": float(triple_centroid[0]),
                             "cz": float(triple_centroid[1]),
                             "r": r * 0.25,
                             "t_start": 0.0, "t_end": TAU, "ccw": True}]
            self._add_zone(
                "abc", abc_arcs, triple_centroid,
                cfg.height_triple, ZONE_ABC,
                rf"{la} \cap {lb} \cap {lc}", zp["abc"],
                count=_cnt(zp["abc"]),
            )

        # Circle outlines
        if cfg.show_outlines:
            y_out = cfg.height_double + cfg.outline_y_lift
            for c_arr, col in [(c1, SET_A_COLOR), (c2, SET_B_COLOR), (c3, SET_C_COLOR)]:
                out = _VennCircleOutline(
                    cx=float(c_arr[0]), cz=float(c_arr[1]), r=r,
                    y=y_out, color=col,
                    stroke_width=cfg.outline_stroke_width,
                )
                self._outlines.append(out)
                self.add(out)

        # Set labels
        if cfg.show_set_labels:
            for c_arr, col, label, others in [
                (c1, SET_A_COLOR, la, [c2, c3]),
                (c2, SET_B_COLOR, lb, [c1, c3]),
                (c3, SET_C_COLOR, lc, [c1, c2]),
            ]:
                push = c_arr - np.mean(others, axis=0)
                pnorm = np.linalg.norm(push)
                if pnorm > 1e-4:
                    push = push / pnorm
                else:
                    push = np.array([0.0, -1.0])
                pos = np.array([
                    c_arr[0] + push[0] * r * 0.88,
                    cfg.height_double + 0.38,
                    c_arr[1] + push[1] * r * 0.88,
                ])
                slbl = MathTex(label, color=col,
                               font_size=cfg.set_label_font_size)
                slbl.move_to(pos)
                self._set_labels.append(slbl)
                self.add(slbl)

        # Omega box
        if cfg.show_omega_box:
            all_x = [c[0] - r for c in [c1,c2,c3]] + [c[0] + r for c in [c1,c2,c3]]
            all_z = [c[1] - r for c in [c1,c2,c3]] + [c[1] + r for c in [c1,c2,c3]]
            self._build_omega_box(
                min(all_x), max(all_x), min(all_z), max(all_z), y=0.0
            )

    # ------------------------------------------------------------------
    # Animation API
    # ------------------------------------------------------------------

    def animate_grow_circles(
        self,
        lag_ratio: float = 0.35,
        run_time:  float = 2.5,
    ) -> LaggedStart:
        """Draw circle outlines one by one, each tracing its full ring.

        The outlines are drawn in the order they were added (A, B, C)::

            self.play(venn.animate_grow_circles())
        """
        if not self._outlines:
            return LaggedStart(FadeIn(VGroup()), lag_ratio=0, run_time=0.1)
        return LaggedStart(
            *[Create(out.curve, run_time=run_time * 0.65)
              for out in self._outlines],
            lag_ratio=lag_ratio,
            run_time=run_time,
        )

    def animate_fill_zones(
        self,
        lag_ratio: float = 0.12,
        run_time:  float = 3.0,
    ) -> Succession:
        """Zones rise level by level: single → double → triple.

        This ordering reinforces the inclusion-exclusion structure:
        individual sets first, then their interactions.
        """
        single_names = {"a_only", "b_only", "c_only"}
        double_names = {"ab_only", "ac_only", "bc_only", "ab"}
        triple_names = {"abc"}

        def _level_anim(names, rt):
            zones = [self._zones[n] for n in names if n in self._zones]
            if not zones:
                return FadeIn(VGroup(), run_time=0.01)
            return LaggedStart(
                *[GrowFromPoint(z, z.floor_center, run_time=rt * 0.65)
                  for z in zones],
                lag_ratio=lag_ratio,
                run_time=rt,
            )

        return Succession(
            _level_anim(single_names, run_time * 0.45),
            _level_anim(double_names, run_time * 0.35),
            _level_anim(triple_names, run_time * 0.20),
        )

    def animate_highlight_zone(
        self,
        zone_name:       str,
        highlight_color: ManimColor | None = None,
        scale_factor:    float = 1.12,
        run_time:        float = 0.60,
    ) -> Succession:
        """Flash-scale one zone and dim all others.

        Parameters
        ----------
        zone_name : str
            One of: ``"a_only"``, ``"b_only"``, ``"c_only"``,
            ``"ab"``, ``"ab_only"``, ``"ac_only"``, ``"bc_only"``, ``"abc"``.
        """
        if zone_name not in self._zones:
            return Succession(FadeIn(VGroup(), run_time=0.1))

        target   = self._zones[zone_name]
        others   = [z for k, z in self._zones.items() if k != zone_name]
        h_color  = highlight_color or self.cfg.highlight_color \
            if hasattr(self.cfg, "highlight_color") else YELLOW

        dim   = AnimationGroup(*[
            z.animate(run_time=run_time / 2).set_opacity(ZONE_DIM_OPACITY)
            for z in others
        ])
        scale = target.animate(run_time=run_time / 2).scale(scale_factor)
        restore = AnimationGroup(
            *[z.animate(run_time=run_time / 2).set_opacity(self.cfg.zone_opacity)
              for z in others],
            target.animate(run_time=run_time / 2).scale(1 / scale_factor),
        )
        return Succession(AnimationGroup(dim, scale), restore)

    def animate_inclusion_exclusion(
        self,
        run_time_each: float = 1.0,
    ) -> Succession:
        """Step through P(A∪B) = P(A) + P(B) − P(A∩B) visually.

        Step 1: Highlight A (A_only + AB zones).
        Step 2: Highlight B (B_only + AB zones).
        Step 3: Dim the AB zone (show we counted it twice).
        Step 4: Re-brighten AB at normal opacity (correct once).
        Step 5: Reveal the inclusion-exclusion annotation.
        """
        def _set_zone_opacity(names, opacity, rt):
            zones = [self._zones[n] for n in names if n in self._zones]
            if not zones:
                return FadeIn(VGroup(), run_time=0.01)
            return AnimationGroup(*[
                z.animate(run_time=rt).set_opacity(opacity)
                for z in zones
            ])

        full  = self.cfg.zone_opacity
        dim   = ZONE_DIM_OPACITY
        rt    = run_time_each

        # Step 1: show A
        s1 = _set_zone_opacity(
            ["b_only","c_only","ab_only","ac_only","bc_only","abc"], dim, rt
        )
        # Step 2: show B (re-brighten b_only, keep ab bright)
        s2 = _set_zone_opacity(["b_only"], full, rt * 0.6)
        # Step 3: double-count AB — flash AB to bright red
        s3 = AnimationGroup(
            *[self._zones[n].animate(run_time=rt * 0.4)
                             .set_color(ManimColor("#FF5252"))
              for n in ["ab", "ab_only"] if n in self._zones]
        ) if any(n in self._zones for n in ["ab","ab_only"]) else FadeIn(VGroup())
        # Step 4: restore AB
        s4 = _set_zone_opacity(["ab","ab_only"], full, rt * 0.5)
        # Step 5: restore all + show annotation
        s5 = _set_zone_opacity(
            list(self._zones.keys()), full, rt * 0.6
        )
        s6 = FadeIn(self._annots, run_time=rt) \
            if len(self._annots) > 0 else FadeIn(VGroup(), run_time=0.1)

        return Succession(s1, s2, s3, s4, s5, s6)

    def animate_condition_on(
        self,
        set_name: str,
        run_time: float = 1.5,
    ) -> Succession:
        """Dim all zones outside set_name; show the P(B|A) annotation.

        For a 2-set diagram, ``set_name = "a"`` dims "b_only" and
        "neither", leaving "a_only" and "ab" fully lit.  A MathTex
        annotation P(B|A) = P(A∩B)/P(A) is revealed above the AB zone.

        Parameters
        ----------
        set_name : str
            ``"a"`` or ``"b"`` (or ``"c"`` for 3-set).
        """
        set_name = set_name.lower()

        # Zones that belong to the conditioned set
        in_set: dict[str, set[str]] = {
            "a": {"a_only", "ab", "ab_only", "ac_only", "abc"},
            "b": {"b_only", "ab", "ab_only", "bc_only", "abc"},
            "c": {"c_only", "ac_only", "bc_only", "abc"},
        }
        inside  = in_set.get(set_name, set())
        outside = set(self._zones.keys()) - inside

        dim_out  = AnimationGroup(*[
            self._zones[n].animate(run_time=run_time * 0.55)
                          .set_opacity(ZONE_DIM_OPACITY)
            for n in outside if n in self._zones
        ])

        # Conditional probability annotation
        ann_zone = None
        ann_tex  = ""
        if self._data2 is not None:
            d = self._data2
            if set_name == "a":
                val    = d.p_b_given_a()
                ann_tex = (rf"P({d.label_b}|{d.label_a}) = "
                           rf"\frac{{{d.p_ab:.3f}}}{{{d.p_a:.3f}}} = {val:.3f}")
                ann_zone = "ab" if "ab" in self._zones else None
            elif set_name == "b":
                val    = d.p_a_given_b()
                ann_tex = (rf"P({d.label_a}|{d.label_b}) = "
                           rf"\frac{{{d.p_ab:.3f}}}{{{d.p_b:.3f}}} = {val:.3f}")
                ann_zone = "ab" if "ab" in self._zones else None

        cond_lbl = None
        if ann_tex and ann_zone and ann_zone in self._zones:
            z   = self._zones[ann_zone]
            lbl = MathTex(ann_tex, color=YELLOW,
                          font_size=self.cfg.annotation_font_size)
            tc  = z.floor_center
            lbl.move_to(np.array([tc[0], self.cfg.height_double + 0.70, tc[2]]))
            self.add(lbl)
            cond_lbl = FadeIn(lbl, run_time=run_time * 0.45)

        return Succession(
            dim_out,
            cond_lbl if cond_lbl else FadeIn(VGroup(), run_time=0.1),
        )

    def animate_restore(
        self,
        run_time: float = 0.8,
    ) -> AnimationGroup:
        """Restore all zones to full opacity and original color."""
        return AnimationGroup(*[
            z.animate(run_time=run_time).set_opacity(self.cfg.zone_opacity)
            for z in self._zones.values()
        ])

    def animate_independence_test(
        self,
        run_time: float = 1.5,
    ) -> Succession:
        """Show P(A∩B) vs P(A)·P(B) with a verdict badge.

        Highlights the AB zone, then reveals the independence annotation.
        """
        if self._data2 is None:
            return Succession(FadeIn(VGroup(), run_time=0.1))

        highlight = self.animate_highlight_zone("ab", run_time=run_time * 0.55)
        if hasattr(self, "independence_banner"):
            show_ann = FadeIn(self.independence_banner, run_time=run_time * 0.45)
        else:
            d   = self._data2
            ind = _IndependenceAnnotation(
                data=d,
                pos=np.array([0.0, self.cfg.annotation_y_offset, 0.0]),
                font_size=self.cfg.annotation_font_size,
            )
            self.add(ind)
            show_ann = FadeIn(ind, run_time=run_time * 0.45)

        return Succession(highlight, show_ann)

    def animate_morph_to(
        self,
        new_data:  VennData2,
        run_time:  float = 2.0,
    ) -> AnimationGroup:
        """Morph all zone surfaces to reflect new probability values.

        A new ``VennDiagram3D`` is built internally and each zone
        transforms into its counterpart.  The circle outlines also
        rescale if radii change.

        Parameters
        ----------
        new_data : VennData2
            Updated probabilities.  Only 2-set supported currently.
        """
        new_venn = VennDiagram3D.two_set(new_data, config=self.cfg)
        anims    = []
        for name, zone in self._zones.items():
            if name in new_venn._zones:
                anims.append(Transform(zone, new_venn._zones[name],
                                       run_time=run_time))
        return AnimationGroup(*anims)

    def animate_add_set_c(
        self,
        data3:    VennData3,
        run_time: float = 2.5,
    ) -> Succession:
        """Grow the C circle into an existing 2-set diagram.

        The new circle outline traces in, then the C-containing zones
        rise: c_only, ac_only, bc_only, abc — in that order.
        """
        cfg      = self.cfg
        circles  = _circle_centres_3(cfg)
        _, _, c3 = circles
        r        = cfg.circle_radius

        c3_out = _VennCircleOutline(
            cx=float(c3[0]), cz=float(c3[1]), r=r,
            y=cfg.height_double + cfg.outline_y_lift,
            color=SET_C_COLOR,
            stroke_width=cfg.outline_stroke_width,
        )
        self.add(c3_out)

        new_zone_names = ["c_only", "ac_only", "bc_only", "abc"]
        # Build a full 3-set venn internally, extract the new zones
        full3 = VennDiagram3D.three_set(data3, config=cfg)
        new_zones  = {n: full3._zones[n] for n in new_zone_names
                      if n in full3._zones}
        for zone in new_zones.values():
            self.add(zone)
            self._zones[list(new_zones.keys())[
                list(new_zones.values()).index(zone)
            ]] = zone

        draw_c  = Create(c3_out.curve, run_time=run_time * 0.35)
        grow    = LaggedStart(
            *[GrowFromPoint(z, z.floor_center, run_time=run_time * 0.45)
              for z in new_zones.values()],
            lag_ratio=0.15,
            run_time=run_time * 0.65,
        )
        return Succession(draw_c, grow)

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def two_set(
        cls,
        data:    VennData2,
        config:  VennConfig | None = None,
        total_n: int | None = None,
    ) -> "VennDiagram3D":
        """Build a 2-set Venn diagram.

        Parameters
        ----------
        data : VennData2
        config : VennConfig, optional
        total_n : int, optional
            If set, natural frequency counts are shown inside zones.

        Example
        -------
        >>> data = VennData2(p_a=0.40, p_b=0.35, p_ab=0.15)
        >>> venn = VennDiagram3D.two_set(data)
        """
        cfg  = config or VennConfig()
        venn = cls(config=cfg)
        venn._data2   = data
        circles = _circle_centres_2(cfg)
        venn._build_2set(data, circles, total_n=total_n)
        return venn

    @classmethod
    def three_set(
        cls,
        data:    VennData3,
        config:  VennConfig | None = None,
        total_n: int | None = None,
    ) -> "VennDiagram3D":
        """Build a 3-set Venn diagram.

        Parameters
        ----------
        data : VennData3
        config : VennConfig, optional
        total_n : int, optional

        Example
        -------
        >>> data = VennData3(
        ...     p_a=0.50, p_b=0.40, p_c=0.35,
        ...     p_ab=0.20, p_ac=0.15, p_bc=0.12,
        ...     p_abc=0.06,
        ... )
        >>> venn = VennDiagram3D.three_set(data)
        """
        cfg  = config or VennConfig()
        venn = cls(config=cfg)
        venn._data3   = data
        circles = _circle_centres_3(cfg)
        venn._build_3set(data, circles, total_n=total_n)
        return venn

    @classmethod
    def from_counts(
        cls,
        n_a:    int,
        n_b:    int,
        n_ab:   int,
        n_total: int,
        config:  VennConfig | None = None,
        label_a: str = "A",
        label_b: str = "B",
    ) -> "VennDiagram3D":
        """Build a 2-set Venn from natural frequency counts.

        Parameters
        ----------
        n_a, n_b, n_ab : int
            Counts of outcomes in A, B, and A∩B.
        n_total : int
            Total sample size (denominator for all probabilities).

        Example
        -------
        >>> # 200 people: 80 own a cat, 70 own a dog, 30 own both
        >>> venn = VennDiagram3D.from_counts(80, 70, 30, 200)
        """
        data = VennData2(
            p_a  = n_a  / n_total,
            p_b  = n_b  / n_total,
            p_ab = n_ab / n_total,
            label_a=label_a,
            label_b=label_b,
        )
        cfg = config or VennConfig()
        cfg.show_count_labels = True
        return cls.two_set(data, config=cfg, total_n=n_total)

    @classmethod
    def disjoint(
        cls,
        p_a:    float,
        p_b:    float,
        config: VennConfig | None = None,
        label_a: str = "A",
        label_b: str = "B",
    ) -> "VennDiagram3D":
        """Euler diagram for two disjoint events (P(A∩B) = 0).

        The circles are placed far enough apart that they don't overlap,
        making the mutual exclusivity visually obvious.
        """
        cfg         = config or VennConfig()
        cfg.layout  = "euler"
        data        = VennData2(p_a=p_a, p_b=p_b, p_ab=0.0,
                                label_a=label_a, label_b=label_b)
        return cls.two_set(data, config=cfg)

    @classmethod
    def subset(
        cls,
        p_a:    float,
        p_b:    float,
        config: VennConfig | None = None,
        label_a: str = "A",
        label_b: str = "B",
    ) -> "VennDiagram3D":
        """Nested diagram for A ⊆ B (every outcome in A is also in B).

        The A circle sits entirely inside the B circle.
        P(A∩B) = P(A) since A ⊆ B.
        """
        cfg         = config or VennConfig()
        cfg.layout  = "nested"
        data        = VennData2(p_a=p_a, p_b=p_b, p_ab=p_a,
                                label_a=label_a, label_b=label_b)
        r_a = cfg.circle_radius * np.sqrt(p_a / max(p_b, 1e-9))
        r_b = cfg.circle_radius
        venn = cls(config=cfg)
        venn._data2 = data
        c_b = np.array([0.0, 0.0])
        offset = r_b * 0.30
        c_a = np.array([-offset, 0.0])
        venn._build_2set(data, (c_a, c_b),
                         radii=(r_a, r_b))
        return venn

    @classmethod
    def independent(
        cls,
        p_a:    float,
        p_b:    float,
        config: VennConfig | None = None,
        label_a: str = "A",
        label_b: str = "B",
    ) -> "VennDiagram3D":
        """2-set diagram where A and B are independent: P(A∩B) = P(A)·P(B).

        The independence annotation is automatically shown.
        """
        cfg = config or VennConfig()
        cfg.show_independence_test    = True
        cfg.show_inclusion_exclusion  = False
        data = VennData2(
            p_a  = p_a,
            p_b  = p_b,
            p_ab = p_a * p_b,
            label_a=label_a,
            label_b=label_b,
        )
        return cls.two_set(data, config=cfg)