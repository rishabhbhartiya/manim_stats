"""
manim_stats/probability/sample_space.py
=========================================
Production-quality 3D sample space and event visualization for Manim.

Objects
-------

SampleSpaceConfig  (dataclass)
    Every visual and layout parameter in one place.

_SampleSpaceBox  (VGroup)
    The Ω boundary rendered as a 3D transparent box: 12 visible edges
    built from Line3D, semi-transparent face panels (front + top only,
    so the interior stays visible), 8 corner dot markers, and a floating
    MathTex Ω label.  The interior floor has a faint grid.

_EventRegion3D  (VGroup)
    An event A ⊆ Ω rendered as a raised 3D prism platform.
    Supports three footprint shapes:
      "rect"    — rectangular prism, placed by (x0, z0, width, depth)
      "ellipse" — Surface of revolution (smooth elliptical top)
      "poly"    — arbitrary polygon footprint from a vertex list
    All variants share the same shaded-face treatment (top / front /
    right) from the module's standard _ShadeBox idiom.
    Carries: event label (MathTex above), P(A) annotation, optional
    fill pattern (solid / hatched / gradient).

_IntersectionRegion  (VGroup)
    Visual overlay showing A ∩ B: a raised platform colored distinctly,
    sitting slightly higher than both parent regions, labeled P(A ∩ B).

_ComplementRegion  (VGroup)
    Ω \ A — the space outside an event region, rendered as a floor-level
    shaded panel filling the box minus the event footprint, with a
    hatched diagonal pattern to distinguish it from the event.

_ProbabilityAxis  (VGroup)
    A decorated [0, 1] number line on the floor plane with tick marks
    at 0, 1/6, 1/4, 1/3, 1/2, 2/3, 3/4, 5/6, 1 and floating labels.
    Event probability markers (colored vertical lines + labels) can be
    added for P(A), P(B), P(A∪B), P(A∩B).

_OutcomeGrid  (VGroup)
    A rectangular grid of Sphere glyphs, one per outcome in a discrete
    sample space.  Outcomes can be colored by event membership:
      no event  → grey,  A only → blue,  B only → red,
      A∩B       → purple,  neither → dark grey.
    Supports card deck (4×13), dice (1×6), gene pairs, custom grids.

_VennZone  (VGroup)
    One of the 7 atomic regions of a 3-circle Venn diagram, rendered as
    a Surface patch with its own color, opacity, and label.

SampleSpace3D  (VGroup)
    Main class.  Composes the box, event regions, optional Venn layout,
    optional discrete outcome grid, and probability axis.

    Factory classmethods:
      two_event_venn(p_a, p_b, p_ab)     – 2-circle Venn inside Ω box
      three_event_venn(p_a,…)            – 3-circle with all 7 zones
      from_dice(event_A, event_B)        – discrete Ω={1…6} grid
      from_cards(event_A, event_B)       – 52-card layout grid
      conditional_highlight(p_a,p_b,p_ab)– P(B|A) conditional visual

    Animation suite:
      animate_build_space()              – box boundary grows edge by edge
      animate_add_event(event)           – event platform rises from floor
      animate_intersection(a, b, inter)  – intersection region materializes
      animate_complement(event, comp)    – complement shading sweeps in
      animate_show_union(a, b, union)    – union region builds from A + B
      animate_sweep_probability(p)       – vertical plane sweeps to P=p
      animate_highlight_outcome(i, j)    – flash one discrete outcome
      animate_venn_build()               – circles grow then labels appear
      animate_conditional(a, b, inter)   – dim Ω minus A then show B∩A fraction
      animate_show_operation(op)         – morph between ∩ / ∪ / complement
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from manim import (
    WHITE, BLACK, GRAY, GRAY_A, GRAY_B, GRAY_C, GRAY_D, GRAY_E,
    BLUE, BLUE_E, RED, RED_E, GREEN, GREEN_E, YELLOW, ORANGE, PURPLE,
    UP, DOWN, LEFT, RIGHT,
    PI, TAU,
    VGroup, VMobject,
    Line3D, Polygon, Sphere, Dot3D, Arrow3D,
    Surface,
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

OMEGA_COLOR    = ManimColor("#37474F")   # dark slate  – Ω boundary
OMEGA_FACE     = ManimColor("#1C2A30")   # near-black  – box face fill
GRID_COLOR     = ManimColor("#455A64")   # blue-grey   – floor grid
CORNER_COLOR   = ManimColor("#546E7A")   # lighter slate – corner dots

EVENT_A_COLOR  = ManimColor("#1565C0")   # blue        – event A
EVENT_B_COLOR  = ManimColor("#B71C1C")   # red         – event B
EVENT_C_COLOR  = ManimColor("#2E7D32")   # green       – event C
INTER_AB_COLOR = ManimColor("#7B1FA2")   # purple      – A ∩ B
INTER_ABC_COLOR= ManimColor("#E65100")   # orange      – A ∩ B ∩ C
UNION_COLOR    = ManimColor("#00838F")   # teal        – A ∪ B
COMP_COLOR     = ManimColor("#263238")   # very dark   – complement
NEITHER_COLOR  = ManimColor("#212121")   # near-black  – neither region

OUTCOME_NONE   = ManimColor("#546E7A")   # grey        – no event
OUTCOME_A      = ManimColor("#1E88E5")   # blue        – in A
OUTCOME_B      = ManimColor("#E53935")   # red         – in B
OUTCOME_AB     = ManimColor("#8E24AA")   # purple      – in A ∩ B

AXIS_COLOR     = GRAY_C
LABEL_COLOR    = ManimColor("#ECEFF1")

FACE_DARKEN_SIDE  = 0.38
FACE_DARKEN_RIGHT = 0.55
FACE_DARKEN_BACK  = 0.65


def _dk(c: ManimColor, f: float) -> ManimColor:
    return interpolate_color(c, BLACK, f)

def _lt(c: ManimColor, f: float) -> ManimColor:
    return interpolate_color(c, WHITE, f)


# ---------------------------------------------------------------------------
# SampleSpaceConfig
# ---------------------------------------------------------------------------

@dataclass
class SampleSpaceConfig:
    """All visual and layout parameters for SampleSpace3D.

    Box geometry
    ------------
    box_width : float
        X extent of the Ω box.
    box_depth : float
        Z extent of the Ω box.
    box_height : float
        Y extent (visual height of the bounding box walls).
    box_edge_width : float
        Stroke width of the 12 box edges.
    box_face_opacity : float
        Opacity of the two visible box face panels (front + top).
    show_corner_dots : bool
        Place small Dot3D markers at each of the 8 box corners.
    show_floor_grid : bool
        Render a faint grid on the floor (y=0) inside the box.
    floor_grid_n : int
        Number of grid lines in each direction.
    omega_label_font_size : int
        Font size of the floating Ω label.

    Event regions
    -------------
    event_height : float
        Default height of a rectangular event prism.
    event_opacity : float
        Fill opacity of event platform faces.
    event_edge_width : float
        Stroke width of event prism edges.
    ellipse_resolution : int
        Mesh resolution for elliptical event surfaces.
    show_event_labels : bool
        Show event name labels above each region.
    show_prob_labels : bool
        Show P(A) annotations above event regions.
    label_font_size : int
        Font size for event labels.
    prob_label_font_size : int
        Font size for P(A) labels.

    Intersection / union / complement
    ----------------------------------
    inter_height_boost : float
        How much higher the intersection prism sits above the event height.
    inter_opacity : float
        Fill opacity of intersection / union overlays.
    comp_hatch_n : int
        Number of diagonal hatch lines in the complement region.
    comp_opacity : float
        Fill opacity of the complement shaded panel.

    Probability axis
    ----------------
    show_prob_axis : bool
        Render the [0,1] probability number line on the floor.
    axis_length : float
        Length of the probability axis in Manim units.
    axis_y_offset : float
        Y offset (should be 0 for floor placement).
    axis_z_offset : float
        Z position of the axis (typically at the front of the box).

    Discrete outcome grid
    ---------------------
    outcome_radius : float
        Radius of Sphere / Dot3D glyphs in the outcome grid.
    outcome_spacing : float
        Centre-to-centre spacing between outcomes.
    outcome_opacity : float
        Opacity of outcome glyphs.

    Venn diagram
    ------------
    venn_circle_radius : float
        Radius of each Venn circle (used for 2- and 3-set layouts).
    venn_overlap : float
        Horizontal overlap between adjacent circles as a fraction of radius.
    venn_height : float
        Height of the Venn platform prisms.
    venn_label_font_size : int
        Font size of Venn zone probability labels.

    Layout
    ------
    origin : np.ndarray
        World position of the front-left-bottom corner of the Ω box.
    """

    # ---- box ----
    box_width:            float = 7.00
    box_depth:            float = 5.00
    box_height:           float = 0.08
    box_edge_width:       float = 1.80
    box_face_opacity:     float = 0.10
    show_corner_dots:     bool  = True
    show_floor_grid:      bool  = True
    floor_grid_n:         int   = 8
    omega_label_font_size: int  = 36

    # ---- event regions ----
    event_height:         float = 0.22
    event_opacity:        float = 0.62
    event_edge_width:     float = 0.70
    ellipse_resolution:   int   = 28
    show_event_labels:    bool  = True
    show_prob_labels:     bool  = True
    label_font_size:      int   = 26
    prob_label_font_size: int   = 22

    # ---- set operations ----
    inter_height_boost:   float = 0.12
    inter_opacity:        float = 0.80
    comp_hatch_n:         int   = 14
    comp_opacity:         float = 0.28

    # ---- probability axis ----
    show_prob_axis:  bool  = False
    axis_length:     float = 6.50
    axis_y_offset:   float = 0.00
    axis_z_offset:   float = 0.00

    # ---- discrete grid ----
    outcome_radius:  float = 0.13
    outcome_spacing: float = 0.44
    outcome_opacity: float = 0.90

    # ---- Venn ----
    venn_circle_radius: float = 1.60
    venn_overlap:       float = 0.50
    venn_height:        float = 0.18
    venn_label_font_size: int = 20

    # ---- layout ----
    origin: np.ndarray = field(
        default_factory=lambda: np.array([-3.5, 0.0, -2.5])
    )


# ---------------------------------------------------------------------------
# _SampleSpaceBox  —  the Ω boundary
# ---------------------------------------------------------------------------

class _SampleSpaceBox(VGroup):
    """3D bounding box for the sample space Ω.

    Rendered as 12 Line3D edges + two semi-transparent face panels
    (front face and top face) so the interior remains fully visible.
    Eight Dot3D corner markers and a floating Ω label complete the
    visual cue that "everything inside is the sample space."
    """

    def __init__(self, cfg: SampleSpaceConfig):
        super().__init__()
        ox, oy, oz = cfg.origin
        w, h, d    = cfg.box_width, cfg.box_height, cfg.box_depth
        ew         = cfg.box_edge_width

        # 8 corners
        corners = {
            "FBL": np.array([ox,     oy,     oz    ]),
            "FBR": np.array([ox + w, oy,     oz    ]),
            "FTL": np.array([ox,     oy + h, oz    ]),
            "FTR": np.array([ox + w, oy + h, oz    ]),
            "BBL": np.array([ox,     oy,     oz + d]),
            "BBR": np.array([ox + w, oy,     oz + d]),
            "BTL": np.array([ox,     oy + h, oz + d]),
            "BTR": np.array([ox + w, oy + h, oz + d]),
        }
        self._corners = corners

        # 12 edges
        edges = [
            ("FBL","FBR"), ("FBL","FTL"), ("FBR","FTR"), ("FTL","FTR"),
            ("BBL","BBR"), ("BBL","BTL"), ("BBR","BTR"), ("BTL","BTR"),
            ("FBL","BBL"), ("FBR","BBR"), ("FTL","BTL"), ("FTR","BTR"),
        ]
        self.edges: list[Line3D] = []
        for a, b in edges:
            line = Line3D(
                start=corners[a], end=corners[b],
                color=OMEGA_COLOR, stroke_width=ew,
            )
            line.set_opacity(0.90)
            self.add(line)
            self.edges.append(line)

        # Front face panel (semi-transparent)
        front = Polygon(
            corners["FBL"], corners["FBR"],
            corners["FTR"], corners["FTL"],
            color=OMEGA_FACE,
        )
        front.set_fill(color=OMEGA_FACE, opacity=cfg.box_face_opacity)
        front.set_stroke(width=0)
        self.add(front)

        # Top face panel
        top = Polygon(
            corners["FTL"], corners["FTR"],
            corners["BTR"], corners["BTL"],
            color=OMEGA_FACE,
        )
        top.set_fill(color=OMEGA_FACE, opacity=cfg.box_face_opacity * 0.6)
        top.set_stroke(width=0)
        self.add(top)

        # Corner dots
        if cfg.show_corner_dots:
            for pt in corners.values():
                dot = Dot3D(point=pt, radius=0.035, color=CORNER_COLOR)
                dot.set_opacity(0.70)
                self.add(dot)

        # Floor grid
        if cfg.show_floor_grid:
            n = cfg.floor_grid_n
            for i in range(n + 1):
                t  = i / n
                xv = ox + t * w
                zv = oz + t * d
                self.add(Line3D(
                    start=np.array([xv, oy, oz    ]),
                    end  =np.array([xv, oy, oz + d]),
                    color=GRID_COLOR, stroke_width=0.4,
                ).set_opacity(0.25))
                self.add(Line3D(
                    start=np.array([ox,     oy, zv]),
                    end  =np.array([ox + w, oy, zv]),
                    color=GRID_COLOR, stroke_width=0.4,
                ).set_opacity(0.25))

        # Ω label — top-right corner of the front-top edge
        omega = MathTex(r"\Omega", color=OMEGA_COLOR,
                        font_size=cfg.omega_label_font_size)
        omega.move_to(corners["FTR"] + np.array([0.35, 0.20, 0]))
        self.add(omega)
        self.omega_label = omega

        # Store geometry for use by children
        self._ox, self._oy, self._oz = ox, oy, oz
        self._w,  self._h,  self._d  = w,  h,  d


# ---------------------------------------------------------------------------
# _EventRegion3D  —  a single event platform
# ---------------------------------------------------------------------------

class _EventRegion3D(VGroup):
    """A raised prism platform representing one event A ⊆ Ω.

    Shape variants
    --------------
    ``"rect"``
        Rectangular footprint.  Requires ``x0``, ``z0``, ``width``, ``depth``.
    ``"ellipse"``
        Elliptical footprint built as a Surface of revolution.
        Requires ``cx``, ``cz``, ``rx`` (x-radius), ``rz`` (z-radius).
    ``"poly"``
        Arbitrary polygon footprint.  Requires ``vertices`` — a list of
        (x, z) pairs in world space.

    All variants sit at y = floor_y + height for their top face.
    """

    def __init__(
        self,
        shape:       str,
        color:       ManimColor,
        cfg:         SampleSpaceConfig,
        floor_y:     float = 0.0,
        height:      float | None = None,
        label:       str   = "",
        prob:        float | None = None,
        # rect params
        x0:    float = 0.0,
        z0:    float = 0.0,
        width: float = 1.0,
        depth: float = 1.0,
        # ellipse params
        cx: float = 0.0,
        cz: float = 0.0,
        rx: float = 1.0,
        rz: float = 1.0,
        # poly params
        vertices: list[tuple[float, float]] | None = None,
    ):
        super().__init__()
        self._cfg    = cfg
        self._color  = color
        self._shape  = shape
        self._label  = label
        self._prob   = prob

        h  = height if height is not None else cfg.event_height
        fy = floor_y

        top_color   = color
        front_color = _dk(color, FACE_DARKEN_SIDE)
        right_color = _dk(color, FACE_DARKEN_RIGHT)

        def _face(pts: list[np.ndarray],
                  col:  ManimColor,
                  opac: float) -> Polygon:
            p = Polygon(*pts, color=col)
            p.set_fill(color=col, opacity=opac)
            p.set_stroke(color=_dk(col, 0.50),
                         width=cfg.event_edge_width,
                         opacity=0.55)
            return p

        # ---- rectangular prism ----
        if shape == "rect":
            x1, z1 = x0 + width, z0 + depth
            TFL = np.array([x0, fy + h, z0])
            TFR = np.array([x1, fy + h, z0])
            TBL = np.array([x0, fy + h, z1])
            TBR = np.array([x1, fy + h, z1])
            AFL = np.array([x0, fy,     z0])
            AFR = np.array([x1, fy,     z0])
            ABL = np.array([x0, fy,     z1])
            ABR = np.array([x1, fy,     z1])

            self.top_face   = _face([TFL, TFR, TBR, TBL], top_color,   cfg.event_opacity)
            self.front_face = _face([AFL, AFR, TFR, TFL], front_color, cfg.event_opacity)
            self.right_face = _face([AFR, ABR, TBR, TFR], right_color, cfg.event_opacity)
            self.add(self.front_face, self.right_face, self.top_face)

            self._top_center = np.array([
                x0 + width / 2, fy + h, z0 + depth / 2
            ])

        # ---- elliptical surface ----
        elif shape == "ellipse":
            def top_surf(u: float, v: float) -> np.ndarray:
                theta = u * TAU
                r_u   = v          # v ∈ [0,1] = radial fraction
                return np.array([
                    cx + rx * r_u * np.cos(theta),
                    fy + h,
                    cz + rz * r_u * np.sin(theta),
                ])

            surf = Surface(
                top_surf,
                u_range=[0, 1],
                v_range=[0, 1],
                resolution=(cfg.ellipse_resolution, cfg.ellipse_resolution // 2),
            )
            surf.set_style(
                fill_color=top_color,
                fill_opacity=cfg.event_opacity,
                stroke_color=_dk(top_color, 0.45),
                stroke_width=cfg.event_edge_width * 0.6,
                stroke_opacity=0.50,
            )
            self.top_face = surf
            self.add(surf)

            # Side wall: thin elliptical ring cylinder
            def side_wall(u: float, v: float) -> np.ndarray:
                theta = u * TAU
                y_v   = fy + v * h
                return np.array([
                    cx + rx * np.cos(theta),
                    y_v,
                    cz + rz * np.sin(theta),
                ])

            wall = Surface(
                side_wall,
                u_range=[0, 1],
                v_range=[0, 1],
                resolution=(cfg.ellipse_resolution, 4),
            )
            wall.set_style(
                fill_color=front_color,
                fill_opacity=cfg.event_opacity * 0.75,
                stroke_width=0,
            )
            self.add(wall)

            self._top_center = np.array([cx, fy + h, cz])

        # ---- polygon footprint ----
        elif shape == "poly":
            if not vertices:
                vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]

            top_pts = [np.array([vx, fy + h, vz]) for vx, vz in vertices]
            bot_pts = [np.array([vx, fy,     vz]) for vx, vz in vertices]

            self.top_face = _face(top_pts, top_color, cfg.event_opacity)
            self.add(self.top_face)

            # Side faces: one per edge of the polygon
            n = len(vertices)
            for i in range(n):
                j = (i + 1) % n
                side = _face(
                    [bot_pts[i], bot_pts[j], top_pts[j], top_pts[i]],
                    front_color if i == 0 else right_color,
                    cfg.event_opacity * 0.75,
                )
                self.add(side)

            cx_avg = np.mean([vx for vx, _ in vertices])
            cz_avg = np.mean([vz for _, vz in vertices])
            self._top_center = np.array([cx_avg, fy + h, cz_avg])

        else:
            raise ValueError(f"Unknown event shape: '{shape}'")

        # ---- event label ----
        self._label_group = VGroup()
        if cfg.show_event_labels and label:
            evt_lbl = MathTex(label, color=_lt(color, 0.28),
                              font_size=cfg.label_font_size)
            evt_lbl.move_to(self._top_center + np.array([0, 0.32, 0]))
            self._label_group.add(evt_lbl)
            self.event_label = evt_lbl

        if cfg.show_prob_labels and prob is not None:
            p_lbl = MathTex(
                rf"P = {prob:.3f}",
                color=_lt(color, 0.15),
                font_size=cfg.prob_label_font_size,
            )
            p_lbl.move_to(self._top_center + np.array([0, 0.60, 0]))
            self._label_group.add(p_lbl)
            self.prob_label = p_lbl

        self.add(self._label_group)

    @property
    def top_center(self) -> np.ndarray:
        return self._top_center.copy()

    @property
    def floor_center(self) -> np.ndarray:
        c = self._top_center.copy()
        c[1] = 0.0
        return c


# ---------------------------------------------------------------------------
# _IntersectionRegion  —  A ∩ B overlay
# ---------------------------------------------------------------------------

class _IntersectionRegion(_EventRegion3D):
    """A ∩ B rendered as a raised platform above both parent regions.

    Inherits _EventRegion3D and adds a MathTex A∩B label + P(A∩B).
    The height is the parent event_height + inter_height_boost so it
    visually "pops" above both A and B.
    """

    def __init__(
        self,
        cfg:     SampleSpaceConfig,
        floor_y: float = 0.0,
        prob:    float | None = None,
        label_a: str = "A",
        label_b: str = "B",
        **region_kwargs,
    ):
        lbl = rf"{label_a} \cap {label_b}"
        super().__init__(
            color   = INTER_AB_COLOR,
            cfg     = cfg,
            floor_y = floor_y,
            height  = cfg.event_height + cfg.inter_height_boost,
            label   = lbl,
            prob    = prob,
            **region_kwargs,
        )


# ---------------------------------------------------------------------------
# _ComplementRegion  —  Ω \ A
# ---------------------------------------------------------------------------

class _ComplementRegion(VGroup):
    """The complement of event A: fills Omega minus A on the floor plane.

    Rendered as a floor-level shaded panel spanning the full Ω footprint,
    overlaid with diagonal hatch lines.  The event region itself is left
    uncovered so the two regions are visually distinct.
    """

    def __init__(
        self,
        box:  _SampleSpaceBox,
        cfg:  SampleSpaceConfig,
        label: str = r"A^c",
        prob:  float | None = None,
    ):
        super().__init__()
        ox, oy, oz = box._ox, box._oy, box._oz
        w,  h,  d  = box._w,  box._h,  box._d
        y          = oy + 0.001   # just above the floor

        # Full Ω footprint quad
        panel = Polygon(
            np.array([ox,     y, oz    ]),
            np.array([ox + w, y, oz    ]),
            np.array([ox + w, y, oz + d]),
            np.array([ox,     y, oz + d]),
            color=COMP_COLOR,
        )
        panel.set_fill(color=COMP_COLOR, opacity=cfg.comp_opacity)
        panel.set_stroke(width=0)
        self.add(panel)

        # Diagonal hatch lines
        n    = cfg.comp_hatch_n
        span = w + d
        for k in range(n):
            t  = (k + 0.5) / n
            # Lines run from (ox + t*w, oz) → (ox, oz + t*d) direction
            xstart = ox + t * span
            zstart = oz
            xend   = ox
            zend   = oz + t * span
            # Clamp to box bounds
            x0c = np.clip(xstart, ox, ox + w)
            z0c = oz if xstart <= ox + w else oz + (xstart - ox - w)
            x1c = np.clip(xend,   ox, ox + w)
            z1c = np.clip(zend,   oz, oz + d)
            if abs(x0c - x1c) < 1e-4 and abs(z0c - z1c) < 1e-4:
                continue
            line = Line3D(
                start=np.array([x0c, y + 0.001, z0c]),
                end  =np.array([x1c, y + 0.001, z1c]),
                color=_lt(COMP_COLOR, 0.25),
                stroke_width=0.45,
            )
            line.set_opacity(cfg.comp_opacity * 1.3)
            self.add(line)

        # Label
        if cfg.show_event_labels:
            lbl = MathTex(label, color=_lt(COMP_COLOR, 0.50),
                          font_size=cfg.label_font_size)
            lbl.move_to(np.array([ox + w * 0.85, y + 0.28, oz + d * 0.15]))
            self.add(lbl)
            if prob is not None and cfg.show_prob_labels:
                p_lbl = MathTex(rf"P = {prob:.3f}",
                                color=_lt(COMP_COLOR, 0.40),
                                font_size=cfg.prob_label_font_size)
                p_lbl.move_to(np.array([ox + w * 0.85, y + 0.52, oz + d * 0.15]))
                self.add(p_lbl)


# ---------------------------------------------------------------------------
# _ProbabilityAxis  —  decorated [0,1] number line
# ---------------------------------------------------------------------------

class _ProbabilityAxis(VGroup):
    """A [0, 1] probability axis with labelled ticks and event markers.

    Notable tick positions: 0, 1/6, 1/4, 1/3, 1/2, 2/3, 3/4, 5/6, 1.
    Event probability markers are vertical line + MathTex label above.
    """

    NOTABLE_TICKS = [
        (0.0,       "0"),
        (1/6,       r"\tfrac{1}{6}"),
        (1/4,       r"\tfrac{1}{4}"),
        (1/3,       r"\tfrac{1}{3}"),
        (1/2,       r"\tfrac{1}{2}"),
        (2/3,       r"\tfrac{2}{3}"),
        (3/4,       r"\tfrac{3}{4}"),
        (5/6,       r"\tfrac{5}{6}"),
        (1.0,       "1"),
    ]

    def __init__(
        self,
        x_center: float,
        y_pos:    float,
        z_pos:    float,
        length:   float,
        cfg:      SampleSpaceConfig,
    ):
        super().__init__()
        x0 = x_center - length / 2
        x1 = x_center + length / 2

        # Main axis line
        axis = Arrow3D(
            start=np.array([x0 - 0.15, y_pos, z_pos]),
            end  =np.array([x1 + 0.15, y_pos, z_pos]),
            color=AXIS_COLOR, stroke_width=1.5,
        )
        self.add(axis)

        # Axis label
        ax_lbl = MathTex(r"P(\cdot)", color=AXIS_COLOR, font_size=22)
        ax_lbl.move_to(np.array([x1 + 0.55, y_pos, z_pos]))
        self.add(ax_lbl)

        # Notable ticks
        for val, tex in self.NOTABLE_TICKS:
            xv   = x0 + val * length
            tick = Line3D(
                start=np.array([xv, y_pos,        z_pos]),
                end  =np.array([xv, y_pos - 0.10, z_pos]),
                color=AXIS_COLOR, stroke_width=0.8,
            )
            self.add(tick)
            lbl = MathTex(tex, color=AXIS_COLOR, font_size=16)
            lbl.move_to(np.array([xv, y_pos - 0.28, z_pos]))
            self.add(lbl)

        self._x0     = x0
        self._length = length
        self._y      = y_pos
        self._z      = z_pos

        # Event marker group (populated by add_event_marker)
        self._markers = VGroup()
        self.add(self._markers)

    def add_event_marker(
        self,
        prob:   float,
        label:  str,
        color:  ManimColor,
        height: float = 0.35,
    ) -> Line3D:
        """Add a vertical marker at P = prob and return it."""
        xv  = self._x0 + np.clip(prob, 0.0, 1.0) * self._length
        m   = Line3D(
            start=np.array([xv, self._y,          self._z]),
            end  =np.array([xv, self._y + height, self._z]),
            color=color, stroke_width=2.2,
        )
        m.set_opacity(0.90)
        lbl = MathTex(label, color=color, font_size=20)
        lbl.move_to(np.array([xv, self._y + height + 0.22, self._z]))
        grp = VGroup(m, lbl)
        self._markers.add(grp)
        return grp


# ---------------------------------------------------------------------------
# _OutcomeGrid  —  discrete sample space of spheres
# ---------------------------------------------------------------------------

class _OutcomeGrid(VGroup):
    """A rectangular grid of spheres, one per outcome, color-coded by event.

    Parameters
    ----------
    outcomes : list[list[str]]
        2D list of outcome labels. outcomes[row][col].
    membership : list[list[str]]
        Parallel 2D list. Each entry is one of:
        ``""`` / ``"none"`` — not in any event
        ``"A"``             — in event A only
        ``"B"``             — in event B only
        ``"AB"``            — in A ∩ B
    center : np.ndarray
        World position of the grid center.
    cfg : SampleSpaceConfig
    floor_y : float
    show_labels : bool
        Show outcome text labels beside each sphere.
    """

    MEMBERSHIP_COLORS = {
        "":     OUTCOME_NONE,
        "none": OUTCOME_NONE,
        "A":    OUTCOME_A,
        "B":    OUTCOME_B,
        "AB":   OUTCOME_AB,
        "C":    EVENT_C_COLOR,
        "AC":   ManimColor("#1A237E"),
        "BC":   ManimColor("#880E4F"),
        "ABC":  INTER_ABC_COLOR,
    }

    def __init__(
        self,
        outcomes:   list[list[str]],
        membership: list[list[str]],
        center:     np.ndarray,
        cfg:        SampleSpaceConfig,
        floor_y:    float = 0.0,
        show_labels: bool = True,
    ):
        super().__init__()
        rows = len(outcomes)
        cols = max(len(r) for r in outcomes)
        sp   = cfg.outcome_spacing
        r    = cfg.outcome_radius

        # Compute grid top-left
        x_start = center[0] - (cols - 1) * sp / 2
        z_start = center[2] - (rows - 1) * sp / 2
        y       = floor_y + r + 0.01

        self._glyphs: list[list[Dot3D | None]] = []

        for row_i, (outcome_row, mem_row) in enumerate(
            zip(outcomes, membership)
        ):
            glyph_row = []
            for col_j, (label, mem) in enumerate(
                zip(outcome_row, mem_row)
            ):
                color = self.MEMBERSHIP_COLORS.get(mem, OUTCOME_NONE)
                pos   = np.array([
                    x_start + col_j * sp,
                    y,
                    z_start + row_i * sp,
                ])
                dot = Dot3D(point=pos, radius=r, color=color)
                dot.set_opacity(cfg.outcome_opacity)
                self.add(dot)
                glyph_row.append(dot)

                if show_labels and label:
                    lbl = Text(label, color=_lt(color, 0.35), font_size=12)
                    lbl.move_to(pos + np.array([0, r + 0.10, 0]))
                    self.add(lbl)

            self._glyphs.append(glyph_row)

    def get_glyph(self, row: int, col: int) -> Dot3D | None:
        """Return the glyph at grid position (row, col)."""
        if 0 <= row < len(self._glyphs) and 0 <= col < len(self._glyphs[row]):
            return self._glyphs[row][col]
        return None


# ---------------------------------------------------------------------------
# _VennZone  —  one atomic region of a Venn diagram
# ---------------------------------------------------------------------------

class _VennZone(_EventRegion3D):
    """One of the atomic regions of a Venn diagram (A only, B only, A∩B, etc.).

    Thin platform prism with a label.  Inherits _EventRegion3D.
    """

    def __init__(
        self,
        shape_kwargs: dict,
        color:        ManimColor,
        cfg:          SampleSpaceConfig,
        label:        str = "",
        prob:         float | None = None,
        floor_y:      float = 0.0,
    ):
        super().__init__(
            color   = color,
            cfg     = cfg,
            floor_y = floor_y,
            height  = cfg.venn_height,
            label   = label,
            prob    = prob,
            **shape_kwargs,
        )


# ---------------------------------------------------------------------------
# Main SampleSpace3D
# ---------------------------------------------------------------------------

class SampleSpace3D(VGroup):
    """A detailed 3D sample space for Manim probability animations.

    Composes the Ω bounding box, event platforms, set operation overlays,
    optional discrete outcome grid, and probability axis.

    Basic usage (continuous events)
    --------------------------------
    >>> from manim import *
    >>> from manim_stats.probability.sample_space import (
    ...     SampleSpace3D, SampleSpaceConfig
    ... )
    >>>
    >>> class MyScene(ThreeDScene):
    ...     def construct(self):
    ...         cfg  = SampleSpaceConfig()
    ...         ss   = SampleSpace3D.two_event_venn(
    ...                    p_a=0.40, p_b=0.35, p_ab=0.15, config=cfg)
    ...         self.set_camera_orientation(phi=55*DEGREES, theta=-45*DEGREES)
    ...         self.play(ss.animate_build_space())
    ...         self.play(ss.animate_venn_build())

    Parameters
    ----------
    config : SampleSpaceConfig, optional
    """

    def __init__(self, config: SampleSpaceConfig | None = None):
        super().__init__()
        self.cfg   = config or SampleSpaceConfig()
        self._box  = _SampleSpaceBox(self.cfg)
        self.add(self._box)

        # Public sub-group references (populated by add_* methods)
        self.event_regions: list[_EventRegion3D]      = []
        self._event_group   = VGroup()
        self._overlay_group = VGroup()
        self._grid_group    = VGroup()
        self._axis_group    = VGroup()
        self.add(self._event_group, self._overlay_group,
                 self._grid_group, self._axis_group)

    # ------------------------------------------------------------------
    # Builder methods (add objects incrementally)
    # ------------------------------------------------------------------

    def add_rect_event(
        self,
        x0:    float,
        z0:    float,
        width: float,
        depth: float,
        color: ManimColor = EVENT_A_COLOR,
        label: str = "A",
        prob:  float | None = None,
    ) -> _EventRegion3D:
        """Add a rectangular event region and return it."""
        ev = _EventRegion3D(
            shape="rect", color=color, cfg=self.cfg,
            floor_y=self.cfg.origin[1],
            label=label, prob=prob,
            x0=x0, z0=z0, width=width, depth=depth,
        )
        self.event_regions.append(ev)
        self._event_group.add(ev)
        return ev

    def add_ellipse_event(
        self,
        cx:    float,
        cz:    float,
        rx:    float,
        rz:    float,
        color: ManimColor = EVENT_A_COLOR,
        label: str = "A",
        prob:  float | None = None,
    ) -> _EventRegion3D:
        """Add an elliptical event region and return it."""
        ev = _EventRegion3D(
            shape="ellipse", color=color, cfg=self.cfg,
            floor_y=self.cfg.origin[1],
            label=label, prob=prob,
            cx=cx, cz=cz, rx=rx, rz=rz,
        )
        self.event_regions.append(ev)
        self._event_group.add(ev)
        return ev

    def add_intersection(
        self,
        shape_kwargs: dict,
        prob:         float | None = None,
        label_a:      str = "A",
        label_b:      str = "B",
    ) -> _IntersectionRegion:
        """Add an intersection overlay and return it."""
        inter = _IntersectionRegion(
            cfg=self.cfg,
            floor_y=self.cfg.origin[1],
            prob=prob,
            label_a=label_a,
            label_b=label_b,
            **shape_kwargs,
        )
        self._overlay_group.add(inter)
        return inter

    def add_complement(
        self,
        label: str = r"A^c",
        prob:  float | None = None,
    ) -> _ComplementRegion:
        """Add a complement (Omega minus A) overlay and return it."""
        comp = _ComplementRegion(
            box=self._box, cfg=self.cfg,
            label=label, prob=prob,
        )
        self._overlay_group.add(comp)
        return comp

    def add_outcome_grid(
        self,
        outcomes:    list[list[str]],
        membership:  list[list[str]],
        center:      np.ndarray | None = None,
        show_labels: bool = True,
    ) -> _OutcomeGrid:
        """Add a discrete outcome grid and return it."""
        if center is None:
            ox, oy, oz = self.cfg.origin
            center = np.array([
                ox + self.cfg.box_width / 2,
                oy,
                oz + self.cfg.box_depth / 2,
            ])
        grid = _OutcomeGrid(
            outcomes=outcomes,
            membership=membership,
            center=center,
            cfg=self.cfg,
            floor_y=self.cfg.origin[1],
            show_labels=show_labels,
        )
        self._grid_group.add(grid)
        return grid

    def add_probability_axis(self) -> _ProbabilityAxis:
        """Add the [0,1] probability axis along the front edge and return it."""
        ox, oy, oz = self.cfg.origin
        axis = _ProbabilityAxis(
            x_center=ox + self.cfg.box_width / 2,
            y_pos   =oy + self.cfg.axis_y_offset,
            z_pos   =oz + self.cfg.axis_z_offset - 0.50,
            length  =self.cfg.axis_length,
            cfg     =self.cfg,
        )
        self._axis_group.add(axis)
        self.prob_axis = axis
        return axis

    # ------------------------------------------------------------------
    # Animation API
    # ------------------------------------------------------------------

    def animate_build_space(
        self,
        lag_ratio: float = 0.06,
        run_time:  float = 2.0,
    ) -> LaggedStart:
        """Grow the Ω box boundary edge by edge from the front-left corner.

        The 12 edges trace out in a logical order: bottom square →
        top square → vertical pillars, giving a satisfying "box
        materialising" effect::

            self.play(ss.animate_build_space())
        """
        return LaggedStart(
            *[Create(edge, run_time=run_time * 0.45)
              for edge in self._box.edges],
            lag_ratio=lag_ratio,
            run_time=run_time,
        )

    def animate_add_event(
        self,
        event:    _EventRegion3D,
        run_time: float = 1.2,
    ) -> GrowFromPoint:
        """Raise an event platform from the floor.

        The platform rises upward from its floor footprint centre.
        """
        return GrowFromPoint(
            event,
            point=event.floor_center,
            run_time=run_time,
        )

    def animate_intersection(
        self,
        inter:    _IntersectionRegion,
        run_time: float = 1.0,
    ) -> GrowFromPoint:
        """Materialise the intersection region above the parent events."""
        return GrowFromPoint(
            inter,
            point=inter.floor_center,
            run_time=run_time,
        )

    def animate_complement(
        self,
        comp:     _ComplementRegion,
        run_time: float = 1.2,
    ) -> FadeIn:
        """Sweep the complement shading across Omega minus A."""
        return FadeIn(comp, run_time=run_time)

    def animate_show_union(
        self,
        event_a:  _EventRegion3D,
        event_b:  _EventRegion3D,
        union:    _EventRegion3D,
        run_time: float = 1.4,
    ) -> Succession:
        """Build the union region: briefly highlight A and B then fade in union."""
        pulse_a  = event_a.animate(run_time=run_time * 0.20).scale(1.04)
        pulse_b  = event_b.animate(run_time=run_time * 0.20).scale(1.04)
        restore  = AnimationGroup(
            event_a.animate(run_time=run_time * 0.15).scale(1 / 1.04),
            event_b.animate(run_time=run_time * 0.15).scale(1 / 1.04),
        )
        show     = FadeIn(union, run_time=run_time * 0.45)
        return Succession(
            AnimationGroup(pulse_a, pulse_b),
            restore,
            show,
        )

    def animate_sweep_probability(
        self,
        p:        float,
        run_time: float = 1.8,
    ) -> Create:
        """Sweep a translucent vertical plane from x=0 to x=P·box_width.

        Visualises the "probability as area fraction" interpretation.
        The plane is created here and added to the scene.
        """
        ox, oy, oz = self.cfg.origin
        target_x   = ox + np.clip(p, 0.0, 1.0) * self.cfg.box_width
        h          = self.cfg.box_height
        d          = self.cfg.box_depth

        sweep_plane = Polygon(
            np.array([target_x, oy,     oz    ]),
            np.array([target_x, oy + h, oz    ]),
            np.array([target_x, oy + h, oz + d]),
            np.array([target_x, oy,     oz + d]),
            color=YELLOW,
        )
        sweep_plane.set_fill(color=YELLOW, opacity=0.12)
        sweep_plane.set_stroke(color=YELLOW, width=1.5, opacity=0.70)
        self.add(sweep_plane)

        lbl = MathTex(rf"P = {p:.3f}", color=YELLOW, font_size=22)
        lbl.move_to(np.array([target_x, oy + h + 0.30, oz + d / 2]))
        self.add(lbl)

        return Create(sweep_plane, run_time=run_time)

    def animate_highlight_outcome(
        self,
        grid:     _OutcomeGrid,
        row:      int,
        col:      int,
        highlight_color: ManimColor = YELLOW,
        scale_factor:    float      = 2.0,
        run_time:        float      = 0.5,
    ) -> Succession:
        """Flash and scale one outcome sphere in a discrete grid."""
        glyph = grid.get_glyph(row, col)
        if glyph is None:
            return Succession(FadeIn(VGroup(), run_time=0.1))
        orig_color = glyph.get_color()
        return Succession(
            AnimationGroup(
                glyph.animate(run_time=run_time / 2)
                     .scale(scale_factor)
                     .set_color(highlight_color),
            ),
            AnimationGroup(
                glyph.animate(run_time=run_time / 2)
                     .scale(1 / scale_factor)
                     .set_color(orig_color),
            ),
        )

    def animate_venn_build(
        self,
        lag_ratio: float = 0.25,
        run_time:  float = 3.0,
    ) -> LaggedStart:
        """Grow event regions then labels: platforms rise, labels fade in.

        Designed for use with the Venn factory classmethods where
        multiple _EventRegion3D objects have been added.
        """
        region_anims = [
            GrowFromPoint(ev, point=ev.floor_center,
                          run_time=run_time * 0.55)
            for ev in self.event_regions
        ]
        label_anims = [
            FadeIn(ev._label_group, run_time=run_time * 0.30)
            for ev in self.event_regions
        ]
        return LaggedStart(
            *region_anims,
            *label_anims,
            lag_ratio=lag_ratio,
            run_time=run_time,
        )

    def animate_conditional(
        self,
        event_a:   _EventRegion3D,
        event_b:   _EventRegion3D,
        inter:     _IntersectionRegion,
        run_time:  float = 2.5,
    ) -> Succession:
        """Visualise P(B|A): dim Ω minus A, then highlight B∩A fraction.

        Step 1 — dim everything outside A (set opacity of B and box to low)
        Step 2 — grow the intersection region above A
        Step 3 — show a P(B|A) annotation above the intersection

        Parameters
        ----------
        event_a, event_b : _EventRegion3D
        inter : _IntersectionRegion
            The A ∩ B overlay.
        """
        dim_outside = AnimationGroup(
            self._box.animate(run_time=run_time * 0.30)
                .set_opacity(0.18),
            event_b.animate(run_time=run_time * 0.30)
                .set_opacity(0.15),
        )
        grow_inter = GrowFromPoint(
            inter, point=inter.floor_center, run_time=run_time * 0.40
        )
        if inter._prob is not None:
            p_ab = inter._prob
            p_a  = event_a._prob or 1e-9
            cond = p_ab / p_a
            cond_lbl = MathTex(
                rf"P(B|A) = \frac{{P(A \cap B)}}{{P(A)}} = "
                rf"\frac{{{p_ab:.3f}}}{{{p_a:.3f}}} = {cond:.3f}",
                color=INTER_AB_COLOR,
                font_size=self.cfg.label_font_size,
            )
            cond_lbl.move_to(inter.top_center + np.array([0, 0.55, 0]))
            self.add(cond_lbl)
            show_label = FadeIn(cond_lbl, run_time=run_time * 0.30)
        else:
            show_label = FadeIn(VGroup(), run_time=0.1)

        return Succession(dim_outside, grow_inter, show_label)

    def animate_show_operation(
        self,
        operation:       str,
        event_a:         _EventRegion3D,
        event_b:         _EventRegion3D | None = None,
        result_region:   _EventRegion3D | None = None,
        run_time:        float = 1.5,
    ) -> AnimationGroup | Succession:
        """Morph the scene to highlight a set operation.

        Parameters
        ----------
        operation : str
            One of ``"intersection"``, ``"union"``, ``"complement"``,
            ``"difference"`` (A \\ B).
        event_a, event_b : _EventRegion3D
        result_region : _EventRegion3D | None
            Pre-built result region to reveal; if None, just dims.
        """
        anims = []

        if operation == "complement":
            # Dim A, brighten everything else
            anims.append(event_a.animate(run_time=run_time)
                                 .set_opacity(0.15))
            if result_region:
                anims.append(FadeIn(result_region, run_time=run_time))

        elif operation == "intersection":
            # Dim non-overlapping parts
            if event_b:
                anims.append(
                    AnimationGroup(
                        event_a.animate(run_time=run_time * 0.5)
                               .set_opacity(0.25),
                        event_b.animate(run_time=run_time * 0.5)
                               .set_opacity(0.25),
                    )
                )
            if result_region:
                anims.append(
                    GrowFromPoint(result_region,
                                  point=result_region.floor_center,
                                  run_time=run_time * 0.5)
                )

        elif operation == "union":
            if result_region:
                anims.append(FadeIn(result_region, run_time=run_time))

        elif operation == "difference":
            # A \ B: dim B portion
            if event_b:
                anims.append(event_b.animate(run_time=run_time).set_opacity(0.12))

        return AnimationGroup(*anims) if len(anims) != 1 else anims[0]

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def two_event_venn(
        cls,
        p_a:    float,
        p_b:    float,
        p_ab:   float,
        config: SampleSpaceConfig | None = None,
        label_a: str = "A",
        label_b: str = "B",
    ) -> "SampleSpace3D":
        """Classic 2-circle Venn diagram inside the Ω box.

        Lays out three event regions:
          - A only  (left ellipse, minus overlap)
          - B only  (right ellipse, minus overlap)
          - A ∩ B   (central overlap, raised higher)

        The "A only" and "B only" regions use rectangular footprints
        for simplicity; the intersection uses the same footprint
        centred between them.

        Parameters
        ----------
        p_a, p_b : float
            Marginal probabilities P(A) and P(B).
        p_ab : float
            Joint probability P(A ∩ B).  Must satisfy p_ab ≤ min(p_a, p_b).
        """
        cfg  = config or SampleSpaceConfig()
        ss   = cls(config=cfg)
        ox, oy, oz = cfg.origin
        w,  d      = cfg.box_width, cfg.box_depth
        r          = cfg.venn_circle_radius

        cx_a = ox + w * 0.35
        cx_b = ox + w * 0.65
        cz   = oz + d * 0.50

        # A-only ellipse (left)
        ev_a = ss.add_ellipse_event(
            cx=cx_a, cz=cz, rx=r, rz=r * 0.85,
            color=EVENT_A_COLOR,
            label=label_a, prob=p_a,
        )
        # B-only ellipse (right)
        ev_b = ss.add_ellipse_event(
            cx=cx_b, cz=cz, rx=r, rz=r * 0.85,
            color=EVENT_B_COLOR,
            label=label_b, prob=p_b,
        )
        # A ∩ B intersection — centred between A and B
        cx_ab = (cx_a + cx_b) / 2
        inter = ss.add_intersection(
            shape_kwargs=dict(
                shape="ellipse",
                cx=cx_ab, cz=cz,
                rx=(cx_b - cx_a) * 0.28,
                rz=r * 0.55,
            ),
            prob=p_ab,
            label_a=label_a,
            label_b=label_b,
        )
        ss._inter_ab = inter

        # Probability axis
        if cfg.show_prob_axis:
            axis = ss.add_probability_axis()
            axis.add_event_marker(p_a,  rf"P({label_a})",              EVENT_A_COLOR)
            axis.add_event_marker(p_b,  rf"P({label_b})",              EVENT_B_COLOR)
            axis.add_event_marker(p_ab, rf"P({label_a}\cap{label_b})", INTER_AB_COLOR)
            p_aub = p_a + p_b - p_ab
            axis.add_event_marker(p_aub, rf"P({label_a}\cup{label_b})", UNION_COLOR)

        return ss

    @classmethod
    def three_event_venn(
        cls,
        p_a:   float,
        p_b:   float,
        p_c:   float,
        p_ab:  float,
        p_ac:  float,
        p_bc:  float,
        p_abc: float,
        config: SampleSpaceConfig | None = None,
    ) -> "SampleSpace3D":
        """3-circle Venn diagram with all 7 atomic zones.

        Zones rendered as elliptical platforms at their approximate
        geometric centroid positions.  Heights are stacked:
            single-event   → event_height
            two-event inter → event_height + inter_height_boost
            triple inter   → event_height + 2×inter_height_boost

        Parameters
        ----------
        p_a, p_b, p_c : float
            Marginal probabilities.
        p_ab, p_ac, p_bc : float
            Pairwise joint probabilities.
        p_abc : float
            Triple joint probability P(A ∩ B ∩ C).
        """
        cfg = config or SampleSpaceConfig()
        ss  = cls(config=cfg)
        ox, oy, oz = cfg.origin
        w,  d      = cfg.box_width, cfg.box_depth
        r          = cfg.venn_circle_radius * 0.85

        # Centres of the three circles (equilateral triangle)
        tri_r  = r * 0.75
        angles = [90, 90 + 120, 90 + 240]
        cx_mid = ox + w / 2
        cz_mid = oz + d / 2

        centres = [
            np.array([
                cx_mid + tri_r * np.cos(np.radians(a)),
                oy,
                cz_mid - tri_r * np.sin(np.radians(a)),
            ])
            for a in angles
        ]

        colors  = [EVENT_A_COLOR, EVENT_B_COLOR, EVENT_C_COLOR]
        labels  = ["A", "B", "C"]
        probs   = [p_a, p_b, p_c]

        for i, (center, color, label, prob) in enumerate(
            zip(centres, colors, labels, probs)
        ):
            ev = ss.add_ellipse_event(
                cx=center[0], cz=center[2],
                rx=r, rz=r * 0.90,
                color=color, label=label, prob=prob,
            )

        # Pairwise intersections — midpoints between circle centres
        inter_pairs = [
            (0, 1, INTER_AB_COLOR, "A∩B", p_ab),
            (0, 2, ManimColor("#00695C"), "A∩C", p_ac),
            (1, 2, ManimColor("#880E4F"), "B∩C", p_bc),
        ]
        h_boost = cfg.inter_height_boost

        for i, j, color, lbl, prob in inter_pairs:
            mid = (centres[i] + centres[j]) / 2
            ss.add_intersection(
                shape_kwargs=dict(
                    shape="ellipse",
                    cx=mid[0], cz=mid[2],
                    rx=r * 0.35, rz=r * 0.30,
                ),
                prob=prob,
                label_a=labels[i],
                label_b=labels[j],
            )

        # Triple intersection at centroid
        triple_centre = np.mean(centres, axis=0)
        triple = _EventRegion3D(
            shape="ellipse",
            color=INTER_ABC_COLOR,
            cfg=cfg,
            floor_y=oy,
            height=cfg.event_height + 2 * h_boost,
            label=r"A\cap B\cap C",
            prob=p_abc,
            cx=triple_centre[0],
            cz=triple_centre[2],
            rx=r * 0.20,
            rz=r * 0.18,
        )
        ss._overlay_group.add(triple)

        return ss

    @classmethod
    def from_dice(
        cls,
        event_a:    set[int] | None = None,
        event_b:    set[int] | None = None,
        config:     SampleSpaceConfig | None = None,
    ) -> "SampleSpace3D":
        """Discrete sample space Ω = {1, 2, 3, 4, 5, 6}.

        Outcomes are arranged in a 1×6 grid of spheres, color-coded
        by event membership.

        Parameters
        ----------
        event_a : set[int], optional
            Dice outcomes in event A.  E.g. ``{2, 4, 6}`` for even numbers.
        event_b : set[int], optional
            Dice outcomes in event B.  E.g. ``{1, 2, 3}`` for ≤ 3.
        """
        cfg      = config or SampleSpaceConfig()
        ss       = cls(config=cfg)
        event_a  = event_a or set()
        event_b  = event_b or set()

        outcomes   = [[str(k) for k in range(1, 7)]]
        membership = [[
            "AB" if k in event_a and k in event_b else
            "A"  if k in event_a else
            "B"  if k in event_b else
            ""
            for k in range(1, 7)
        ]]

        ox, oy, oz = cfg.origin
        centre = np.array([ox + cfg.box_width / 2, oy, oz + cfg.box_depth / 2])
        ss.add_outcome_grid(outcomes, membership, center=centre)

        # Probability annotations
        n = 6
        p_a  = len(event_a) / n
        p_b  = len(event_b) / n
        p_ab = len(event_a & event_b) / n
        if cfg.show_prob_axis:
            axis = ss.add_probability_axis()
            if p_a  > 0: axis.add_event_marker(p_a,  "P(A)", EVENT_A_COLOR)
            if p_b  > 0: axis.add_event_marker(p_b,  "P(B)", EVENT_B_COLOR)
            if p_ab > 0: axis.add_event_marker(p_ab, r"P(A\cap B)", INTER_AB_COLOR)

        return ss

    @classmethod
    def from_cards(
        cls,
        event_a:  set[str] | None = None,
        event_b:  set[str] | None = None,
        config:   SampleSpaceConfig | None = None,
    ) -> "SampleSpace3D":
        """Discrete sample space: full 52-card deck in a 4×13 grid.

        Parameters
        ----------
        event_a : set[str], optional
            Card identifiers in event A.  Format: "A♠", "K♥", "2♣", etc.
        event_b : set[str], optional
            Card identifiers in event B.
        """
        cfg     = config or SampleSpaceConfig()
        ss      = cls(config=cfg)
        event_a = event_a or set()
        event_b = event_b or set()

        suits  = ["♠", "♥", "♦", "♣"]
        values = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]

        outcomes   = [[f"{v}{s}" for v in values] for s in suits]
        membership = [[
            "AB" if f"{v}{s}" in event_a and f"{v}{s}" in event_b else
            "A"  if f"{v}{s}" in event_a else
            "B"  if f"{v}{s}" in event_b else
            ""
            for v in values
        ] for s in suits]

        ox, oy, oz = cfg.origin
        centre = np.array([ox + cfg.box_width / 2, oy, oz + cfg.box_depth / 2])
        ss.add_outcome_grid(outcomes, membership, center=centre, show_labels=False)

        return ss

    @classmethod
    def conditional_highlight(
        cls,
        p_a:    float,
        p_b:    float,
        p_ab:   float,
        config: SampleSpaceConfig | None = None,
    ) -> "SampleSpace3D":
        """Pre-built scene for visualising P(B|A).

        Lays out a 2-event Venn with A and B, ready for
        ``animate_conditional()`` to be called::

            ss = SampleSpace3D.conditional_highlight(0.4, 0.35, 0.15)
            self.play(ss.animate_build_space())
            self.play(ss.animate_venn_build())
            self.play(ss.animate_conditional(
                ss.event_regions[0],
                ss.event_regions[1],
                ss._inter_ab,
            ))
        """
        base = cls.two_event_venn(p_a, p_b, p_ab, config=config)
        return base