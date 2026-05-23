"""
manim_stats/probability/bayes.py
==================================
Production-quality Bayesian reasoning visualizations for Manim.

Objects
-------

BayesBox3D
    The classic 2×2 Bayes table rendered as a 3D mosaic of prisms.
    Cell footprint area (width × depth) encodes joint probability P(H∩E).
    Cell height encodes the likelihood ratio for that cell.
    Four labeled partitions: TP, FP, FN, TN — each a shaded prism.
    Axes labeled with P(H)/P(¬H) and P(E|H)/P(E|¬H).
    Posterior P(H|E) displayed as an annotated bracket along one edge.

PriorPosteriorBar3D
    A vertical probability bar split into two color bands (H vs ¬H).
    Displays the prior split, then morphs to the posterior after evidence
    via animate_update().  The "squeeze/expand" motion makes the
    Bayesian shift viscerally visible.

LikelihoodPanel3D
    Side-by-side pair of bars: P(E|H) and P(E|¬H).
    A ratio bracket above them shows LR = P(E|H) / P(E|¬H) with color
    coding — green for LR > 1 (evidence supports H), red for LR < 1.

NaturalFrequencyTree3D
    Tree where each branch is a 3D prism whose cross-sectional area
    encodes the absolute count of people/items in that branch.
    Root: population N.  Level 1: N·P(H) with H vs ¬H branches.
    Level 2: further splits by P(E|H) and P(E|¬H).
    Leaf nodes show icon-array style dot grids.

SequentialBayesUpdater3D
    A horizontal probability axis [0, 1] with a marker at the current
    prior.  Each piece of evidence shifts the marker left or right via
    animate_add_evidence().  A trail of ghost markers records the history.
    The Bayes factor for each update is displayed above the axis.

BayesFormulaBanner
    Floating MathTex rendering of Bayes' theorem with each term in a
    distinct color matching the visual objects in the scene.
    animate_highlight_term() pulses one term at a time.

BayesConfig
    Dataclass controlling all visual parameters.

Animation suite
---------------
All objects expose named animate_* methods returning Manim Animations
suitable for self.play(...).  Key animations:

  BayesBox3D
    .animate_build_box()        – cells rise from floor one by one
    .animate_highlight_cell()   – flash one partition prism
    .animate_reveal_posterior() – bracket and label slide into place
    .animate_sweep_evidence()   – color-wave across the E+ column

  PriorPosteriorBar3D
    .animate_build()            – bar grows upward from floor
    .animate_update(p_h, evidence) – morph split to posterior

  LikelihoodPanel3D
    .animate_build()            – both bars grow simultaneously
    .animate_highlight_lr()     – pulse the LR bracket

  NaturalFrequencyTree3D
    .animate_grow_tree()        – branches extend level by level
    .animate_drop_icons()       – icon dots rain into leaf nodes

  SequentialBayesUpdater3D
    .animate_axis()             – axis and prior marker appear
    .animate_add_evidence(lr)   – update marker with Bayes factor

  BayesFormulaBanner
    .animate_appear()           – formula writes itself
    .animate_highlight_term(t)  – pulse term t (0–4)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import stats as scipy_stats

from manim import (
    WHITE, BLACK, GRAY, GRAY_A, GRAY_B, GRAY_C, GRAY_D, GRAY_E,
    BLUE, BLUE_E, RED, RED_E, GREEN, GREEN_E, YELLOW, ORANGE, PURPLE,
    UP, DOWN, LEFT, RIGHT,
    PI, TAU,
    VGroup, VMobject,
    Line3D, Polygon, Sphere, Dot3D, Arrow3D,
    Text, MathTex,
    FadeIn, FadeOut, GrowFromPoint, GrowFromCenter, Transform,
    Create, Write, AnimationGroup, Succession, LaggedStart,
    interpolate_color,
    ManimColor,
    DEGREES,
)


# ---------------------------------------------------------------------------
# Color semantics for Bayesian reasoning
# ---------------------------------------------------------------------------

# Hypothesis colors
H_TRUE_COLOR    = ManimColor("#1565C0")   # blue    – H is true
H_FALSE_COLOR   = ManimColor("#B71C1C")   # red     – H is false

# Evidence colors
E_POS_COLOR     = ManimColor("#2E7D32")   # green   – E is present / test +
E_NEG_COLOR     = ManimColor("#4E342E")   # brown   – E is absent  / test −

# Cell (joint probability) colors — one per quadrant
TP_COLOR        = ManimColor("#1976D2")   # blue    – H true  & E+ (true positive)
FP_COLOR        = ManimColor("#EF5350")   # red     – H false & E+ (false positive)
FN_COLOR        = ManimColor("#64B5F6")   # lt blue – H true  & E− (false negative)
TN_COLOR        = ManimColor("#EF9A9A")   # lt red  – H false & E− (true negative)

# Likelihood ratio
LR_POS_COLOR    = ManimColor("#00E676")   # green   – LR > 1 (evidence supports H)
LR_NEG_COLOR    = ManimColor("#FF5252")   # red     – LR < 1 (evidence against H)
LR_NULL_COLOR   = GRAY_C                  # grey    – LR ≈ 1 (no information)

# Posterior update
PRIOR_COLOR     = ManimColor("#78909C")   # slate
POSTERIOR_COLOR = ManimColor("#FFD600")   # yellow

# Formula term colors (matching visual objects)
TERM_COLORS = [
    ManimColor("#FFD600"),   # P(H|E)   – posterior
    ManimColor("#1976D2"),   # P(E|H)   – likelihood
    ManimColor("#1565C0"),   # P(H)     – prior
    ManimColor("#FF5252"),   # P(E|¬H)  – false positive rate
    ManimColor("#B71C1C"),   # P(¬H)    – complement prior
]

FACE_DARKEN_SIDE  = 0.38
FACE_DARKEN_RIGHT = 0.55


def _dk(c: ManimColor, f: float) -> ManimColor:
    return interpolate_color(c, BLACK, f)

def _lt(c: ManimColor, f: float) -> ManimColor:
    return interpolate_color(c, WHITE, f)

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(v, lo, hi))


# ---------------------------------------------------------------------------
# Core Bayesian math helpers
# ---------------------------------------------------------------------------

def bayes_update(prior: float, lr: float) -> float:
    """Bayesian update via likelihood ratio on probability (not odds).

    posterior = (prior * lr) / (prior * lr + (1 − prior))

    Parameters
    ----------
    prior : float
        P(H) before evidence.
    lr : float
        Likelihood ratio P(E|H) / P(E|¬H).

    Returns
    -------
    float
        Posterior P(H|E).
    """
    prior    = _clamp(prior, 1e-9, 1 - 1e-9)
    odds_pre = prior / (1 - prior)
    odds_post = odds_pre * lr
    return odds_post / (1 + odds_post)


def compute_joint_probs(
    p_h:    float,
    p_e_h:  float,
    p_e_nh: float,
) -> dict[str, float]:
    """Compute the four joint probabilities for a 2×2 Bayes table.

    Returns
    -------
    dict with keys "tp", "fp", "fn", "tn" and "p_e", "posterior"
    """
    p_nh   = 1 - p_h
    tp     = p_h  * p_e_h           # P(H ∩ E+)
    fn     = p_h  * (1 - p_e_h)     # P(H ∩ E−)
    fp     = p_nh * p_e_nh          # P(¬H ∩ E+)
    tn     = p_nh * (1 - p_e_nh)    # P(¬H ∩ E−)
    p_e    = tp + fp                 # P(E+) = total positive rate
    post   = tp / p_e if p_e > 1e-12 else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                p_e=p_e, posterior=post,
                p_h=p_h, p_nh=p_nh,
                p_e_h=p_e_h, p_e_nh=p_e_nh,
                lr=p_e_h / p_e_nh if p_e_nh > 1e-12 else float("inf"))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BayesConfig:
    """Visual configuration for all Bayesian objects in this module.

    BayesBox3D
    ----------
    box_scale_x : float
        Total width of the Bayes box (maps P(H) axis) in Manim units.
    box_scale_z : float
        Total depth of the Bayes box (maps P(E|·) axis) in Manim units.
    box_max_height : float
        Maximum prism height (cell with largest LR gets this height).
        Set to 0 for a flat mosaic (no height encoding).
    box_gap : float
        Gap between adjacent prisms (visual separation).
    show_cell_labels : bool
        Render probability annotations inside each cell.
    show_axis_labels : bool
        Render P(H)/P(¬H) and P(E|H)/P(E|¬H) axis labels.
    show_posterior_bracket : bool
        Render the posterior bracket + annotation on the E+ row.
    cell_label_font_size : int
        Font size for cell value annotations.
    axis_label_font_size : int
        Font size for axis labels.

    PriorPosteriorBar3D
    -------------------
    bar_width : float
        Width of the prior/posterior bar prism.
    bar_depth : float
        Depth (Z) of the bar prism.
    bar_max_height : float
        Full height of the bar (represents probability = 1).
    show_probability_labels : bool
        Show floating P(H) / P(¬H) labels on each band.

    LikelihoodPanel3D
    -----------------
    lr_bar_width : float
        Width of each likelihood bar.
    lr_bar_spacing : float
        Gap between the two likelihood bars.
    lr_max_height : float
        Height corresponding to probability = 1.
    show_lr_bracket : bool
        Render the LR ratio bracket above the two bars.

    NaturalFrequencyTree3D
    ----------------------
    tree_population : int
        Total population N at the root.
    tree_level_spacing : float
        Vertical distance between tree levels.
    tree_branch_gap : float
        Gap between sibling branches at the same level.
    tree_max_width : float
        Maximum width of the root node.
    show_icon_arrays : bool
        Draw dot grids in leaf nodes.
    icon_radius : float
        Radius of each icon dot.

    SequentialBayesUpdater3D
    ------------------------
    axis_length : float
        Length of the [0, 1] probability axis.
    marker_radius : float
        Radius of the prior/posterior marker sphere.
    ghost_opacity : float
        Opacity of historical ghost markers.
    show_bayes_factor : bool
        Display the LR above the axis at each update.

    Shared
    ------
    show_formula_banner : bool
        Show the BayesFormulaBanner floating above the main object.
    font_size : int
        Default font size for labels.
    axes_color : ManimColor
        Color for axis lines and ticks.
    """

    # ---- BayesBox3D ----
    box_scale_x:            float = 6.0
    box_scale_z:            float = 4.0
    box_max_height:         float = 1.20
    box_gap:                float = 0.06
    show_cell_labels:       bool  = True
    show_axis_labels:       bool  = True
    show_posterior_bracket: bool  = True
    cell_label_font_size:   int   = 20
    axis_label_font_size:   int   = 22

    # ---- PriorPosteriorBar3D ----
    bar_width:                float = 0.80
    bar_depth:                float = 0.40
    bar_max_height:           float = 4.50
    show_probability_labels:  bool  = True

    # ---- LikelihoodPanel3D ----
    lr_bar_width:    float = 0.70
    lr_bar_spacing:  float = 0.55
    lr_max_height:   float = 3.50
    show_lr_bracket: bool  = True

    # ---- NaturalFrequencyTree3D ----
    tree_population:    int   = 1000
    tree_level_spacing: float = 2.20
    tree_branch_gap:    float = 0.30
    tree_max_width:     float = 5.00
    show_icon_arrays:   bool  = True
    icon_radius:        float = 0.045

    # ---- SequentialBayesUpdater3D ----
    axis_length:       float = 7.00
    marker_radius:     float = 0.14
    ghost_opacity:     float = 0.28
    show_bayes_factor: bool  = True

    # ---- Shared ----
    show_formula_banner: bool       = True
    font_size:           int        = 22
    axes_color:          ManimColor = GRAY_B


# ---------------------------------------------------------------------------
# _ShadeBox  —  shared 3-face shaded prism (reused across all objects)
# ---------------------------------------------------------------------------

class _ShadeBox(VGroup):
    """A 3D prism with physically-shaded top / front / right faces.

    Coordinate layout:
        x : [x0, x0 + width]
        y : [0,  height    ]
        z : [z0, z0 + depth]
    """

    def __init__(
        self,
        x0:        float,
        z0:        float,
        width:     float,
        depth:     float,
        height:    float,
        top_color: ManimColor,
        opacity:   float = 0.90,
        edge_w:    float = 0.6,
        edge_op:   float = 0.40,
    ):
        super().__init__()
        h = max(height, 0.004)
        s_w, s_d = width, depth

        front_color = _dk(top_color, FACE_DARKEN_SIDE)
        right_color = _dk(top_color, FACE_DARKEN_RIGHT)

        # 8 corners
        AFL = np.array([x0,        0, z0       ])
        AFR = np.array([x0 + s_w,  0, z0       ])
        ABL = np.array([x0,        0, z0 + s_d ])
        ABR = np.array([x0 + s_w,  0, z0 + s_d ])
        TFL = np.array([x0,        h, z0       ])
        TFR = np.array([x0 + s_w,  h, z0       ])
        TBL = np.array([x0,        h, z0 + s_d ])
        TBR = np.array([x0 + s_w,  h, z0 + s_d ])

        def _face(pts, color):
            p = Polygon(*pts, color=color)
            p.set_fill(color=color, opacity=opacity)
            p.set_stroke(color=_dk(color, 0.50), width=edge_w, opacity=edge_op)
            return p

        self.top_face   = _face([TFL, TFR, TBR, TBL], top_color)
        self.front_face = _face([AFL, AFR, TFR, TFL], front_color)
        self.right_face = _face([AFR, ABR, TBR, TFR], right_color)
        self.add(self.front_face, self.right_face, self.top_face)

        self._top_center = np.array([
            x0 + width  / 2,
            h,
            z0 + depth  / 2,
        ])

    @property
    def top_center(self) -> np.ndarray:
        return self._top_center.copy()

    @property
    def floor_center(self) -> np.ndarray:
        c = self._top_center.copy()
        c[1] = 0.0
        return c


# ---------------------------------------------------------------------------
# _PosteriorBracket  —  annotated bracket on the E+ edge of the Bayes box
# ---------------------------------------------------------------------------

class _PosteriorBracket(VGroup):
    """A vertical bracket along the front edge of the E+ column,
    labeled with the posterior probability P(H|E+)."""

    def __init__(
        self,
        x0:       float,   # start of the TP cell
        x1:       float,   # end of the TP cell (= start of FP cell at x1)
        x2:       float,   # full width (x0 + total box width)
        y_height: float,   # height to draw the bracket
        z_pos:    float,
        posterior: float,
        font_size: int,
    ):
        super().__init__()

        # Vertical ticks at x0, x1, x2
        for xv, label, color in [
            (x0, "",          POSTERIOR_COLOR),
            (x1, f"P(H|E⁺)\n= {posterior:.3f}", POSTERIOR_COLOR),
        ]:
            tick = Line3D(
                start=np.array([xv, y_height,        z_pos]),
                end  =np.array([xv, y_height + 0.20, z_pos]),
                color=color, stroke_width=1.8,
            )
            self.add(tick)

        # Horizontal bar from x0 → x1
        horiz = Line3D(
            start=np.array([x0, y_height + 0.20, z_pos]),
            end  =np.array([x1, y_height + 0.20, z_pos]),
            color=POSTERIOR_COLOR, stroke_width=1.8,
        )
        self.add(horiz)

        # Label
        lbl = MathTex(
            rf"P(H|E^+) = {posterior:.3f}",
            color=POSTERIOR_COLOR,
            font_size=font_size,
        )
        lbl.move_to(np.array([(x0 + x1) / 2, y_height + 0.55, z_pos]))
        self.add(lbl)


# ---------------------------------------------------------------------------
# BayesBox3D
# ---------------------------------------------------------------------------

class BayesBox3D(VGroup):
    """2×2 Bayesian mosaic rendered as shaded 3D prisms.

    The footprint of each prism encodes the joint probability
    (width = P(H) or P(¬H),  depth = P(E|H) or P(E|¬H)).
    Height encodes the likelihood ratio of that cell, giving a 3D
    "landscape" where tall peaks show the most diagnostic quadrants.

    Layout (viewed from above):
        ┌───────────────┬────────────────┐  ← E+ row (front, z=0)
        │  TP  P(H∩E+)  │  FP  P(¬H∩E+) │
        ├───────────────┼────────────────┤  ← E− row (back)
        │  FN  P(H∩E−)  │  TN  P(¬H∩E−) │
        └───────────────┴────────────────┘
           H column          ¬H column

    Parameters
    ----------
    p_h : float
        Prior P(H).
    p_e_h : float
        Sensitivity P(E+|H).
    p_e_nh : float
        False positive rate P(E+|¬H).
    config : BayesConfig, optional
    """

    def __init__(
        self,
        p_h:    float,
        p_e_h:  float,
        p_e_nh: float,
        config: BayesConfig | None = None,
    ):
        super().__init__()
        self.cfg   = config or BayesConfig()
        self.stats = compute_joint_probs(
            _clamp(p_h, 1e-4, 1 - 1e-4),
            _clamp(p_e_h,  1e-4, 1 - 1e-4),
            _clamp(p_e_nh, 1e-4, 1 - 1e-4),
        )
        self._build()

    # ------------------------------------------------------------------

    def _build(self) -> None:
        cfg   = self.cfg
        s     = self.stats
        sx    = cfg.box_scale_x
        sz    = cfg.box_scale_z
        g     = cfg.box_gap / 2

        # Column widths  (x axis = H vs ¬H)
        w_h   = s["p_h"]  * sx
        w_nh  = s["p_nh"] * sx

        # Row depths  (z axis = E+ vs E−)
        d_ep  = s["p_e_h"]       * sz   # depth of E+ row for H column
        d_en  = (1 - s["p_e_h"]) * sz

        d_fp  = s["p_e_nh"]      * sz   # depth of E+ row for ¬H column
        d_fn  = (1 - s["p_e_nh"])* sz

        # Cell heights (encode likelihood ratio of each cell)
        # TP height = max_height (it is the most diagnostic)
        # We scale by joint prob so larger cells don't dominate
        max_h = cfg.box_max_height
        lr    = s["lr"]
        cell_heights = {
            "tp": max_h,
            "fn": max_h * (1 / max(lr, 1.0)),
            "fp": max_h * (1 / max(lr, 1.0)),
            "tn": max_h * 0.30,
        }

        # ---- Four prisms --------------------------------------------
        #  TP: x ∈ [0, w_h],         z ∈ [0, d_ep]
        #  FP: x ∈ [w_h+g, sx],      z ∈ [0, d_fp]
        #  FN: x ∈ [0, w_h],         z ∈ [d_ep+g, sz]
        #  TN: x ∈ [w_h+g, sx],      z ∈ [d_fp+g, sz]

        self.tp_cell = _ShadeBox(
            x0=0,      z0=0,
            width=w_h - g,  depth=d_ep - g,
            height=cell_heights["tp"], top_color=TP_COLOR,
        )
        self.fp_cell = _ShadeBox(
            x0=w_h + g,         z0=0,
            width=w_nh - g,     depth=d_fp - g,
            height=cell_heights["fp"], top_color=FP_COLOR,
        )
        self.fn_cell = _ShadeBox(
            x0=0,      z0=d_ep + g,
            width=w_h - g,  depth=d_en - g,
            height=cell_heights["fn"], top_color=FN_COLOR,
        )
        self.tn_cell = _ShadeBox(
            x0=w_h + g,     z0=d_fp + g,
            width=w_nh - g, depth=d_fn - g,
            height=cell_heights["tn"], top_color=TN_COLOR,
        )
        self._cells = [self.tp_cell, self.fp_cell,
                       self.fn_cell, self.tn_cell]
        for cell in self._cells:
            self.add(cell)

        # ---- Cell labels --------------------------------------------
        self._cell_labels = VGroup()
        if cfg.show_cell_labels:
            cell_data = [
                (self.tp_cell, "TP", s["tp"],  WHITE),
                (self.fp_cell, "FP", s["fp"],  WHITE),
                (self.fn_cell, "FN", s["fn"],  WHITE),
                (self.tn_cell, "TN", s["tn"],  WHITE),
            ]
            for cell, name, prob, color in cell_data:
                tc  = cell.top_center
                # Name tag
                tag = Text(name, color=color, font_size=cfg.cell_label_font_size + 4,
                           weight="BOLD")
                tag.move_to(tc + np.array([0, 0.22, 0]))
                # Probability value
                val = MathTex(
                    rf"{prob:.4f}",
                    color=_lt(color, 0.30),
                    font_size=cfg.cell_label_font_size,
                )
                val.move_to(tc + np.array([0, 0.50, 0]))
                self._cell_labels.add(tag, val)
        self.add(self._cell_labels)

        # ---- Axis dimension labels ----------------------------------
        self._axis_labels = VGroup()
        if cfg.show_axis_labels:
            # H / ¬H labels on x axis (at z = sz + 0.3)
            h_lbl = MathTex(
                rf"P(H) = {s['p_h']:.2f}",
                color=H_TRUE_COLOR,
                font_size=cfg.axis_label_font_size,
            )
            h_lbl.move_to(np.array([w_h / 2, -0.15, sz + 0.45]))

            nh_lbl = MathTex(
                rf"P(\neg H) = {s['p_nh']:.2f}",
                color=H_FALSE_COLOR,
                font_size=cfg.axis_label_font_size,
            )
            nh_lbl.move_to(np.array([w_h + w_nh / 2, -0.15, sz + 0.45]))

            # P(E+|H) / P(E+|¬H) labels on z axis (at x = -0.3)
            ep_h_lbl = MathTex(
                rf"P(E^+|H) = {s['p_e_h']:.2f}",
                color=E_POS_COLOR,
                font_size=cfg.axis_label_font_size,
            )
            ep_h_lbl.move_to(np.array([-1.10, -0.10, d_ep / 2]))

            ep_nh_lbl = MathTex(
                rf"P(E^+|\neg H) = {s['p_e_nh']:.2f}",
                color=FP_COLOR,
                font_size=cfg.axis_label_font_size,
            )
            ep_nh_lbl.move_to(np.array([-1.30, -0.10, d_fp / 2]))

            # LR badge
            lr_color = LR_POS_COLOR if s["lr"] > 1 else (
                LR_NEG_COLOR if s["lr"] < 1 else LR_NULL_COLOR
            )
            lr_lbl = MathTex(
                rf"LR = {s['lr']:.2f}",
                color=lr_color,
                font_size=cfg.axis_label_font_size + 4,
            )
            lr_lbl.move_to(np.array([sx / 2, cfg.box_max_height + 0.55, sz / 2]))

            self._axis_labels.add(h_lbl, nh_lbl, ep_h_lbl, ep_nh_lbl, lr_lbl)
        self.add(self._axis_labels)

        # ---- Posterior bracket --------------------------------------
        self._posterior_bracket: _PosteriorBracket | None = None
        if cfg.show_posterior_bracket:
            self._posterior_bracket = _PosteriorBracket(
                x0=0,
                x1=w_h,
                x2=sx,
                y_height=cell_heights["tp"] + 0.05,
                z_pos=0.0,
                posterior=s["posterior"],
                font_size=cfg.axis_label_font_size,
            )
            self.add(self._posterior_bracket)

        # Store metrics for animations
        self._w_h  = w_h
        self._sx   = sx
        self._sz   = sz
        self._d_ep = d_ep

    # ------------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------------

    def animate_build_box(
        self,
        lag_ratio: float = 0.20,
        run_time:  float = 2.5,
    ) -> LaggedStart:
        """Cells rise from the floor one by one: TP → FP → FN → TN.

        The build order conveys the structure: first the diagnostic
        cells (high evidence value), then the irrelevant ones.
        """
        order = [self.tp_cell, self.fp_cell, self.fn_cell, self.tn_cell]
        return LaggedStart(
            *[GrowFromPoint(cell, point=cell.floor_center, run_time=run_time * 0.6)
              for cell in order],
            lag_ratio=lag_ratio,
            run_time=run_time,
        )

    def animate_highlight_cell(
        self,
        cell_name:       str,
        highlight_color: ManimColor = YELLOW,
        scale_factor:    float      = 1.10,
        run_time:        float      = 0.55,
    ) -> Succession:
        """Flash and scale one cell.

        Parameters
        ----------
        cell_name : str
            One of ``"tp"``, ``"fp"``, ``"fn"``, ``"tn"``.
        """
        cell_map = {"tp": self.tp_cell, "fp": self.fp_cell,
                    "fn": self.fn_cell, "tn": self.tn_cell}
        cell = cell_map[cell_name.lower()]
        orig_top   = cell.top_face.get_fill_color()
        orig_front = cell.front_face.get_fill_color()
        orig_right = cell.right_face.get_fill_color()

        flash = AnimationGroup(
            cell.animate(run_time=run_time / 2).scale(scale_factor),
            cell.top_face.animate(run_time=run_time / 2)
                .set_fill(color=highlight_color),
        )
        restore = AnimationGroup(
            cell.animate(run_time=run_time / 2).scale(1 / scale_factor),
            cell.top_face.animate(run_time=run_time / 2)
                .set_fill(color=orig_top),
        )
        return Succession(flash, restore)

    def animate_reveal_posterior(
        self,
        run_time: float = 1.0,
    ) -> FadeIn:
        """Fade the posterior bracket annotation into view."""
        if self._posterior_bracket is None:
            return FadeIn(VGroup(), run_time=0.1)
        return FadeIn(self._posterior_bracket, run_time=run_time)

    def animate_sweep_evidence(
        self,
        run_time:  float = 1.2,
        lag_ratio: float = 0.30,
    ) -> LaggedStart:
        """Color-wave sweep across the E+ column (TP + FP) then dim E−.

        Visually "selects" the positive-evidence partition to show
        how only the E+ column contributes to the posterior.
        """
        return LaggedStart(
            self.tp_cell.animate(run_time=run_time * 0.5)
                .set_fill(color=E_POS_COLOR),
            self.fp_cell.animate(run_time=run_time * 0.5)
                .set_fill(color=_lt(E_POS_COLOR, 0.45)),
            AnimationGroup(
                self.fn_cell.animate(run_time=run_time * 0.4)
                    .set_opacity(0.25),
                self.tn_cell.animate(run_time=run_time * 0.4)
                    .set_opacity(0.25),
            ),
            lag_ratio=lag_ratio,
            run_time=run_time,
        )


# ---------------------------------------------------------------------------
# PriorPosteriorBar3D
# ---------------------------------------------------------------------------

class PriorPosteriorBar3D(VGroup):
    """A vertical probability bar split into H (bottom) and ¬H (top) bands.

    The bar height represents the full probability space [0, 1].
    The split point encodes P(H).  After calling animate_update(),
    the split morphs to the posterior P(H|E).

    Parameters
    ----------
    p_h : float
        Initial prior P(H) — height of the blue (H) band.
    x_pos : float
        World X position of the bar center.
    config : BayesConfig, optional
    """

    def __init__(
        self,
        p_h:    float,
        x_pos:  float = 0.0,
        config: BayesConfig | None = None,
    ):
        super().__init__()
        self.cfg       = config or BayesConfig()
        self._p_h      = _clamp(p_h)
        self._x_pos    = x_pos
        self._build(p_h=p_h, animate=False)

    def _build(self, p_h: float, animate: bool = False) -> None:
        cfg   = self.cfg
        p_h   = _clamp(p_h)
        p_nh  = 1 - p_h
        w     = cfg.bar_width
        d     = cfg.bar_depth
        H     = cfg.bar_max_height
        xc    = self._x_pos - w / 2
        zc    = -d / 2

        h_h   = H * p_h
        h_nh  = H * p_nh

        self.h_band = _ShadeBox(
            x0=xc, z0=zc,
            width=w, depth=d,
            height=h_h,
            top_color=H_TRUE_COLOR,
        )
        self.nh_band = _ShadeBox(
            x0=xc, z0=zc,
            width=w, depth=d,
            height=h_nh,
            top_color=H_FALSE_COLOR,
        )
        # ¬H sits on top of H
        self.nh_band.shift(np.array([0, h_h, 0]))

        self.add(self.h_band, self.nh_band)

        # Dividing line at the split
        self._split_line = Line3D(
            start=np.array([xc,     h_h, zc    ]),
            end  =np.array([xc + w, h_h, zc + d]),
            color=WHITE, stroke_width=1.8,
        )
        self.add(self._split_line)

        # Probability labels
        self._bar_labels = VGroup()
        if cfg.show_probability_labels:
            lbl_h = MathTex(
                rf"P(H) = {p_h:.3f}",
                color=_lt(H_TRUE_COLOR, 0.30),
                font_size=cfg.font_size,
            )
            lbl_h.move_to(np.array([xc + w + 0.55, h_h / 2, 0]))

            lbl_nh = MathTex(
                rf"P(\neg H) = {p_nh:.3f}",
                color=_lt(H_FALSE_COLOR, 0.30),
                font_size=cfg.font_size,
            )
            lbl_nh.move_to(np.array([xc + w + 0.65, h_h + h_nh / 2, 0]))

            self._bar_labels.add(lbl_h, lbl_nh)
        self.add(self._bar_labels)

        # Y axis tick marks at 0, 0.25, 0.5, 0.75, 1.0
        self._ticks = VGroup()
        ax_x = xc - 0.12
        for frac in [0.0, 0.25, 0.50, 0.75, 1.0]:
            yw = frac * H
            tick = Line3D(
                start=np.array([ax_x,        yw, 0]),
                end  =np.array([ax_x - 0.12, yw, 0]),
                color=cfg.axes_color, stroke_width=0.8,
            )
            lbl = Text(f"{frac:.2f}", color=cfg.axes_color,
                       font_size=cfg.font_size - 4)
            lbl.move_to(np.array([ax_x - 0.38, yw, 0]))
            self._ticks.add(tick, lbl)
        self.add(self._ticks)

        self._h_h  = h_h
        self._h_nh = h_nh

    # ------------------------------------------------------------------

    def animate_build(self, run_time: float = 1.5) -> LaggedStart:
        """Grow the bar from the floor upward: H band first, then ¬H."""
        return LaggedStart(
            GrowFromPoint(self.h_band,  point=self.h_band.floor_center,
                          run_time=run_time * 0.55),
            GrowFromPoint(self.nh_band, point=self.h_band.top_center,
                          run_time=run_time * 0.45),
            lag_ratio=0.40,
            run_time=run_time,
        )

    def animate_update(
        self,
        p_h_new:  float,
        run_time: float = 1.5,
    ) -> AnimationGroup:
        """Morph the bar split from the current prior to a new probability.

        Typically called with the posterior after seeing evidence::

            post = bayes_update(prior=0.01, lr=10.0)
            self.play(bar.animate_update(p_h_new=post))
        """
        cfg   = self.cfg
        p_h_new = _clamp(p_h_new)
        H     = cfg.bar_max_height
        xc    = self._x_pos - cfg.bar_width / 2
        zc    = -cfg.bar_depth / 2

        # Build target geometry objects
        target_h  = _ShadeBox(
            x0=xc, z0=zc,
            width=cfg.bar_width, depth=cfg.bar_depth,
            height=H * p_h_new,
            top_color=H_TRUE_COLOR,
        )
        target_nh = _ShadeBox(
            x0=xc, z0=zc,
            width=cfg.bar_width, depth=cfg.bar_depth,
            height=H * (1 - p_h_new),
            top_color=H_FALSE_COLOR,
        )
        target_nh.shift(np.array([0, H * p_h_new, 0]))

        new_split = Line3D(
            start=np.array([xc,              H * p_h_new, zc                  ]),
            end  =np.array([xc + cfg.bar_width, H * p_h_new, zc + cfg.bar_depth]),
            color=POSTERIOR_COLOR, stroke_width=2.2,
        )

        return AnimationGroup(
            Transform(self.h_band,       target_h,  run_time=run_time),
            Transform(self.nh_band,      target_nh, run_time=run_time),
            Transform(self._split_line,  new_split, run_time=run_time),
        )


# ---------------------------------------------------------------------------
# LikelihoodPanel3D
# ---------------------------------------------------------------------------

class LikelihoodPanel3D(VGroup):
    """Side-by-side bars for P(E|H) and P(E|¬H) with an LR bracket.

    Parameters
    ----------
    p_e_h : float
        P(E+|H) — sensitivity.
    p_e_nh : float
        P(E+|¬H) — false positive rate.
    x_pos : float
        World X of the panel centre.
    config : BayesConfig, optional
    """

    def __init__(
        self,
        p_e_h:  float,
        p_e_nh: float,
        x_pos:  float = 0.0,
        config: BayesConfig | None = None,
    ):
        super().__init__()
        self.cfg    = config or BayesConfig()
        self.p_e_h  = _clamp(p_e_h)
        self.p_e_nh = _clamp(p_e_nh)
        self._x_pos = x_pos
        self._build()

    def _build(self) -> None:
        cfg    = self.cfg
        H      = cfg.lr_max_height
        w      = cfg.lr_bar_width
        sp     = cfg.lr_bar_spacing
        d      = 0.30
        xc     = self._x_pos - (w + sp / 2)

        # P(E|H) bar
        self.h_bar = _ShadeBox(
            x0=xc,     z0=-d / 2,
            width=w,   depth=d,
            height=H * self.p_e_h,
            top_color=E_POS_COLOR,
        )
        # P(E|¬H) bar
        self.nh_bar = _ShadeBox(
            x0=xc + w + sp,  z0=-d / 2,
            width=w,          depth=d,
            height=H * self.p_e_nh,
            top_color=FP_COLOR,
        )
        self.add(self.h_bar, self.nh_bar)

        # Bar labels underneath
        for bar, text, color in [
            (self.h_bar,  rf"P(E^+|H)\n= {self.p_e_h:.3f}",   E_POS_COLOR),
            (self.nh_bar, rf"P(E^+|\neg H)\n= {self.p_e_nh:.3f}", FP_COLOR),
        ]:
            fc = bar.floor_center
            lbl = MathTex(text, color=color, font_size=cfg.font_size - 2)
            lbl.move_to(fc + np.array([0, -0.35, 0]))
            self.add(lbl)

        # LR bracket + value
        self._lr_bracket = VGroup()
        if cfg.show_lr_bracket:
            lr     = self.p_e_h / max(self.p_e_nh, 1e-9)
            y_tp   = H * self.p_e_h   + 0.12
            y_fp   = H * self.p_e_nh  + 0.12
            y_brk  = max(y_tp, y_fp)  + 0.20

            lr_color = LR_POS_COLOR if lr > 1 else (
                LR_NEG_COLOR if lr < 1 else LR_NULL_COLOR
            )
            x_left  = xc + w / 2
            x_right = xc + w + sp + w / 2

            for xv in [x_left, x_right]:
                tick = Line3D(
                    start=np.array([xv, y_brk - 0.12, 0]),
                    end  =np.array([xv, y_brk,         0]),
                    color=lr_color, stroke_width=1.6,
                )
                self._lr_bracket.add(tick)

            horiz = Line3D(
                start=np.array([x_left,  y_brk, 0]),
                end  =np.array([x_right, y_brk, 0]),
                color=lr_color, stroke_width=1.6,
            )
            self._lr_bracket.add(horiz)

            lr_lbl = MathTex(
                rf"LR = \frac{{P(E^+|H)}}{{P(E^+|\neg H)}} = {lr:.2f}",
                color=lr_color,
                font_size=cfg.font_size,
            )
            lr_lbl.move_to(np.array([(x_left + x_right) / 2, y_brk + 0.45, 0]))
            self._lr_bracket.add(lr_lbl)
            self.add(self._lr_bracket)

    # ------------------------------------------------------------------

    def animate_build(self, run_time: float = 1.5) -> AnimationGroup:
        """Both bars grow simultaneously from the floor."""
        return AnimationGroup(
            GrowFromPoint(self.h_bar,  self.h_bar.floor_center,  run_time=run_time),
            GrowFromPoint(self.nh_bar, self.nh_bar.floor_center, run_time=run_time),
        )

    def animate_highlight_lr(
        self,
        run_time: float = 0.8,
    ) -> Succession:
        """Pulse the LR bracket to draw attention to the ratio."""
        bright = self._lr_bracket.animate(run_time=run_time / 2).scale(1.12)
        dim    = self._lr_bracket.animate(run_time=run_time / 2).scale(1 / 1.12)
        return Succession(bright, dim)


# ---------------------------------------------------------------------------
# NaturalFrequencyTree3D
# ---------------------------------------------------------------------------

class NaturalFrequencyTree3D(VGroup):
    """A tree of prisms where branch area encodes absolute count.

    Structure
    ---------
    Level 0  (root)  :  N people total
    Level 1  (H/¬H)  :  N·P(H) with H  |  N·P(¬H) without H
    Level 2  (E/¬E)  :  each level-1 branch splits by P(E|·)

    Leaf nodes (level 2) show icon-array dot grids if
    ``config.show_icon_arrays = True``.

    Parameters
    ----------
    p_h : float
    p_e_h : float
    p_e_nh : float
    config : BayesConfig, optional
    """

    def __init__(
        self,
        p_h:    float,
        p_e_h:  float,
        p_e_nh: float,
        config: BayesConfig | None = None,
    ):
        super().__init__()
        self.cfg    = config or BayesConfig()
        self.stats  = compute_joint_probs(
            _clamp(p_h, 1e-4, 1 - 1e-4),
            _clamp(p_e_h,  1e-4, 1 - 1e-4),
            _clamp(p_e_nh, 1e-4, 1 - 1e-4),
        )
        self._build()

    def _build(self) -> None:
        cfg   = self.cfg
        s     = self.stats
        N     = cfg.tree_population
        ls    = cfg.tree_level_spacing
        mw    = cfg.tree_max_width
        g     = cfg.tree_branch_gap
        d     = 0.28   # uniform prism depth

        # Counts at each node
        n_h   = round(N * s["p_h"])
        n_nh  = N - n_h
        n_tp  = round(n_h  * s["p_e_h"])
        n_fn  = n_h  - n_tp
        n_fp  = round(n_nh * s["p_e_nh"])
        n_tn  = n_nh - n_fp

        # Width proportional to count / N
        def _w(n): return mw * (n / N)

        # ---- Level 0: root ------------------------------------------
        root_w   = mw
        root_x0  = -root_w / 2
        self.root_bar = _ShadeBox(
            x0=root_x0, z0=-d / 2,
            width=root_w, depth=d,
            height=0.30,
            top_color=GRAY_C,
        )
        self.add(self.root_bar)
        self._add_count_label(self.root_bar, N, GRAY_B, cfg.font_size)

        # ---- Connector line from root to level 1 --------------------
        root_top_x = 0.0
        y_root_top = 0.30

        # ---- Level 1: H and ¬H --------------------------------------
        w_h   = _w(n_h)
        w_nh  = _w(n_nh)
        total_l1_w = w_h + g + w_nh
        x0_h  = -total_l1_w / 2
        x0_nh = x0_h + w_h + g
        y1    = -(ls)

        self.h_bar = _ShadeBox(
            x0=x0_h, z0=-d / 2,
            width=w_h, depth=d, height=0.30,
            top_color=H_TRUE_COLOR,
        )
        self.nh_bar = _ShadeBox(
            x0=x0_nh, z0=-d / 2,
            width=w_nh, depth=d, height=0.30,
            top_color=H_FALSE_COLOR,
        )
        for bar in [self.h_bar, self.nh_bar]:
            bar.shift(np.array([0, y1, 0]))
        self.add(self.h_bar, self.nh_bar)
        self._add_count_label(self.h_bar,  n_h,  H_TRUE_COLOR,  cfg.font_size)
        self._add_count_label(self.nh_bar, n_nh, H_FALSE_COLOR, cfg.font_size)

        # Connectors: root → h_bar centre, root → nh_bar centre
        for bar, name in [(self.h_bar, "H"), (self.nh_bar, "¬H")]:
            fc   = bar.floor_center
            conn = Line3D(
                start=np.array([0,       y_root_top,  0]),
                end  =np.array([fc[0],   y1 + 0.30,   0]),
                color=GRAY_C, stroke_width=1.2,
            )
            self.add(conn)
            mid  = (np.array([0, y_root_top, 0]) + fc) / 2
            lbl  = Text(name,
                        color=H_TRUE_COLOR if name == "H" else H_FALSE_COLOR,
                        font_size=cfg.font_size - 2)
            lbl.move_to(mid + np.array([0.22, 0, 0]))
            self.add(lbl)

        # ---- Level 2: four leaf nodes (TP, FN, FP, TN) --------------
        y2   = y1 - ls
        gap2 = g * 0.7

        nodes_l2 = [
            (n_tp, TP_COLOR,  x0_h,                "TP", "E+|H",   s["tp"]),
            (n_fn, FN_COLOR,  x0_h  + _w(n_tp) + gap2, "FN", "E−|H",  s["fn"]),
            (n_fp, FP_COLOR,  x0_nh,               "FP", "E+|¬H",  s["fp"]),
            (n_tn, TN_COLOR,  x0_nh + _w(n_fp) + gap2, "TN", "E−|¬H", s["tn"]),
        ]
        self.leaf_bars = []
        for count, color, xstart, name, branch_lbl, prob in nodes_l2:
            if count == 0:
                continue
            w_leaf = _w(count)
            bar    = _ShadeBox(
                x0=xstart, z0=-d / 2,
                width=w_leaf, depth=d, height=0.30,
                top_color=color,
            )
            bar.shift(np.array([0, y2, 0]))
            self.add(bar)
            self.leaf_bars.append(bar)
            self._add_count_label(bar, count, WHITE, cfg.font_size - 2)

            # Branch label
            branch_lbl_obj = Text(
                f"{branch_lbl}\n{prob:.4f}",
                color=color, font_size=cfg.font_size - 4,
            )
            branch_lbl_obj.move_to(
                bar.floor_center + np.array([0, -0.50, 0])
            )
            self.add(branch_lbl_obj)

            # Connector from parent level-1 bar
            parent_bar = self.h_bar if name in ("TP", "FN") else self.nh_bar
            conn = Line3D(
                start=parent_bar.floor_center + np.array([0, y1, 0]),
                end  =bar.floor_center + np.array([0, 0.30, 0]),
                color=GRAY_C, stroke_width=1.0,
            )
            self.add(conn)

            # Icon array in leaf
            if cfg.show_icon_arrays and count > 0:
                self._build_icon_array(bar, count, N, color)

        # Posterior annotation
        if n_tp + n_fp > 0:
            post_pct = n_tp / (n_tp + n_fp)
            post_lbl = MathTex(
                rf"P(H|E^+) = \frac{{{n_tp}}}{{{n_tp + n_fp}}} = {post_pct:.3f}",
                color=POSTERIOR_COLOR,
                font_size=cfg.font_size,
            )
            post_lbl.move_to(np.array([0, y2 - 0.70, 0]))
            self.add(post_lbl)

    def _add_count_label(
        self,
        bar:      _ShadeBox,
        count:    int,
        color:    ManimColor,
        font_size: int,
    ) -> None:
        lbl = Text(f"n = {count}", color=color, font_size=font_size)
        lbl.move_to(bar.top_center + np.array([0, 0.22, 0]))
        self.add(lbl)

    def _build_icon_array(
        self,
        bar:    _ShadeBox,
        count:  int,
        total:  int,
        color:  ManimColor,
    ) -> None:
        """Draw a small grid of dots inside the leaf bar footprint."""
        cfg      = self.cfg
        max_dots = 25   # cap for visual clarity
        n_dots   = min(count, max_dots)
        radius   = cfg.icon_radius
        tc       = bar.top_center
        cols     = min(n_dots, 5)
        rows     = (n_dots + cols - 1) // cols
        spacing  = radius * 2.8
        x_start  = tc[0] - (cols - 1) * spacing / 2
        z_pos    = tc[2]

        for k in range(n_dots):
            row_k = k // cols
            col_k = k %  cols
            dot   = Dot3D(
                point=np.array([
                    x_start + col_k * spacing,
                    tc[1] + 0.12 + row_k * spacing,
                    z_pos,
                ]),
                radius=radius,
                color=color,
            )
            dot.set_opacity(0.85)
            self.add(dot)

    # ------------------------------------------------------------------

    def animate_grow_tree(
        self,
        run_time: float = 3.0,
    ) -> Succession:
        """Extend branches level by level: root → L1 → L2."""
        root_anim = GrowFromPoint(
            self.root_bar, self.root_bar.floor_center,
            run_time=run_time * 0.25,
        )
        l1_anim = AnimationGroup(
            GrowFromPoint(self.h_bar,  self.h_bar.floor_center,
                          run_time=run_time * 0.35),
            GrowFromPoint(self.nh_bar, self.nh_bar.floor_center,
                          run_time=run_time * 0.35),
        )
        l2_anim = LaggedStart(
            *[GrowFromPoint(bar, bar.floor_center, run_time=run_time * 0.30)
              for bar in self.leaf_bars],
            lag_ratio=0.20,
            run_time=run_time * 0.40,
        )
        return Succession(root_anim, l1_anim, l2_anim)


# ---------------------------------------------------------------------------
# SequentialBayesUpdater3D
# ---------------------------------------------------------------------------

class SequentialBayesUpdater3D(VGroup):
    """A probability axis with a marker that updates with each evidence.

    Maintains a running prior. Each call to animate_add_evidence()
    returns an animation that moves the marker and records a ghost.

    Parameters
    ----------
    prior : float
        Starting prior P(H).
    x_center : float
        World X of the axis center.
    y_pos : float
        World Y of the axis.
    config : BayesConfig, optional
    """

    def __init__(
        self,
        prior:    float,
        x_center: float = 0.0,
        y_pos:    float = 0.0,
        config:   BayesConfig | None = None,
    ):
        super().__init__()
        self.cfg       = config or BayesConfig()
        self._prior    = _clamp(prior)
        self._x_center = x_center
        self._y        = y_pos
        self._history: list[float] = [self._prior]
        self._build()

    def _build(self) -> None:
        cfg  = self.cfg
        L    = cfg.axis_length
        x0   = self._x_center - L / 2
        x1   = self._x_center + L / 2
        y    = self._y

        # Axis line
        axis = Line3D(
            start=np.array([x0, y, 0]),
            end  =np.array([x1, y, 0]),
            color=cfg.axes_color, stroke_width=1.5,
        )
        self.add(axis)

        # Tick marks and labels at 0, 0.1, ..., 1.0
        for k in range(11):
            t  = k / 10
            xv = x0 + t * L
            tick = Line3D(
                start=np.array([xv, y,        0]),
                end  =np.array([xv, y - 0.10, 0]),
                color=cfg.axes_color, stroke_width=0.8,
            )
            lbl = Text(f"{t:.1f}", color=cfg.axes_color,
                       font_size=cfg.font_size - 6)
            lbl.move_to(np.array([xv, y - 0.28, 0]))
            self.add(tick, lbl)

        # Axis label
        axis_lbl = MathTex(r"P(H)", color=cfg.axes_color,
                           font_size=cfg.font_size)
        axis_lbl.move_to(np.array([x1 + 0.50, y, 0]))
        self.add(axis_lbl)

        # Prior marker
        self._marker = self._make_marker(
            self._prior, POSTERIOR_COLOR, cfg.marker_radius
        )
        self.add(self._marker)

        # Ghosts group
        self._ghosts = VGroup()
        self.add(self._ghosts)

        # BF label group
        self._bf_labels = VGroup()
        self.add(self._bf_labels)

        self._x0 = x0
        self._L  = L

    def _prob_to_x(self, p: float) -> float:
        return self._x0 + _clamp(p) * self._L

    def _make_marker(
        self,
        p:      float,
        color:  ManimColor,
        radius: float,
    ) -> Sphere:
        m = Sphere(radius=radius, resolution=(10, 10))
        m.set_color(color)
        m.set_opacity(0.95)
        m.move_to(np.array([self._prob_to_x(p), self._y, 0]))
        return m

    # ------------------------------------------------------------------

    def animate_axis(self, run_time: float = 1.0) -> FadeIn:
        """Fade the axis and prior marker into view."""
        return FadeIn(self, run_time=run_time)

    def animate_add_evidence(
        self,
        lr:       float,
        label:    str = "",
        run_time: float = 1.2,
    ) -> Succession:
        """Apply one Bayesian update and animate the marker shift.

        A ghost marker is left at the old position.
        A Bayes factor annotation appears above the axis.

        Parameters
        ----------
        lr : float
            Likelihood ratio P(E|H) / P(E|¬H) for this piece of evidence.
        label : str
            Optional label for this evidence item (e.g. "Test +").

        Returns a ``Succession`` animation::

            for lr in [10.0, 5.0, 0.2]:
                self.play(updater.animate_add_evidence(lr))
        """
        cfg         = self.cfg
        old_p       = self._prior
        new_p       = bayes_update(old_p, lr)
        self._prior = new_p
        self._history.append(new_p)

        # Ghost at old position
        ghost = self._make_marker(old_p, PRIOR_COLOR, cfg.marker_radius * 0.75)
        ghost.set_opacity(cfg.ghost_opacity)
        self._ghosts.add(ghost)

        # Connecting trail line old → new
        old_x = self._prob_to_x(old_p)
        new_x = self._prob_to_x(new_p)
        trail = Line3D(
            start=np.array([old_x, self._y, 0]),
            end  =np.array([new_x, self._y, 0]),
            color=POSTERIOR_COLOR, stroke_width=1.8,
        )

        # Bayes factor label above the midpoint
        bf_color = LR_POS_COLOR if lr > 1 else (
            LR_NEG_COLOR if lr < 1 else LR_NULL_COLOR
        )
        bf_text  = f"LR = {lr:.2f}"
        if label:
            bf_text = f"{label}: {bf_text}"
        bf_lbl = Text(bf_text, color=bf_color, font_size=cfg.font_size - 4)
        mid_x  = (old_x + new_x) / 2
        bf_lbl.move_to(np.array([mid_x, self._y + 0.45, 0]))
        self._bf_labels.add(bf_lbl)

        # Target marker at new position
        new_marker = self._make_marker(new_p, POSTERIOR_COLOR, cfg.marker_radius)

        build_ghost   = FadeIn(ghost, run_time=run_time * 0.15)
        draw_trail    = Create(trail, run_time=run_time * 0.40)
        move_marker   = Transform(self._marker, new_marker, run_time=run_time * 0.45)
        show_bf       = FadeIn(bf_lbl, run_time=run_time * 0.20)

        self.add(trail)

        return Succession(
            build_ghost,
            AnimationGroup(draw_trail, show_bf),
            move_marker,
        )

    def animate_reset(
        self,
        new_prior: float,
        run_time:  float = 0.8,
    ) -> AnimationGroup:
        """Jump the marker back to a new prior, clearing all ghosts."""
        self._prior = _clamp(new_prior)
        self._history = [self._prior]
        target = self._make_marker(new_prior, POSTERIOR_COLOR,
                                   self.cfg.marker_radius)
        return AnimationGroup(
            Transform(self._marker, target, run_time=run_time),
            FadeOut(self._ghosts,   run_time=run_time * 0.5),
            FadeOut(self._bf_labels, run_time=run_time * 0.5),
        )


# ---------------------------------------------------------------------------
# BayesFormulaBanner
# ---------------------------------------------------------------------------

class BayesFormulaBanner(VGroup):
    """Floating MathTex rendering of Bayes' theorem with color-coded terms.

    The five terms are colored to match the visual objects in the scene:
        P(H|E)  ← posterior      [gold]
        P(E|H)  ← likelihood     [blue]
        P(H)    ← prior          [blue]
        P(E|¬H) ← FP rate        [red]
        P(¬H)   ← comp. prior    [red]

    Parameters
    ----------
    pos : np.ndarray
        World position of the banner center.
    font_size : int
    show_numerical : bool
        If True (and stats are provided), show the numerical substitution
        below the symbolic formula.
    stats : dict | None
        Output of ``compute_joint_probs()`` for numerical display.
    """

    def __init__(
        self,
        pos:             np.ndarray = np.array([0.0, 2.5, 0.0]),
        font_size:       int        = 36,
        show_numerical:  bool       = False,
        stats:           dict | None = None,
    ):
        super().__init__()

        # Symbolic formula
        self.formula = MathTex(
            r"P(H|E)",
            r"=",
            r"\frac{",
            r"P(E|H)",
            r"\cdot",
            r"P(H)",
            r"}{",
            r"P(E|H)",
            r"\cdot",
            r"P(H)",
            r"+",
            r"P(E|\neg H)",
            r"\cdot",
            r"P(\neg H)",
            r"}",
            font_size=font_size,
        )

        # Color individual terms
        # Index mapping (rough — depends on MathTex split):
        color_map = {
            0:  TERM_COLORS[0],   # P(H|E)
            3:  TERM_COLORS[1],   # P(E|H) numerator
            5:  TERM_COLORS[2],   # P(H) numerator
            7:  TERM_COLORS[1],   # P(E|H) denominator
            9:  TERM_COLORS[2],   # P(H) denominator
            11: TERM_COLORS[3],   # P(E|¬H)
            13: TERM_COLORS[4],   # P(¬H)
        }
        for idx, color in color_map.items():
            if idx < len(self.formula):
                self.formula[idx].set_color(color)

        self.formula.move_to(pos)
        self.add(self.formula)

        # Numerical substitution row
        if show_numerical and stats:
            s    = stats
            num  = MathTex(
                rf"= \frac{{{s['p_e_h']:.3f} \times {s['p_h']:.3f}}}"
                rf"{{{s['p_e_h']:.3f} \times {s['p_h']:.3f} + "
                rf"{s['p_e_nh']:.3f} \times {s['p_nh']:.3f}}}"
                rf"= {s['posterior']:.4f}",
                font_size=font_size - 6,
                color=GRAY_B,
            )
            num.next_to(self.formula, DOWN, buff=0.30)
            self.add(num)
            self.numerical = num

    # ------------------------------------------------------------------

    def animate_appear(self, run_time: float = 2.0) -> Write:
        """Write the formula with the Manim Write animation."""
        return Write(self.formula, run_time=run_time)

    def animate_highlight_term(
        self,
        term_index:  int,
        run_time:    float = 0.5,
        scale_factor: float = 1.20,
    ) -> Succession:
        """Pulse-scale one term of the formula to draw attention.

        Parameters
        ----------
        term_index : int
            0 = P(H|E) / posterior
            1 = P(E|H) / likelihood
            2 = P(H)   / prior
            3 = P(E|¬H)/ FP rate
            4 = P(¬H)  / complement prior
        """
        # Map logical term index to MathTex sub-object indices
        term_subobject_map = {0: [0], 1: [3, 7], 2: [5, 9], 3: [11], 4: [13]}
        indices = term_subobject_map.get(term_index, [0])

        pulse_in  = AnimationGroup(*[
            self.formula[i].animate(run_time=run_time / 2).scale(scale_factor)
            for i in indices if i < len(self.formula)
        ])
        pulse_out = AnimationGroup(*[
            self.formula[i].animate(run_time=run_time / 2).scale(1 / scale_factor)
            for i in indices if i < len(self.formula)
        ])
        return Succession(pulse_in, pulse_out)

    def animate_highlight_all_terms(
        self,
        run_time_each: float = 0.45,
    ) -> Succession:
        """Sequentially pulse each of the 5 logical terms in order:
        posterior → likelihood → prior → FP rate → complement prior.

        Useful as a "walking tour" of the formula before the main
        Bayesian scene begins.
        """
        return Succession(*[
            self.animate_highlight_term(i, run_time=run_time_each)
            for i in range(5)
        ])


# ---------------------------------------------------------------------------
# Convenience factory: full Bayes scene bundle
# ---------------------------------------------------------------------------

def build_bayes_scene(
    p_h:    float,
    p_e_h:  float,
    p_e_nh: float,
    config: BayesConfig | None = None,
) -> dict:
    """Instantiate and position all five Bayesian objects for a scene.

    Returns a dict with keys:
        "box"        – BayesBox3D (centred at origin)
        "prior_bar"  – PriorPosteriorBar3D (left of box)
        "lr_panel"   – LikelihoodPanel3D   (right of box)
        "tree"       – NaturalFrequencyTree3D (below box)
        "updater"    – SequentialBayesUpdater3D (above box)
        "formula"    – BayesFormulaBanner (top)
        "stats"      – dict from compute_joint_probs()

    Typical use::

        bundle = build_bayes_scene(p_h=0.01, p_e_h=0.90, p_e_nh=0.05)
        self.play(bundle["box"].animate_build_box())
        self.play(bundle["formula"].animate_appear())
    """
    cfg   = config or BayesConfig()
    stats = compute_joint_probs(
        _clamp(p_h, 1e-4, 1 - 1e-4),
        _clamp(p_e_h,  1e-4, 1 - 1e-4),
        _clamp(p_e_nh, 1e-4, 1 - 1e-4),
    )

    box     = BayesBox3D(p_h, p_e_h, p_e_nh, config=cfg)
    bar     = PriorPosteriorBar3D(p_h, x_pos=-(cfg.box_scale_x / 2 + 1.60),
                                  config=cfg)
    lr      = LikelihoodPanel3D(p_e_h, p_e_nh,
                                x_pos=cfg.box_scale_x / 2 + 1.50, config=cfg)
    tree    = NaturalFrequencyTree3D(p_h, p_e_h, p_e_nh, config=cfg)
    tree.shift(np.array([cfg.box_scale_x / 2, 0, cfg.box_scale_z + 1.5]))

    updater = SequentialBayesUpdater3D(
        prior=p_h,
        y_pos=cfg.box_max_height + 1.20,
        config=cfg,
    )
    formula = BayesFormulaBanner(
        pos=np.array([cfg.box_scale_x / 2, cfg.box_max_height + 2.50, 0]),
        show_numerical=True,
        stats=stats,
    )

    return dict(
        box=box, prior_bar=bar, lr_panel=lr,
        tree=tree, updater=updater, formula=formula,
        stats=stats,
    )