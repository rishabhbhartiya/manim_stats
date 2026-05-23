"""
manim_stats/probability/prob_tree.py
======================================
Production-quality 3D probability tree visualizations for Manim.

Objects
-------

ProbTreeNode  (dataclass)
    The logical building block.  Stores event name, conditional
    probability, children, optional label text, leaf flag, and any
    metadata needed for layout and rendering.  Trees are composed
    entirely of these; no separate "edge" dataclass is needed.

ProbTreeConfig  (dataclass)
    Every visual and layout parameter in one place.

_NodeGlyph  (VGroup)
    3D sphere whose radius scales with the cumulative path probability
    (root = largest, rare leaves = smallest).  Carries:
      - Sphere body colored by probability via a cold→hot gradient
      - Event label (Text) floating below
      - Conditional probability expression (MathTex) beside the edge
      - Optional terminal halo ring (annular Polygon) for leaf nodes
      - Probability badge (Text) above the sphere

_EdgeGlyph  (VGroup)
    Beveled 3D line from parent centre → child centre.
    Stroke width proportional to the transition probability so thick
    edges = high-probability transitions.
    Carries a centered probability label offset away from the line to
    avoid collisions, and a small arrowhead at the child end.

_PathLabel  (VGroup)
    At each leaf: full product expression  P = p₁ · p₂ · … · pₙ
    rendered as MathTex, plus a mini horizontal bar whose length
    encodes the final path probability.

_ConditionalTable  (VGroup)
    Floating 3D grid beside the tree showing all leaf-path
    conditional probabilities in a compact table.

_LayoutEngine
    Reingold-Tilford inspired recursive layout algorithm.
    Assigns each node a (x, y) position such that:
      - Nodes at the same depth share the same y coordinate
      - Subtrees never overlap (minimum horizontal separation enforced)
      - The whole tree is centred at x = 0
    Returns a dict mapping node id → np.ndarray world position.

ProbabilityTree3D  (VGroup)
    Main class.  Accepts a ProbTreeNode root and builds the full
    3D scene: glyphs, edges, path labels, optional table.

    Factory classmethods:
      from_dict(d)               – nested dict  {name: {p: …, children: {…}}}
      from_bayes(p_h, p_e_h, …)  – 2-level Bayes tree (H/¬H → E+/E−)
      from_bernoulli(n, p)       – n-flip coin tree, all 2ⁿ paths
      from_conditional_table(M)  – tree from a conditional probability matrix

    Animation suite:
      animate_grow_level(k)          – fan-out nodes + edges at depth k
      animate_grow_tree()            – level-by-level reveal
      animate_trace_path(path_names) – glow-pulse along a root→leaf path
      animate_highlight_event(name)  – given observed leaf, highlight
                                       all paths reaching it; dim others
      animate_prune(name)            – fade incompatible branches
      animate_label_paths()          – path products appear at leaves
      animate_morph_probs(new_root)  – update probabilities + geometry
      animate_reveal_table()         – conditional table fades in
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence
import uuid

import numpy as np

from manim import (
    WHITE, BLACK, GRAY, GRAY_A, GRAY_B, GRAY_C, GRAY_D, GRAY_E,
    BLUE, BLUE_E, RED, RED_E, GREEN, GREEN_E, YELLOW, ORANGE, PURPLE,
    UP, DOWN, LEFT, RIGHT,
    PI, TAU,
    VGroup, VMobject,
    Line3D, Polygon, Sphere, Dot3D, Arrow3D, Annulus,
    Text, MathTex,
    FadeIn, FadeOut, GrowFromCenter, GrowFromPoint, Transform,
    Create, Write, AnimationGroup, Succession, LaggedStart,
    interpolate_color,
    ManimColor,
    DEGREES,
)


# ---------------------------------------------------------------------------
# Palette & helpers
# ---------------------------------------------------------------------------

# Node color: probability cold (rare) → hot (common)
NODE_COLD    = ManimColor("#1A237E")   # dark indigo   – p near 0
NODE_MID     = ManimColor("#1565C0")   # blue          – p ~ 0.25
NODE_WARM    = ManimColor("#E65100")   # deep orange   – p ~ 0.75
NODE_HOT     = ManimColor("#B71C1C")   # dark red      – p near 1

# Edge & label accents
EDGE_COLOR   = ManimColor("#546E7A")   # blue-grey
LABEL_COLOR  = ManimColor("#ECEFF1")   # near-white
PATH_COLOR   = ManimColor("#FFD600")   # gold – highlighted path
PRUNE_OPACITY = 0.12                   # dimmed-out branches

# Leaf halo ring
HALO_COLOR   = ManimColor("#26C6DA")

# Table colors
TABLE_HEADER = ManimColor("#37474F")
TABLE_ROW_A  = ManimColor("#263238")
TABLE_ROW_B  = ManimColor("#1C2A30")

FACE_DARKEN  = 0.35


def _dk(c: ManimColor, f: float) -> ManimColor:
    return interpolate_color(c, BLACK, f)

def _lt(c: ManimColor, f: float) -> ManimColor:
    return interpolate_color(c, WHITE, f)

def _node_color(p: float) -> ManimColor:
    """Map cumulative path probability p ∈ [0,1] to a color."""
    p = float(np.clip(p, 0.0, 1.0))
    stops = [NODE_COLD, NODE_MID, NODE_WARM, NODE_HOT]
    n     = len(stops) - 1
    seg   = min(int(p * n), n - 1)
    t     = p * n - seg
    return interpolate_color(stops[seg], stops[seg + 1], t)


# ---------------------------------------------------------------------------
# ProbTreeNode  — logical tree structure
# ---------------------------------------------------------------------------

@dataclass
class ProbTreeNode:
    """One node in a probability tree.

    Parameters
    ----------
    name : str
        Event identifier; used as a display label and for path lookups.
    prob : float
        Conditional probability P(this event | parent event).
        For the root node, use the marginal probability (or 1.0 if the
        root represents the entire sample space).
    children : list[ProbTreeNode]
        Child nodes.  Empty list = leaf node.
    label : str, optional
        Display label override.  Defaults to ``name``.
    description : str, optional
        Longer description shown in the conditional table.
    color : ManimColor | None
        Override the automatic probability-gradient color.
    """
    name:        str
    prob:        float
    children:    list["ProbTreeNode"] = field(default_factory=list)
    label:       str                  = ""
    description: str                  = ""
    color:       ManimColor | None    = None
    _id:         str                  = field(default_factory=lambda: str(uuid.uuid4())[:8],
                                              init=False, repr=False)

    def __post_init__(self):
        self.prob  = float(np.clip(self.prob, 0.0, 1.0))
        self.label = self.label or self.name

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def iter_nodes(self) -> Iterator["ProbTreeNode"]:
        """Pre-order traversal of all nodes in this subtree."""
        yield self
        for child in self.children:
            yield from child.iter_nodes()

    def iter_paths(
        self,
        _current: list["ProbTreeNode"] | None = None,
    ) -> Iterator[list["ProbTreeNode"]]:
        """Yield all root→leaf paths as lists of nodes."""
        _current = (_current or []) + [self]
        if self.is_leaf:
            yield _current
        else:
            for child in self.children:
                yield from child.iter_paths(_current)

    def path_probability(self, path: list["ProbTreeNode"]) -> float:
        """Product of conditional probabilities along a path."""
        p = 1.0
        for node in path:
            p *= node.prob
        return p

    def depth(self) -> int:
        """Maximum depth of this subtree (leaf = 0)."""
        if self.is_leaf:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def nodes_at_depth(self, d: int) -> list["ProbTreeNode"]:
        """Return all nodes exactly d levels below this root."""
        if d == 0:
            return [self]
        out = []
        for child in self.children:
            out.extend(child.nodes_at_depth(d - 1))
        return out


# ---------------------------------------------------------------------------
# ProbTreeConfig
# ---------------------------------------------------------------------------

@dataclass
class ProbTreeConfig:
    """All visual and layout parameters for ProbabilityTree3D.

    Layout
    ------
    level_height : float
        Vertical distance between depth levels (Y axis).
    min_sibling_gap : float
        Minimum horizontal gap between sibling node centres.
    min_subtree_gap : float
        Minimum horizontal gap between adjacent subtrees.
    x_scale : float
        Global horizontal scale multiplier applied after layout.
    root_y : float
        World Y position of the root node.

    Node glyphs
    -----------
    root_radius : float
        Sphere radius for the root node (depth 0).
    min_radius : float
        Minimum sphere radius (leaf nodes at max depth).
    use_prob_size : bool
        Scale node radius by the cumulative path probability to the
        node, interpolated between min_radius and root_radius.
    node_opacity : float
        Fill opacity of node spheres.
    node_resolution : tuple[int,int]
        Sphere mesh resolution (u, v segments).
    show_halo : bool
        Render a faint halo ring around leaf nodes.
    halo_ratio : float
        Outer radius of the halo as a multiple of the leaf node radius.
    halo_opacity : float
        Opacity of the leaf halo ring.

    Edge glyphs
    -----------
    edge_base_width : float
        Stroke width for an edge with transition probability = 1.0.
    edge_min_width : float
        Minimum stroke width regardless of probability.
    show_edge_arrows : bool
        Render a small arrowhead at the child end of each edge.
    arrow_tip_length : float
        Length of the arrowhead tip.
    edge_opacity : float
        Opacity of edge lines.

    Labels
    ------
    show_event_labels : bool
        Show event name below each node sphere.
    show_edge_probs : bool
        Show conditional probability alongside each edge.
    show_path_probs : bool
        Show full path probability product at each leaf.
    show_path_bars : bool
        Show a mini horizontal bar at each leaf encoding path P.
    path_bar_max_width : float
        Maximum width of the leaf path probability bar.
    path_bar_height : float
        Height of the path probability bar prism.
    event_label_font_size : int
        Font size for node event labels.
    edge_prob_font_size : int
        Font size for edge probability labels.
    path_prob_font_size : int
        Font size for path product labels.

    Conditional table
    -----------------
    show_table : bool
        Render the floating conditional probability table.
    table_x_offset : float
        How far right of the tree the table is placed.
    table_row_height : float
        Height of each table row.
    table_col_width : float
        Width of the path / probability columns.
    table_font_size : int
        Font size inside the table.

    Highlight
    ---------
    highlight_color : ManimColor
        Color used for path tracing and event highlighting.
    highlight_stroke_width : float
        Extra stroke width added to highlighted edges.
    """

    # ---- layout ----
    level_height:     float = 2.20
    min_sibling_gap:  float = 0.55
    min_subtree_gap:  float = 0.40
    x_scale:          float = 1.00
    root_y:           float = 0.00

    # ---- node glyphs ----
    root_radius:      float         = 0.22
    min_radius:       float         = 0.10
    use_prob_size:    bool          = True
    node_opacity:     float         = 0.92
    node_resolution:  tuple         = (12, 12)
    show_halo:        bool          = True
    halo_ratio:       float         = 1.85
    halo_opacity:     float         = 0.20

    # ---- edge glyphs ----
    edge_base_width:  float = 3.00
    edge_min_width:   float = 0.80
    show_edge_arrows: bool  = True
    arrow_tip_length: float = 0.12
    edge_opacity:     float = 0.80

    # ---- labels ----
    show_event_labels:    bool  = True
    show_edge_probs:      bool  = True
    show_path_probs:      bool  = True
    show_path_bars:       bool  = True
    path_bar_max_width:   float = 1.40
    path_bar_height:      float = 0.08
    event_label_font_size: int  = 20
    edge_prob_font_size:   int  = 18
    path_prob_font_size:   int  = 16

    # ---- conditional table ----
    show_table:       bool       = False
    table_x_offset:   float      = 1.20
    table_row_height: float      = 0.40
    table_col_width:  float      = 1.80
    table_font_size:  int        = 17

    # ---- highlight ----
    highlight_color:        ManimColor = PATH_COLOR
    highlight_stroke_width: float      = 4.00


# ---------------------------------------------------------------------------
# _LayoutEngine  — Reingold-Tilford style recursive layout
# ---------------------------------------------------------------------------

class _LayoutEngine:
    """Assign (x, y) positions to every node, preventing subtree overlap.

    Algorithm
    ---------
    1. Post-order traversal: compute the *contour width* of each subtree
       (the horizontal span needed to accommodate all descendants with
       ``min_sibling_gap`` between siblings).
    2. In-order assignment: for each internal node, centre its children
       horizontally under it; spread them by the contour widths.
    3. Root is placed at x = 0.
    4. Y positions are simply  -depth * level_height  (root at top).

    All positions are in abstract layout units; the caller applies
    ``x_scale`` and ``root_y`` offset.
    """

    def __init__(self, cfg: ProbTreeConfig):
        self.cfg      = cfg
        self._widths: dict[str, float] = {}   # node._id → subtree width
        self._pos:    dict[str, np.ndarray] = {}

    def layout(self, root: ProbTreeNode) -> dict[str, np.ndarray]:
        """Return a dict: node._id → np.ndarray([x, y, 0])."""
        self._compute_widths(root)
        self._assign_positions(root, x=0.0, depth=0)
        return self._pos

    def _compute_widths(self, node: ProbTreeNode) -> float:
        """Return the horizontal span required by this subtree."""
        cfg = self.cfg
        if node.is_leaf:
            self._widths[node._id] = cfg.min_sibling_gap
            return cfg.min_sibling_gap

        child_widths = [self._compute_widths(c) for c in node.children]
        total = sum(child_widths) + cfg.min_subtree_gap * (len(node.children) - 1)
        # Node must be at least as wide as a single sibling gap
        total = max(total, cfg.min_sibling_gap)
        self._widths[node._id] = total
        return total

    def _assign_positions(
        self,
        node:  ProbTreeNode,
        x:     float,
        depth: int,
    ) -> None:
        cfg = self.cfg
        y   = -(depth * cfg.level_height) * cfg.x_scale + cfg.root_y
        self._pos[node._id] = np.array([x * cfg.x_scale, y, 0.0])

        if node.is_leaf:
            return

        # Lay children out centred under this node
        child_widths = [self._widths[c._id] for c in node.children]
        total_w      = sum(child_widths) + cfg.min_subtree_gap * (len(node.children) - 1)
        cursor       = x - total_w / 2

        for child, cw in zip(node.children, child_widths):
            child_x = cursor + cw / 2
            self._assign_positions(child, child_x, depth + 1)
            cursor += cw + cfg.min_subtree_gap


# ---------------------------------------------------------------------------
# _NodeGlyph
# ---------------------------------------------------------------------------

class _NodeGlyph(VGroup):
    """3D sphere + labels for one probability tree node.

    Parameters
    ----------
    node : ProbTreeNode
    pos : np.ndarray
        World position (x, y, z) of this node.
    cumulative_prob : float
        Product of conditional probabilities from root to this node.
        Used to size and color the sphere.
    cfg : ProbTreeConfig
    depth : int
        Depth of this node (root = 0); used for radius interpolation.
    max_depth : int
        Maximum depth of the full tree; used for radius interpolation.
    """

    def __init__(
        self,
        node:             ProbTreeNode,
        pos:              np.ndarray,
        cumulative_prob:  float,
        cfg:              ProbTreeConfig,
        depth:            int,
        max_depth:        int,
    ):
        super().__init__()
        self._node  = node
        self._pos   = pos
        self._cfg   = cfg

        # Radius: interpolate root_radius → min_radius by depth ratio
        if max_depth > 0:
            t      = depth / max_depth
            radius = cfg.root_radius * (1 - t) + cfg.min_radius * t
        else:
            radius = cfg.root_radius

        if cfg.use_prob_size:
            # Also scale by cumulative probability so rare paths shrink
            p_scale = 0.5 + 0.5 * cumulative_prob   # clamp shrinkage to 50%
            radius  = max(radius * p_scale, cfg.min_radius)

        self._radius = radius

        # Sphere color
        color = node.color if node.color is not None else _node_color(cumulative_prob)

        sphere = Sphere(radius=radius, resolution=cfg.node_resolution)
        sphere.set_color(color)
        sphere.set_opacity(cfg.node_opacity)
        sphere.move_to(pos)
        self.add(sphere)
        self.sphere = sphere
        self._color = color

        # Event label (below sphere)
        if cfg.show_event_labels and node.label:
            lbl = Text(node.label, color=LABEL_COLOR,
                       font_size=cfg.event_label_font_size)
            lbl.move_to(pos + np.array([0, -radius - 0.22, 0]))
            self.add(lbl)
            self.event_label = lbl

        # Probability badge (above sphere, shows the conditional prob)
        prob_txt = Text(f"{node.prob:.3f}",
                        color=_lt(color, 0.35),
                        font_size=cfg.edge_prob_font_size - 2)
        prob_txt.move_to(pos + np.array([0, radius + 0.18, 0]))
        self.add(prob_txt)
        self.prob_badge = prob_txt

        # Leaf halo ring
        if node.is_leaf and cfg.show_halo:
            halo = Annulus(
                inner_radius=radius * 1.10,
                outer_radius=radius * cfg.halo_ratio,
                color=HALO_COLOR,
            )
            halo.set_fill(color=HALO_COLOR, opacity=cfg.halo_opacity)
            halo.set_stroke(color=HALO_COLOR, width=0.5, opacity=cfg.halo_opacity * 1.5)
            halo.move_to(pos)
            self.add(halo)
            self.halo = halo


# ---------------------------------------------------------------------------
# _EdgeGlyph
# ---------------------------------------------------------------------------

class _EdgeGlyph(VGroup):
    """Beveled Line3D from parent → child with probability label.

    Stroke width scales with the transition probability so that
    visually thicker edges immediately signal high-probability transitions.
    """

    def __init__(
        self,
        parent_pos:  np.ndarray,
        child_pos:   np.ndarray,
        child_node:  ProbTreeNode,
        parent_radius: float,
        child_radius:  float,
        cfg:         ProbTreeConfig,
    ):
        super().__init__()
        p = child_node.prob

        # Shorten the line so it doesn't overlap the sphere glyphs
        direction = child_pos - parent_pos
        dist      = np.linalg.norm(direction)
        if dist < 1e-6:
            return
        unit      = direction / dist
        start     = parent_pos + unit * (parent_radius + 0.04)
        end_pt    = child_pos  - unit * (child_radius  + 0.04)

        stroke_w  = max(
            cfg.edge_min_width,
            p * cfg.edge_base_width,
        )

        if cfg.show_edge_arrows:
            edge = Arrow3D(
                start=start,
                end  =end_pt,
                color=EDGE_COLOR,
                stroke_width=stroke_w,
                tip_length=cfg.arrow_tip_length,
            )
        else:
            edge = Line3D(
                start=start,
                end  =end_pt,
                color=EDGE_COLOR,
                stroke_width=stroke_w,
            )

        edge.set_opacity(cfg.edge_opacity)
        self.add(edge)
        self.line = edge

        # Probability label: positioned at 40% along the edge, offset
        if cfg.show_edge_probs:
            mid     = start + 0.42 * (end_pt - start)
            # Perpendicular offset to avoid sitting on the line
            perp    = np.array([-unit[1], unit[0], 0]) * 0.28
            lbl_pos = mid + perp

            lbl = MathTex(
                rf"{p:.3f}",
                color=_lt(EDGE_COLOR, 0.40),
                font_size=cfg.edge_prob_font_size,
            )
            lbl.move_to(lbl_pos)
            self.add(lbl)
            self.prob_label = lbl

        self._start  = start
        self._end    = end_pt
        self._color  = EDGE_COLOR


# ---------------------------------------------------------------------------
# _PathLabel
# ---------------------------------------------------------------------------

class _PathLabel(VGroup):
    """Full path-probability product expression at a leaf node.

    Shows  P = p₁ · p₂ · … · pₙ = <value>  as MathTex, plus
    a mini horizontal bar whose length encodes the path probability.
    """

    def __init__(
        self,
        path:       list[ProbTreeNode],
        leaf_pos:   np.ndarray,
        leaf_radius: float,
        cfg:        ProbTreeConfig,
    ):
        super().__init__()
        # Compute path probability
        path_p = 1.0
        for node in path:
            path_p *= node.prob

        # Build a compact product string: p₁ × p₂ × … = value
        factors = " \\times ".join(f"{n.prob:.3f}" for n in path)
        tex_str = rf"P = {factors} = {path_p:.4f}"

        lbl = MathTex(tex_str,
                      color=_lt(PATH_COLOR, 0.20),
                      font_size=cfg.path_prob_font_size)
        lbl_pos = leaf_pos + np.array([0, -leaf_radius - 0.55, 0])
        lbl.move_to(lbl_pos)
        self.add(lbl)
        self.label = lbl

        # Mini bar
        if cfg.show_path_bars:
            bar_w  = cfg.path_bar_max_width * path_p
            bar_h  = cfg.path_bar_height
            color  = _node_color(path_p)
            bar_x0 = leaf_pos[0] - bar_w / 2
            bar_y0 = lbl_pos[1]  - 0.30
            bar_z0 = -bar_h / 2

            bar = Polygon(
                np.array([bar_x0,         bar_y0,           bar_z0]),
                np.array([bar_x0 + bar_w, bar_y0,           bar_z0]),
                np.array([bar_x0 + bar_w, bar_y0 + bar_h,   bar_z0]),
                np.array([bar_x0,         bar_y0 + bar_h,   bar_z0]),
                color=color,
            )
            bar.set_fill(color=color, opacity=0.88)
            bar.set_stroke(color=_dk(color, 0.35), width=0.5)
            self.add(bar)
            self.bar = bar

        self._path_prob = path_p


# ---------------------------------------------------------------------------
# _ConditionalTable
# ---------------------------------------------------------------------------

class _ConditionalTable(VGroup):
    """Floating grid showing all leaf paths and their probabilities.

    Columns: Path  |  P(path)  |  % of total
    One row per leaf path.
    """

    def __init__(
        self,
        paths:    list[list[ProbTreeNode]],
        x_pos:    float,
        y_top:    float,
        cfg:      ProbTreeConfig,
    ):
        super().__init__()
        rh  = cfg.table_row_height
        cw  = cfg.table_col_width
        fs  = cfg.table_font_size
        z   = 0.0

        headers = ["Path", "P(path)", "% total"]
        n_cols  = len(headers)

        # Total probability (should sum to ~1)
        all_probs = []
        for path in paths:
            p = 1.0
            for node in path:
                p *= node.prob
            all_probs.append(p)
        total = sum(all_probs)

        # Header row
        for j, hdr in enumerate(headers):
            cell_x = x_pos + j * cw
            cell_y = y_top
            bg = Polygon(
                np.array([cell_x,      cell_y - rh, z]),
                np.array([cell_x + cw, cell_y - rh, z]),
                np.array([cell_x + cw, cell_y,      z]),
                np.array([cell_x,      cell_y,      z]),
                color=TABLE_HEADER,
            )
            bg.set_fill(color=TABLE_HEADER, opacity=0.90)
            bg.set_stroke(color=GRAY_D, width=0.5)
            self.add(bg)
            lbl = Text(hdr, color=WHITE, font_size=fs, weight="BOLD")
            lbl.move_to(np.array([cell_x + cw / 2, cell_y - rh / 2, z]))
            self.add(lbl)

        # Data rows
        for row_idx, (path, path_p) in enumerate(zip(paths, all_probs)):
            y_row = y_top - (row_idx + 1) * rh
            bg_color = TABLE_ROW_A if row_idx % 2 == 0 else TABLE_ROW_B

            # Path string: "A → B → C"
            path_str = " → ".join(n.label for n in path)
            pct_str  = f"{100 * path_p / total:.1f}%"

            row_data = [path_str, f"{path_p:.5f}", pct_str]
            for j, val in enumerate(row_data):
                cell_x = x_pos + j * cw
                bg = Polygon(
                    np.array([cell_x,      y_row - rh, z]),
                    np.array([cell_x + cw, y_row - rh, z]),
                    np.array([cell_x + cw, y_row,      z]),
                    np.array([cell_x,      y_row,      z]),
                    color=bg_color,
                )
                bg.set_fill(color=bg_color, opacity=0.85)
                bg.set_stroke(color=GRAY_D, width=0.4)
                self.add(bg)
                text_color = _node_color(path_p) if j == 1 else LABEL_COLOR
                lbl = Text(val, color=text_color, font_size=fs)
                lbl.move_to(np.array([cell_x + cw / 2, y_row - rh / 2, z]))
                self.add(lbl)

        # Total row
        y_total = y_top - (len(paths) + 1) * rh
        for j, val in enumerate(["TOTAL", f"{total:.5f}", "100%"]):
            cell_x = x_pos + j * cw
            bg = Polygon(
                np.array([cell_x,      y_total - rh, z]),
                np.array([cell_x + cw, y_total - rh, z]),
                np.array([cell_x + cw, y_total,      z]),
                np.array([cell_x,      y_total,      z]),
                color=TABLE_HEADER,
            )
            bg.set_fill(color=TABLE_HEADER, opacity=0.90)
            bg.set_stroke(color=GRAY_D, width=0.5)
            self.add(bg)
            lbl = Text(val, color=YELLOW, font_size=fs, weight="BOLD")
            lbl.move_to(np.array([cell_x + cw / 2, y_total - rh / 2, z]))
            self.add(lbl)


# ---------------------------------------------------------------------------
# Main ProbabilityTree3D
# ---------------------------------------------------------------------------

class ProbabilityTree3D(VGroup):
    """A detailed 3D probability tree for Manim statistics animations.

    Nodes are shaded spheres whose size encodes cumulative path
    probability.  Edges are beveled lines whose thickness encodes
    transition probability.  Leaves carry full path-product labels
    and optional mini probability bars.

    Basic usage
    -----------
    >>> from manim import *
    >>> from manim_stats.probability.prob_tree import (
    ...     ProbabilityTree3D, ProbTreeNode, ProbTreeConfig
    ... )
    >>>
    >>> class MyScene(ThreeDScene):
    ...     def construct(self):
    ...         root = ProbTreeNode("S", 1.0, children=[
    ...             ProbTreeNode("A", 0.3, children=[
    ...                 ProbTreeNode("X", 0.6),
    ...                 ProbTreeNode("Y", 0.4),
    ...             ]),
    ...             ProbTreeNode("B", 0.7, children=[
    ...                 ProbTreeNode("X", 0.2),
    ...                 ProbTreeNode("Y", 0.8),
    ...             ]),
    ...         ])
    ...         cfg  = ProbTreeConfig(show_path_probs=True)
    ...         tree = ProbabilityTree3D(root, config=cfg)
    ...         self.set_camera_orientation(phi=60*DEGREES, theta=-50*DEGREES)
    ...         self.play(tree.animate_grow_tree())

    Parameters
    ----------
    root : ProbTreeNode
        Root of the probability tree.
    config : ProbTreeConfig, optional
    """

    def __init__(
        self,
        root:   ProbTreeNode,
        config: ProbTreeConfig | None = None,
    ):
        super().__init__()
        self.cfg        = config or ProbTreeConfig()
        self.root       = root
        self._max_depth = root.depth()
        self._build()

    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------

    def _build(self) -> None:
        cfg = self.cfg

        # ---- 1. Layout -----------------------------------------------
        engine   = _LayoutEngine(cfg)
        self._positions: dict[str, np.ndarray] = engine.layout(self.root)

        # ---- 2. Compute cumulative probabilities ---------------------
        self._cum_probs: dict[str, float] = {}
        self._parent_map: dict[str, ProbTreeNode | None] = {self.root._id: None}

        def _compute_cum(node: ProbTreeNode, cum: float) -> None:
            self._cum_probs[node._id] = cum * node.prob
            for child in node.children:
                self._parent_map[child._id] = node
                _compute_cum(child, cum * node.prob)

        _compute_cum(self.root, 1.0)

        # ---- 3. Compute node depths ---------------------------------
        self._depths: dict[str, int] = {}

        def _set_depth(node: ProbTreeNode, d: int) -> None:
            self._depths[node._id] = d
            for child in node.children:
                _set_depth(child, d + 1)

        _set_depth(self.root, 0)

        # ---- 4. Node glyphs (grouped by depth for animation) --------
        self._node_glyphs:  dict[str, _NodeGlyph]  = {}
        self._depth_groups: dict[int, VGroup]       = {}
        self._edge_glyphs:  dict[str, _EdgeGlyph]  = {}

        for node in self.root.iter_nodes():
            pos   = self._positions[node._id]
            cum   = self._cum_probs[node._id]
            depth = self._depths[node._id]

            glyph = _NodeGlyph(
                node=node,
                pos=pos,
                cumulative_prob=cum,
                cfg=cfg,
                depth=depth,
                max_depth=self._max_depth,
            )
            self._node_glyphs[node._id] = glyph

            if depth not in self._depth_groups:
                self._depth_groups[depth] = VGroup()
            self._depth_groups[depth].add(glyph)

        # ---- 5. Edge glyphs (also grouped by depth of the child) ----
        self._edge_depth_groups: dict[int, VGroup] = {}

        for node in self.root.iter_nodes():
            parent_pos    = self._positions[node._id]
            parent_glyph  = self._node_glyphs[node._id]
            parent_radius = parent_glyph._radius

            for child in node.children:
                child_pos    = self._positions[child._id]
                child_glyph  = self._node_glyphs[child._id]
                child_radius = child_glyph._radius
                child_depth  = self._depths[child._id]

                edge = _EdgeGlyph(
                    parent_pos=parent_pos,
                    child_pos=child_pos,
                    child_node=child,
                    parent_radius=parent_radius,
                    child_radius=child_radius,
                    cfg=cfg,
                )
                self._edge_glyphs[child._id] = edge

                if child_depth not in self._edge_depth_groups:
                    self._edge_depth_groups[child_depth] = VGroup()
                self._edge_depth_groups[child_depth].add(edge)

        # Add all edges then all nodes (edges behind nodes)
        for d in sorted(self._edge_depth_groups.keys()):
            self.add(self._edge_depth_groups[d])
        for d in sorted(self._depth_groups.keys()):
            self.add(self._depth_groups[d])

        # ---- 6. Path probability labels at leaves -------------------
        self._path_labels: list[_PathLabel] = []
        self._path_labels_group = VGroup()

        if cfg.show_path_probs:
            for path in self.root.iter_paths():
                leaf      = path[-1]
                leaf_pos  = self._positions[leaf._id]
                leaf_r    = self._node_glyphs[leaf._id]._radius
                plbl      = _PathLabel(path, leaf_pos, leaf_r, cfg)
                self._path_labels.append(plbl)
                self._path_labels_group.add(plbl)
            self.add(self._path_labels_group)

        # ---- 7. Total probability Σ = 1 annotation -----------------
        all_paths = list(self.root.iter_paths())
        if all_paths:
            total = sum(
                np.prod([n.prob for n in path]) for path in all_paths
            )
            sigma_lbl = MathTex(
                rf"\sum P(\text{{path}}) = {total:.4f}",
                color=GRAY_B,
                font_size=cfg.edge_prob_font_size,
            )
            # Place below the deepest node
            y_min = min(
                self._positions[n._id][1]
                for n in self.root.iter_nodes()
            )
            sigma_lbl.move_to(np.array([0, y_min - 0.65, 0]))
            self.add(sigma_lbl)
            self.sigma_label = sigma_lbl

        # ---- 8. Conditional table -----------------------------------
        self._table: _ConditionalTable | None = None
        if cfg.show_table and all_paths:
            x_right = max(
                self._positions[n._id][0]
                for n in self.root.iter_nodes()
            )
            y_top = self._positions[self.root._id][1]
            self._table = _ConditionalTable(
                paths=all_paths,
                x_pos=x_right + cfg.table_x_offset,
                y_top=y_top,
                cfg=cfg,
            )
            self.add(self._table)

        # ---- Store all paths for animation use ----------------------
        self._all_paths = all_paths

    # ------------------------------------------------------------------
    # Helpers: node/path lookup
    # ------------------------------------------------------------------

    def _find_node(self, name: str) -> ProbTreeNode | None:
        """Find the first node with the given name by pre-order search."""
        for node in self.root.iter_nodes():
            if node.name == name:
                return node
        return None

    def _paths_through(self, name: str) -> list[list[ProbTreeNode]]:
        """Return all root→leaf paths that pass through a node named `name`."""
        return [p for p in self._all_paths
                if any(n.name == name for n in p)]

    def _ancestor_chain(self, node: ProbTreeNode) -> list[ProbTreeNode]:
        """Return the chain [root, …, node] using the parent map."""
        chain = [node]
        current = node
        while self._parent_map.get(current._id) is not None:
            current = self._parent_map[current._id]
            chain.append(current)
        return list(reversed(chain))

    # ------------------------------------------------------------------
    # Animation API
    # ------------------------------------------------------------------

    def animate_grow_level(
        self,
        depth:     int,
        lag_ratio: float = 0.12,
        run_time:  float = 1.2,
    ) -> LaggedStart:
        """Fan out all nodes and their incoming edges at a given depth.

        Nodes grow from their parent's position; edges trace outward.

        Parameters
        ----------
        depth : int
            0 = root node only;  1 = first level of children; etc.
        """
        anims = []

        if depth in self._depth_groups:
            for glyph in self._depth_groups[depth].submobjects:
                parent_id = self._parent_map.get(glyph._node._id)
                if parent_id is not None:
                    origin = self._positions[parent_id._id]
                else:
                    origin = glyph._pos
                anims.append(
                    GrowFromPoint(glyph, point=origin, run_time=run_time * 0.7)
                )

        if depth in self._edge_depth_groups:
            for edge in self._edge_depth_groups[depth].submobjects:
                anims.append(Create(edge, run_time=run_time * 0.60))

        if not anims:
            return LaggedStart(FadeIn(VGroup()), lag_ratio=0, run_time=0.1)

        return LaggedStart(*anims, lag_ratio=lag_ratio, run_time=run_time)

    def animate_grow_tree(
        self,
        lag_ratio_levels: float = 0.10,
        run_time:         float = 4.0,
    ) -> Succession:
        """Reveal the tree level by level from root to leaves.

        Each level's nodes and edges grow together before the next level
        begins, making the branching structure immediately legible::

            self.play(tree.animate_grow_tree())
        """
        n_levels = self._max_depth + 1
        per_level_time = run_time / max(n_levels, 1)

        level_anims = [
            self.animate_grow_level(d, run_time=per_level_time)
            for d in range(n_levels)
        ]
        return Succession(*level_anims)

    def animate_trace_path(
        self,
        path_names: list[str],
        run_time:   float = 2.5,
        lag_ratio:  float = 0.25,
    ) -> LaggedStart:
        """Highlight one root→leaf path by pulsing each edge in sequence.

        Each edge along the path is temporarily recolored and thickened;
        the corresponding node glyph scales up.

        Parameters
        ----------
        path_names : list[str]
            Node names along the path, e.g. ``["S", "A", "X"]``.

        Example
        -------
        >>> self.play(tree.animate_trace_path(["S", "A", "X"]))
        """
        cfg    = self.cfg
        anims  = []
        nodes  = [self._find_node(name) for name in path_names]
        nodes  = [n for n in nodes if n is not None]

        for i, node in enumerate(nodes):
            glyph = self._node_glyphs.get(node._id)
            if glyph:
                # Scale up the node
                pulse = Succession(
                    glyph.animate(run_time=run_time * 0.12 / max(len(nodes), 1))
                         .scale(1.30),
                    glyph.animate(run_time=run_time * 0.10 / max(len(nodes), 1))
                         .scale(1 / 1.30),
                )
                anims.append(pulse)

            # Highlight the edge leading to this node (if not root)
            if i > 0:
                edge = self._edge_glyphs.get(node._id)
                if edge and hasattr(edge, "line"):
                    orig_color  = edge._color
                    orig_width  = cfg.edge_base_width * node.prob
                    highlight = AnimationGroup(
                        edge.line.animate(run_time=run_time * 0.15 / max(len(nodes), 1))
                            .set_color(cfg.highlight_color)
                            .set_stroke(width=cfg.highlight_stroke_width),
                    )
                    restore = AnimationGroup(
                        edge.line.animate(run_time=run_time * 0.10 / max(len(nodes), 1))
                            .set_color(orig_color)
                            .set_stroke(width=max(cfg.edge_min_width,
                                                  node.prob * cfg.edge_base_width)),
                    )
                    anims.append(Succession(highlight, restore))

        return LaggedStart(*anims, lag_ratio=lag_ratio, run_time=run_time)

    def animate_highlight_event(
        self,
        observed_name: str,
        run_time:      float = 1.5,
    ) -> AnimationGroup:
        """Given an observed event, highlight compatible paths; dim others.

        Useful for demonstrating backward Bayesian reasoning: "We observed
        X — which prior branches are now more/less plausible?"

        All paths that pass through the observed node are fully lit;
        all other nodes and edges are dimmed to ``PRUNE_OPACITY``.

        Parameters
        ----------
        observed_name : str
            Name of the observed event node.
        """
        compatible_paths = self._paths_through(observed_name)
        compatible_ids   = {
            n._id
            for path in compatible_paths
            for n in path
        }

        dim_anims  = []
        show_anims = []

        for node in self.root.iter_nodes():
            glyph = self._node_glyphs.get(node._id)
            if glyph is None:
                continue
            if node._id in compatible_ids:
                show_anims.append(
                    glyph.animate(run_time=run_time).set_opacity(1.0)
                )
            else:
                dim_anims.append(
                    glyph.animate(run_time=run_time).set_opacity(PRUNE_OPACITY)
                )

            edge = self._edge_glyphs.get(node._id)
            if edge is None:
                continue
            if node._id in compatible_ids:
                show_anims.append(
                    edge.animate(run_time=run_time)
                        .set_opacity(1.0)
                        .set_color(self.cfg.highlight_color)
                )
            else:
                dim_anims.append(
                    edge.animate(run_time=run_time).set_opacity(PRUNE_OPACITY)
                )

        return AnimationGroup(*dim_anims, *show_anims)

    def animate_prune(
        self,
        incompatible_name: str,
        run_time:          float = 1.2,
    ) -> AnimationGroup:
        """Fade out all branches that pass through an incompatible node.

        Opposite of animate_highlight_event: call this when an event is
        ruled out, and every path through that event disappears.

        Parameters
        ----------
        incompatible_name : str
            Name of the node whose branches should be pruned.
        """
        pruned_paths  = self._paths_through(incompatible_name)
        pruned_ids    = {n._id for path in pruned_paths for n in path}

        anims = []
        for node_id in pruned_ids:
            glyph = self._node_glyphs.get(node_id)
            if glyph:
                anims.append(
                    FadeOut(glyph, run_time=run_time)
                )
            edge = self._edge_glyphs.get(node_id)
            if edge:
                anims.append(
                    FadeOut(edge, run_time=run_time)
                )
        return AnimationGroup(*anims)

    def animate_label_paths(
        self,
        lag_ratio: float = 0.20,
        run_time:  float = 2.5,
    ) -> LaggedStart:
        """Reveal path-probability product labels at leaves one by one.

        Labels appear left to right (sorted by leaf X position).
        """
        if not self._path_labels:
            return LaggedStart(FadeIn(VGroup()), lag_ratio=0, run_time=0.1)

        # Sort labels left to right by leaf X position
        sorted_labels = sorted(
            self._path_labels,
            key=lambda pl: self._positions[
                pl.label.get_center()[0]   # approximate by label x
                if False else
                # find the corresponding leaf
                min(
                    (self._positions[n._id][0], n._id)
                    for n in self.root.iter_nodes()
                    if n.is_leaf
                )[1]
            ][0]
            if False else 0,  # fallback: original order
        )

        return LaggedStart(
            *[FadeIn(lbl, shift=DOWN * 0.10, run_time=run_time * 0.5)
              for lbl in self._path_labels],
            lag_ratio=lag_ratio,
            run_time=run_time,
        )

    def animate_reveal_table(
        self,
        run_time: float = 1.2,
    ) -> FadeIn:
        """Fade the conditional probability table into view."""
        if self._table is None:
            return FadeIn(VGroup(), run_time=0.1)
        return FadeIn(self._table, run_time=run_time)

    def animate_morph_probs(
        self,
        new_root:  ProbTreeNode,
        run_time:  float = 2.0,
    ) -> AnimationGroup:
        """Smoothly morph the tree to a new root with updated probabilities.

        Node spheres and edge widths all transition to reflect the new
        probability values.  The tree structure (shape) must be identical.

        Parameters
        ----------
        new_root : ProbTreeNode
            A root with the same topology but different probability values.
        """
        new_tree = ProbabilityTree3D(new_root, config=self.cfg)
        anims    = []

        for node, new_node in zip(
            self.root.iter_nodes(), new_root.iter_nodes()
        ):
            old_glyph = self._node_glyphs.get(node._id)
            new_glyph = new_tree._node_glyphs.get(new_node._id)
            if old_glyph and new_glyph:
                anims.append(Transform(old_glyph, new_glyph, run_time=run_time))

            old_edge = self._edge_glyphs.get(node._id)
            new_edge = new_tree._edge_glyphs.get(new_node._id)
            if old_edge and new_edge:
                anims.append(Transform(old_edge, new_edge, run_time=run_time))

        return AnimationGroup(*anims)

    def animate_restore(
        self,
        run_time: float = 0.8,
    ) -> AnimationGroup:
        """Restore all nodes and edges to full opacity after dimming/pruning."""
        anims = []
        for node in self.root.iter_nodes():
            glyph = self._node_glyphs.get(node._id)
            if glyph:
                anims.append(glyph.animate(run_time=run_time).set_opacity(1.0))
            edge = self._edge_glyphs.get(node._id)
            if edge:
                anims.append(
                    edge.animate(run_time=run_time)
                        .set_opacity(self.cfg.edge_opacity)
                        .set_color(EDGE_COLOR)
                )
        return AnimationGroup(*anims)

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        d:      dict,
        config: ProbTreeConfig | None = None,
    ) -> "ProbabilityTree3D":
        """Build a tree from a nested dict.

        Dict format (recursive)::

            {
                "name": "Root",
                "prob": 1.0,
                "label": "S",          # optional display label
                "children": [
                    {
                        "name": "A",
                        "prob": 0.3,
                        "children": [
                            {"name": "X", "prob": 0.6},
                            {"name": "Y", "prob": 0.4},
                        ]
                    },
                    {
                        "name": "B",
                        "prob": 0.7,
                        "children": [
                            {"name": "X", "prob": 0.2},
                            {"name": "Y", "prob": 0.8},
                        ]
                    },
                ]
            }
        """
        def _parse(node_dict: dict) -> ProbTreeNode:
            children = [
                _parse(c) for c in node_dict.get("children", [])
            ]
            return ProbTreeNode(
                name     = node_dict.get("name", "?"),
                prob     = float(node_dict.get("prob", 1.0)),
                children = children,
                label    = node_dict.get("label", node_dict.get("name", "?")),
                description = node_dict.get("description", ""),
            )

        return cls(_parse(d), config=config)

    @classmethod
    def from_bayes(
        cls,
        p_h:    float,
        p_e_h:  float,
        p_e_nh: float,
        config: ProbTreeConfig | None = None,
    ) -> "ProbabilityTree3D":
        """Two-level Bayes tree: H/¬H splitting into E+/E−.

        Matches the logical structure of ``BayesBox3D`` so the two
        objects can be used together in the same scene.

        Parameters
        ----------
        p_h : float    P(H)
        p_e_h : float  P(E+|H)
        p_e_nh : float P(E+|¬H)
        """
        p_nh   = 1 - p_h
        p_en_h = 1 - p_e_h
        p_en_nh = 1 - p_e_nh

        root = ProbTreeNode("S", 1.0, label="Sample\nSpace", children=[
            ProbTreeNode("H",   p_h,   label=f"H\n({p_h:.2f})",   color=ManimColor("#1565C0"),
                         children=[
                ProbTreeNode("E+|H",  p_e_h,   label=f"E⁺\n({p_e_h:.2f})",  color=ManimColor("#2E7D32")),
                ProbTreeNode("E-|H",  p_en_h,  label=f"E⁻\n({p_en_h:.2f})", color=ManimColor("#4E342E")),
            ]),
            ProbTreeNode("¬H",  p_nh,  label=f"¬H\n({p_nh:.2f})",  color=ManimColor("#B71C1C"),
                         children=[
                ProbTreeNode("E+|¬H", p_e_nh,  label=f"E⁺\n({p_e_nh:.2f})",  color=ManimColor("#EF5350")),
                ProbTreeNode("E-|¬H", p_en_nh, label=f"E⁻\n({p_en_nh:.2f})", color=ManimColor("#795548")),
            ]),
        ])
        return cls(root, config=config)

    @classmethod
    def from_bernoulli(
        cls,
        n:      int,
        p:      float,
        config: ProbTreeConfig | None = None,
    ) -> "ProbabilityTree3D":
        """n-flip coin tree with all 2ⁿ paths.

        Each level branches into H (heads, prob=p) and T (tails, prob=1−p).
        Path labels show the cumulative probability of the full sequence.

        Parameters
        ----------
        n : int
            Number of coin flips.  Recommended n ≤ 4 for readability.
        p : float
            Probability of Heads on each flip.
        """
        q = 1 - p

        def _coin_subtree(depth: int) -> list[ProbTreeNode]:
            if depth == 0:
                return []
            return [
                ProbTreeNode("H", p, label=f"H\n({p:.2f})",
                             color=ManimColor("#1565C0"),
                             children=_coin_subtree(depth - 1)),
                ProbTreeNode("T", q, label=f"T\n({q:.2f})",
                             color=ManimColor("#B71C1C"),
                             children=_coin_subtree(depth - 1)),
            ]

        root = ProbTreeNode(
            "Start", 1.0, label="Start",
            children=_coin_subtree(n),
        )
        return cls(root, config=config)

    @classmethod
    def from_conditional_table(
        cls,
        table:        np.ndarray,
        row_names:    list[str],
        col_names:    list[str],
        row_priors:   list[float] | None = None,
        config:       ProbTreeConfig | None = None,
    ) -> "ProbabilityTree3D":
        """Build a 2-level tree from a conditional probability matrix.

        Parameters
        ----------
        table : (R, C) ndarray
            ``table[i, j] = P(col_j | row_i)``.  Each row should sum to 1.
        row_names : list[str]
            Names for the first-level nodes (R names).
        col_names : list[str]
            Names for the second-level leaf nodes (C names).
        row_priors : list[float] | None
            Prior probabilities for each row event.  Defaults to uniform.

        Example
        -------
        >>> # P(symptom | disease) table
        >>> table = np.array([[0.8, 0.2], [0.3, 0.7]])
        >>> tree = ProbabilityTree3D.from_conditional_table(
        ...     table,
        ...     row_names=["Disease A", "Disease B"],
        ...     col_names=["Symptom +", "Symptom −"],
        ...     row_priors=[0.4, 0.6],
        ... )
        """
        table = np.asarray(table, dtype=float)
        R, C  = table.shape

        if row_priors is None:
            row_priors = [1.0 / R] * R

        children = []
        for i, (rname, rprior) in enumerate(zip(row_names, row_priors)):
            leaf_nodes = [
                ProbTreeNode(
                    name=f"{rname}→{col_names[j]}",
                    prob=float(table[i, j]),
                    label=col_names[j],
                )
                for j in range(C)
            ]
            children.append(ProbTreeNode(
                name=rname,
                prob=float(rprior),
                label=rname,
                children=leaf_nodes,
            ))

        root = ProbTreeNode("S", 1.0, label="S", children=children)
        return cls(root, config=config)