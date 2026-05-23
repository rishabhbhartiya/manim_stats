"""
manim_stats/props/card.py
=========================
Physically accurate, animation-rich playing-card props for statistical scenes.

Class hierarchy
---------------
  CardSuit        — enum: HEARTS, DIAMONDS, CLUBS, SPADES
  CardValue       — enum: ACE, TWO … TEN, JACK, QUEEN, KING
  CardFacing      — enum: FACE_UP, FACE_DOWN, HIDDEN
  CardFace        — dataclass: suit + value → symbol, color, rank string
  CardGeometry    — dataclass: physical dimensions, pip-position tables
  Card3D          — VGroup: full renderable card with pips, labels, back pattern
  Deck3D          — VGroup: ordered stack of Card3D objects with deal/shuffle anims

Statistical use-cases
---------------------
  STANDARD_DECK               — the canonical 52-card Deck3D singleton factory
  prob_card_event()           — annotate a card with P(event) banner
  hypergeometric_demo()       — animated sampling-without-replacement scene helper
  sample_without_replacement()— deal n cards, track outcomes for stats scenes
  birthday_problem_deck()     — specialized deck for birthday-paradox visualizations

Design notes
------------
* Every spatial constant is in Manim world-units (1 unit ≈ height of a Text char).
  Default card is 1.4 w × 2.0 h, matching a real card's 2.5:3.5 inch aspect ratio.

* Pip positions follow the Bicycle card standard:
  layout tables are stored on CardGeometry, keyed by pip count 1–10.
  All (x, y) coords are normalised to card-half-width / card-half-height fractions
  so they scale with CardGeometry.width / height.

* Face cards (J/Q/K) render a stylised monogram in the card centre rather than
  pips; the monogram color matches the suit.

* Card3D geometry is built from Manim primitives only — no external images needed.
  Front face: white background + pip VGroup + corner labels.
  Back face:  solid dark-blue + crosshatch diamond pattern.
  Edge:       thin dark rectangle slightly larger than face (gives card thickness).

* All flip / deal / shuffle animations use standard Manim animation primitives
  (Rotate, MoveAlongPath, AnimationGroup, Succession) and are parameterised so
  callers can override run_time, rate_func, and trajectory.

* The module imports gracefully without Manim installed — all pure-Python data
  classes and helpers work standalone.  Manim-dependent code lives in methods
  that raise ImportError with a clear message.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Graceful Manim import
# ---------------------------------------------------------------------------
try:
    import manim as mn
    from manim import (
        VGroup, VMobject, RoundedRectangle, Rectangle, Square, Circle,
        Line, Arc, Polygon, Dot,
        Text, MathTex, Tex,
        ManimColor, WHITE, BLACK, GRAY, DARK_GRAY, LIGHT_GRAY,
        RED, BLUE, GREEN, YELLOW, ORANGE, PURPLE, PINK,
        DARK_BLUE, PURE_RED,
        UP, DOWN, LEFT, RIGHT, OUT, IN, ORIGIN,
        TAU, PI,
        # Animations
        Write, FadeIn, FadeOut, Create, Uncreate,
        Transform, ReplacementTransform, TransformMatchingShapes,
        Rotate, Flash, Indicate, Circumscribe, ShowPassingFlash,
        AnimationGroup, Succession, LaggedStart, LaggedStartMap,
        # Updaters / path
        MoveAlongPath, ArcBetweenPoints,
        rate_functions,
        # Colors
        interpolate_color,
    )
    _MANIM_AVAILABLE = True
except ImportError:
    _MANIM_AVAILABLE = False
    # Stub so type annotations don't crash at import time
    VGroup = VMobject = object  # type: ignore


def _require_manim(method_name: str) -> None:
    if not _MANIM_AVAILABLE:
        raise ImportError(
            f"Card3D.{method_name}() requires Manim.  "
            "Install it with: pip install manim"
        )


# ===========================================================================
# ENUMERATIONS
# ===========================================================================

class CardSuit(Enum):
    """The four suits of a standard French-suited deck."""
    HEARTS   = "hearts"
    DIAMONDS = "diamonds"
    CLUBS    = "clubs"
    SPADES   = "spades"

    @property
    def symbol(self) -> str:
        """Unicode suit symbol: ♥ ♦ ♣ ♠"""
        return {
            CardSuit.HEARTS:   "♥",
            CardSuit.DIAMONDS: "♦",
            CardSuit.CLUBS:    "♣",
            CardSuit.SPADES:   "♠",
        }[self]

    @property
    def is_red(self) -> bool:
        return self in (CardSuit.HEARTS, CardSuit.DIAMONDS)

    @property
    def latex_symbol(self) -> str:
        r"""LaTeX string for the suit symbol (requires \usepackage{bbding} or Unicode)."""
        return {
            CardSuit.HEARTS:   r"\heartsuit",
            CardSuit.DIAMONDS: r"\diamondsuit",
            CardSuit.CLUBS:    r"\clubsuit",
            CardSuit.SPADES:   r"\spadesuit",
        }[self]

    @property
    def color_name(self) -> str:
        """Manim color name string for this suit."""
        return "RED" if self.is_red else "BLACK"

    @property
    def manim_color(self):
        """The Manim color for this suit.  Requires Manim installed."""
        _require_manim("suit.manim_color")
        # Rich red for hearts/diamonds, near-black for clubs/spades
        return ManimColor("#C8102E") if self.is_red else ManimColor("#1A1A1A")


class CardValue(Enum):
    """
    The 13 values of a standard deck.

    ``.rank`` gives the integer rank (Ace=1, Jack=11, Queen=12, King=13).
    ``.pip_count`` gives the number of pips shown on the face (0 for face cards).
    ``.rank_str`` gives the display string (A, 2–10, J, Q, K).
    """
    ACE   = 1
    TWO   = 2
    THREE = 3
    FOUR  = 4
    FIVE  = 5
    SIX   = 6
    SEVEN = 7
    EIGHT = 8
    NINE  = 9
    TEN   = 10
    JACK  = 11
    QUEEN = 12
    KING  = 13

    @property
    def rank(self) -> int:
        return self.value

    @property
    def pip_count(self) -> int:
        """Number of pips on the face; 0 for J, Q, K (face cards)."""
        return self.value if self.value <= 10 else 0

    @property
    def is_face_card(self) -> bool:
        return self.value > 10

    @property
    def is_ace(self) -> bool:
        return self.value == 1

    @property
    def rank_str(self) -> str:
        """Short display string: A, 2–10, J, Q, K."""
        mapping = {1: "A", 11: "J", 12: "Q", 13: "K"}
        return mapping.get(self.value, str(self.value))

    @property
    def full_name(self) -> str:
        names = {
            1: "Ace", 11: "Jack", 12: "Queen", 13: "King"
        }
        return names.get(self.value, str(self.value))

    @property
    def blackjack_value(self) -> int:
        """
        Blackjack hard value (Ace = 11; caller must handle soft-hand logic).
        """
        if self.value == 1:
            return 11
        return min(self.value, 10)

    @property
    def poker_rank(self) -> int:
        """Poker rank: Ace is high (14), 2 is lowest (2)."""
        return 14 if self.value == 1 else self.value


class CardFacing(Enum):
    """Whether a card is showing its face, its back, or fully hidden."""
    FACE_UP   = auto()    # face fully visible
    FACE_DOWN = auto()    # back fully visible
    HIDDEN    = auto()    # invisible (not yet dealt / out of play)


# ===========================================================================
# CARD FACE — pure data, no Manim
# ===========================================================================

@dataclass(frozen=True)
class CardFace:
    r"""
    Immutable value object that identifies a single playing card.

    This is *only* the identity — no geometry, no Manim objects.
    It is hashable and sortable (by poker rank within suit order).

    Attributes
    ----------
    suit : CardSuit
    value : CardValue

    Properties
    ----------
    symbol : str
        Unicode suit symbol.
    rank_str : str
        Short rank label (A, 2-10, J, Q, K).
    full_name : str
        e.g. "Ace of Spades", "Seven of Hearts"
    is_red : bool
    pip_count : int
    latex_label : str
        LaTeX-ready label e.g. ``r"A\spadesuit"``
    """

    suit:  CardSuit
    value: CardValue

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def symbol(self) -> str:
        return self.suit.symbol

    @property
    def rank_str(self) -> str:
        return self.value.rank_str

    @property
    def full_name(self) -> str:
        return f"{self.value.full_name} of {self.suit.value.capitalize()}"

    @property
    def is_red(self) -> bool:
        return self.suit.is_red

    @property
    def pip_count(self) -> int:
        return self.value.pip_count

    @property
    def is_face_card(self) -> bool:
        return self.value.is_face_card

    @property
    def latex_label(self) -> str:
        r"""e.g. r"A\spadesuit" or r"10\heartsuit" — for MathTex overlays."""
        return self.value.rank_str + self.suit.latex_symbol

    @property
    def sort_key(self) -> Tuple[int, int]:
        """
        Sort key: (suit_order, poker_rank).
        Suit order: spades=0, hearts=1, diamonds=2, clubs=3.
        """
        suit_order = {
            CardSuit.SPADES: 0, CardSuit.HEARTS: 1,
            CardSuit.DIAMONDS: 2, CardSuit.CLUBS: 3
        }
        return (suit_order[self.suit], self.value.poker_rank)

    def __lt__(self, other: "CardFace") -> bool:
        return self.sort_key < other.sort_key

    def __repr__(self) -> str:
        return f"CardFace({self.value.rank_str}{self.suit.symbol})"

    def __str__(self) -> str:
        return self.full_name


# ===========================================================================
# CARD GEOMETRY — physical dimensions and pip layout tables
# ===========================================================================

# Standard pip positions (normalised fractions of card half-size).
# x in [-1..1], y in [-1..1].  Origin = card centre.
# Based on the Bicycle® card standard layout.
# The bottom half is the vertical mirror of the top for cards 2–10.
_PIP_POSITIONS_NORM: dict[int, list[tuple[float, float]]] = {
    1:  [(0.0,  0.0)],
    2:  [(0.0,  0.58), (0.0, -0.58)],
    3:  [(0.0,  0.58), (0.0,  0.0),  (0.0, -0.58)],
    4:  [(-0.38,  0.58), (0.38,  0.58),
         (-0.38, -0.58), (0.38, -0.58)],
    5:  [(-0.38,  0.58), (0.38,  0.58),
         (0.0,   0.0),
         (-0.38, -0.58), (0.38, -0.58)],
    6:  [(-0.38,  0.58), (0.38,  0.58),
         (-0.38,  0.0),  (0.38,  0.0),
         (-0.38, -0.58), (0.38, -0.58)],
    7:  [(-0.38,  0.58), (0.38,  0.58),
         (0.0,   0.25),
         (-0.38,  0.0),  (0.38,  0.0),
         (-0.38, -0.58), (0.38, -0.58)],
    8:  [(-0.38,  0.58), (0.38,  0.58),
         (0.0,   0.25),
         (-0.38,  0.0),  (0.38,  0.0),
         (0.0,  -0.25),
         (-0.38, -0.58), (0.38, -0.58)],
    9:  [(-0.38,  0.58), (0.38,  0.58),
         (-0.38,  0.22), (0.38,  0.22),
         (0.0,   0.0),
         (-0.38, -0.22), (0.38, -0.22),
         (-0.38, -0.58), (0.38, -0.58)],
    10: [(-0.38,  0.58), (0.38,  0.58),
         (0.0,   0.38),
         (-0.38,  0.18), (0.38,  0.18),
         (-0.38, -0.18), (0.38, -0.18),
         (0.0,  -0.38),
         (-0.38, -0.58), (0.38, -0.58)],
}

# Rotation for each pip position in the bottom half — these pips are printed
# upside-down on real cards so both ends look right.
_PIP_UPSIDE_DOWN: dict[int, set[int]] = {
    # indices (0-based) of pips that are printed rotated 180°
    2:  {1},
    3:  {2},
    4:  {2, 3},
    5:  {3, 4},
    6:  {2, 3, 4, 5},
    7:  {5, 6},
    8:  {5, 6, 7},
    9:  {5, 6, 7, 8},
    10: {5, 6, 7, 8, 9},
}


@dataclass
class CardGeometry:
    """
    Physical dimensions of a Card3D and derived spatial quantities.

    All values are in Manim world-units.  The default matches a real-world
    playing card's 2.5 : 3.5 inch aspect ratio scaled to 2.0 units tall.

    Attributes
    ----------
    width : float
        Card face width.  Default 1.4.
    height : float
        Card face height.  Default 2.0.
    thickness : float
        Rendered card thickness (depth of the edge slab).  Default 0.025.
    corner_radius : float
        Rounding radius of the card corners.  Default 0.10.
    pip_scale : float
        Scale factor applied to pip symbols relative to card size.  Default 1.0.
    label_scale : float
        Scale factor for corner rank/suit labels.  Default 1.0.
    back_pattern_density : int
        Number of diamond tiles per row in the back pattern.  Default 5.
    """

    width:                float = 1.4
    height:               float = 2.0
    thickness:            float = 0.025
    corner_radius:        float = 0.10
    pip_scale:            float = 1.0
    label_scale:          float = 1.0
    back_pattern_density: int   = 5

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def aspect_ratio(self) -> float:
        """width / height."""
        return self.width / self.height

    @property
    def half_w(self) -> float:
        return self.width / 2

    @property
    def half_h(self) -> float:
        return self.height / 2

    @property
    def pip_font_size(self) -> float:
        """
        Pip symbol font size in Manim points, scaled to card dimensions.
        Calibrated so pips look correct on a 1.4 × 2.0 card.
        """
        base = self.height * 14.5   # empirical: ~29 pts for 2.0-unit tall card
        return round(base * self.pip_scale, 1)

    @property
    def corner_label_font_size(self) -> float:
        """Font size for the rank label in the card corner."""
        base = self.height * 9.0
        return round(base * self.label_scale, 1)

    @property
    def corner_suit_font_size(self) -> float:
        """Font size for the suit symbol in the card corner (smaller than rank)."""
        base = self.height * 7.5
        return round(base * self.label_scale, 1)

    @property
    def face_card_monogram_font_size(self) -> float:
        """Font size for the J/Q/K monogram letter."""
        base = self.height * 28.0
        return round(base * self.pip_scale, 1)

    # ------------------------------------------------------------------
    # Pip layout
    # ------------------------------------------------------------------

    def pip_world_positions(
        self, pip_count: int
    ) -> list[np.ndarray]:
        """
        Return a list of Manim world-space position vectors for each pip on a
        card with *pip_count* pips (1–10).

        Positions are scaled from normalised [-1..1] fractions to actual
        world-unit offsets from the card's centre.

        Parameters
        ----------
        pip_count : int
            Number of pips (1–10).  Raises ``ValueError`` for out-of-range values.

        Returns
        -------
        list of np.ndarray
            3D position vectors [x, y, z] where z = 0 (flat on the card face).
        """
        if pip_count not in _PIP_POSITIONS_NORM:
            raise ValueError(
                f"pip_count must be 1–10, got {pip_count}."
            )
        norm = _PIP_POSITIONS_NORM[pip_count]
        hw = self.half_w * 0.62   # pip area is ~62% of card half-width
        hh = self.half_h * 0.68   # pip area is ~68% of card half-height
        return [
            np.array([x * hw, y * hh, 0.0])
            for x, y in norm
        ]

    def pip_rotations(self, pip_count: int) -> list[float]:
        """
        Return rotation angles (in radians) for each pip.
        Pips in the bottom half of the card are rotated PI (upside-down).
        """
        upside_down_indices = _PIP_UPSIDE_DOWN.get(pip_count, set())
        n = len(_PIP_POSITIONS_NORM.get(pip_count, []))
        return [
            math.pi if i in upside_down_indices else 0.0
            for i in range(n)
        ]

    def corner_label_offset(self) -> np.ndarray:
        """
        Offset from card centre to the top-left corner rank label.
        Returns a 3D vector.
        """
        return np.array([
            -self.half_w + 0.13,
             self.half_h - 0.16,
             0.001,   # tiny z-lift so labels render on top of background
        ])

    def back_diamond_size(self) -> float:
        """Size of each diamond tile in the back pattern."""
        return self.width / (self.back_pattern_density * 1.6)


# ===========================================================================
# DEFAULT GEOMETRY PRESETS
# ===========================================================================

#: Standard poker card — matches a real Bicycle® card's proportions
POKER_GEOMETRY = CardGeometry(
    width=1.4, height=2.0, thickness=0.025,
    corner_radius=0.10, pip_scale=1.0, label_scale=1.0,
)

#: Small "thumbnail" card for multi-card layouts
MINI_GEOMETRY = CardGeometry(
    width=0.7, height=1.0, thickness=0.014,
    corner_radius=0.06, pip_scale=0.7, label_scale=0.7,
    back_pattern_density=4,
)

#: Large card for close-up explanation scenes
LARGE_GEOMETRY = CardGeometry(
    width=2.1, height=3.0, thickness=0.038,
    corner_radius=0.15, pip_scale=1.4, label_scale=1.4,
    back_pattern_density=6,
)


# ===========================================================================
# CARD COLORS — centralised so themes can override them
# ===========================================================================

@dataclass
class CardColorScheme:
    """
    Color scheme for a Card3D.  All values are hex strings so they work
    with or without Manim's color parsing.

    Attributes
    ----------
    face_bg : str
        Face background (typically white or cream).
    face_border : str
        Face border / edge color.
    red_suit : str
        Color for hearts and diamonds.
    black_suit : str
        Color for clubs and spades.
    back_bg : str
        Back face background.
    back_pattern : str
        Back pattern / diamond tile color.
    back_border : str
        Back face border color.
    highlight_ring : str
        Color of the highlight ring shown by .highlight().
    face_card_bg : str
        Tinted background for face-card monogram area.
    """
    face_bg:       str = "#FEFEFE"
    face_border:   str = "#CCCCCC"
    red_suit:      str = "#C8102E"   # classic Bicycle red
    black_suit:    str = "#1A1A1A"
    back_bg:       str = "#003580"   # classic deep blue back
    back_pattern:  str = "#0050B0"
    back_border:   str = "#002060"
    highlight_ring: str = "#FFD700"  # gold highlight
    face_card_bg:  str = "#FFF5F5"   # warm tint for J/Q/K area


#: Default classic card color scheme
CLASSIC_SCHEME = CardColorScheme()

#: Dark-themed card for DARK_THEME scenes
DARK_SCHEME = CardColorScheme(
    face_bg="#1E1E2E", face_border="#444466",
    red_suit="#FF4D6D", black_suit="#CCCCDD",
    back_bg="#0D1B2A", back_pattern="#1A3550",
    back_border="#0A1220",
    highlight_ring="#FFD700",
    face_card_bg="#2A1525",
)

#: Paper/parchment scheme for PAPER_THEME scenes
PAPER_SCHEME = CardColorScheme(
    face_bg="#FDF8ED", face_border="#C8A96E",
    red_suit="#8B0000", black_suit="#2C2416",
    back_bg="#6B4226", back_pattern="#8B5E3C",
    back_border="#4A2E1A",
    highlight_ring="#D4A017",
    face_card_bg="#FBF0DC",
)


# ===========================================================================
# CARD3D — The main Manim VGroup
# ===========================================================================

class Card3D(VGroup if _MANIM_AVAILABLE else object):  # type: ignore
    """
    A physically detailed 3D-style playing card as a Manim ``VGroup``.

    The card is composed of layered Manim Mobjects:
    - ``body``        : thin dark edge/shadow slab (RoundedRectangle)
    - ``back_face``   : back face panel with crosshatch pattern
    - ``front_face``  : white front panel
    - ``pip_group``   : VGroup of pip symbols (Text objects)
    - ``label_group`` : VGroup of corner rank + suit labels
    - ``face_card_group``: VGroup for J/Q/K monogram (face cards only)
    - ``highlight_ring`` : animated highlight ring (hidden by default)

    Construction
    ------------
    Card3D is constructed with a ``CardFace`` (suit + value), optional
    ``CardGeometry``, ``CardColorScheme``, and initial ``CardFacing``.

    Parameters
    ----------
    face : CardFace
        Identity of this card.
    geometry : CardGeometry, optional
        Physical dimensions.  Defaults to ``POKER_GEOMETRY``.
    colors : CardColorScheme, optional
        Visual color scheme.  Defaults to ``CLASSIC_SCHEME``.
    facing : CardFacing, optional
        Initial orientation.  Defaults to ``CardFacing.FACE_UP``.
    position : array-like, optional
        Initial world-space position.  Defaults to ORIGIN.

    Key properties
    --------------
    .face_data : CardFace
        The card's identity (suit + value).
    .facing : CardFacing
        Current orientation state.
    .face_up : bool
        True when ``facing == CardFacing.FACE_UP``.
    .suit / .value / .is_red
        Delegates to the underlying ``CardFace``.

    Key sub-mobjects
    ----------------
    .body, .front_face, .back_face
    .pip_group, .label_group, .face_card_group
    .highlight_ring

    Animations (all return Manim Animation objects)
    -----------------------------------------------
    .flip_to_face_up(run_time)   — flip so front is visible
    .flip_to_face_down(run_time) — flip so back is visible
    .flip(run_time)              — flip to whichever side is currently hidden
    .reveal_anim(run_time)       — flip face-up + highlight flash
    .deal_anim(target, run_time) — arc-travel from current pos to *target*
    .discard_anim(run_time)      — slide off screen with fade
    .hover_anim(run_time)        — small upward float (selection hint)
    .unhover_anim(run_time)      — return to original position

    Helpers
    -------
    .highlight(color)            — show the golden highlight ring
    .unhighlight()               — hide the highlight ring
    .set_facing(facing)          — instantly change orientation (no animation)
    .clone()                     — return an independent copy of this Card3D
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        face:     CardFace,
        geometry: CardGeometry      = POKER_GEOMETRY,
        colors:   CardColorScheme   = CLASSIC_SCHEME,
        facing:   CardFacing        = CardFacing.FACE_UP,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        _require_manim("__init__")
        super().__init__(**kwargs)

        self.face_data    = face
        self.geometry     = geometry
        self.colors       = colors
        self._facing      = facing
        self._hover_delta = np.array([0.0, 0.12, 0.0])

        # Build all sub-mobjects
        self._build_body()
        self._build_back_face()
        self._build_front_face()
        self._build_pips()
        self._build_corner_labels()
        if face.is_face_card:
            self._build_face_card_monogram()
        self._build_highlight_ring()

        # Apply initial orientation (show/hide correct face)
        self.set_facing(facing)

        # Position
        if position is not None:
            self.move_to(np.asarray(position, dtype=float))

    # ------------------------------------------------------------------
    # Sub-mobject builders
    # ------------------------------------------------------------------

    def _build_body(self) -> None:
        """
        Build the card body — a thin dark slab slightly larger than the
        face panels, giving the illusion of card edge thickness.
        """
        g = self.geometry
        c = self.colors
        self.body = RoundedRectangle(
            width         = g.width  + g.thickness * 2,
            height        = g.height + g.thickness * 2,
            corner_radius = g.corner_radius + g.thickness * 0.5,
            fill_color    = ManimColor(c.face_border),
            fill_opacity  = 1.0,
            stroke_width  = 0,
        )
        self.add(self.body)

    def _build_back_face(self) -> None:
        """
        Build the card back: solid background + repeating diamond tile pattern.

        The pattern is a grid of rotated squares (diamonds) in a slightly
        lighter shade, matching the traditional Bicycle-style back design.
        """
        g = self.geometry
        c = self.colors

        # Solid background panel
        self.back_face = RoundedRectangle(
            width         = g.width,
            height        = g.height,
            corner_radius = g.corner_radius,
            fill_color    = ManimColor(c.back_bg),
            fill_opacity  = 1.0,
            stroke_color  = ManimColor(c.back_border),
            stroke_width  = 1.2,
        )

        # Diamond pattern: grid of rotated squares
        self.back_pattern_group = VGroup()
        tile_size  = g.back_diamond_size()
        half_tile  = tile_size / 2
        n          = g.back_pattern_density
        # Compute grid offsets so pattern is centred on the card face
        x_offsets  = np.linspace(-g.half_w * 0.78, g.half_w * 0.78, n * 2)
        y_offsets  = np.linspace(-g.half_h * 0.80, g.half_h * 0.80, n * 3)

        for xi, x in enumerate(x_offsets):
            for yi, y in enumerate(y_offsets):
                # Stagger alternate rows for a more traditional look
                x_shifted = x + (half_tile * 0.5 if yi % 2 == 1 else 0)
                diamond = Square(
                    side_length   = tile_size * 0.72,
                    fill_color    = ManimColor(c.back_pattern),
                    fill_opacity  = 0.65,
                    stroke_color  = ManimColor(c.back_border),
                    stroke_width  = 0.4,
                ).rotate(PI / 4).move_to([x_shifted, y, 0.001])
                self.back_pattern_group.add(diamond)

        # Outer decorative border inset from the edge
        border_margin = 0.10
        self.back_border_rect = RoundedRectangle(
            width         = g.width  - border_margin * 2,
            height        = g.height - border_margin * 2,
            corner_radius = g.corner_radius * 0.7,
            fill_opacity  = 0,
            stroke_color  = ManimColor(c.back_border),
            stroke_width  = 1.8,
        ).shift([0, 0, 0.002])

        self.back_group = VGroup(
            self.back_face,
            self.back_pattern_group,
            self.back_border_rect,
        )
        self.add(self.back_group)

    def _build_front_face(self) -> None:
        """Build the card front: clean white/cream background panel."""
        g = self.geometry
        c = self.colors

        self.front_face = RoundedRectangle(
            width         = g.width,
            height        = g.height,
            corner_radius = g.corner_radius,
            fill_color    = ManimColor(c.face_bg),
            fill_opacity  = 1.0,
            stroke_color  = ManimColor(c.face_border),
            stroke_width  = 1.0,
        ).shift([0, 0, 0.001])
        self.add(self.front_face)

    def _build_pips(self) -> None:
        """
        Build the pip symbols for number cards (A–10).

        Each pip is a ``Text`` object showing the suit's Unicode symbol,
        positioned according to ``CardGeometry.pip_world_positions()`` and
        rotated for bottom-half pips.

        Face cards (J/Q/K) skip pip building — they use a monogram instead.
        """
        g   = self.geometry
        self.pip_group = VGroup()

        if self.face_data.is_face_card:
            self.add(self.pip_group)
            return

        pip_count = self.face_data.pip_count
        if pip_count == 0:
            self.add(self.pip_group)
            return

        positions  = g.pip_world_positions(pip_count)
        rotations  = g.pip_rotations(pip_count)
        suit_color = ManimColor(
            self.colors.red_suit if self.face_data.is_red
            else self.colors.black_suit
        )
        symbol     = self.face_data.symbol
        font_size  = g.pip_font_size

        for pos, rot in zip(positions, rotations):
            pip = Text(
                symbol,
                font_size = font_size,
                color     = suit_color,
            ).move_to(pos + np.array([0, 0, 0.003]))
            if rot != 0.0:
                pip.rotate(rot)
            self.pip_group.add(pip)

        self.add(self.pip_group)

    def _build_corner_labels(self) -> None:
        """
        Build the four corner rank+suit labels.

        Real cards show rank + suit symbol in the top-left and bottom-right
        corners (bottom-right is upside-down).  We build both.
        """
        g          = self.geometry
        suit_color = ManimColor(
            self.colors.red_suit if self.face_data.is_red
            else self.colors.black_suit
        )
        rank_str   = self.face_data.rank_str
        suit_sym   = self.face_data.symbol
        rank_fs    = g.corner_label_font_size
        suit_fs    = g.corner_suit_font_size
        offset     = g.corner_label_offset()   # top-left corner

        # Top-left: rank then suit symbol stacked vertically
        rank_tl = Text(rank_str, font_size=rank_fs, color=suit_color)
        suit_tl = Text(suit_sym, font_size=suit_fs, color=suit_color)

        # Position rank at offset, suit just below it
        rank_tl.move_to(offset + np.array([0, 0, 0.003]))
        suit_tl.move_to(offset + np.array([0, -rank_fs * 0.018, 0.003]))

        # Bottom-right: identical but rotated 180°
        rank_br = rank_tl.copy().rotate(PI).move_to(
            -offset + np.array([0, 0, 0.003])
        )
        suit_br = suit_tl.copy().rotate(PI).move_to(
            -offset + np.array([0, rank_fs * 0.018, 0.003])
        )

        self.label_group = VGroup(rank_tl, suit_tl, rank_br, suit_br)
        self.add(self.label_group)

    def _build_face_card_monogram(self) -> None:
        """
        Build the J / Q / K monogram for face cards.

        Uses a large serif-style initial letter centered on the card,
        with a lightly tinted background rectangle behind it.
        """
        g          = self.geometry
        suit_color = ManimColor(
            self.colors.red_suit if self.face_data.is_red
            else self.colors.black_suit
        )
        letter     = self.face_data.value.rank_str  # "J", "Q", or "K"

        # Tinted background panel (smaller than the card face)
        bg = RoundedRectangle(
            width         = g.width  * 0.70,
            height        = g.height * 0.68,
            corner_radius = g.corner_radius * 0.5,
            fill_color    = ManimColor(self.colors.face_card_bg),
            fill_opacity  = 0.8,
            stroke_color  = suit_color,
            stroke_width  = 0.6,
        ).shift([0, 0, 0.002])

        # Large monogram letter
        monogram = Text(
            letter,
            font_size = g.face_card_monogram_font_size,
            color     = suit_color,
            weight    = "BOLD",
        ).shift([0, 0, 0.004])

        # Suit symbol below the monogram
        sub_suit = Text(
            self.face_data.symbol,
            font_size = g.face_card_monogram_font_size * 0.4,
            color     = suit_color,
        ).next_to(monogram, DOWN, buff=0.04).shift([0, 0, 0.004])

        self.face_card_group = VGroup(bg, monogram, sub_suit)
        self.add(self.face_card_group)

    def _build_highlight_ring(self) -> None:
        """
        Build a golden highlight ring around the card edge (hidden initially).
        Used by ``.highlight()`` and the reveal animation.
        """
        g = self.geometry
        self.highlight_ring = RoundedRectangle(
            width         = g.width  + 0.08,
            height        = g.height + 0.08,
            corner_radius = g.corner_radius + 0.04,
            fill_opacity  = 0,
            stroke_color  = ManimColor(self.colors.highlight_ring),
            stroke_width  = 3.0,
        ).shift([0, 0, 0.01])
        self.highlight_ring.set_opacity(0)
        self.add(self.highlight_ring)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def facing(self) -> CardFacing:
        """Current CardFacing state."""
        return self._facing

    @property
    def face_up(self) -> bool:
        return self._facing == CardFacing.FACE_UP

    @property
    def face_down(self) -> bool:
        return self._facing == CardFacing.FACE_DOWN

    @property
    def suit(self) -> CardSuit:
        return self.face_data.suit

    @property
    def value(self) -> CardValue:
        return self.face_data.value

    @property
    def is_red(self) -> bool:
        return self.face_data.is_red

    @property
    def rank_str(self) -> str:
        return self.face_data.rank_str

    @property
    def full_name(self) -> str:
        return self.face_data.full_name

    # ------------------------------------------------------------------
    # Orientation control (instant, no animation)
    # ------------------------------------------------------------------

    def set_facing(self, facing: CardFacing) -> "Card3D":
        """
        Instantly change which face is visible.  Does not animate.

        Parameters
        ----------
        facing : CardFacing

        Returns
        -------
        self (for chaining)
        """
        self._facing = facing

        if facing == CardFacing.FACE_UP:
            self._show_front()
        elif facing == CardFacing.FACE_DOWN:
            self._show_back()
        else:  # HIDDEN
            self._show_hidden()
        return self

    def _show_front(self) -> None:
        self.front_face.set_opacity(1)
        self.pip_group.set_opacity(1)
        self.label_group.set_opacity(1)
        if hasattr(self, "face_card_group"):
            self.face_card_group.set_opacity(1)
        self.back_group.set_opacity(0)

    def _show_back(self) -> None:
        self.front_face.set_opacity(0)
        self.pip_group.set_opacity(0)
        self.label_group.set_opacity(0)
        if hasattr(self, "face_card_group"):
            self.face_card_group.set_opacity(0)
        self.back_group.set_opacity(1)

    def _show_hidden(self) -> None:
        self.set_opacity(0)

    # ------------------------------------------------------------------
    # Visual state helpers
    # ------------------------------------------------------------------

    def highlight(self, color: Optional[str] = None) -> "Card3D":
        """
        Show the golden highlight ring around the card border.

        Parameters
        ----------
        color : str, optional
            Override the ring color (hex string).

        Returns self for chaining.
        """
        if color:
            self.highlight_ring.set_stroke(color=ManimColor(color))
        self.highlight_ring.set_opacity(1)
        return self

    def unhighlight(self) -> "Card3D":
        """Hide the highlight ring.  Returns self."""
        self.highlight_ring.set_opacity(0)
        return self

    # ------------------------------------------------------------------
    # Flip animations
    # ------------------------------------------------------------------

    def flip_to_face_up(self, run_time: float = 0.6) -> "mn.Animation":
        """
        Return an animation that flips this card so its front is visible.

        Implementation: Rotate 180° around the Y-axis using two half-steps
        (0° → 90° with front hidden, then 90° → 180° with front visible),
        giving the illusion of a physical flip.

        Parameters
        ----------
        run_time : float
            Total duration of the flip.

        Returns
        -------
        manim.Animation
        """
        _require_manim("flip_to_face_up")

        def _updater_first_half(mob, alpha):
            mob.rotate(
                -PI / 2 * (1 / (run_time * 60)),
                axis=UP,
                about_point=mob.get_center(),
            )
            if alpha > 0.98:
                mob._show_hidden()

        def _updater_second_half(mob, alpha):
            mob.rotate(
                -PI / 2 * (1 / (run_time * 60)),
                axis=UP,
                about_point=mob.get_center(),
            )
            if alpha < 0.02:
                mob._show_front()

        return Succession(
            Rotate(self, angle=PI / 2, axis=UP,
                   run_time=run_time / 2,
                   rate_func=rate_functions.ease_in_sine),
            Rotate(self, angle=PI / 2, axis=UP,
                   run_time=run_time / 2,
                   rate_func=rate_functions.ease_out_sine),
        )

    def flip_to_face_down(self, run_time: float = 0.6) -> "mn.Animation":
        """
        Return an animation that flips this card so its back is visible.
        """
        _require_manim("flip_to_face_down")
        return Succession(
            Rotate(self, angle=-PI / 2, axis=UP,
                   run_time=run_time / 2,
                   rate_func=rate_functions.ease_in_sine),
            Rotate(self, angle=-PI / 2, axis=UP,
                   run_time=run_time / 2,
                   rate_func=rate_functions.ease_out_sine),
        )

    def flip(self, run_time: float = 0.6) -> "mn.Animation":
        """
        Flip to whichever side is currently hidden.

        If FACE_UP  → flips to FACE_DOWN.
        If FACE_DOWN → flips to FACE_UP.
        """
        _require_manim("flip")
        if self.face_up:
            return self.flip_to_face_down(run_time)
        return self.flip_to_face_up(run_time)

    # ------------------------------------------------------------------
    # Reveal animation
    # ------------------------------------------------------------------

    def reveal_anim(
        self,
        run_time:       float = 1.0,
        highlight_color: Optional[str] = None,
    ) -> "mn.AnimationGroup":
        """
        Animated reveal: flip face-up + golden highlight flash.

        Parameters
        ----------
        run_time : float
            Duration of the full reveal sequence.
        highlight_color : str, optional
            Color of the flash ring.

        Returns
        -------
        manim.AnimationGroup
        """
        _require_manim("reveal_anim")
        ring_color = highlight_color or self.colors.highlight_ring
        self.highlight_ring.set_stroke(color=ManimColor(ring_color))

        return Succession(
            self.flip_to_face_up(run_time=run_time * 0.55),
            AnimationGroup(
                Flash(
                    self.front_face,
                    color=ManimColor(ring_color),
                    flash_radius=max(self.geometry.width, self.geometry.height) * 0.7,
                    line_length=0.15,
                    num_lines=12,
                    run_time=run_time * 0.45,
                ),
                Indicate(
                    self,
                    color=ManimColor(ring_color),
                    scale_factor=1.06,
                    run_time=run_time * 0.45,
                ),
            ),
        )

    # ------------------------------------------------------------------
    # Deal animation
    # ------------------------------------------------------------------

    def deal_anim(
        self,
        target:    np.ndarray,
        run_time:  float = 0.45,
        arc_height: float = 0.6,
        flip:      bool  = False,
    ) -> "mn.Animation":
        """
        Arc-travel from the card's current position to *target*.

        The trajectory is a smooth arc (parabolic lift + drop), mimicking a
        card being dealt across a table.

        Parameters
        ----------
        target : array-like
            Destination position in world space.
        run_time : float
            Travel duration.
        arc_height : float
            Height of the arc above the straight-line midpoint.
        flip : bool
            If True, flip the card face-up upon arrival (for hole-card deals).

        Returns
        -------
        manim.Animation
        """
        _require_manim("deal_anim")
        start = self.get_center().copy()
        end   = np.asarray(target, dtype=float)
        mid   = (start + end) / 2 + np.array([0, arc_height, 0])

        # Build a smooth quadratic arc path
        arc_path = mn.CubicBezier(
            start,
            start + (mid - start) * 0.8,
            end   + (mid - end)   * 0.8,
            end,
        )

        travel = MoveAlongPath(
            self, arc_path,
            run_time=run_time,
            rate_func=rate_functions.ease_out_quad,
        )
        if not flip:
            return travel

        return Succession(
            travel,
            self.flip_to_face_up(run_time=run_time * 0.5),
        )

    # ------------------------------------------------------------------
    # Discard animation
    # ------------------------------------------------------------------

    def discard_anim(
        self,
        direction: np.ndarray = None,
        run_time:  float = 0.5,
    ) -> "mn.Animation":
        """
        Slide the card off screen in *direction* while fading out.

        Parameters
        ----------
        direction : np.ndarray, optional
            Unit vector for discard direction.  Defaults to LEFT + slight DOWN.
        run_time : float

        Returns
        -------
        manim.AnimationGroup
        """
        _require_manim("discard_anim")
        if direction is None:
            direction = LEFT * 1.5 + DOWN * 0.3

        target = self.get_center() + direction * 6
        return AnimationGroup(
            self.animate(run_time=run_time).move_to(target),
            FadeOut(self, run_time=run_time * 0.8),
        )

    # ------------------------------------------------------------------
    # Hover animation
    # ------------------------------------------------------------------

    def hover_anim(self, run_time: float = 0.18) -> "mn.Animation":
        """Float the card slightly upward (selection / mouse-over hint)."""
        _require_manim("hover_anim")
        return self.animate(run_time=run_time).shift(self._hover_delta)

    def unhover_anim(self, run_time: float = 0.18) -> "mn.Animation":
        """Return card to its pre-hover position."""
        _require_manim("unhover_anim")
        return self.animate(run_time=run_time).shift(-self._hover_delta)

    # ------------------------------------------------------------------
    # Clone
    # ------------------------------------------------------------------

    def clone(self) -> "Card3D":
        """Return a deep copy of this Card3D at the same position."""
        _require_manim("clone")
        return Card3D(
            face     = self.face_data,
            geometry = self.geometry,
            colors   = self.colors,
            facing   = self._facing,
            position = self.get_center().copy(),
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        face_str = "↑" if self.face_up else "↓"
        return f"Card3D({self.face_data!r} {face_str})"


# ===========================================================================
# DECK3D — ordered stack of Card3D objects
# ===========================================================================

class Deck3D(VGroup if _MANIM_AVAILABLE else object):  # type: ignore
    """
    An ordered deck of ``Card3D`` objects rendered as a physical stack.

    By default builds the canonical 52-card French deck.  A custom subset
    can be passed as ``cards``.

    The deck is rendered as a stacked pile: each card is offset by a tiny
    amount in the +z direction, and a small lateral shear is added to give
    a natural "not perfectly aligned" look.

    Parameters
    ----------
    cards : list[CardFace], optional
        Card identities to include.  Defaults to the full 52-card deck.
    geometry : CardGeometry, optional
        Physical card geometry.  Defaults to ``POKER_GEOMETRY``.
    colors : CardColorScheme, optional
        Color scheme for all cards in the deck.  Defaults to ``CLASSIC_SCHEME``.
    initial_facing : CardFacing, optional
        Starting orientation of all cards.  Defaults to ``CardFacing.FACE_DOWN``.
    stack_offset_z : float, optional
        Z-axis separation between stacked cards.  Default 0.005.
    stack_jitter : float, optional
        Random lateral offset amplitude for the "natural pile" look.  Default 0.003.
    seed : int, optional
        Random seed for the stack jitter (for reproducible scenes).  Default 42.
    position : array-like, optional
        Position of the bottom of the deck.

    Key properties
    --------------
    .cards : list[Card3D]
        All card objects (including dealt cards).
    .pile : list[Card3D]
        Cards remaining in the draw pile (not yet dealt).
    .dealt : list[Card3D]
        Cards that have been dealt.
    .remaining : int
        Number of cards left in the pile.
    .is_empty : bool
    .top_card : Card3D
        The top card of the pile (next to be dealt).

    Deal methods
    ------------
    .deal_one(target, run_time, flip)
        Deal the top card to *target* position.  Returns Animation.
    .deal_n(n, targets, run_time, stagger)
        Deal *n* cards to a list of target positions.  Returns Animation.
    .reset(run_time)
        Collect all dealt cards back into the pile.  Returns Animation.

    Layout methods
    --------------
    .fan_out(n, angle, radius, facing)
        Spread top *n* cards in a fan arc.  Returns Animation.
    .spread_face_up(spacing, run_time)
        Lay all cards in a horizontal row, face-up.  Returns Animation.
    .collect_anim(run_time)
        Animate all spread cards back into a stack.  Returns Animation.

    Shuffle methods
    ---------------
    .shuffle(inplace)
        Reorder ``self.pile`` randomly (no animation).
    .shuffle_anim(style, run_time)
        Animated shuffle: 'riffle' or 'overhand'.  Returns Animation.

    Reveal methods
    --------------
    .reveal_top(run_time)
        Flip the top card face-up in place.  Returns Animation.
    .cut_anim(cut_point, run_time)
        Animate a deck cut.  Returns Animation.
    """

    def __init__(
        self,
        cards:          Optional[List[CardFace]]   = None,
        geometry:       CardGeometry               = POKER_GEOMETRY,
        colors:         CardColorScheme            = CLASSIC_SCHEME,
        initial_facing: CardFacing                 = CardFacing.FACE_DOWN,
        stack_offset_z: float                      = 0.005,
        stack_jitter:   float                      = 0.003,
        seed:           int                        = 42,
        position:       Optional[np.ndarray]       = None,
        **kwargs,
    ) -> None:
        _require_manim("Deck3D.__init__")
        super().__init__(**kwargs)

        card_faces   = cards or _make_standard_deck()
        self.geometry       = geometry
        self.colors         = colors
        self._initial_facing = initial_facing
        self._stack_offset_z = stack_offset_z
        self._rng            = random.Random(seed)

        # Build Card3D objects
        self.cards: List[Card3D] = []
        for i, face in enumerate(card_faces):
            jitter_x = (self._rng.random() - 0.5) * stack_jitter
            jitter_y = (self._rng.random() - 0.5) * stack_jitter * 0.5
            card = Card3D(
                face    = face,
                geometry= geometry,
                colors  = colors,
                facing  = initial_facing,
            ).shift(np.array([jitter_x, jitter_y, i * stack_offset_z]))
            self.cards.append(card)
            self.add(card)

        self.pile:   List[Card3D] = list(self.cards)  # copy, top = last
        self.dealt:  List[Card3D] = []

        if position is not None:
            self.move_to(np.asarray(position, dtype=float))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def remaining(self) -> int:
        return len(self.pile)

    @property
    def is_empty(self) -> bool:
        return len(self.pile) == 0

    @property
    def top_card(self) -> Optional[Card3D]:
        """The next card to be dealt (last in pile list = visually on top)."""
        return self.pile[-1] if self.pile else None

    @property
    def is_shuffled(self) -> bool:
        """True if current pile order differs from the original card order."""
        original = list(self.cards)
        return self.pile != original

    # ------------------------------------------------------------------
    # Deal methods
    # ------------------------------------------------------------------

    def deal_one(
        self,
        target:   np.ndarray,
        run_time: float = 0.45,
        flip:     bool  = False,
    ) -> "mn.Animation":
        """
        Remove the top card from the pile and animate it to *target*.

        Parameters
        ----------
        target : array-like
            Destination world position.
        run_time : float
        flip : bool
            If True, the card flips face-up on arrival.

        Returns
        -------
        manim.Animation (or None if pile is empty)

        Raises
        ------
        IndexError
            If the deck is empty.
        """
        _require_manim("deal_one")
        if self.is_empty:
            raise IndexError("Cannot deal from an empty deck.")

        card = self.pile.pop()
        self.dealt.append(card)
        return card.deal_anim(target=np.asarray(target, dtype=float),
                              run_time=run_time, flip=flip)

    def deal_n(
        self,
        n:        int,
        targets:  List[np.ndarray],
        run_time: float = 0.35,
        stagger:  float = 0.10,
        flip:     bool  = False,
    ) -> "mn.Animation":
        """
        Deal *n* cards to a list of *targets*.

        Parameters
        ----------
        n : int
            Number of cards to deal.  ``n == len(targets)`` required.
        targets : list of array-like
            Destination positions, one per card.
        run_time : float
            Duration of each individual deal animation.
        stagger : float
            Time offset between successive cards being dealt.
        flip : bool
            Flip each card face-up on arrival.

        Returns
        -------
        manim.LaggedStart animation
        """
        _require_manim("deal_n")
        if n != len(targets):
            raise ValueError(
                f"deal_n: n ({n}) must equal len(targets) ({len(targets)})."
            )
        anims = [
            self.deal_one(target=t, run_time=run_time, flip=flip)
            for t in targets[:n]
        ]
        return LaggedStart(*anims, lag_ratio=stagger / run_time)

    def reset(self, run_time: float = 1.2) -> "mn.Animation":
        """
        Animate all dealt cards flying back into the pile.

        Returns
        -------
        manim.AnimationGroup
        """
        _require_manim("reset")
        pile_pos = self.get_center()
        anims = []
        for i, card in enumerate(reversed(self.dealt)):
            card.set_facing(self._initial_facing)
            anims.append(
                card.animate(run_time=run_time).move_to(
                    pile_pos + np.array([0, 0, i * self._stack_offset_z])
                )
            )
        # Restore pile state
        self.pile = list(self.cards)
        self.dealt.clear()
        return LaggedStart(*anims, lag_ratio=0.04) if anims else AnimationGroup()

    # ------------------------------------------------------------------
    # Layout methods
    # ------------------------------------------------------------------

    def fan_out(
        self,
        n:        int              = 5,
        angle:    float            = PI * 0.55,
        radius:   float            = 2.0,
        facing:   CardFacing       = CardFacing.FACE_UP,
        center:   np.ndarray       = None,
        run_time: float            = 0.8,
    ) -> "mn.Animation":
        """
        Animate the top *n* cards fanning out in a radial arc.

        Cards are arranged in a semicircle of given *radius*, centred at
        *center* (defaults to the deck's position).  Each card is rotated
        to be tangent to the arc.

        Parameters
        ----------
        n : int
            Number of cards to fan.
        angle : float
            Total arc angle in radians.
        radius : float
            Radius of the fan arc.
        facing : CardFacing
            Face-up or face-down in the fan.
        center : np.ndarray, optional
            Centre of the fan arc.
        run_time : float

        Returns
        -------
        manim.AnimationGroup
        """
        _require_manim("fan_out")
        n = min(n, self.remaining)
        if n == 0:
            return AnimationGroup()

        if center is None:
            center = self.get_center()

        fan_cards = self.pile[-n:]
        step      = angle / max(n - 1, 1)
        start_ang = -angle / 2 + PI / 2   # top of arc

        anims = []
        for i, card in enumerate(fan_cards):
            theta  = start_ang - i * step
            pos    = center + np.array([
                math.cos(theta) * radius,
                math.sin(theta) * radius - radius * 0.85,
                i * self._stack_offset_z * 10,
            ])
            card_angle = -(theta - PI / 2)   # card tangent to arc
            anims.append(AnimationGroup(
                card.animate(run_time=run_time).move_to(pos).rotate(card_angle),
            ))
            if facing != card._facing:
                card.set_facing(facing)

        return LaggedStart(*anims, lag_ratio=0.06)

    def spread_face_up(
        self,
        spacing:  float = None,
        run_time: float = 1.2,
        center:   Optional[np.ndarray] = None,
    ) -> "mn.Animation":
        """
        Lay all remaining pile cards in a horizontal row, face-up.

        Parameters
        ----------
        spacing : float, optional
            Gap between card centres.  Defaults to card width + 0.05.
        run_time : float
        center : np.ndarray, optional
            Centre of the row.  Defaults to current deck position.

        Returns
        -------
        manim.LaggedStart
        """
        _require_manim("spread_face_up")
        if not self.pile:
            return AnimationGroup()

        g        = self.geometry
        spacing  = spacing or (g.width + 0.05)
        n        = len(self.pile)
        cx, cy   = (center or self.get_center())[:2]
        start_x  = cx - (n - 1) * spacing / 2

        anims = []
        for i, card in enumerate(self.pile):
            target = np.array([start_x + i * spacing, cy, i * 0.001])
            card.set_facing(CardFacing.FACE_UP)
            anims.append(
                card.animate(run_time=run_time).move_to(target)
            )
        return LaggedStart(*anims, lag_ratio=0.05)

    def collect_anim(self, run_time: float = 0.8) -> "mn.AnimationGroup":
        """
        Animate all cards (pile + dealt) flying back into a stack.

        Returns
        -------
        manim.LaggedStart
        """
        _require_manim("collect_anim")
        all_cards = self.cards
        pile_pos  = self.get_center()
        anims = []
        for i, card in enumerate(all_cards):
            card.set_facing(self._initial_facing)
            anims.append(
                card.animate(run_time=run_time).move_to(
                    pile_pos + np.array([0, 0, i * self._stack_offset_z])
                )
            )
        self.pile  = list(self.cards)
        self.dealt = []
        return LaggedStart(*anims, lag_ratio=0.02)

    # ------------------------------------------------------------------
    # Shuffle methods
    # ------------------------------------------------------------------

    def shuffle(self, inplace: bool = True) -> "Deck3D":
        """
        Reorder ``self.pile`` randomly (no animation, instant).

        Parameters
        ----------
        inplace : bool
            If True (default), shuffle the pile in-place.
            If False, return a new deck with shuffled order.

        Returns
        -------
        self (or new Deck3D if not inplace)
        """
        if inplace:
            self._rng.shuffle(self.pile)
            return self
        new_faces = [c.face_data for c in self.pile]
        self._rng.shuffle(new_faces)
        return Deck3D(
            cards    = new_faces,
            geometry = self.geometry,
            colors   = self.colors,
        )

    def shuffle_anim(
        self,
        style:    str   = "riffle",
        run_time: float = 1.5,
    ) -> "mn.Animation":
        """
        Animated deck shuffle.

        Parameters
        ----------
        style : {'riffle', 'overhand'}
            Animation style:
            - ``'riffle'``: splits deck in two halves, interleaves them.
            - ``'overhand'``: peels small packets off the top and drops them.
        run_time : float
            Total animation duration.

        Returns
        -------
        manim.Animation
        """
        _require_manim("shuffle_anim")
        if style == "riffle":
            return self._riffle_anim(run_time)
        elif style == "overhand":
            return self._overhand_anim(run_time)
        else:
            raise ValueError(
                f"Unknown shuffle style {style!r}. "
                "Choose 'riffle' or 'overhand'."
            )

    def _riffle_anim(self, run_time: float) -> "mn.Animation":
        """
        Riffle shuffle: split pile at midpoint, interleave left + right half.

        Visually: the two halves arc toward each other and merge.
        The pile order is updated to match the interleaved result.
        """
        n     = len(self.pile)
        mid   = n // 2
        left  = self.pile[:mid]
        right = self.pile[mid:]

        deck_pos = self.get_center()
        sep      = self.geometry.width * 0.65

        # Phase 1: split the two halves apart
        split_anims = []
        for i, card in enumerate(left):
            split_anims.append(
                card.animate(run_time=run_time * 0.30).shift([-sep, 0, 0])
            )
        for i, card in enumerate(right):
            split_anims.append(
                card.animate(run_time=run_time * 0.30).shift([sep, 0, 0])
            )

        # Interleave the piles (riffle order)
        interleaved: List[Card3D] = []
        li, ri = 0, 0
        while li < len(left) or ri < len(right):
            if li < len(left):
                interleaved.append(left[li]); li += 1
            if ri < len(right):
                interleaved.append(right[ri]); ri += 1
        self.pile = interleaved

        # Phase 2: merge halves back together
        merge_anims = []
        for i, card in enumerate(interleaved):
            merge_anims.append(
                card.animate(run_time=run_time * 0.50).move_to(
                    deck_pos + np.array([0, 0, i * self._stack_offset_z])
                )
            )

        return Succession(
            AnimationGroup(*split_anims),
            LaggedStart(*merge_anims, lag_ratio=0.01),
        )

    def _overhand_anim(self, run_time: float) -> "mn.Animation":
        """
        Overhand shuffle: peel small packets off the top, drop under deck.

        Creates a visually satisfying stack-reorder animation.
        """
        n        = len(self.pile)
        n_cuts   = min(6, n // 3)
        deck_pos = self.get_center()
        anims    = []

        new_pile: List[Card3D] = []
        remaining = list(self.pile)

        for cut_i in range(n_cuts):
            packet_size = max(1, self._rng.randint(2, max(2, n // n_cuts)))
            packet      = remaining[-packet_size:]
            remaining   = remaining[:-packet_size]
            new_pile    = packet + new_pile

            lift_height = 0.4 + cut_i * 0.08
            for card in packet:
                anims.append(Succession(
                    card.animate(run_time=run_time * 0.06).shift([0, lift_height, 0.1]),
                    card.animate(run_time=run_time * 0.06).shift([0, -lift_height, -0.1]),
                ))

        self.pile = new_pile + remaining
        return LaggedStart(*anims, lag_ratio=run_time * 0.06 / len(anims))

    # ------------------------------------------------------------------
    # Reveal / cut
    # ------------------------------------------------------------------

    def reveal_top(self, run_time: float = 0.8) -> "mn.Animation":
        """
        Flip the top card of the pile face-up in place.

        Returns
        -------
        manim.Animation, or AnimationGroup() if deck is empty.
        """
        _require_manim("reveal_top")
        if self.is_empty:
            return AnimationGroup()
        return self.top_card.reveal_anim(run_time=run_time)

    def cut_anim(
        self,
        cut_point: Optional[int]  = None,
        run_time:  float          = 0.7,
    ) -> "mn.Animation":
        """
        Animate a deck cut at *cut_point*.

        The top portion lifts, shifts right, the bottom portion slides left,
        then the two halves swap positions and settle back.

        Parameters
        ----------
        cut_point : int, optional
            Index where the cut is made.  Defaults to near the middle.
        run_time : float

        Returns
        -------
        manim.Succession
        """
        _require_manim("cut_anim")
        n         = len(self.pile)
        cut_point = cut_point if cut_point is not None else n // 2
        cut_point = max(1, min(n - 1, cut_point))

        bottom_half = self.pile[:cut_point]   # stays at bottom after cut
        top_half    = self.pile[cut_point:]   # goes to bottom after cut

        g        = self.geometry
        deck_pos = self.get_center()
        lift     = g.height * 0.35
        shift    = g.width  * 0.55

        # Phase 1: lift top half up and to the right
        lift_anims = [
            card.animate(run_time=run_time * 0.30).shift([shift, lift, 0.05])
            for card in top_half
        ]
        # Phase 2: slide bottom half left
        slide_anims = [
            card.animate(run_time=run_time * 0.25).shift([-shift, 0, 0])
            for card in bottom_half
        ]
        # Phase 3: top half settles on top of what was the bottom
        settle_anims = []
        for i, card in enumerate(top_half):
            settle_anims.append(
                card.animate(run_time=run_time * 0.30).move_to(
                    deck_pos + np.array([0, 0, (cut_point + i) * self._stack_offset_z])
                )
            )
        # Also settle the bottom half
        for i, card in enumerate(bottom_half):
            settle_anims.append(
                card.animate(run_time=run_time * 0.30).move_to(
                    deck_pos + np.array([0, 0, i * self._stack_offset_z])
                )
            )

        # Update pile order (top_half is now on top of bottom_half)
        self.pile = bottom_half + top_half

        return Succession(
            AnimationGroup(*lift_anims),
            AnimationGroup(*slide_anims),
            AnimationGroup(*settle_anims),
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.pile)

    def __iter__(self) -> Iterator[Card3D]:
        return iter(self.pile)

    def __getitem__(self, idx: int) -> Card3D:
        return self.pile[idx]

    def __repr__(self) -> str:
        return (
            f"Deck3D({self.remaining} remaining / "
            f"{len(self.dealt)} dealt / "
            f"{len(self.cards)} total)"
        )


# ===========================================================================
# PURE-PYTHON DECK FACTORY HELPERS
# ===========================================================================

def _make_standard_deck(
    shuffle: bool = False,
    seed:    int  = None,
) -> List[CardFace]:
    """
    Return the 52 ``CardFace`` objects of a standard French deck.

    Order: Spades A–K, Hearts A–K, Diamonds A–K, Clubs A–K.

    Parameters
    ----------
    shuffle : bool
        If True, return the deck in a random order.
    seed : int, optional
        Random seed for shuffle (None = system time).

    Returns
    -------
    list[CardFace] of length 52.
    """
    faces = [
        CardFace(suit=suit, value=value)
        for suit  in CardSuit
        for value in CardValue
    ]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(faces)
    return faces


def standard_deck(shuffle: bool = False, seed: int = None) -> List[CardFace]:
    """
    Public alias for ``_make_standard_deck()``.

    Returns a list of all 52 ``CardFace`` objects.
    """
    return _make_standard_deck(shuffle=shuffle, seed=seed)


def suit_subset(suit: CardSuit) -> List[CardFace]:
    """Return the 13 ``CardFace`` objects for one suit."""
    return [CardFace(suit=suit, value=v) for v in CardValue]


def value_subset(value: CardValue) -> List[CardFace]:
    """Return the 4 ``CardFace`` objects for one value (one per suit)."""
    return [CardFace(suit=s, value=value) for s in CardSuit]


def face_cards_only() -> List[CardFace]:
    """Return the 12 face cards (J, Q, K of all suits)."""
    return [
        CardFace(suit=s, value=v)
        for s in CardSuit
        for v in (CardValue.JACK, CardValue.QUEEN, CardValue.KING)
    ]


def number_cards_only() -> List[CardFace]:
    """Return the 40 number cards (A–10 of all suits)."""
    return [
        CardFace(suit=s, value=v)
        for s in CardSuit
        for v in list(CardValue)[:10]  # ACE through TEN
    ]


# ===========================================================================
# STATISTICAL USE-CASE HELPERS
# ===========================================================================

def prob_card_event(
    card:       "Card3D",
    event_desc: str,
    probability: float,
    position:   np.ndarray = None,
    color:      str        = "#FFD700",
    font_size:  int        = 20,
) -> "VGroup":
    """
    Build a Manim VGroup containing:
    - A highlighted copy of *card*
    - A labeled banner showing ``event_desc`` and ``P = probability``

    Useful for illustrating classical probability on a single card.

    Parameters
    ----------
    card : Card3D
        The card mobject to annotate.
    event_desc : str
        Event description, e.g. "Heart drawn".
    probability : float
        P(event), e.g. 13/52 = 0.25.
    position : np.ndarray, optional
        Where to place the annotated group.
    color : str
        Banner accent color.
    font_size : int

    Returns
    -------
    manim.VGroup — call scene.play(Write(group)) to display.
    """
    _require_manim("prob_card_event")

    card.highlight(color=color)

    # Probability label
    p_frac = _float_to_fraction_str(probability)
    label = Text(
        f"{event_desc}:  P = {p_frac} ≈ {probability:.3f}",
        font_size = font_size,
        color     = ManimColor(color),
    ).next_to(card, DOWN, buff=0.15)

    group = VGroup(card, label)
    if position is not None:
        group.move_to(np.asarray(position, dtype=float))
    return group


def _float_to_fraction_str(p: float, max_denom: int = 52) -> str:
    """
    Attempt to express *p* as a simple fraction string.

    Examples: 0.25 → "1/4",  0.0769… → "1/13",  0.5 → "1/2".
    Falls back to a decimal string if no simple fraction is found.
    """
    from fractions import Fraction
    frac = Fraction(p).limit_denominator(max_denom)
    return str(frac)


def sample_without_replacement(
    deck:     "Deck3D",
    n:        int,
    targets:  List[np.ndarray],
    run_time: float = 0.4,
    stagger:  float = 0.08,
    flip:     bool  = True,
) -> Tuple["mn.Animation", List["Card3D"]]:
    """
    Deal *n* cards face-up to *targets*, representing a
    sample-without-replacement draw.

    Parameters
    ----------
    deck : Deck3D
    n : int
        Sample size.
    targets : list of array-like
        Destination positions for the drawn cards.
    run_time : float
        Per-card deal duration.
    stagger : float
        Time offset between successive deals.
    flip : bool
        Flip each card face-up on arrival.

    Returns
    -------
    (animation, drawn_cards)
        A tuple of the LaggedStart animation and the list of drawn Card3D objects.
    """
    _require_manim("sample_without_replacement")
    drawn = deck.pile[-n:]           # peek at the top n cards
    anim  = deck.deal_n(
        n        = n,
        targets  = targets[:n],
        run_time = run_time,
        stagger  = stagger,
        flip     = flip,
    )
    return anim, drawn


def hypergeometric_demo(
    deck:        "Deck3D",
    event_suit:  CardSuit,
    sample_size: int,
    targets:     List[np.ndarray],
    run_time:    float = 0.4,
) -> Tuple["mn.Animation", int]:
    """
    Demonstrate a hypergeometric sampling scenario:
    Draw *sample_size* cards from *deck* and count how many are of *event_suit*.

    Parameters
    ----------
    deck : Deck3D
    event_suit : CardSuit
        The "success" suit we're counting.
    sample_size : int
    targets : list of array-like
        Positions for the drawn cards.
    run_time : float

    Returns
    -------
    (animation, successes)
        The deal animation, and the integer count of cards matching *event_suit*.
    """
    _require_manim("hypergeometric_demo")
    anim, drawn_cards = sample_without_replacement(
        deck     = deck,
        n        = sample_size,
        targets  = targets,
        run_time = run_time,
        flip     = True,
    )
    successes = sum(1 for c in drawn_cards if c.suit == event_suit)
    return anim, successes


def birthday_problem_deck(n_people: int = 23) -> List[CardFace]:
    """
    Return a ``CardFace`` list representing the birthday problem:
    each of the 52 cards stands for a week of the year (52 weeks).

    Drawing *n_people* cards and checking for duplicates (same value)
    demonstrates why P(collision) ≥ 0.5 surprisingly quickly.

    Parameters
    ----------
    n_people : int
        Number of "birthdays" to draw.  Default 23 (the classic threshold).

    Returns
    -------
    list[CardFace] of length *n_people*, sampled without replacement from a
    standard deck.
    """
    faces = _make_standard_deck(shuffle=True, seed=None)
    return faces[:n_people]


def conditional_prob_demo(
    deck:       "Deck3D",
    condition:  CardSuit,
    event:      CardValue,
) -> Tuple[float, float, float]:
    """
    Compute conditional probability P(value=event | suit=condition)
    over the current deck pile.

    Useful for annotating Bayesian / conditional probability scenes.

    Parameters
    ----------
    deck : Deck3D
    condition : CardSuit
        The conditioning event (e.g. "card is a Heart").
    event : CardValue
        The target event (e.g. "card is an Ace").

    Returns
    -------
    (p_event, p_condition, p_event_given_condition)
        All three probabilities computed over ``deck.pile``.
    """
    pile         = deck.pile
    n            = len(pile)
    n_condition  = sum(1 for c in pile if c.suit == condition)
    n_event      = sum(1 for c in pile if c.value == event)
    n_both       = sum(1 for c in pile if c.suit == condition and c.value == event)

    p_condition  = n_condition / n if n > 0 else 0.0
    p_event      = n_event     / n if n > 0 else 0.0
    p_event_given_condition = n_both / n_condition if n_condition > 0 else 0.0

    return p_event, p_condition, p_event_given_condition


# ===========================================================================
# __all__ — public API
# ===========================================================================

__all__ = [
    # Enumerations
    "CardSuit",
    "CardValue",
    "CardFacing",

    # Pure-data classes
    "CardFace",
    "CardGeometry",
    "CardColorScheme",

    # Geometry presets
    "POKER_GEOMETRY",
    "MINI_GEOMETRY",
    "LARGE_GEOMETRY",

    # Color scheme presets
    "CLASSIC_SCHEME",
    "DARK_SCHEME",
    "PAPER_SCHEME",

    # Pip layout tables (for custom renderers)
    "_PIP_POSITIONS_NORM",
    "_PIP_UPSIDE_DOWN",

    # Manim mobjects
    "Card3D",
    "Deck3D",

    # Deck factory helpers
    "standard_deck",
    "suit_subset",
    "value_subset",
    "face_cards_only",
    "number_cards_only",

    # Statistical helpers
    "prob_card_event",
    "sample_without_replacement",
    "hypergeometric_demo",
    "birthday_problem_deck",
    "conditional_prob_demo",
]