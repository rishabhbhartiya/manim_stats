"""
manim_stats/core/base.py
========================
Master base layer for the Manim Statistics Extension.

Every asset in this library inherits from StatsObject3D which
provides the full suite of shared systems:

    • Theme engine          — 5 built-in palettes + custom
    • Material system       — matte / glass / metallic / emissive / holographic
    • Camera awareness      — billboard facing, depth-sorted labels
    • Label attachment      — anchored, auto-repositioning text/formula labels
    • Animation protocol    — build / update / highlight / exit contracts
    • Data binding          — attach numpy / scipy objects, subscribe to changes
    • Highlight & selection — pulse, outline glow, color-shift, shake
    • Composition helpers   — snap-to-grid, align, distribute, bounding-box
    • Config dataclasses    — one import covers all subsystem config
"""

from __future__ import annotations

import copy
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, ClassVar, Dict, Generator,
    Iterable, List, Optional, Sequence, Tuple, Union
)

import numpy as np
from colour import Color

from manim import (
    # Core mobjects
    VGroup, Group, VMobject, Mobject,
    # 3-D primitives
    ThreeDAxes, Surface, Arrow3D, Line3D, Dot3D,
    Sphere, Cylinder, Cone, Prism,
    # Text & math
    Text, MathTex, Tex, MarkupText,
    # 2-D overlays still useful in 3-D scenes
    RoundedRectangle, Rectangle, Circle, Dot,
    SurroundingRectangle, DashedLine,
    # Animation base classes
    Animation, AnimationGroup, Succession,
    Create, FadeIn, FadeOut, Transform,
    Write, DrawBorderThenFill,
    # Value trackers & updaters
    ValueTracker, always_redraw,
    # Constants & helpers
    ORIGIN, UP, DOWN, LEFT, RIGHT, IN, OUT,
    X_AXIS, Y_AXIS, Z_AXIS,
    WHITE, BLACK, GRAY, GRAY_A, GRAY_B, GRAY_C,
    BLUE, BLUE_A, BLUE_B, BLUE_C, BLUE_D, BLUE_E,
    RED, RED_A, RED_B, RED_C, RED_D, RED_E,
    GREEN, GREEN_A, GREEN_B, GREEN_C, GREEN_D, GREEN_E,
    YELLOW, YELLOW_A, YELLOW_B, YELLOW_C, YELLOW_D, YELLOW_E,
    ORANGE, PURPLE, TEAL, MAROON, GOLD, PINK,
    DEGREES, PI, TAU,
    # Utility
    interpolate_color, color_to_rgba, rgba_to_color,
    normalize, angle_of_vector, rotation_matrix,
    there_and_back, smooth, rush_from, rush_into,
    # Scene config
    config,
    rate_functions,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ENUMERATIONS
# ─────────────────────────────────────────────────────────────────────────────

class ThemeMode(Enum):
    DARK         = auto()   # Rich dark background, vibrant palette
    LIGHT        = auto()   # Off-white, muted palette for prints
    NEON         = auto()   # Pure black, electric neon accents
    PUBLICATION  = auto()   # Greyscale-first, colour used sparingly
    SOLARIZED    = auto()   # Solarized base, warm accent family


class MaterialStyle(Enum):
    MATTE        = auto()   # Flat, no specularity — clean academic look
    GLASS        = auto()   # Semi-transparent with light refraction hint
    METALLIC     = auto()   # High specular, reflects environment
    EMISSIVE     = auto()   # Glows from within — highlights, emphasis
    HOLOGRAPHIC  = auto()   # Animated rainbow sheen (for UI panels)
    WIREFRAME    = auto()   # Edges only — skeletal / deconstructed


class LabelAnchor(Enum):
    """Cardinal + diagonal + custom anchors for label placement."""
    TOP          = auto()
    BOTTOM       = auto()
    LEFT         = auto()
    RIGHT        = auto()
    TOP_LEFT     = auto()
    TOP_RIGHT    = auto()
    BOTTOM_LEFT  = auto()
    BOTTOM_RIGHT = auto()
    CENTER       = auto()
    FLOAT        = auto()   # Caller supplies explicit offset vector


class HighlightStyle(Enum):
    PULSE        = auto()   # Scale up-down once
    GLOW         = auto()   # Emissive outline ring
    COLOR_SHIFT  = auto()   # Temporarily change fill/stroke colour
    SHAKE        = auto()   # Short horizontal oscillation (for errors)
    OUTLINE      = auto()   # Persistent dashed outline
    FLASH        = auto()   # Brief white flash then restore


class BuildStyle(Enum):
    """How a complex asset renders itself for the first time."""
    GROW_FROM_CENTER  = auto()
    DRAW_BORDER       = auto()
    FADE_UP           = auto()   # FadeIn + shift UP
    UNFOLD            = auto()   # Axis by axis sequential reveal
    CASCADE           = auto()   # Sub-parts appear with staggered delay
    TYPEWRITER        = auto()   # Labels write themselves in sequence
    ASSEMBLE          = auto()   # Parts fly in from different directions


class DataUpdateMode(Enum):
    """How the asset reacts when bound data changes."""
    TRANSFORM   = auto()   # Smooth Transform to new geometry
    REDRAW      = auto()   # Tear-down and rebuild (for topology changes)
    MORPH       = auto()   # MorphingAnimation between old and new surfaces


# ─────────────────────────────────────────────────────────────────────────────
# 2.  THEME ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StatsColorPalette:
    """
    Complete colour specification for one theme mode.

    Naming convention mirrors Manim's own colour constants but is
    scoped to stats-specific semantics:

        primary      — main data colour (bars, curves, surfaces)
        secondary    — second data series
        accent       — attention / highlight
        positive     — probability mass, accepted region, p < alpha
        negative     — rejection region, error, warning
        neutral      — gridlines, tick marks, axis spines
        background   — scene background
        surface      — panel / table backgrounds
        text_primary — main label text
        text_secondary — axis tick labels, minor annotations
        distribution_*  — preset palette for up to 8 simultaneous distributions
    """
    # Semantic singles
    primary:          str = "#4FC3F7"   # sky blue
    secondary:        str = "#F48FB1"   # rose
    accent:           str = "#FFD54F"   # amber
    positive:         str = "#81C784"   # soft green
    negative:         str = "#E57373"   # soft red
    neutral:          str = "#78909C"   # blue-grey
    background:       str = "#0D1117"   # near-black
    surface:          str = "#161B22"   # dark card
    surface_raised:   str = "#21262D"   # elevated card
    text_primary:     str = "#E6EDF3"   # near-white
    text_secondary:   str = "#8B949E"   # muted grey
    border:           str = "#30363D"   # subtle border

    # Gradient endpoints (used in distribution surfaces, heatmaps)
    gradient_low:     str = "#1A237E"   # deep blue
    gradient_mid:     str = "#4FC3F7"   # sky
    gradient_high:    str = "#FFCA28"   # gold

    # Per-distribution palette (index 0–7)
    distribution_palette: List[str] = field(default_factory=lambda: [
        "#4FC3F7",  # 0 — sky blue
        "#F48FB1",  # 1 — rose
        "#A5D6A7",  # 2 — mint
        "#FFD54F",  # 3 — amber
        "#CE93D8",  # 4 — lavender
        "#80DEEA",  # 5 — cyan
        "#FFAB91",  # 6 — peach
        "#EF9A9A",  # 7 — salmon
    ])

    def dist_color(self, index: int) -> str:
        """Cycle through the distribution palette."""
        return self.distribution_palette[index % len(self.distribution_palette)]


class StatsTheme:
    """
    Global theme registry.  Use ``StatsTheme.set(ThemeMode.NEON)`` at the
    top of any scene to instantly switch the appearance of ALL assets.

    Themes are applied lazily — assets read ``StatsTheme.current`` at
    construction time, so you can also change mid-scene with an animation.
    """

    _palettes: ClassVar[Dict[ThemeMode, StatsColorPalette]] = {

        ThemeMode.DARK: StatsColorPalette(),   # defaults are DARK

        ThemeMode.LIGHT: StatsColorPalette(
            primary         = "#1565C0",
            secondary       = "#AD1457",
            accent          = "#F57F17",
            positive        = "#2E7D32",
            negative        = "#B71C1C",
            neutral         = "#546E7A",
            background      = "#F8F9FA",
            surface         = "#FFFFFF",
            surface_raised  = "#F1F3F4",
            text_primary    = "#1C2128",
            text_secondary  = "#57606A",
            border          = "#D0D7DE",
            gradient_low    = "#E3F2FD",
            gradient_mid    = "#1565C0",
            gradient_high   = "#F57F17",
            distribution_palette=[
                "#1565C0","#AD1457","#2E7D32","#F57F17",
                "#6A1B9A","#00838F","#BF360C","#827717",
            ],
        ),

        ThemeMode.NEON: StatsColorPalette(
            primary         = "#00FFF0",
            secondary       = "#FF00C8",
            accent          = "#FFFF00",
            positive        = "#00FF88",
            negative        = "#FF3366",
            neutral         = "#444466",
            background      = "#000005",
            surface         = "#0A0A1A",
            surface_raised  = "#10102A",
            text_primary    = "#FFFFFF",
            text_secondary  = "#8888BB",
            border          = "#222244",
            gradient_low    = "#000055",
            gradient_mid    = "#00FFF0",
            gradient_high   = "#FFFF00",
            distribution_palette=[
                "#00FFF0","#FF00C8","#00FF88","#FFFF00",
                "#FF8800","#AA00FF","#00BBFF","#FF3366",
            ],
        ),

        ThemeMode.PUBLICATION: StatsColorPalette(
            primary         = "#2C2C2C",
            secondary       = "#777777",
            accent          = "#C0392B",
            positive        = "#27AE60",
            negative        = "#C0392B",
            neutral         = "#AAAAAA",
            background      = "#FAFAFA",
            surface         = "#FFFFFF",
            surface_raised  = "#F0F0F0",
            text_primary    = "#1A1A1A",
            text_secondary  = "#666666",
            border          = "#CCCCCC",
            gradient_low    = "#EEEEEE",
            gradient_mid    = "#777777",
            gradient_high   = "#2C2C2C",
            distribution_palette=[
                "#2C2C2C","#C0392B","#27AE60","#2980B9",
                "#8E44AD","#E67E22","#1ABC9C","#F39C12",
            ],
        ),

        ThemeMode.SOLARIZED: StatsColorPalette(
            primary         = "#268BD2",
            secondary       = "#D33682",
            accent          = "#B58900",
            positive        = "#859900",
            negative        = "#DC322F",
            neutral         = "#586E75",
            background      = "#002B36",
            surface         = "#073642",
            surface_raised  = "#0C4A58",
            text_primary    = "#FDF6E3",
            text_secondary  = "#93A1A1",
            border          = "#073642",
            gradient_low    = "#073642",
            gradient_mid    = "#268BD2",
            gradient_high   = "#B58900",
            distribution_palette=[
                "#268BD2","#D33682","#859900","#B58900",
                "#2AA198","#6C71C4","#CB4B16","#DC322F",
            ],
        ),
    }

    _current_mode:    ClassVar[ThemeMode]          = ThemeMode.DARK
    _custom_palette:  ClassVar[Optional[StatsColorPalette]] = None
    _change_callbacks: ClassVar[List[Callable]]    = []

    # ── public API ────────────────────────────────────────────────────────

    @classmethod
    def set(cls, mode: ThemeMode) -> None:
        """Switch global theme; fires all registered callbacks."""
        cls._current_mode   = mode
        cls._custom_palette = None
        for cb in cls._change_callbacks:
            cb(mode)

    @classmethod
    def set_custom(cls, palette: StatsColorPalette) -> None:
        """Install a fully custom palette."""
        cls._custom_palette = palette
        for cb in cls._change_callbacks:
            cb(None)

    @classmethod
    @property
    def current(cls) -> StatsColorPalette:
        if cls._custom_palette is not None:
            return cls._custom_palette
        return cls._palettes[cls._current_mode]

    @classmethod
    @property
    def mode(cls) -> ThemeMode:
        return cls._current_mode

    @classmethod
    def on_change(cls, callback: Callable) -> None:
        """Register a callable(mode) that fires on every theme switch."""
        cls._change_callbacks.append(callback)

    @classmethod
    def get_gradient(
        cls,
        n: int,
        low_key: str = "gradient_low",
        high_key: str = "gradient_high",
    ) -> List[str]:
        """
        Return ``n`` evenly-spaced hex colours between two theme gradient keys.
        """
        palette = cls.current
        low  = getattr(palette, low_key)
        high = getattr(palette, high_key)
        return [
            Color(low).interpolate(Color(high), t).hex_l
            for t in np.linspace(0, 1, n)
        ]

    @classmethod
    def get_diverging(cls, n: int) -> List[str]:
        """
        Return ``n`` colours along low → mid → high (diverging colour map).
        Useful for signed residuals, z-scores, correlations.
        """
        palette = cls.current
        low, mid, high = palette.gradient_low, palette.gradient_mid, palette.gradient_high
        half = n // 2
        left  = [Color(low).interpolate(Color(mid), t).hex_l
                 for t in np.linspace(0, 1, half)]
        right = [Color(mid).interpolate(Color(high), t).hex_l
                 for t in np.linspace(0, 1, n - half)]
        return left + right[1:]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MATERIAL SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MaterialConfig:
    """
    Fully parameterised material specification.

    In Manim CE opacity and stroke are the primary visual knobs;
    this dataclass maps higher-level material concepts onto those
    primitives while also carrying metadata that AnimationProtocol
    and HighlightSystem use.
    """
    style:              MaterialStyle = MaterialStyle.MATTE

    # Opacity / transparency
    fill_opacity:       float = 1.0
    stroke_opacity:     float = 1.0
    stroke_width:       float = 2.0

    # Colour overrides — None → inherit from theme
    fill_color:         Optional[str] = None
    stroke_color:       Optional[str] = None

    # Emissive properties (glow ring)
    emissive_color:     Optional[str] = None  # None → same as fill
    emissive_opacity:   float = 0.4
    emissive_radius:    float = 0.08          # extra stroke width for glow

    # Glass / refraction
    glass_ior:          float = 1.45          # index of refraction hint
    glass_tint:         Optional[str] = None  # tint applied over content

    # Metallic
    metallic_sheen:     float = 0.7           # 0=dull, 1=mirror
    metallic_sheen_color: str = "#FFFFFF"

    # Holographic (animated only — used via updater)
    holo_speed:         float = 1.0           # full rainbow cycle per second
    holo_saturation:    float = 0.9

    # Wireframe
    wireframe_density:  int   = 4             # line subdivisions per face

    # Shadow
    cast_shadow:        bool  = False
    shadow_opacity:     float = 0.25
    shadow_blur:        float = 0.3

    # ── preset factories ──────────────────────────────────────────────────

    @classmethod
    def matte(cls, color: Optional[str] = None, opacity: float = 1.0) -> "MaterialConfig":
        return cls(style=MaterialStyle.MATTE,
                   fill_color=color, fill_opacity=opacity,
                   stroke_width=1.5)

    @classmethod
    def glass(cls, tint: Optional[str] = None, opacity: float = 0.35) -> "MaterialConfig":
        return cls(style=MaterialStyle.GLASS,
                   fill_color=tint, fill_opacity=opacity,
                   stroke_width=1.0, stroke_opacity=0.6,
                   glass_tint=tint)

    @classmethod
    def metallic(cls, color: Optional[str] = None, sheen: float = 0.7) -> "MaterialConfig":
        return cls(style=MaterialStyle.METALLIC,
                   fill_color=color, fill_opacity=1.0,
                   stroke_width=0.8, metallic_sheen=sheen)

    @classmethod
    def emissive(cls, color: Optional[str] = None,
                 glow_opacity: float = 0.5) -> "MaterialConfig":
        return cls(style=MaterialStyle.EMISSIVE,
                   fill_color=color, fill_opacity=1.0,
                   stroke_width=2.5, stroke_opacity=1.0,
                   emissive_color=color, emissive_opacity=glow_opacity)

    @classmethod
    def holographic(cls, speed: float = 1.0) -> "MaterialConfig":
        return cls(style=MaterialStyle.HOLOGRAPHIC,
                   fill_opacity=0.6, stroke_opacity=0.9,
                   stroke_width=1.5, holo_speed=speed)

    @classmethod
    def wireframe(cls, color: Optional[str] = None) -> "MaterialConfig":
        return cls(style=MaterialStyle.WIREFRAME,
                   fill_color=color, fill_opacity=0.0,
                   stroke_width=1.2, stroke_opacity=0.8)


class MaterialApplicator:
    """
    Applies a ``MaterialConfig`` to any ``VMobject`` or ``Mobject``.

    For Manim CE (which has no native PBR shading pipeline) we approximate
    higher-level materials through:

        GLASS        → reduced opacity + lighter stroke
        METALLIC     → gradient fill between fill_color and sheen_color
        EMISSIVE     → bright stroke + an extra glow ring (added to parent)
        HOLOGRAPHIC  → time-based hue rotation via an updater
        WIREFRAME    → zero fill opacity + fine stroke mesh
    """

    @staticmethod
    def apply(mob: VMobject, mat: MaterialConfig,
              theme: Optional[StatsColorPalette] = None) -> VMobject:
        """
        Mutate *mob* in-place according to *mat*.
        Returns *mob* for chaining.
        """
        t = theme or StatsTheme.current

        fill   = mat.fill_color   or t.primary
        stroke = mat.stroke_color or t.border

        if mat.style == MaterialStyle.MATTE:
            mob.set_fill(fill,   opacity=mat.fill_opacity)
            mob.set_stroke(stroke, width=mat.stroke_width,
                           opacity=mat.stroke_opacity)

        elif mat.style == MaterialStyle.GLASS:
            tint = mat.glass_tint or fill
            mob.set_fill(tint,   opacity=mat.fill_opacity)
            mob.set_stroke(tint, width=mat.stroke_width,
                           opacity=mat.stroke_opacity * 1.4)

        elif mat.style == MaterialStyle.METALLIC:
            # Two-stop gradient: fill → sheen
            mob.set_fill(fill, opacity=mat.fill_opacity)
            mob.set_sheen_direction(UP)
            mob.set_sheen(mat.metallic_sheen,
                          direction=UP)
            mob.set_stroke(stroke, width=mat.stroke_width * 0.6)

        elif mat.style == MaterialStyle.EMISSIVE:
            glow_color = mat.emissive_color or fill
            mob.set_fill(glow_color, opacity=mat.fill_opacity)
            mob.set_stroke(glow_color,
                           width=mat.stroke_width + mat.emissive_radius * 20,
                           opacity=mat.emissive_opacity)

        elif mat.style == MaterialStyle.WIREFRAME:
            mob.set_fill(fill, opacity=0.0)
            mob.set_stroke(fill, width=mat.stroke_width, opacity=mat.stroke_opacity)

        elif mat.style == MaterialStyle.HOLOGRAPHIC:
            # Initial state; time-based updater must be added by caller
            mob.set_fill(fill,   opacity=mat.fill_opacity)
            mob.set_stroke(fill, width=mat.stroke_width,
                           opacity=mat.stroke_opacity)

        return mob

    @staticmethod
    def add_holographic_updater(mob: VMobject, mat: MaterialConfig) -> None:
        """
        Attach a per-frame updater that cycles hue for HOLOGRAPHIC style.
        The updater reads scene elapsed time via the mobject's
        ``_holo_start_time`` attribute (set here).
        """
        mob._holo_start_time = time.time()

        def _holo_update(m: VMobject, dt: float) -> None:
            elapsed = time.time() - m._holo_start_time
            hue = (elapsed * mat.holo_speed) % 1.0
            c   = Color(hsl=(hue, mat.holo_saturation, 0.6))
            m.set_fill(c.hex_l,   opacity=mat.fill_opacity)
            m.set_stroke(c.hex_l, width=mat.stroke_width,
                         opacity=mat.stroke_opacity)

        mob.add_updater(_holo_update)

    @staticmethod
    def add_shadow(mob: VMobject, mat: MaterialConfig) -> VGroup:
        """
        Return a VGroup([shadow_mob, mob]).
        Shadow is a slightly-scaled, dark, blurred copy of the silhouette.
        """
        if not mat.cast_shadow:
            return VGroup(mob)

        shadow = mob.copy()
        shadow.set_fill("#000000",    opacity=mat.shadow_opacity)
        shadow.set_stroke("#000000",  width=0)
        shadow.shift(DOWN * 0.05 + RIGHT * 0.05)
        shadow.scale(1 + mat.shadow_blur * 0.1)
        return VGroup(shadow, mob)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LABEL ATTACHMENT SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LabelConfig:
    """Full specification for a single attached label."""

    text:           str  = ""
    is_math:        bool = False            # True → MathTex, else Text
    anchor:         LabelAnchor = LabelAnchor.TOP
    offset:         np.ndarray  = field(default_factory=lambda: np.array([0., 0., 0.]))
    font_size:      float = 24
    color:          Optional[str] = None    # None → theme.text_primary
    background:     bool  = False           # pill-shaped background behind text
    bg_color:       Optional[str] = None    # None → theme.surface
    bg_opacity:     float = 0.75
    bg_padding:     float = 0.15
    always_face_camera: bool = True         # billboard mode
    depth_offset:   float = 0.01           # push in front of surface
    visible:        bool  = True


class LabelAttachment:
    """
    Manages all labels attached to a ``StatsObject3D``.

    Labels are stored as a dict[str, Mobject] so they can be
    individually shown/hidden/updated by key.
    """

    def __init__(self, parent: "StatsObject3D") -> None:
        self._parent   = parent
        self._labels:  Dict[str, Tuple[LabelConfig, Mobject]] = {}
        self._group:   VGroup = VGroup()

    # ── public API ────────────────────────────────────────────────────────

    def add(self, key: str, cfg: LabelConfig) -> Mobject:
        """Build, position, and register a label. Returns the mobject."""
        mob = self._build_label_mob(cfg)
        self._position_label(mob, cfg)
        self._labels[key] = (cfg, mob)
        self._group.add(mob)
        return mob

    def remove(self, key: str) -> None:
        if key in self._labels:
            _, mob = self._labels.pop(key)
            self._group.remove(mob)

    def update_text(self, key: str, new_text: str) -> None:
        if key not in self._labels:
            return
        cfg, old_mob = self._labels[key]
        cfg = copy.copy(cfg)
        cfg.text = new_text
        new_mob = self._build_label_mob(cfg)
        self._position_label(new_mob, cfg)
        self._group.remove(old_mob)
        self._labels[key] = (cfg, new_mob)
        self._group.add(new_mob)

    def show(self, key: str) -> None:
        if key in self._labels:
            self._labels[key][1].set_opacity(1)

    def hide(self, key: str) -> None:
        if key in self._labels:
            self._labels[key][1].set_opacity(0)

    def get_mob(self, key: str) -> Optional[Mobject]:
        return self._labels[key][1] if key in self._labels else None

    @property
    def group(self) -> VGroup:
        return self._group

    def reposition_all(self) -> None:
        """Call after parent has been moved/scaled."""
        for key, (cfg, mob) in self._labels.items():
            self._position_label(mob, cfg)

    # ── private helpers ───────────────────────────────────────────────────

    def _build_label_mob(self, cfg: LabelConfig) -> Mobject:
        t = StatsTheme.current
        color = cfg.color or t.text_primary

        if cfg.is_math:
            text_mob = MathTex(cfg.text, font_size=cfg.font_size,
                               color=color)
        else:
            text_mob = Text(cfg.text, font_size=cfg.font_size,
                            color=color)

        if cfg.background:
            bg_color = cfg.bg_color or t.surface
            bg = RoundedRectangle(
                corner_radius=0.1,
                width=text_mob.width + cfg.bg_padding * 2,
                height=text_mob.height + cfg.bg_padding * 2,
            )
            bg.set_fill(bg_color, opacity=cfg.bg_opacity)
            bg.set_stroke(t.border, width=1.0)
            bg.move_to(text_mob)
            mob = VGroup(bg, text_mob)
        else:
            mob = text_mob

        return mob

    def _position_label(self, mob: Mobject, cfg: LabelConfig) -> None:
        """Place *mob* relative to the parent's bounding box."""
        parent = self._parent
        bb     = parent.get_bounding_box()         # [min, mid, max]
        min_pt, mid_pt, max_pt = bb[0], bb[1], bb[2]

        anchor_map: Dict[LabelAnchor, np.ndarray] = {
            LabelAnchor.TOP:          mid_pt + UP    * (max_pt[1] - mid_pt[1] + 0.2),
            LabelAnchor.BOTTOM:       mid_pt + DOWN  * (mid_pt[1] - min_pt[1] + 0.2),
            LabelAnchor.LEFT:         mid_pt + LEFT  * (mid_pt[0] - min_pt[0] + 0.2),
            LabelAnchor.RIGHT:        mid_pt + RIGHT * (max_pt[0] - mid_pt[0] + 0.2),
            LabelAnchor.TOP_LEFT:     np.array([min_pt[0], max_pt[1], mid_pt[2]]) + UP*0.15 + LEFT*0.15,
            LabelAnchor.TOP_RIGHT:    np.array([max_pt[0], max_pt[1], mid_pt[2]]) + UP*0.15 + RIGHT*0.15,
            LabelAnchor.BOTTOM_LEFT:  np.array([min_pt[0], min_pt[1], mid_pt[2]]) + DOWN*0.15 + LEFT*0.15,
            LabelAnchor.BOTTOM_RIGHT: np.array([max_pt[0], min_pt[1], mid_pt[2]]) + DOWN*0.15 + RIGHT*0.15,
            LabelAnchor.CENTER:       mid_pt.copy(),
            LabelAnchor.FLOAT:        parent.get_center() + cfg.offset,
        }

        pos = anchor_map.get(cfg.anchor, mid_pt.copy())
        pos = pos + cfg.offset + OUT * cfg.depth_offset
        mob.move_to(pos)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DATA BINDING LAYER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BoundData:
    """
    Wraps a numpy array (or any array-like) with metadata used by
    ``StatsObject3D`` to drive geometry updates.

    ``schema`` is an open dict for assets to store domain-specific
    metadata (e.g. axis labels, bin counts, distribution params).
    """
    data:               Any                            # np.ndarray, scipy rv, …
    label:              str            = "x"
    update_mode:        DataUpdateMode = DataUpdateMode.TRANSFORM
    schema:             Dict[str, Any] = field(default_factory=dict)
    _subscribers:       List[Callable] = field(default_factory=list, repr=False)

    def subscribe(self, callback: Callable[["BoundData"], None]) -> None:
        """Register a callback fired whenever .update() is called."""
        self._subscribers.append(callback)

    def update(self, new_data: Any) -> None:
        """Replace data and notify all subscribers."""
        self.data = new_data
        for cb in self._subscribers:
            cb(self)

    # Convenience: expose numpy stats if data is array-like
    @property
    def as_array(self) -> np.ndarray:
        return np.asarray(self.data)

    @property
    def n(self) -> int:
        return len(self.as_array)

    @property
    def mean(self) -> float:
        return float(np.mean(self.as_array))

    @property
    def std(self) -> float:
        return float(np.std(self.as_array, ddof=1)) if self.n > 1 else 0.0

    @property
    def min(self) -> float:
        return float(np.min(self.as_array))

    @property
    def max(self) -> float:
        return float(np.max(self.as_array))


# ─────────────────────────────────────────────────────────────────────────────
# 6.  ANIMATION PROTOCOL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnimationConfig:
    """
    Shared animation parameters threaded through all protocol methods.
    Every public animation factory on ``StatsObject3D`` accepts this.
    """
    run_time:       float          = 1.5
    lag_ratio:      float          = 0.1       # stagger for composite anims
    rate_func:      Callable       = smooth
    build_style:    BuildStyle     = BuildStyle.CASCADE
    update_mode:    DataUpdateMode = DataUpdateMode.TRANSFORM
    color_override: Optional[str]  = None      # temporary colour during anim


class AnimationProtocol(ABC):
    """
    Contract that every ``StatsObject3D`` must fulfil.

    Each method returns a Manim ``Animation`` (or ``AnimationGroup``)
    so the caller can use it directly in ``scene.play()``.
    """

    @abstractmethod
    def animate_build(self, cfg: AnimationConfig) -> Animation:
        """First-time reveal of the entire object."""
        ...

    @abstractmethod
    def animate_update(self, new_data: Any,
                       cfg: AnimationConfig) -> Animation:
        """Smoothly morph to reflect changed data."""
        ...

    @abstractmethod
    def animate_highlight(self, style: HighlightStyle,
                          cfg: AnimationConfig) -> Animation:
        """Draw attention to this object (pulse, glow, shake …)."""
        ...

    @abstractmethod
    def animate_exit(self, cfg: AnimationConfig) -> Animation:
        """Graceful removal from the scene."""
        ...

    # ── default implementations (can be overridden) ───────────────────────

    def _default_build(self, cfg: AnimationConfig) -> Animation:
        """
        Fallback build: FadeIn with upward shift.
        Used by leaf objects that don't need a custom build.
        """
        return FadeIn(self, shift=UP * 0.3, run_time=cfg.run_time,
                      rate_func=cfg.rate_func)

    def _default_exit(self, cfg: AnimationConfig) -> Animation:
        return FadeOut(self, shift=DOWN * 0.2, run_time=cfg.run_time * 0.7)

    def _cascade_build(self, submobs: List[Mobject],
                       cfg: AnimationConfig) -> Animation:
        """
        Staggered FadeIn for a list of sub-mobjects.
        The most common composite build style.
        """
        anims = [FadeIn(m, shift=UP * 0.15) for m in submobs]
        return AnimationGroup(*anims,
                              lag_ratio=cfg.lag_ratio,
                              run_time=cfg.run_time,
                              rate_func=cfg.rate_func)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  HIGHLIGHT & SELECTION SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

class HighlightSystem:
    """
    Stateless factory: produces Manim animations for each ``HighlightStyle``.
    All methods return an ``Animation`` or ``AnimationGroup``.
    """

    @staticmethod
    def pulse(mob: VMobject, cfg: AnimationConfig,
              scale_factor: float = 1.15) -> Animation:
        from manim import ScaleInPlace
        up   = ScaleInPlace(mob, scale_factor,
                            run_time=cfg.run_time * 0.4,
                            rate_func=rush_from)
        down = ScaleInPlace(mob, 1 / scale_factor,
                            run_time=cfg.run_time * 0.6,
                            rate_func=rush_into)
        return Succession(up, down)

    @staticmethod
    def glow(mob: VMobject, cfg: AnimationConfig) -> Animation:
        """
        Animate stroke width up then back down, simulating a glow bloom.
        """
        from manim import UpdateFromAlphaFunc
        original_sw = mob.get_stroke_width()
        glow_sw     = original_sw + 8

        def updater(m: VMobject, alpha: float) -> None:
            t  = there_and_back(alpha)
            sw = interpolate(original_sw, glow_sw, t)
            m.set_stroke(width=sw)

        return UpdateFromAlphaFunc(mob, updater,
                                   run_time=cfg.run_time,
                                   rate_func=smooth)

    @staticmethod
    def color_shift(mob: VMobject, cfg: AnimationConfig,
                    target_color: Optional[str] = None) -> Animation:
        from manim import UpdateFromAlphaFunc
        t          = StatsTheme.current
        orig_fill  = mob.get_fill_color()
        dest_color = target_color or t.accent

        def updater(m: VMobject, alpha: float) -> None:
            a = there_and_back(alpha)
            c = interpolate_color(orig_fill, dest_color, a)
            m.set_fill(c)

        return UpdateFromAlphaFunc(mob, updater,
                                   run_time=cfg.run_time,
                                   rate_func=smooth)

    @staticmethod
    def shake(mob: VMobject, cfg: AnimationConfig,
              amplitude: float = 0.08, cycles: int = 4) -> Animation:
        from manim import UpdateFromAlphaFunc
        origin = mob.get_center().copy()

        def updater(m: VMobject, alpha: float) -> None:
            phase = alpha * cycles * TAU
            offset = RIGHT * amplitude * np.sin(phase) * (1 - alpha)
            m.move_to(origin + offset)

        return UpdateFromAlphaFunc(mob, updater,
                                   run_time=cfg.run_time,
                                   rate_func=smooth)

    @staticmethod
    def outline(mob: VMobject, cfg: AnimationConfig) -> VGroup:
        """
        Return a persistent dashed outline ``VMobject`` surrounding *mob*.
        Add it to the scene; remove it when no longer needed.
        """
        t       = StatsTheme.current
        outline = SurroundingRectangle(mob,
                                       color=t.accent,
                                       buff=0.12,
                                       stroke_width=2.0,
                                       corner_radius=0.06)
        outline.set_stroke(t.accent, opacity=0.9)
        outline.set_fill(opacity=0)
        return outline

    @staticmethod
    def flash(mob: VMobject, cfg: AnimationConfig) -> Animation:
        from manim import Flash
        t = StatsTheme.current
        return Flash(mob,
                     color=t.accent,
                     flash_radius=mob.width * 0.7,
                     num_lines=12,
                     run_time=cfg.run_time * 0.6)


def interpolate(a: float, b: float, t: float) -> float:
    """Linear interpolate between a and b."""
    return a + (b - a) * t


# ─────────────────────────────────────────────────────────────────────────────
# 8.  COMPOSITION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

class CompositionHelper:
    """
    Static helpers for snapping, aligning, and distributing
    ``StatsObject3D`` instances in 3-D space.
    """

    # ── alignment ─────────────────────────────────────────────────────────

    @staticmethod
    def align_left(mobs: Sequence[Mobject]) -> None:
        if not mobs: return
        x = min(m.get_left()[0] for m in mobs)
        for m in mobs:
            m.shift(RIGHT * (x - m.get_left()[0]))

    @staticmethod
    def align_right(mobs: Sequence[Mobject]) -> None:
        if not mobs: return
        x = max(m.get_right()[0] for m in mobs)
        for m in mobs:
            m.shift(RIGHT * (x - m.get_right()[0]))

    @staticmethod
    def align_top(mobs: Sequence[Mobject]) -> None:
        if not mobs: return
        y = max(m.get_top()[1] for m in mobs)
        for m in mobs:
            m.shift(UP * (y - m.get_top()[1]))

    @staticmethod
    def align_bottom(mobs: Sequence[Mobject]) -> None:
        if not mobs: return
        y = min(m.get_bottom()[1] for m in mobs)
        for m in mobs:
            m.shift(UP * (y - m.get_bottom()[1]))

    @staticmethod
    def center_on(mob: Mobject, target: np.ndarray) -> None:
        mob.move_to(target)

    # ── distribution ──────────────────────────────────────────────────────

    @staticmethod
    def distribute_horizontally(mobs: Sequence[Mobject],
                                 spacing: float = 0.5) -> None:
        if len(mobs) < 2: return
        total_w = sum(m.width for m in mobs) + spacing * (len(mobs) - 1)
        x_start = -total_w / 2
        x = x_start
        for m in mobs:
            m.move_to(np.array([x + m.width / 2, m.get_center()[1],
                                 m.get_center()[2]]))
            x += m.width + spacing

    @staticmethod
    def distribute_vertically(mobs: Sequence[Mobject],
                               spacing: float = 0.5) -> None:
        if len(mobs) < 2: return
        total_h = sum(m.height for m in mobs) + spacing * (len(mobs) - 1)
        y_start = total_h / 2
        y = y_start
        for m in mobs:
            m.move_to(np.array([m.get_center()[0], y - m.height / 2,
                                 m.get_center()[2]]))
            y -= m.height + spacing

    @staticmethod
    def arrange_in_grid(mobs: Sequence[Mobject],
                        cols: int,
                        h_spacing: float = 0.5,
                        v_spacing: float = 0.5) -> None:
        for i, m in enumerate(mobs):
            row, col = divmod(i, cols)
            m.move_to(np.array([col * h_spacing, -row * v_spacing, 0.0]))

    # ── snap ──────────────────────────────────────────────────────────────

    @staticmethod
    def snap_to_grid(mob: Mobject, grid_size: float = 0.5) -> None:
        c = mob.get_center()
        snapped = np.array([
            round(c[0] / grid_size) * grid_size,
            round(c[1] / grid_size) * grid_size,
            round(c[2] / grid_size) * grid_size,
        ])
        mob.move_to(snapped)

    # ── bounding box ──────────────────────────────────────────────────────

    @staticmethod
    def combined_bounding_box(
        mobs: Sequence[Mobject],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (min_corner, max_corner) of the union bounding box."""
        mins = np.array([m.get_bounding_box()[0] for m in mobs])
        maxs = np.array([m.get_bounding_box()[2] for m in mobs])
        return mins.min(axis=0), maxs.max(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  CAMERA AWARENESS
# ─────────────────────────────────────────────────────────────────────────────

class CameraAwareness:
    """
    Mixin that gives an object the ability to orient labels / flat panels
    toward the active camera in a ``ThreeDScene``.

    Usage
    -----
    Add ``CameraAwareness`` to the MRO of your asset, then call
    ``self.register_camera(scene)`` inside the scene's ``construct``.
    Labels/billboards are then updated each frame via an updater.
    """

    _camera_ref: Optional[Any] = None   # set to scene.camera by register_camera

    def register_camera(self, scene: Any) -> None:
        """Bind this object to the scene's camera."""
        self._camera_ref = scene.camera

    def get_camera_direction(self) -> np.ndarray:
        """Unit vector FROM the object TO the camera."""
        if self._camera_ref is None:
            return OUT
        cam_pos  = self._camera_ref.get_location()
        self_pos = self.get_center()  # type: ignore[attr-defined]
        diff     = cam_pos - self_pos
        norm     = np.linalg.norm(diff)
        return diff / norm if norm > 1e-6 else OUT

    def face_camera(self, mob: VMobject) -> None:
        """
        Rotate *mob* so its normal points toward the camera.
        Call this inside an updater for live billboard behaviour.
        """
        if self._camera_ref is None:
            return
        cam_dir = self.get_camera_direction()
        # Manim's rotate_to_face_camera-style logic
        angle    = angle_of_vector(cam_dir[:2])
        mob.rotate(-mob.get_angle() + angle, about_point=mob.get_center())

    def add_billboard_updater(self, mob: VMobject) -> None:
        """Attach a persistent updater so *mob* always faces the camera."""
        def _update(m: VMobject, dt: float) -> None:
            self.face_camera(m)
        mob.add_updater(_update)


# ─────────────────────────────────────────────────────────────────────────────
# 10.  STATS OBJECT 3D  (master base class)
# ─────────────────────────────────────────────────────────────────────────────

class StatsObject3D(VGroup, AnimationProtocol, CameraAwareness):
    """
    Master base class for every asset in the Manim Statistics Extension.

    Inheritance chain
    -----------------
    StatsObject3D → VGroup → VMobject → Mobject
        ↳ AnimationProtocol   (abstract build/update/highlight/exit)
        ↳ CameraAwareness     (mixin for billboard labels)

    Subclass responsibilities
    -------------------------
    1.  Override ``_build_geometry()`` — create sub-mobjects, add to self
    2.  Override ``animate_build()``, ``animate_update()``,
        ``animate_highlight()``, ``animate_exit()`` OR call the
        default helpers provided here.

    Constructor parameters
    ----------------------
    material    : MaterialConfig   — surface appearance
    anim_cfg    : AnimationConfig  — default animation settings
    label_cfgs  : dict[str, LabelConfig]  — labels to attach at build time
    theme_mode  : ThemeMode | None — override global theme for this object
    """

    # Class-level registry so StatsTheme.set() can redraw all live objects
    _registry: ClassVar[List["StatsObject3D"]] = []

    def __init__(
        self,
        material:    MaterialConfig                 = None,
        anim_cfg:    AnimationConfig                = None,
        label_cfgs:  Dict[str, LabelConfig]         = None,
        theme_mode:  Optional[ThemeMode]            = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # ── subsystems ────────────────────────────────────────────────────
        self._material    = material  or MaterialConfig()
        self._anim_cfg    = anim_cfg  or AnimationConfig()
        self._labels      = LabelAttachment(self)
        self._data_slots: Dict[str, BoundData] = {}
        self._built:      bool = False
        self._selected:   bool = False

        # Per-object theme override
        self._theme_mode  = theme_mode
        if theme_mode is not None:
            # Temporarily switch the global theme to get the right palette,
            # then restore.  (Thread-unsafe in parallel rendering; acceptable.)
            self._palette = StatsTheme._palettes[theme_mode]
        else:
            self._palette = StatsTheme.current

        # Register for theme-change callbacks
        StatsTheme.on_change(self._on_theme_change)
        StatsObject3D._registry.append(self)

        # ── build ─────────────────────────────────────────────────────────
        self._build_geometry()

        # Attach any labels supplied at construction time
        if label_cfgs:
            for key, cfg in label_cfgs.items():
                self._labels.add(key, cfg)
            self.add(self._labels.group)

        self._built = True

    # ── subclass contract ─────────────────────────────────────────────────

    def _build_geometry(self) -> None:
        """
        Override in subclasses to construct all sub-mobjects.

        Typical pattern::

            def _build_geometry(self) -> None:
                body = Sphere(radius=0.5)
                MaterialApplicator.apply(body, self._material, self._palette)
                self.add(body)
        """
        pass   # base class has no geometry of its own

    # ── animation protocol (default implementations) ──────────────────────

    def animate_build(self, cfg: Optional[AnimationConfig] = None) -> Animation:
        cfg = cfg or self._anim_cfg
        submobs = list(self.submobjects)
        if not submobs:
            return self._default_build(cfg)

        if cfg.build_style == BuildStyle.CASCADE:
            return self._cascade_build(submobs, cfg)
        elif cfg.build_style == BuildStyle.FADE_UP:
            return FadeIn(self, shift=UP * 0.3,
                          run_time=cfg.run_time, rate_func=cfg.rate_func)
        elif cfg.build_style == BuildStyle.DRAW_BORDER:
            return DrawBorderThenFill(self,
                                      run_time=cfg.run_time,
                                      rate_func=cfg.rate_func)
        elif cfg.build_style == BuildStyle.GROW_FROM_CENTER:
            from manim import GrowFromCenter
            return GrowFromCenter(self, run_time=cfg.run_time,
                                  rate_func=cfg.rate_func)
        elif cfg.build_style == BuildStyle.UNFOLD:
            return self._unfold_build(submobs, cfg)
        else:
            return self._default_build(cfg)

    def animate_update(
        self, new_data: Any, cfg: Optional[AnimationConfig] = None,
    ) -> Animation:
        cfg = cfg or self._anim_cfg
        # Default: rebuild geometry, then Transform into the new state
        old_copy = self.copy()
        self._build_geometry()
        return Transform(old_copy, self,
                         run_time=cfg.run_time,
                         rate_func=cfg.rate_func)

    def animate_highlight(
        self,
        style: HighlightStyle = HighlightStyle.PULSE,
        cfg:   Optional[AnimationConfig] = None,
    ) -> Animation:
        cfg = cfg or self._anim_cfg
        hs  = HighlightSystem
        if style == HighlightStyle.PULSE:
            return hs.pulse(self, cfg)
        elif style == HighlightStyle.GLOW:
            return hs.glow(self, cfg)
        elif style == HighlightStyle.COLOR_SHIFT:
            return hs.color_shift(self, cfg,
                                  target_color=cfg.color_override)
        elif style == HighlightStyle.SHAKE:
            return hs.shake(self, cfg)
        elif style == HighlightStyle.FLASH:
            return hs.flash(self, cfg)
        else:
            return hs.pulse(self, cfg)

    def animate_exit(self, cfg: Optional[AnimationConfig] = None) -> Animation:
        cfg = cfg or self._anim_cfg
        return self._default_exit(cfg)

    # ── data binding ──────────────────────────────────────────────────────

    def bind_data(self, key: str, data: BoundData) -> None:
        """
        Attach a ``BoundData`` object under *key*.
        Subscribes the asset's ``_on_data_update`` callback.
        """
        self._data_slots[key] = data
        data.subscribe(lambda bd: self._on_data_update(key, bd))

    def get_data(self, key: str) -> Optional[BoundData]:
        return self._data_slots.get(key)

    def _on_data_update(self, key: str, bd: BoundData) -> None:
        """
        Fired when bound data changes.
        Subclasses override this to react — e.g. trigger a geometry rebuild.
        Base implementation emits a warning if the scene isn't playing.
        """
        warnings.warn(
            f"{self.__class__.__name__}: data '{key}' updated but "
            f"_on_data_update not overridden. Call animate_update() manually.",
            stacklevel=2,
        )

    # ── label convenience wrappers ────────────────────────────────────────

    def add_label(self, key: str, cfg: LabelConfig) -> Mobject:
        mob = self._labels.add(key, cfg)
        if self._labels.group not in self.submobjects:
            self.add(self._labels.group)
        return mob

    def remove_label(self, key: str) -> None:
        self._labels.remove(key)

    def update_label(self, key: str, new_text: str) -> None:
        self._labels.update_text(key, new_text)

    def show_label(self, key: str) -> None:
        self._labels.show(key)

    def hide_label(self, key: str) -> None:
        self._labels.hide(key)

    # ── material update ───────────────────────────────────────────────────

    def set_material(self, mat: MaterialConfig) -> "StatsObject3D":
        """Re-apply a new material to all VMobject children."""
        self._material = mat
        for sub in self.get_all_points_defining_boundary():
            if isinstance(sub, VMobject):
                MaterialApplicator.apply(sub, mat, self._palette)
        return self

    # ── composition helpers (instance-level shortcuts) ────────────────────

    @property
    def composition(self) -> CompositionHelper:
        return CompositionHelper

    # ── selection ─────────────────────────────────────────────────────────

    def select(self) -> None:
        """Mark as selected — adds a glow outline."""
        if not self._selected:
            self._selected     = True
            self._outline_mob  = HighlightSystem.outline(
                self, self._anim_cfg)
            self.add(self._outline_mob)

    def deselect(self) -> None:
        if self._selected:
            self._selected = False
            if hasattr(self, "_outline_mob"):
                self.remove(self._outline_mob)

    # ── internal ──────────────────────────────────────────────────────────

    def _on_theme_change(self, mode: Optional[ThemeMode]) -> None:
        """
        Callback fired by ``StatsTheme.set()``.
        If this object does not have a per-object override, it rebuilds
        its palette reference and re-applies materials.
        """
        if self._theme_mode is None:
            self._palette = StatsTheme.current
            # Trigger a lightweight colour-only refresh
            self._refresh_colors()

    def _refresh_colors(self) -> None:
        """
        Re-apply the current palette to all VMobject children.
        Subclasses with complex coloring logic should override this.
        """
        MaterialApplicator.apply(self, self._material, self._palette)

    def _unfold_build(self, submobs: List[Mobject],
                      cfg: AnimationConfig) -> Animation:
        """Reveal sub-mobjects axis by axis (X→Y→Z groups)."""
        # Simple approximation: split submobs into thirds and stagger
        third = max(1, len(submobs) // 3)
        groups = [
            submobs[:third],
            submobs[third:2*third],
            submobs[2*third:],
        ]
        stages = []
        for g in groups:
            if g:
                stages.append(AnimationGroup(
                    *[FadeIn(m, shift=OUT * 0.1) for m in g],
                    run_time=cfg.run_time / 3,
                ))
        return Succession(*stages, lag_ratio=0)

    def __del__(self) -> None:
        """Clean up registry reference on garbage collection."""
        try:
            StatsObject3D._registry.remove(self)
        except (ValueError, AttributeError):
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 11.  CONVENIENCE BASE VARIANTS
# ─────────────────────────────────────────────────────────────────────────────

class StatsSurface3D(StatsObject3D):
    """
    Base for any asset that is primarily a ``Surface`` (distribution curves,
    probability density surfaces, bivariate normals).

    Adds:
        • ``u_range``, ``v_range`` resolution control
        • ``checkerboard_colors`` from theme palette
        • Automatic shaded region support
    """

    DEFAULT_RESOLUTION: ClassVar[Tuple[int, int]] = (64, 64)

    def __init__(
        self,
        u_range: Tuple[float, float] = (0, 1),
        v_range: Tuple[float, float] = (0, 1),
        resolution: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> None:
        self._u_range    = u_range
        self._v_range    = v_range
        self._resolution = resolution or self.DEFAULT_RESOLUTION
        super().__init__(**kwargs)

    def _make_surface(
        self,
        func: Callable[[float, float], np.ndarray],
    ) -> Surface:
        """Build a ``Surface`` from a parametric function."""
        surf = Surface(
            func,
            u_range=self._u_range,
            v_range=self._v_range,
            resolution=self._resolution,
            fill_opacity=self._material.fill_opacity,
        )
        t = self._palette
        surf.set_fill_by_checkerboard(t.primary, t.secondary, opacity=0.85)
        return surf

    def shade_region(
        self,
        func: Callable[[float, float], np.ndarray],
        u_range: Tuple[float, float],
        color: Optional[str] = None,
        opacity: float = 0.55,
    ) -> Surface:
        """
        Create a highlighted sub-region of the surface (e.g. area under curve,
        rejection region).  Returns a new Surface; caller adds it to the scene.
        """
        t   = self._palette
        col = color or t.positive
        region = Surface(
            func,
            u_range=u_range,
            v_range=self._v_range,
            resolution=(max(16, self._resolution[0]//2), self._resolution[1]),
        )
        region.set_fill(col, opacity=opacity)
        region.set_stroke(col, width=1.5)
        return region


class StatsChart3D(StatsObject3D):
    """
    Base for chart-type assets (bar charts, histograms, scatter plots).

    Adds:
        • ``axes`` reference (``ThreeDAxes``)
        • Coordinate-to-scene-point conversion helpers
        • Automatic axis label placement
    """

    def __init__(
        self,
        x_range: Tuple[float, float, float] = (-5, 5, 1),
        y_range: Tuple[float, float, float] = (0, 1, 0.2),
        z_range: Tuple[float, float, float] = (-5, 5, 1),
        axis_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self._x_range    = x_range
        self._y_range    = y_range
        self._z_range    = z_range
        self._axis_config = axis_config or {}
        self.axes: Optional[ThreeDAxes] = None   # set in _build_geometry
        super().__init__(**kwargs)

    def _build_axes(self) -> ThreeDAxes:
        t = self._palette
        default_cfg: Dict[str, Any] = {
            "x_range": list(self._x_range),
            "y_range": list(self._y_range),
            "z_range": list(self._z_range),
            "tips": True,
            "axis_config": {
                "color": t.neutral,
                "stroke_width": 1.8,
                "include_numbers": True,
                "decimal_number_config": {
                    "color": t.text_secondary,
                    "font_size": 22,
                },
            },
        }
        default_cfg.update(self._axis_config)
        axes = ThreeDAxes(**default_cfg)
        axes.set_color(t.neutral)
        return axes

    def c2p(self, x: float, y: float, z: float = 0.0) -> np.ndarray:
        """Convert data coordinates to scene 3-D point."""
        if self.axes is None:
            raise RuntimeError("Axes not yet built. Call _build_axes() first.")
        return self.axes.c2p(x, y, z)

    def add_axis_labels(
        self,
        x_label: str = "x",
        y_label: str = "y",
        z_label: str = "z",
        is_math: bool = False,
    ) -> VGroup:
        """Attach axis labels and return them as a VGroup."""
        if self.axes is None:
            raise RuntimeError("Axes not yet built.")
        t   = self._palette
        cls = MathTex if is_math else Text
        xl  = cls(x_label, font_size=28, color=t.text_primary)
        yl  = cls(y_label, font_size=28, color=t.text_primary)
        zl  = cls(z_label, font_size=28, color=t.text_primary)
        xl.next_to(self.axes.x_axis, RIGHT + DOWN * 0.3)
        yl.next_to(self.axes.y_axis, UP)
        zl.next_to(self.axes.z_axis, OUT)
        grp = VGroup(xl, yl, zl)
        self.add(grp)
        return grp


class StatsProp3D(StatsObject3D):
    """
    Base for physical prop assets (coins, dice, urns, cards).

    Adds:
        • ``scale_factor`` for uniform resizing
        • ``face_labels`` dict for labelling each face
        • ``throw()`` / ``flip()`` animation helpers
    """

    def __init__(
        self,
        scale_factor: float = 1.0,
        face_labels: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> None:
        self._scale_factor = scale_factor
        self._face_labels  = face_labels or {}
        super().__init__(**kwargs)
        if scale_factor != 1.0:
            self.scale(scale_factor)

    def throw(self, cfg: Optional[AnimationConfig] = None) -> Animation:
        """
        Default "throw" animation — arc upward then fall back.
        Subclasses (Die3D, Coin3D) override with physics-like rotations.
        """
        cfg = cfg or self._anim_cfg
        from manim import MoveAlongPath, Arc
        arc = Arc(radius=1.5, start_angle=0, angle=PI)
        return MoveAlongPath(self, arc, run_time=cfg.run_time,
                             rate_func=there_and_back)


# ─────────────────────────────────────────────────────────────────────────────
# 12.  MODULE-LEVEL EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Enumerations
    "ThemeMode",
    "MaterialStyle",
    "LabelAnchor",
    "HighlightStyle",
    "BuildStyle",
    "DataUpdateMode",

    # Theme
    "StatsColorPalette",
    "StatsTheme",

    # Material
    "MaterialConfig",
    "MaterialApplicator",

    # Labels
    "LabelConfig",
    "LabelAttachment",

    # Data
    "BoundData",

    # Animation
    "AnimationConfig",
    "AnimationProtocol",

    # Highlight
    "HighlightSystem",
    "interpolate",

    # Composition
    "CompositionHelper",

    # Camera
    "CameraAwareness",

    # Base classes
    "StatsObject3D",
    "StatsSurface3D",
    "StatsChart3D",
    "StatsProp3D",
]