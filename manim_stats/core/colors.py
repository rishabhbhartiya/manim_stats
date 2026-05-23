"""
manim_stats/core/colors.py
==========================
A full-featured, multi-layer color system for statistical visualization.

Architecture
------------
  Layer 0  StatColor          — single color with accessibility utilities
  Layer 1  ColorFamily        — semantically grouped color ramp (base/light/dark/muted…)
  Layer 2  DistributionPalette— per-distribution-family curated palettes
  Layer 3  StatsTheme         — full scene theme (bg, axes, grid, annotation, …)
  Layer 4  Manim helpers      — gradient ramps, colormaps, diverging maps

Usage
-----
    from manim_stats.core.colors import DARK_THEME, NORMAL_FAMILY, StatColor

    blue = StatColor.from_hex("#185FA5")
    print(blue.luminance)            # 0.107
    print(blue.contrast_ratio(StatColor.from_hex("#ffffff")))  # ≈8.4

    ramp = NORMAL_FAMILY.gradient(n=256)   # list of 256 StatColors
    theme = DARK_THEME
    theme.apply(scene)                     # patches a Manim scene's background etc.
"""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Attempt graceful Manim import — all Manim-specific helpers are conditional
# ---------------------------------------------------------------------------
try:
    from manim import ManimColor          # type: ignore
    _MANIM_AVAILABLE = True
except ImportError:                       # running outside a Manim environment
    _MANIM_AVAILABLE = False
    ManimColor = None                     # placeholder so type hints don't crash


# ===========================================================================
# LAYER 0 — StatColor
# A single, immutable color value with rich utilities.
# ===========================================================================

@dataclass(frozen=True)
class StatColor:
    """
    Immutable color with perceptual and accessibility utilities.

    Attributes
    ----------
    r, g, b : float
        Linear (0–1) channel values *after* sRGB gamma removal.
        Store linear so that luminance / blending math is correct.
    _hex : str
        Canonical 6-digit lowercase hex string (e.g. "#185fa5").
        Always computed from r, g, b — never stored separately.

    Notes
    -----
    We store *linear* RGB internally.  Most color math (luminance,
    contrast ratio, interpolation) requires linear values.  Conversion
    to 8-bit sRGB happens only at display time (``hex``, ``rgb8``).
    """

    r: float   # 0.0–1.0  linear
    g: float
    b: float

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_hex(cls, hex_str: str) -> "StatColor":
        """Parse #RRGGBB or #RGB hex strings → StatColor."""
        h = hex_str.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) != 6:
            raise ValueError(f"Invalid hex color: '{hex_str}'")
        r8, g8, b8 = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return cls(
            r=_srgb_to_linear(r8 / 255),
            g=_srgb_to_linear(g8 / 255),
            b=_srgb_to_linear(b8 / 255),
        )

    @classmethod
    def from_rgb8(cls, r: int, g: int, b: int) -> "StatColor":
        """Create from 8-bit sRGB integers (0–255)."""
        return cls(
            r=_srgb_to_linear(r / 255),
            g=_srgb_to_linear(g / 255),
            b=_srgb_to_linear(b / 255),
        )

    @classmethod
    def from_rgb_float(cls, r: float, g: float, b: float) -> "StatColor":
        """Create from perceptual (sRGB gamma-encoded) floats (0.0–1.0)."""
        return cls(
            r=_srgb_to_linear(r),
            g=_srgb_to_linear(g),
            b=_srgb_to_linear(b),
        )

    @classmethod
    def from_hsl(cls, h: float, s: float, l: float) -> "StatColor":
        """Create from HSL (0–360, 0–1, 0–1)."""
        r, g, b = colorsys.hls_to_rgb(h / 360, l, s)
        return cls.from_rgb_float(r, g, b)

    # ------------------------------------------------------------------
    # Display / export properties
    # ------------------------------------------------------------------

    @property
    def rgb8(self) -> Tuple[int, int, int]:
        """Gamma-encoded 8-bit sRGB tuple (0–255)."""
        return (
            round(_linear_to_srgb(self.r) * 255),
            round(_linear_to_srgb(self.g) * 255),
            round(_linear_to_srgb(self.b) * 255),
        )

    @property
    def hex(self) -> str:
        """Lowercase #rrggbb string."""
        r8, g8, b8 = self.rgb8
        return f"#{r8:02x}{g8:02x}{b8:02x}"

    @property
    def hsl(self) -> Tuple[float, float, float]:
        """HSL tuple — hue 0–360, saturation 0–1, lightness 0–1."""
        r, g, b = (_linear_to_srgb(c) for c in (self.r, self.g, self.b))
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        return (h * 360, s, l)

    @property
    def luminance(self) -> float:
        """
        Relative luminance per WCAG 2.1, §1.4.3.
        Range 0.0 (black) … 1.0 (white).
        """
        return 0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b

    # ------------------------------------------------------------------
    # Accessibility
    # ------------------------------------------------------------------

    def contrast_ratio(self, other: "StatColor") -> float:
        """
        WCAG 2.1 contrast ratio between self and *other*.
        Formula: (L1 + 0.05) / (L2 + 0.05)  where L1 ≥ L2.
        AA normal text requires ≥ 4.5 : 1, AA large ≥ 3 : 1.
        AAA normal ≥ 7 : 1.
        """
        l1, l2 = self.luminance, other.luminance
        if l2 > l1:
            l1, l2 = l2, l1
        return (l1 + 0.05) / (l2 + 0.05)

    def is_accessible(
        self,
        background: "StatColor",
        level: str = "AA",
        large_text: bool = False,
    ) -> bool:
        """
        Return True if this foreground color meets WCAG *level* on *background*.

        Parameters
        ----------
        level : {'AA', 'AAA'}
        large_text : bool
            Large text (≥ 18pt or bold ≥ 14pt) has relaxed thresholds.
        """
        ratio = self.contrast_ratio(background)
        thresholds = {
            ("AA",  False): 4.5,
            ("AA",  True ): 3.0,
            ("AAA", False): 7.0,
            ("AAA", True ): 4.5,
        }
        required = thresholds.get((level, large_text), 4.5)
        return ratio >= required

    def best_label_color(
        self, candidates: Optional[Sequence["StatColor"]] = None
    ) -> "StatColor":
        """
        Pick the candidate with the highest contrast ratio against self
        (use self as background).  Defaults to choosing between white/black.
        """
        if candidates is None:
            candidates = [WHITE, BLACK]
        return max(candidates, key=lambda c: c.contrast_ratio(self))

    # ------------------------------------------------------------------
    # Color manipulation
    # ------------------------------------------------------------------

    def lighten(self, amount: float) -> "StatColor":
        """
        Increase lightness by *amount* (0–1) in HSL space.
        ``amount=0`` → unchanged, ``amount=1`` → white.
        """
        h, s, l = self.hsl
        return StatColor.from_hsl(h, s, min(1.0, l + amount * (1.0 - l)))

    def darken(self, amount: float) -> "StatColor":
        """Decrease lightness by *amount* in HSL space."""
        h, s, l = self.hsl
        return StatColor.from_hsl(h, s, max(0.0, l - amount * l))

    def desaturate(self, amount: float) -> "StatColor":
        """Reduce saturation by *amount* (0–1)."""
        h, s, l = self.hsl
        return StatColor.from_hsl(h, max(0.0, s - amount * s), l)

    def saturate(self, amount: float) -> "StatColor":
        """Increase saturation by *amount* (0–1)."""
        h, s, l = self.hsl
        return StatColor.from_hsl(h, min(1.0, s + amount * (1.0 - s)), l)

    def shift_hue(self, degrees: float) -> "StatColor":
        """Rotate hue by *degrees* (can be negative)."""
        h, s, l = self.hsl
        return StatColor.from_hsl((h + degrees) % 360, s, l)

    def with_alpha(self, alpha: float) -> "StatColorAlpha":
        """Return a new StatColorAlpha with this color and the given *alpha* (0–1)."""
        return StatColorAlpha(color=self, alpha=alpha)

    def mix(self, other: "StatColor", t: float = 0.5) -> "StatColor":
        """
        Linear blend in *linear* RGB space.
        ``t=0`` → self, ``t=1`` → other.  This is perceptually correct
        unlike sRGB blending (which darkens the midpoint).
        """
        return StatColor(
            r=self.r + t * (other.r - self.r),
            g=self.g + t * (other.g - self.g),
            b=self.b + t * (other.b - self.b),
        )

    # ------------------------------------------------------------------
    # Manim integration
    # ------------------------------------------------------------------

    def to_manim(self) -> "ManimColor":
        """
        Convert to a ``manim.ManimColor`` (hex string form).
        Raises ``ImportError`` if Manim is not installed.
        """
        if not _MANIM_AVAILABLE:
            raise ImportError(
                "Manim is not installed.  Install it with 'pip install manim'."
            )
        return ManimColor(self.hex)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"StatColor({self.hex!r})"

    def __str__(self) -> str:
        return self.hex


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StatColorAlpha:
    """A StatColor paired with an opacity value (0–1)."""
    color: StatColor
    alpha: float = 1.0

    @property
    def rgba8(self) -> Tuple[int, int, int, int]:
        r, g, b = self.color.rgb8
        return (r, g, b, round(self.alpha * 255))

    @property
    def hex_alpha(self) -> str:
        r, g, b, a = self.rgba8
        return f"#{r:02x}{g:02x}{b:02x}{a:02x}"

    def to_manim(self) -> "ManimColor":
        if not _MANIM_AVAILABLE:
            raise ImportError("Manim is not installed.")
        return ManimColor(self.hex_alpha)


# ---------------------------------------------------------------------------
# sRGB ↔ linear conversion helpers (IEC 61966-2-1)
# ---------------------------------------------------------------------------

def _srgb_to_linear(c: float) -> float:
    """Gamma-expand one sRGB channel to linear light (0–1)."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    """Gamma-compress one linear channel to sRGB (0–1)."""
    c = max(0.0, min(1.0, c))
    return c * 12.92 if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055


# ===========================================================================
# LAYER 1 — ColorFamily
# A semantically complete color ramp for one hue family.
# ===========================================================================

@dataclass
class ColorFamily:
    """
    A curated ramp for a single hue family, expressed as named stops.

    Each stop name maps to a specific visual role so that code using
    a ColorFamily is self-documenting:

        NORMAL_FAMILY.highlight    # vivid fill for the main curve
        NORMAL_FAMILY.muted        # secondary, background shading
        NORMAL_FAMILY.on_dark      # readable label when placed on a dark bg

    Attributes
    ----------
    name : str
        Human-readable family name, e.g. "Normal / Gaussian".
    base : StatColor
        The primary color — used for main strokes, bars, curve lines.
    light : StatColor
        Lightened variant — used for area fills, shaded regions.
    dark : StatColor
        Darkened variant — used for borders, selected state, emphasis.
    muted : StatColor
        Desaturated variant — used for secondary distributions, grid.
    highlight : StatColor
        High-saturation / high-lightness variant for call-outs, ticks.
    on_dark : StatColor
        Light-enough to read when placed on a very dark background.
    on_light : StatColor
        Dark-enough to read when placed on a white / very light background.
    """

    name:      str
    base:      StatColor
    light:     StatColor
    dark:      StatColor
    muted:     StatColor
    highlight: StatColor
    on_dark:   StatColor     # readable on dark themes
    on_light:  StatColor     # readable on light themes

    # ------------------------------------------------------------------
    # Gradient / ramp helpers
    # ------------------------------------------------------------------

    def gradient(self, n: int = 256) -> list[StatColor]:
        """
        Return *n* StatColors interpolated from ``dark`` → ``base`` → ``light``.
        Useful as a sequential colormap for heatmaps / PDFs.
        """
        half = n // 2
        stops: list[StatColor] = []
        for i in range(half):
            t = i / max(half - 1, 1)
            stops.append(self.dark.mix(self.base, t))
        for i in range(n - half):
            t = i / max(n - half - 1, 1)
            stops.append(self.base.mix(self.light, t))
        return stops

    def diverging(
        self,
        other: "ColorFamily",
        n: int = 256,
        midpoint: "StatColor" = None,
    ) -> list[StatColor]:
        """
        Build a diverging colormap from ``self.dark`` through an optional
        neutral *midpoint* to ``other.dark``.

        Parameters
        ----------
        other : ColorFamily
            The color family at the opposite pole (e.g. REGRESSION_FAMILY).
        n : int
            Number of discrete steps.
        midpoint : StatColor, optional
            Neutral center color.  Defaults to a desaturated mid-gray.
        """
        mid = midpoint or NEUTRAL_MID
        left_steps = n // 2
        right_steps = n - left_steps
        left  = [self.dark.mix(mid, i / max(left_steps - 1, 1))  for i in range(left_steps)]
        right = [mid.mix(other.dark, i / max(right_steps - 1, 1)) for i in range(right_steps)]
        return left + right

    def shade(self, t: float) -> StatColor:
        """
        Interpolate along the full ramp: t=0 → dark, t=0.5 → base, t=1 → light.
        """
        if t <= 0.5:
            return self.dark.mix(self.base, t * 2)
        return self.base.mix(self.light, (t - 0.5) * 2)

    def with_opacity(self, alpha: float) -> "ColorFamilyAlpha":
        """Return a version of this family where every stop has the given alpha."""
        return ColorFamilyAlpha(family=self, alpha=alpha)

    def to_matplotlib_cmap(self, name: Optional[str] = None):
        """
        Build a ``matplotlib.colors.LinearSegmentedColormap`` from this family.
        Requires matplotlib ≥ 3.5.

        Returns
        -------
        matplotlib.colors.LinearSegmentedColormap
        """
        try:
            from matplotlib.colors import LinearSegmentedColormap
        except ImportError:
            raise ImportError("matplotlib is required for to_matplotlib_cmap()")
        colors = [(c.r, c.g, c.b) for c in self.gradient(256)]
        return LinearSegmentedColormap.from_list(name or self.name, colors)

    def __repr__(self) -> str:
        return f"ColorFamily({self.name!r}, base={self.base})"


@dataclass
class ColorFamilyAlpha:
    family: ColorFamily
    alpha: float = 1.0

    def __getattr__(self, item: str):
        """Delegate attribute access to the underlying family, wrapping with alpha."""
        color = getattr(self.family, item)
        if isinstance(color, StatColor):
            return color.with_alpha(self.alpha)
        return color


# ===========================================================================
# LAYER 2 — DistributionPalette
# One palette per statistical family / topic area.
# ===========================================================================

@dataclass
class DistributionPalette:
    """
    Color assignments for a distribution family or statistical topic.

    Each slot is a :class:`ColorFamily` assigned to a semantic role
    within the family so that multi-distribution plots are consistent
    across different scenes.

    Attributes
    ----------
    name : str
        Human-readable group name, e.g. "Normal / Gaussian".
    primary : ColorFamily
        Main distribution / main curve.
    secondary : ColorFamily
        Secondary overlaid distribution, comparison curve.
    shading : ColorFamily
        Area-under-curve fills, region shading.
    annotation : ColorFamily
        Arrows, callout boxes, equation panels.
    critical : ColorFamily
        Rejection regions, outlier highlights, error markers.
    neutral : ColorFamily
        Grid, tick marks, reference lines.
    """

    name:       str
    primary:    ColorFamily
    secondary:  ColorFamily
    shading:    ColorFamily
    annotation: ColorFamily
    critical:   ColorFamily
    neutral:    ColorFamily


# ===========================================================================
# LAYER 3 — StatsTheme
# Full scene theme binding all color decisions to Manim Mobject properties.
# ===========================================================================

class ThemeMode(Enum):
    DARK       = "dark"
    LIGHT      = "light"
    PAPER      = "paper"
    NEON       = "neon"
    PASTEL     = "pastel"
    MONOCHROME = "monochrome"


@dataclass
class StatsTheme:
    """
    A complete visual theme for a Manim stats scene.

    Contains colors for every scene element so that a single
    ``theme.apply(scene)`` call or ``theme.as_kwargs()`` dict
    patches backgrounds, axes, grids, labels, etc. consistently.

    Parameters
    ----------
    mode : ThemeMode
        Named preset this theme belongs to.
    background : StatColor
        Scene / canvas background.
    surface : StatColor
        Card / panel backgrounds (slightly lighter / darker than background).
    grid_major : StatColorAlpha
        Major grid line color (usually semi-transparent).
    grid_minor : StatColorAlpha
        Minor grid line color (more transparent).
    axis : StatColor
        Axis spine color.
    tick : StatColor
        Tick mark color.
    label : StatColor
        Axis / data label text color.
    title : StatColor
        Chart title text color.
    annotation : StatColor
        Arrow and annotation text color.
    highlight : StatColor
        Selection / focus highlight ring color.
    distribution_palettes : dict[str, DistributionPalette]
        Mapping of palette name → DistributionPalette for this theme.
    """

    mode:        ThemeMode
    background:  StatColor
    surface:     StatColor
    grid_major:  StatColorAlpha
    grid_minor:  StatColorAlpha
    axis:        StatColor
    tick:        StatColor
    label:       StatColor
    title:       StatColor
    annotation:  StatColor
    highlight:   StatColor
    distribution_palettes: dict = field(default_factory=dict)

    # ------------------------------------------------------------------

    def apply(self, scene) -> None:
        """
        Patch a Manim *scene* object's background color.

        Requires Manim.  Call in ``scene.setup()`` or ``construct()``.
        """
        if not _MANIM_AVAILABLE:
            raise ImportError("Manim is required for theme.apply(scene).")
        scene.camera.background_color = self.background.to_manim()

    def axes_kwargs(self) -> dict:
        """
        Return a kwargs dict suitable for ``manim.Axes(...)``.
        Covers ``axis_config``, ``tip_shape``, color assignments.
        """
        if not _MANIM_AVAILABLE:
            raise ImportError("Manim is required for axes_kwargs().")
        return {
            "axis_config": {
                "color":            self.axis.to_manim(),
                "tick_size":        0.06,
                "include_tip":      True,
                "tip_width":        0.15,
                "tip_height":       0.15,
                "stroke_width":     2.0,
                "include_numbers":  False,
            },
            "x_axis_config": {"color": self.axis.to_manim()},
            "y_axis_config": {"color": self.axis.to_manim()},
        }

    def number_plane_kwargs(self) -> dict:
        """Kwargs for ``manim.NumberPlane`` matching this theme's grid style."""
        if not _MANIM_AVAILABLE:
            raise ImportError("Manim is required for number_plane_kwargs().")
        return {
            "background_line_style": {
                "stroke_color": self.grid_major.color.to_manim(),
                "stroke_width": 1.0,
                "stroke_opacity": self.grid_major.alpha,
            },
            "faded_line_style": {
                "stroke_color": self.grid_minor.color.to_manim(),
                "stroke_width": 0.5,
                "stroke_opacity": self.grid_minor.alpha,
            },
            "faded_line_ratio": 4,
        }

    def text_style(self) -> dict:
        """Return a dict of ``manim.Text`` / ``MathTex`` color kwargs."""
        if not _MANIM_AVAILABLE:
            raise ImportError("Manim is required.")
        return {"color": self.label.to_manim()}

    def register_custom_palette(
        self, key: str, palette: DistributionPalette
    ) -> None:
        """Add or replace a distribution palette in this theme."""
        self.distribution_palettes[key] = palette

    def palette(self, key: str) -> DistributionPalette:
        """Retrieve a distribution palette by key, with a helpful error."""
        try:
            return self.distribution_palettes[key]
        except KeyError:
            available = ", ".join(self.distribution_palettes.keys())
            raise KeyError(
                f"Palette {key!r} not found in theme {self.mode.value!r}. "
                f"Available: {available}"
            )

    def __repr__(self) -> str:
        return f"StatsTheme(mode={self.mode.value!r})"


# ===========================================================================
# LAYER 4 — Gradient / colormap factory functions
# These are pure-Python; Manim usage is opt-in.
# ===========================================================================

def gradient_ramp(
    colors: Sequence[StatColor],
    n: int = 256,
    weights: Optional[Sequence[float]] = None,
) -> list[StatColor]:
    """
    Build a smooth *n*-step ramp through an arbitrary list of *colors*.

    Parameters
    ----------
    colors : sequence of StatColor
        Key colors.  At least 2 required.
    n : int
        Number of output steps.
    weights : sequence of float, optional
        Relative weight of each segment between consecutive key colors.
        Length must be ``len(colors) - 1``.  Uniform by default.

    Returns
    -------
    list[StatColor] of length *n*.
    """
    if len(colors) < 2:
        raise ValueError("Need at least 2 colors for a gradient ramp.")
    segments = len(colors) - 1
    if weights is None:
        weights = [1.0] * segments
    if len(weights) != segments:
        raise ValueError("len(weights) must equal len(colors) - 1.")
    total_w = sum(weights)
    step_counts = [max(1, round(w / total_w * n)) for w in weights]
    # Adjust last segment so total == n
    step_counts[-1] += n - sum(step_counts)

    ramp: list[StatColor] = []
    for i, (a, b, steps) in enumerate(
        zip(colors[:-1], colors[1:], step_counts)
    ):
        for j in range(steps):
            t = j / max(steps - 1, 1) if steps > 1 else 0.0
            ramp.append(a.mix(b, t))
    return ramp[:n]


def diverging_map(
    low: StatColor,
    high: StatColor,
    n: int = 256,
    mid: Optional[StatColor] = None,
) -> list[StatColor]:
    """
    Create a symmetric diverging colormap low → mid → high.

    Parameters
    ----------
    low, high : StatColor
        Extreme pole colors.
    n : int
        Total number of steps (should be odd for a true midpoint).
    mid : StatColor, optional
        Center neutral color.  Defaults to NEUTRAL_MID.
    """
    center = mid or NEUTRAL_MID
    half = n // 2
    left  = gradient_ramp([low, center], n=half)
    right = gradient_ramp([center, high], n=n - half)
    return left + right


def sequential_map(
    family: ColorFamily,
    n: int = 256,
) -> list[StatColor]:
    """Sequential colormap from a ColorFamily's dark → base → light."""
    return family.gradient(n)


def qualitative_palette(
    families: Sequence[ColorFamily],
    stop: str = "base",
) -> list[StatColor]:
    """
    Extract one color per family for qualitative (categorical) plots.

    Parameters
    ----------
    families : sequence of ColorFamily
    stop : {'base', 'dark', 'light', 'muted', 'highlight', 'on_dark', 'on_light'}
        Which stop to pull from each family.
    """
    return [getattr(f, stop) for f in families]


def heatmap_colormap(
    low: StatColor,
    high: StatColor,
    n: int = 256,
    gamma: float = 1.0,
) -> list[StatColor]:
    """
    Sequential heatmap colormap with optional *gamma* correction for
    emphasis on high values (gamma > 1) or low values (gamma < 1).
    """
    ramp = gradient_ramp([low, high], n=n)
    if gamma != 1.0:
        ramp = [
            low.mix(high, (i / (n - 1)) ** gamma)
            for i in range(n)
        ]
    return ramp


def interpolate_colors(
    c1: StatColor, c2: StatColor, steps: int = 10
) -> list[StatColor]:
    """Return *steps* evenly-spaced colors from c1 to c2 (inclusive)."""
    return [c1.mix(c2, i / max(steps - 1, 1)) for i in range(steps)]


# ===========================================================================
# RAW PALETTE — canonical StatColor singletons
# Named after semantic role, not hue, so the module is theme-agnostic.
# ===========================================================================

# --- Achromatic ---
BLACK        = StatColor.from_hex("#000000")
WHITE        = StatColor.from_hex("#ffffff")
NEUTRAL_MID  = StatColor.from_hex("#9e9e9e")   # WCAG-neutral midpoint

# --- Blues (normal distributions, frequentist inference) ---
BLUE_50      = StatColor.from_hex("#e3f0fc")
BLUE_100     = StatColor.from_hex("#b6d8f7")
BLUE_200     = StatColor.from_hex("#82baf0")
BLUE_400     = StatColor.from_hex("#3a8fd6")
BLUE_600     = StatColor.from_hex("#185fa5")
BLUE_700     = StatColor.from_hex("#0f4b87")
BLUE_800     = StatColor.from_hex("#0a3568")
BLUE_900     = StatColor.from_hex("#051f45")

# --- Teals (exponential / survival analysis) ---
TEAL_50      = StatColor.from_hex("#e0f5ee")
TEAL_100     = StatColor.from_hex("#9de0cb")
TEAL_200     = StatColor.from_hex("#5cc8a4")
TEAL_400     = StatColor.from_hex("#1d9d74")
TEAL_600     = StatColor.from_hex("#0f6e56")
TEAL_700     = StatColor.from_hex("#095040")
TEAL_800     = StatColor.from_hex("#063530")
TEAL_900     = StatColor.from_hex("#02201e")

# --- Ambers (discrete distributions, Poisson, Binomial) ---
AMBER_50     = StatColor.from_hex("#fdf3d9")
AMBER_100    = StatColor.from_hex("#fad882")
AMBER_200    = StatColor.from_hex("#efb230")
AMBER_400    = StatColor.from_hex("#c47b10")
AMBER_600    = StatColor.from_hex("#8b500b")
AMBER_700    = StatColor.from_hex("#6a3a07")
AMBER_800    = StatColor.from_hex("#4c2804")
AMBER_900    = StatColor.from_hex("#2e1602")

# --- Coral/Red (inference, rejection regions, critical values) ---
CORAL_50     = StatColor.from_hex("#fceee9")
CORAL_100    = StatColor.from_hex("#f7c9b6")
CORAL_200    = StatColor.from_hex("#f09a7d")
CORAL_400    = StatColor.from_hex("#d95b31")
CORAL_600    = StatColor.from_hex("#993d1d")
CORAL_700    = StatColor.from_hex("#782c12")
CORAL_800    = StatColor.from_hex("#551d0b")
CORAL_900    = StatColor.from_hex("#320e04")

# --- Purples (regression, Bayesian posterior) ---
PURPLE_50    = StatColor.from_hex("#eeedfe")
PURPLE_100   = StatColor.from_hex("#cecbf6")
PURPLE_200   = StatColor.from_hex("#afa9ec")
PURPLE_400   = StatColor.from_hex("#7f77dd")
PURPLE_600   = StatColor.from_hex("#534ab7")
PURPLE_700   = StatColor.from_hex("#3e3490")
PURPLE_800   = StatColor.from_hex("#2c2269")
PURPLE_900   = StatColor.from_hex("#1a1244")

# --- Pinks (probability, combinatorics) ---
PINK_50      = StatColor.from_hex("#fbeaf0")
PINK_100     = StatColor.from_hex("#f4c0d1")
PINK_200     = StatColor.from_hex("#ed93b1")
PINK_400     = StatColor.from_hex("#d4537e")
PINK_600     = StatColor.from_hex("#993556")
PINK_700     = StatColor.from_hex("#76233f")
PINK_800     = StatColor.from_hex("#53132a")
PINK_900     = StatColor.from_hex("#300618")

# --- Greens (information theory, entropy) ---
GREEN_50     = StatColor.from_hex("#eaf3de")
GREEN_100    = StatColor.from_hex("#c0dd97")
GREEN_200    = StatColor.from_hex("#97c459")
GREEN_400    = StatColor.from_hex("#639922")
GREEN_600    = StatColor.from_hex("#3b6d11")
GREEN_700    = StatColor.from_hex("#275009")
GREEN_800    = StatColor.from_hex("#173404")
GREEN_900    = StatColor.from_hex("#091d01")

# --- Grays (neutral, structural) ---
GRAY_50      = StatColor.from_hex("#f4f2ec")
GRAY_100     = StatColor.from_hex("#d4d2c8")
GRAY_200     = StatColor.from_hex("#b4b2a9")
GRAY_400     = StatColor.from_hex("#888780")
GRAY_600     = StatColor.from_hex("#5f5e5a")
GRAY_700     = StatColor.from_hex("#444440")
GRAY_800     = StatColor.from_hex("#2d2c2a")
GRAY_900     = StatColor.from_hex("#181816")

# --- Scene backgrounds ---
BG_DARK      = StatColor.from_hex("#0f0f14")   # Rich near-black, bluer than gray
BG_MEDIUM    = StatColor.from_hex("#1c1c24")   # Standard Manim dark
BG_LIGHT     = StatColor.from_hex("#f8f7f2")   # Off-white paper
BG_PAPER     = StatColor.from_hex("#fdf8ed")   # Warm parchment


# ===========================================================================
# LAYER 1 — ColorFamily instances
# ===========================================================================

NORMAL_FAMILY = ColorFamily(
    name      = "Normal / Gaussian",
    base      = BLUE_600,
    light     = BLUE_200,
    dark      = BLUE_800,
    muted     = BLUE_100,
    highlight = BLUE_400,
    on_dark   = BLUE_100,
    on_light  = BLUE_800,
)

EXPONENTIAL_FAMILY = ColorFamily(
    name      = "Exponential / Survival",
    base      = TEAL_600,
    light     = TEAL_200,
    dark      = TEAL_800,
    muted     = TEAL_100,
    highlight = TEAL_400,
    on_dark   = TEAL_100,
    on_light  = TEAL_800,
)

GAMMA_FAMILY = ColorFamily(
    name      = "Gamma / Chi-Squared",
    base      = TEAL_400,
    light     = TEAL_100,
    dark      = TEAL_700,
    muted     = TEAL_50,
    highlight = TEAL_200,
    on_dark   = TEAL_50,
    on_light  = TEAL_700,
)

DISCRETE_FAMILY = ColorFamily(
    name      = "Discrete (Binomial / Poisson / Geometric)",
    base      = AMBER_600,
    light     = AMBER_200,
    dark      = AMBER_800,
    muted     = AMBER_100,
    highlight = AMBER_400,
    on_dark   = AMBER_100,
    on_light  = AMBER_800,
)

INFERENCE_FAMILY = ColorFamily(
    name      = "Hypothesis Testing / Inference",
    base      = CORAL_600,
    light     = CORAL_200,
    dark      = CORAL_800,
    muted     = CORAL_100,
    highlight = CORAL_400,
    on_dark   = CORAL_100,
    on_light  = CORAL_800,
)

REGRESSION_FAMILY = ColorFamily(
    name      = "Regression / Correlation",
    base      = PURPLE_600,
    light     = PURPLE_200,
    dark      = PURPLE_800,
    muted     = PURPLE_100,
    highlight = PURPLE_400,
    on_dark   = PURPLE_100,
    on_light  = PURPLE_800,
)

PROBABILITY_FAMILY = ColorFamily(
    name      = "Probability / Combinatorics",
    base      = PINK_600,
    light     = PINK_200,
    dark      = PINK_800,
    muted     = PINK_100,
    highlight = PINK_400,
    on_dark   = PINK_100,
    on_light  = PINK_800,
)

INFORMATION_FAMILY = ColorFamily(
    name      = "Information Theory",
    base      = GREEN_600,
    light     = GREEN_200,
    dark      = GREEN_800,
    muted     = GREEN_100,
    highlight = GREEN_400,
    on_dark   = GREEN_100,
    on_light  = GREEN_800,
)

NEUTRAL_FAMILY = ColorFamily(
    name      = "Neutral / Structural",
    base      = GRAY_600,
    light     = GRAY_200,
    dark      = GRAY_800,
    muted     = GRAY_100,
    highlight = GRAY_400,
    on_dark   = GRAY_100,
    on_light  = GRAY_800,
)

#: All distribution ColorFamilies in a canonical order for qualitative plots.
ALL_FAMILIES: list[ColorFamily] = [
    NORMAL_FAMILY,
    EXPONENTIAL_FAMILY,
    GAMMA_FAMILY,
    DISCRETE_FAMILY,
    INFERENCE_FAMILY,
    REGRESSION_FAMILY,
    PROBABILITY_FAMILY,
    INFORMATION_FAMILY,
]


# ===========================================================================
# LAYER 2 — DistributionPalette instances
# ===========================================================================

NORMAL_PALETTE = DistributionPalette(
    name       = "Normal / Gaussian",
    primary    = NORMAL_FAMILY,
    secondary  = ColorFamily(          # t-distribution companion
        name="Student-t",
        base=BLUE_400, light=BLUE_100, dark=BLUE_700,
        muted=BLUE_50, highlight=BLUE_200,
        on_dark=BLUE_100, on_light=BLUE_700,
    ),
    shading    = ColorFamily(          # area-under-curve fill
        name="Normal area",
        base=BLUE_200, light=BLUE_50, dark=BLUE_400,
        muted=BLUE_50, highlight=BLUE_100,
        on_dark=BLUE_50, on_light=BLUE_600,
    ),
    annotation = NEUTRAL_FAMILY,
    critical   = INFERENCE_FAMILY,
    neutral    = NEUTRAL_FAMILY,
)

DISCRETE_PALETTE = DistributionPalette(
    name       = "Discrete distributions",
    primary    = DISCRETE_FAMILY,
    secondary  = ColorFamily(
        name="Geometric secondary",
        base=AMBER_400, light=AMBER_100, dark=AMBER_700,
        muted=AMBER_50, highlight=AMBER_200,
        on_dark=AMBER_100, on_light=AMBER_700,
    ),
    shading    = ColorFamily(
        name="Discrete area",
        base=AMBER_200, light=AMBER_50, dark=AMBER_400,
        muted=AMBER_50, highlight=AMBER_100,
        on_dark=AMBER_50, on_light=AMBER_600,
    ),
    annotation = NEUTRAL_FAMILY,
    critical   = INFERENCE_FAMILY,
    neutral    = NEUTRAL_FAMILY,
)

INFERENCE_PALETTE = DistributionPalette(
    name       = "Hypothesis testing",
    primary    = INFERENCE_FAMILY,    # rejection region
    secondary  = NORMAL_FAMILY,       # the null distribution
    shading    = ColorFamily(
        name="Acceptance region",
        base=BLUE_200, light=BLUE_50, dark=BLUE_400,
        muted=BLUE_50, highlight=BLUE_100,
        on_dark=BLUE_50, on_light=BLUE_700,
    ),
    annotation = NEUTRAL_FAMILY,
    critical   = INFERENCE_FAMILY,
    neutral    = NEUTRAL_FAMILY,
)

REGRESSION_PALETTE = DistributionPalette(
    name       = "Regression",
    primary    = REGRESSION_FAMILY,   # fitted line / plane
    secondary  = NORMAL_FAMILY,       # scatter points
    shading    = ColorFamily(
        name="CI band",
        base=PURPLE_200, light=PURPLE_50, dark=PURPLE_400,
        muted=PURPLE_50, highlight=PURPLE_100,
        on_dark=PURPLE_50, on_light=PURPLE_600,
    ),
    annotation = NEUTRAL_FAMILY,
    critical   = INFERENCE_FAMILY,
    neutral    = NEUTRAL_FAMILY,
)

BAYES_PALETTE = DistributionPalette(
    name       = "Bayesian",
    primary    = REGRESSION_FAMILY,   # posterior
    secondary  = NORMAL_FAMILY,       # likelihood
    shading    = PROBABILITY_FAMILY,  # prior
    annotation = NEUTRAL_FAMILY,
    critical   = INFERENCE_FAMILY,
    neutral    = NEUTRAL_FAMILY,
)


# ===========================================================================
# LAYER 3 — StatsTheme instances
# ===========================================================================

def _make_dark_theme(palettes: dict) -> StatsTheme:
    return StatsTheme(
        mode        = ThemeMode.DARK,
        background  = BG_DARK,
        surface     = StatColor.from_hex("#161620"),
        grid_major  = GRAY_600.with_alpha(0.30),
        grid_minor  = GRAY_600.with_alpha(0.12),
        axis        = GRAY_400,
        tick        = GRAY_400,
        label       = GRAY_100,
        title       = WHITE,
        annotation  = GRAY_200,
        highlight   = BLUE_400,
        distribution_palettes = palettes,
    )


def _make_light_theme(palettes: dict) -> StatsTheme:
    return StatsTheme(
        mode        = ThemeMode.LIGHT,
        background  = BG_LIGHT,
        surface     = StatColor.from_hex("#eeecea"),
        grid_major  = GRAY_400.with_alpha(0.35),
        grid_minor  = GRAY_400.with_alpha(0.15),
        axis        = GRAY_700,
        tick        = GRAY_600,
        label       = GRAY_800,
        title       = GRAY_900,
        annotation  = GRAY_700,
        highlight   = BLUE_600,
        distribution_palettes = palettes,
    )


def _make_paper_theme(palettes: dict) -> StatsTheme:
    return StatsTheme(
        mode        = ThemeMode.PAPER,
        background  = BG_PAPER,
        surface     = StatColor.from_hex("#f5edd8"),
        grid_major  = AMBER_600.with_alpha(0.20),
        grid_minor  = AMBER_600.with_alpha(0.08),
        axis        = StatColor.from_hex("#4a3e2c"),
        tick        = StatColor.from_hex("#6a5c40"),
        label       = StatColor.from_hex("#4a3e2c"),
        title       = StatColor.from_hex("#2e2418"),
        annotation  = StatColor.from_hex("#5a4e38"),
        highlight   = AMBER_400,
        distribution_palettes = palettes,
    )


def _make_neon_theme(palettes: dict) -> StatsTheme:
    """Neon: jet-black background, highly saturated vivid colors."""
    return StatsTheme(
        mode        = ThemeMode.NEON,
        background  = StatColor.from_hex("#0a0a0a"),
        surface     = StatColor.from_hex("#111118"),
        grid_major  = StatColor.from_hex("#00ffcc").with_alpha(0.12),
        grid_minor  = StatColor.from_hex("#00ffcc").with_alpha(0.05),
        axis        = StatColor.from_hex("#00ffcc"),
        tick        = StatColor.from_hex("#00e6b8"),
        label       = StatColor.from_hex("#ccffee"),
        title       = StatColor.from_hex("#ffffff"),
        annotation  = StatColor.from_hex("#80ffdd"),
        highlight   = StatColor.from_hex("#ff4af8"),
        distribution_palettes = palettes,
    )


def _make_pastel_theme(palettes: dict) -> StatsTheme:
    return StatsTheme(
        mode        = ThemeMode.PASTEL,
        background  = StatColor.from_hex("#f2eeff"),
        surface     = StatColor.from_hex("#e8e0ff"),
        grid_major  = PURPLE_200.with_alpha(0.35),
        grid_minor  = PURPLE_200.with_alpha(0.15),
        axis        = PURPLE_600,
        tick        = PURPLE_400,
        label       = PURPLE_700,
        title       = PURPLE_800,
        annotation  = PURPLE_600,
        highlight   = PINK_400,
        distribution_palettes = palettes,
    )


def _make_monochrome_theme(palettes: dict) -> StatsTheme:
    return StatsTheme(
        mode        = ThemeMode.MONOCHROME,
        background  = StatColor.from_hex("#1a1a1a"),
        surface     = StatColor.from_hex("#242424"),
        grid_major  = GRAY_600.with_alpha(0.40),
        grid_minor  = GRAY_600.with_alpha(0.15),
        axis        = GRAY_200,
        tick        = GRAY_400,
        label       = GRAY_100,
        title       = WHITE,
        annotation  = GRAY_200,
        highlight   = WHITE,
        distribution_palettes = palettes,
    )


# Build shared palette registry once
_DEFAULT_PALETTES: dict = {
    "normal":     NORMAL_PALETTE,
    "discrete":   DISCRETE_PALETTE,
    "inference":  INFERENCE_PALETTE,
    "regression": REGRESSION_PALETTE,
    "bayes":      BAYES_PALETTE,
}

DARK_THEME        = _make_dark_theme(_DEFAULT_PALETTES)
LIGHT_THEME       = _make_light_theme(_DEFAULT_PALETTES)
PAPER_THEME       = _make_paper_theme(_DEFAULT_PALETTES)
NEON_THEME        = _make_neon_theme(_DEFAULT_PALETTES)
PASTEL_THEME      = _make_pastel_theme(_DEFAULT_PALETTES)
MONOCHROME_THEME  = _make_monochrome_theme(_DEFAULT_PALETTES)

THEMES: dict[str, StatsTheme] = {
    "dark":       DARK_THEME,
    "light":      LIGHT_THEME,
    "paper":      PAPER_THEME,
    "neon":       NEON_THEME,
    "pastel":     PASTEL_THEME,
    "monochrome": MONOCHROME_THEME,
}


def get_theme(name: str) -> StatsTheme:
    """
    Retrieve a theme by name (case-insensitive).

    Parameters
    ----------
    name : str
        One of 'dark', 'light', 'paper', 'neon', 'pastel', 'monochrome'.

    Raises
    ------
    KeyError
        If *name* is not a known theme.
    """
    key = name.lower().strip()
    if key not in THEMES:
        raise KeyError(
            f"Unknown theme {name!r}.  "
            f"Available: {', '.join(THEMES.keys())}"
        )
    return THEMES[key]


def register_theme(name: str, theme: StatsTheme) -> None:
    """
    Register a custom theme globally so it can be retrieved with
    :func:`get_theme`.  Overwrites an existing entry if *name* already exists.
    """
    THEMES[name.lower().strip()] = theme


# ===========================================================================
# UTILITY — semantic color lookup shortcuts
# ===========================================================================

class SemanticRole(Enum):
    """Named color roles used across all scenes."""
    PRIMARY     = "primary"       # dominant distribution / main element
    SECONDARY   = "secondary"     # comparison / secondary element
    SHADING     = "shading"       # area fills (PDFs, histograms)
    ANNOTATION  = "annotation"    # arrows, callout text, formula panels
    CRITICAL    = "critical"      # rejection regions, errors, outliers
    NEUTRAL     = "neutral"       # grid, axes, structural elements
    HIGHLIGHT   = "highlight"     # selection, hover, animated emphasis


def resolve_color(
    role: SemanticRole,
    palette_key: str = "normal",
    theme: Optional[StatsTheme] = None,
    stop: str = "base",
) -> StatColor:
    """
    Resolve a semantic color to a StatColor for a given palette and theme.

    Parameters
    ----------
    role : SemanticRole
        The visual role to resolve.
    palette_key : str
        Which DistributionPalette to look in (e.g. 'normal', 'inference').
    theme : StatsTheme, optional
        Theme whose palettes to use.  Defaults to DARK_THEME.
    stop : str
        Which ColorFamily stop to pull ('base', 'light', 'dark', …).

    Examples
    --------
        >>> c = resolve_color(SemanticRole.CRITICAL, 'inference', LIGHT_THEME)
        >>> c.hex
        '#993d1d'
    """
    t = theme or DARK_THEME
    palette = t.palette(palette_key)
    family: ColorFamily = getattr(palette, role.value)
    return getattr(family, stop)


# ===========================================================================
# CONVENIENCE — pre-built colormaps ready for matplotlib / Manim use
# ===========================================================================

#: Sequential blue ramp — normal distribution, frequentist
CMAP_NORMAL        = gradient_ramp([BLUE_800, BLUE_600, BLUE_200, BLUE_50])

#: Sequential teal ramp — survival / exponential
CMAP_EXPONENTIAL   = gradient_ramp([TEAL_800, TEAL_600, TEAL_200, TEAL_50])

#: Sequential amber ramp — discrete distributions
CMAP_DISCRETE      = gradient_ramp([AMBER_800, AMBER_600, AMBER_200, AMBER_50])

#: Diverging: blue (negative) ↔ coral (positive) — for correlation, z-scores
CMAP_DIVERGING_ZS  = diverging_map(BLUE_700, CORAL_700, n=256, mid=GRAY_200)

#: Diverging: purple ↔ teal — for regression residuals
CMAP_DIVERGING_RES = diverging_map(PURPLE_700, TEAL_700, n=256, mid=GRAY_200)

#: Full-spectrum qualitative set — 8 families, one color each
QUALITATIVE_8 = qualitative_palette(ALL_FAMILIES, stop="base")


# ===========================================================================
# __all__  — public API
# ===========================================================================

__all__ = [
    # Core classes
    "StatColor",
    "StatColorAlpha",
    "ColorFamily",
    "ColorFamilyAlpha",
    "DistributionPalette",
    "StatsTheme",
    "ThemeMode",
    "SemanticRole",

    # Color family instances
    "NORMAL_FAMILY",
    "EXPONENTIAL_FAMILY",
    "GAMMA_FAMILY",
    "DISCRETE_FAMILY",
    "INFERENCE_FAMILY",
    "REGRESSION_FAMILY",
    "PROBABILITY_FAMILY",
    "INFORMATION_FAMILY",
    "NEUTRAL_FAMILY",
    "ALL_FAMILIES",

    # Distribution palette instances
    "NORMAL_PALETTE",
    "DISCRETE_PALETTE",
    "INFERENCE_PALETTE",
    "REGRESSION_PALETTE",
    "BAYES_PALETTE",

    # Theme instances
    "DARK_THEME",
    "LIGHT_THEME",
    "PAPER_THEME",
    "NEON_THEME",
    "PASTEL_THEME",
    "MONOCHROME_THEME",
    "THEMES",
    "get_theme",
    "register_theme",

    # Raw palette singletons
    "BLACK", "WHITE", "NEUTRAL_MID",
    "BLUE_50", "BLUE_100", "BLUE_200", "BLUE_400",
    "BLUE_600", "BLUE_700", "BLUE_800", "BLUE_900",
    "TEAL_50", "TEAL_100", "TEAL_200", "TEAL_400",
    "TEAL_600", "TEAL_700", "TEAL_800", "TEAL_900",
    "AMBER_50", "AMBER_100", "AMBER_200", "AMBER_400",
    "AMBER_600", "AMBER_700", "AMBER_800", "AMBER_900",
    "CORAL_50", "CORAL_100", "CORAL_200", "CORAL_400",
    "CORAL_600", "CORAL_700", "CORAL_800", "CORAL_900",
    "PURPLE_50", "PURPLE_100", "PURPLE_200", "PURPLE_400",
    "PURPLE_600", "PURPLE_700", "PURPLE_800", "PURPLE_900",
    "PINK_50", "PINK_100", "PINK_200", "PINK_400",
    "PINK_600", "PINK_700", "PINK_800", "PINK_900",
    "GREEN_50", "GREEN_100", "GREEN_200", "GREEN_400",
    "GREEN_600", "GREEN_700", "GREEN_800", "GREEN_900",
    "GRAY_50", "GRAY_100", "GRAY_200", "GRAY_400",
    "GRAY_600", "GRAY_700", "GRAY_800", "GRAY_900",
    "BG_DARK", "BG_MEDIUM", "BG_LIGHT", "BG_PAPER",

    # Colormap helpers
    "gradient_ramp",
    "diverging_map",
    "sequential_map",
    "qualitative_palette",
    "heatmap_colormap",
    "interpolate_colors",
    "resolve_color",

    # Pre-built colormaps
    "CMAP_NORMAL",
    "CMAP_EXPONENTIAL",
    "CMAP_DISCRETE",
    "CMAP_DIVERGING_ZS",
    "CMAP_DIVERGING_RES",
    "QUALITATIVE_8",
]