"""
Card Probability Statistics Scene
===================================
A multi-act Manim animation demonstrating probability concepts using a deck of cards.

Acts:
  1. Deal the deck, introduce the sample space
  2. Visualise P(Heart) with a Venn diagram
  3. Conditional probability — P(Face | Red)
  4. Binomial distribution — drawing with replacement n times
  5. Hypergeometric distribution — drawing WITHOUT replacement
  6. Bayes update — posterior after seeing a red card

Run with:manim -pql card_probability_scene.py CardProbabilityScene
    
    manim -pqh card_probability_scene.py CardProbabilityScene   # high quality
"""

from __future__ import annotations
import numpy as np
from manim import *

# ── manim_stats imports (adjust path if needed) ──────────────────────────────
from manim_stats.props.card import (
    Card3D,
    Deck3D,
    CardSuit,
    CardValue,
    CardFace,
    CardFacing,
    standard_deck,
    suit_subset,
    face_cards_only,
)

from manim_stats.props.coin import make_coin_row
from manim_stats.probability.venn3d import VennDiagram3D, VennData2, VennConfig
from manim_stats.probability.sample_space import SampleSpace3D, SampleSpaceConfig
from manim_stats.probability.bayes import (
    BayesBox3D, PriorPosteriorBar3D, BayesFormulaBanner,
    compute_joint_probs, build_bayes_scene,
)
from manim_stats.distributions.discrete_dists import (
    BinomialDistribution3D,
    HypergeometricDistribution3D,
)
from manim_stats.ui.labels import StatLabel3D, FormulaPanel3D
from manim_stats.ui.panels import InfoPanel3D, HypothesisPanel3D
from manim_stats.core.base import StatsTheme, StatsColorPalette
from manim_stats.core.colors import get_theme

# ─────────────────────────────────────────────────────────────────────────────
#  Helper constants
# ─────────────────────────────────────────────────────────────────────────────
THEME        = get_theme("dark")
PALETTE      = THEME.palette("discrete")

RED_SUIT     = ManimColor("#E63946")
BLACK_SUIT   = ManimColor("#E8E8E8")
GOLD         = ManimColor("#FFD166")
TEAL         = ManimColor("#06D6A0")
PURPLE_ACC   = ManimColor("#9B5DE5")
BG_COLOR     = ManimColor("#0D1117")

CARD_FACTS = {
    "Total cards":        52,
    "Hearts (♥)":         13,
    "Red cards":          26,
    "Face cards":         12,
    "P(Heart)":           "13/52 = 1/4",
    "P(Face | Red)":      "6/26 = 3/13",
    "P(Ace)":             "4/52 = 1/13",
}

# ─────────────────────────────────────────────────────────────────────────────
#  Scene
# ─────────────────────────────────────────────────────────────────────────────
class CardProbabilityScene(ThreeDScene):
    """Full multi-act card-probability demonstration."""

    # ── camera ───────────────────────────────────────────────────────────────
    def setup(self):
        self.camera.background_color = BG_COLOR
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)

    # ── entry point ──────────────────────────────────────────────────────────
    def construct(self):
        self._act0_title()
        self._act1_deck_and_sample_space()
        self._act2_venn_suit_probability()
        self._act3_conditional_probability()
        self._act4_binomial_with_replacement()
        self._act5_hypergeometric_no_replacement()
        self._act6_bayes_update()
        self._act7_summary()

    # =========================================================================
    #  ACT 0 — Title card
    # =========================================================================
    def _act0_title(self):
        title = Text("Cards & Probability", font="Georgia", color=GOLD)
        title.scale(1.4).move_to(ORIGIN)

        sub = Text(
            "A statistical journey through a 52-card deck",
            font="Georgia", color=WHITE,
        ).scale(0.55).next_to(title, DOWN, buff=0.4)

        deck_icon = self._tiny_deck_icon()
        deck_icon.next_to(sub, DOWN, buff=0.6)

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(sub, shift=UP * 0.3), FadeIn(deck_icon), run_time=1.2)
        self.wait(1.5)
        self.play(FadeOut(VGroup(title, sub, deck_icon)))

    # =========================================================================
    #  ACT 1 — Deck layout & sample space overview
    # =========================================================================
    def _act1_deck_and_sample_space(self):
        self._section_banner("Act 1 · The Sample Space")

        # Build a 52-card grid (face-up, 13 × 4)
        deck = standard_deck(shuffle=False)

        # Arrange cards in a 13 × 4 grid (cols=values, rows=suits)
        suits = [CardSuit.HEARTS, CardSuit.DIAMONDS, CardSuit.CLUBS, CardSuit.SPADES]
        values = list(CardValue)

        card_mobs = VGroup()
        suit_rows: dict[CardSuit, VGroup] = {}

        for row_i, suit in enumerate(suits):
            row = VGroup()
            for col_i, value in enumerate(values):
                card = Card3D(
    face=CardFace(suit=suit, value=value),
    facing=CardFacing.FACE_UP,
)
                card.scale(0.18)
                card.move_to(
                    RIGHT * (col_i - 6) * 0.55
                    + UP  * (1.5 - row_i * 0.75)
                )
                row.add(card)
                card_mobs.add(card)
            suit_rows[suit] = row

        # Title
        title = Text("The 52-Card Sample Space", font="Georgia", color=GOLD)
        title.scale(0.65).to_edge(UP, buff=0.15)

        self.play(Write(title))
        self.play(
            LaggedStart(
                *[FadeIn(c, scale=0.5) for c in card_mobs],
                lag_ratio=0.02,
            ),
            run_time=3,
        )
        self.wait(1)

        # Highlight hearts row
        hearts_row = suit_rows[CardSuit.HEARTS]
        self.play(
            hearts_row.animate.set_color(RED_SUIT),
            run_time=0.8,
        )
        label_h = Text("13 Hearts  →  P(♥) = 13/52 = 0.25", font="Georgia",
                       color=RED_SUIT).scale(0.45)
        label_h.next_to(card_mobs, DOWN, buff=0.35)
        self.play(Write(label_h))
        self.wait(1.2)

        # Highlight face cards
        face_cards_mobs = VGroup(
            *[c for c in card_mobs if c.value.is_face_card]
        )
        self.play(
            face_cards_mobs.animate.set_color(GOLD),
            run_time=0.8,
        )
        label_f = Text("12 Face Cards  →  P(Face) = 12/52 ≈ 0.231", font="Georgia",
                       color=GOLD).scale(0.45)
        label_f.next_to(label_h, DOWN, buff=0.2)
        self.play(Write(label_f))
        self.wait(1.5)

        self.play(FadeOut(VGroup(card_mobs, title, label_h, label_f)))

    # =========================================================================
    #  ACT 2 — Venn diagram: P(Heart) vs P(Red)
    # =========================================================================
    def _act2_venn_suit_probability(self):
        self._section_banner("Act 2 · Venn Diagram  ·  P(Heart) & P(Red)")

        # Hearts ∩ Red = Hearts (13); Red-only = Diamonds (13); neither = 26
        # P(A=Heart) = 13/52,  P(B=Red) = 26/52,  P(A∩B) = 13/52
        venn_data = VennData2(
              p_a=13 / 52,
              p_b=26 / 52,
              p_ab=13 / 52,

              label_a="H",
              label_b="R",
          )

        cfg = VennConfig(
              layout="standard",
              show_inclusion_exclusion=True,
              show_prob_labels=True,
              show_set_labels=True,
          )

        venn = VennDiagram3D.two_set(
              venn_data,
              config=cfg,
              total_n=52,
          )

        venn.scale(0.85).move_to(LEFT * 1.5)

        # Info panel on the right
        panel = InfoPanel3D(
    title="Probability Facts",
    accent_color=PURPLE_ACC,
)
        panel.add_section("Basic", [
            ("P(Heart)",    "13/52 = 0.250"),
            ("P(Red)",      "26/52 = 0.500"),
            ("P(H ∩ Red)",  "13/52 = 0.250"),
            ("P(H | Red)",  "13/26 = 0.500"),
        ])
        panel.scale(0.7).move_to(RIGHT * 3.5)

        self.play(venn.animate_grow_circles(), run_time=1.5)
        self.play(venn.animate_fill_zones(), run_time=1.2)
        self.play(panel.animate_build(), run_time=1)
        self.wait(1.5)

        # Highlight intersection
        self.play(venn.animate_highlight_zone("ab"), run_time=0.8)
        label = Text("Hearts ⊂ Red  →  P(H ∩ Red) = P(H)", font="Georgia",
                     color=WHITE).scale(0.42)
        label.to_edge(DOWN, buff=0.3)
        self.play(Write(label))
        self.wait(2)

        self.play(FadeOut(VGroup(venn, panel, label)))

    # =========================================================================
    #  ACT 3 — Conditional probability: P(Face | Red)
    # =========================================================================
    def _act3_conditional_probability(self):
        self._section_banner("Act 3 · Conditional Probability  ·  P(Face | Red)")

        # Draw only the red cards (26), highlight the 6 red face cards
        red_cards = VGroup()
        face_red  = VGroup()

        suits_red = [CardSuit.HEARTS, CardSuit.DIAMONDS]
        for row_i, suit in enumerate(suits_red):
            for col_i, value in enumerate(CardValue):
                card = Card3D(
    face=CardFace(suit=suit, value=value),
    facing=CardFacing.FACE_UP,
)
                card.scale(0.22)
                card.move_to(
                    RIGHT * (col_i - 6) * 0.60
                    + UP  * (0.5 - row_i * 1.0)
                )
                red_cards.add(card)
                if value.is_face_card:
                    face_red.add(card)

        title = Text("Red cards only  (26 cards)", font="Georgia",
                     color=RED_SUIT).scale(0.55).to_edge(UP, buff=0.2)

        self.play(Write(title))
        self.play(
            LaggedStart(*[FadeIn(c, scale=0.6) for c in red_cards], lag_ratio=0.03),
            run_time=2.5,
        )
        self.wait(0.8)

        # Highlight the 6 face cards
        self.play(face_red.animate.set_color(GOLD), run_time=0.7)

        formula = MathTex(
            r"P(\text{Face} \mid \text{Red}) = \frac{6}{26} = \frac{3}{13} \approx 0.231",
            color=GOLD,
        ).scale(0.75).to_edge(DOWN, buff=0.4)

        self.play(Write(formula))
        self.wait(2)

        # Quick comparison: unconditional P(Face) = 12/52
        formula2 = MathTex(
            r"P(\text{Face}) = \frac{12}{52} = \frac{3}{13} \approx 0.231",
            color=TEAL,
        ).scale(0.75).next_to(formula, UP, buff=0.25)

        note = Text("Same! Suit color and face-card status are INDEPENDENT.",
                    font="Georgia", color=WHITE).scale(0.4)
        note.next_to(formula2, UP, buff=0.2)

        self.play(Write(formula2))
        self.play(Write(note))
        self.wait(2)

        self.play(FadeOut(VGroup(red_cards, title, formula, formula2, note)))

    # =========================================================================
    #  ACT 4 — Binomial: drawing WITH replacement
    # =========================================================================
    def _act4_binomial_with_replacement(self):
        self._section_banner("Act 4 · Binomial Distribution  ·  Draw With Replacement")

        # n=10 draws, p=13/52=0.25 for Hearts
        n, p = 10, 0.25

        axes = ThreeDAxes(
            x_range=[0, n + 1, 1],
            y_range=[0, 0.5, 0.1],
            z_range=[0, 1, 1],
        )

        def data_to_scene_length(length, axis):
            if axis in ("x", 0):
                p1 = axes.c2p(0, 0, 0)
                p2 = axes.c2p(length, 0, 0)
            elif axis in ("y", 1):
                p1 = axes.c2p(0, 0, 0)
                p2 = axes.c2p(0, length, 0)
            else:
                p1 = axes.c2p(0, 0, 0)
                p2 = axes.c2p(0, 0, length)

            return np.linalg.norm(p2 - p1)

        axes.data_to_scene_length = data_to_scene_length

        dist = BinomialDistribution3D(
            axes=axes,
            n=n,
            p=p,
        )       
        self.add(axes)
        dist.scale(0.85).move_to(LEFT * 1)

        label = Text(
            f"Binomial(n={n}, p=1/4)\n"
            "X = # Hearts in 10 draws (with replacement)",
            font="Georgia", color=WHITE,
        ).scale(0.42).to_edge(DOWN, buff=0.35)

        self.move_camera(phi=50 * DEGREES, theta=-30 * DEGREES, run_time=1)
        self.play(dist.animate_build(), run_time=1.5)
        self.play(Write(label))

        # Shade P(X ≥ 4)
        shade = dist.shade_at_least(4)
        self.play(FadeIn(shade), run_time=0.8)
        prob_label = MathTex(
            r"P(X \geq 4) \approx 0.224", color=PURPLE_ACC
        ).scale(0.65).next_to(label, UP, buff=0.15)
        self.play(Write(prob_label))
        self.wait(1.5)

        # Animate sweep p from 0.1 → 0.5
        sweep_lbl = Text("Sweeping p …", font="Georgia",
                         color=GOLD).scale(0.4).next_to(prob_label, UP, buff=0.15)
        self.play(Write(sweep_lbl))
        self.play(dist.animate_p_sweep(0.10, 0.50), run_time=3)
        self.wait(1)

        self.play(FadeOut(VGroup(dist, label, prob_label, sweep_lbl)))

# =========================================================================
#  ACT 5 — Hypergeometric: drawing WITHOUT replacement
# =========================================================================
    def _act5_hypergeometric_no_replacement(self):
        self._section_banner("Act 5 · Hypergeometric  ·  Draw Without Replacement")

        # N=52, K=13 (hearts), n=5 draws
        N, K, n_draw = 52, 13, 5

        axes = ThreeDAxes(
        x_range=[0, n_draw + 1, 1],
        y_range=[0, 0.5, 0.1],
        z_range=[0, 1, 1],
    )

    # ---------------------------------------------------------------------
    # Patch required by manim_stats distribution classes
    # ---------------------------------------------------------------------
        def data_to_scene_length(length, axis):
            if axis in ("x", 0):
                p1 = axes.c2p(0, 0, 0)
                p2 = axes.c2p(length, 0, 0)

            elif axis in ("y", 1):
                p1 = axes.c2p(0, 0, 0)
                p2 = axes.c2p(0, length, 0)

            else:
                p1 = axes.c2p(0, 0, 0)
                p2 = axes.c2p(0, 0, length)

            return np.linalg.norm(p2 - p1)

        axes.data_to_scene_length = data_to_scene_length

    # ---------------------------------------------------------------------
    # Hypergeometric distribution
    # ---------------------------------------------------------------------
        dist = HypergeometricDistribution3D(
        axes=axes,
        M=N,        # population size
        n=K,        # success states
        N=n_draw,   # draws
    )

        self.add(axes)

        dist.scale(0.85).move_to(LEFT * 1)

        label = Text(
            f"Hypergeometric(N=52, K=13, n={n_draw})\n"
            "X = # Hearts in 5 draws WITHOUT replacement",
            font="Georgia",
            color=WHITE,
        ).scale(0.42).to_edge(DOWN, buff=0.35)

        comp_label = Text(
        "Compare: Binomial overestimates variance slightly",
        font="Georgia",
        color=TEAL,
        ).scale(0.38).next_to(label, UP, buff=0.15)

        self.move_camera(phi=60 * DEGREES, theta=-40 * DEGREES, run_time=1)

        self.play(dist.animate_build(), run_time=1.5)
        self.play(Write(label))

    # Optional comparison animation
        comparison = dist.compare_to_binomial()
        self.play(Create(comparison), run_time=1.2)

        self.play(Write(comp_label))
        self.wait(1)

    # Shade P(X = 0)
        if hasattr(dist, "shade_at_most"):
            shade = dist.shade_at_most(0)
            self.play(FadeIn(shade), run_time=0.6)

        p0 = MathTex(
            r"P(X=0) \approx 0.222",
            color=RED_SUIT,
        ).scale(0.6)

        p0.next_to(comp_label, UP, buff=0.15)

        self.play(Write(p0))
        self.wait(1.5)

    # Sweep draw count if supported
        if hasattr(dist, "animate_n_sweep"):
            self.play(
                dist.animate_n_sweep(1, 15),
                run_time=3.5,
            )

        self.wait(1)

        self.play(
        FadeOut(
            VGroup(
                dist,
                axes,
                label,
                comp_label,
                p0,
            )
        )
        )

# =========================================================================
#  ACT 6 — Bayes update: seeing a red card
# =========================================================================
    def _act6_bayes_update(self):
        self._section_banner("Act 6 · Bayes' Theorem  ·  Is It a Heart?")

    # ------------------------------------------------------------
    # Probabilities
    # ------------------------------------------------------------
        p_h = 13 / 52          # Prior P(Heart)
        p_r_h = 1.0            # P(Red | Heart)
        p_r_nh = 13 / 39       # P(Red | Not Heart)

        posterior_val = (
        p_h * p_r_h
        / (p_h * p_r_h + (1 - p_h) * p_r_nh)
        )

    # ------------------------------------------------------------
    # Bayes box
    # ------------------------------------------------------------
        bayes_box = BayesBox3D(
        p_h=p_h,
        p_e_h=p_r_h,
        p_e_nh=p_r_nh,
        )

        bayes_box.scale(0.8)
        bayes_box.move_to(LEFT * 3)

    # ------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------
        heart_lbl = Text(
        "Hypothesis: Heart",
        font="Georgia",
        color=RED_SUIT,
        ).scale(0.38)

        evidence_lbl = Text(
        "Evidence: Red card",
        font="Georgia",
        color=GOLD,
        ).scale(0.38)

        heart_lbl.next_to(bayes_box, UP, buff=0.3)
        evidence_lbl.next_to(bayes_box, DOWN, buff=0.3)

    # ------------------------------------------------------------
    # Formula banner
    # ------------------------------------------------------------
        formula_banner = BayesFormulaBanner()

        formula_banner.scale(0.6)
        formula_banner.to_edge(DOWN, buff=0.3)

    # ------------------------------------------------------------
    # Manual prior/posterior bars
    # ------------------------------------------------------------
        prior_title = Text(
        "Prior",
        font="Georgia",
        color=WHITE,
        ).scale(0.4)

        posterior_title = Text(
        "Posterior",
        font="Georgia",
        color=WHITE,
        ).scale(0.4)

        prior_bar_bg = Rectangle(
        width=0.6,
        height=3,
        color=WHITE,
        )

        prior_bar_fill = Rectangle(
        width=0.6,
        height=3 * p_h,
        fill_color=TEAL,
        fill_opacity=1,
        stroke_width=0,
        )

        prior_bar_fill.align_to(prior_bar_bg, DOWN)

        posterior_bar_bg = Rectangle(
        width=0.6,
        height=3,
        color=WHITE,
    )

        posterior_bar_fill = Rectangle(
        width=0.6,
        height=3 * posterior_val,
        fill_color=GOLD,
        fill_opacity=1,
        stroke_width=0,
    )

        posterior_bar_fill.align_to(posterior_bar_bg, DOWN)

        prior_group = VGroup(
            prior_bar_bg,
            prior_bar_fill,
            prior_title,
    )

        posterior_group = VGroup(
            posterior_bar_bg,
            posterior_bar_fill,
            posterior_title,
    )

        prior_title.next_to(prior_bar_bg, UP, buff=0.2)
        posterior_title.next_to(posterior_bar_bg, UP, buff=0.2)

        bars = VGroup(prior_group, posterior_group)
        bars.arrange(RIGHT, buff=1.2)

        bars.move_to(RIGHT * 3)

    # Numeric labels
        prior_num = MathTex(
        r"0.25",
        color=TEAL,
    ).scale(0.6)

        posterior_num = MathTex(
        r"0.50",
        color=GOLD,
    ).scale(0.6)

        prior_num.next_to(prior_bar_bg, DOWN, buff=0.2)
        posterior_num.next_to(posterior_bar_bg, DOWN, buff=0.2)

    # ------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------
        self.move_camera(
        phi=55 * DEGREES,
        theta=-35 * DEGREES,
        run_time=1,
    )

    # ------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------
        if hasattr(bayes_box, "animate_build_box"):
            self.play(
                bayes_box.animate_build_box(),
                run_time=1.5,
            )
        else:
            self.play(FadeIn(bayes_box))

        self.play(
           FadeIn(heart_lbl),
            FadeIn(evidence_lbl),
        )

        if hasattr(formula_banner, "animate_appear"):
            self.play(
                formula_banner.animate_appear(),
                run_time=1,
            )
        else:
            self.play(FadeIn(formula_banner))

        if hasattr(bayes_box, "animate_reveal_posterior"):
            self.play(
                bayes_box.animate_reveal_posterior(),
                run_time=1,
            )

        self.play(
            FadeIn(prior_group),
            FadeIn(posterior_group),
        )

        self.play(
            Write(prior_num),
            Write(posterior_num),
        )

        self.wait(1)

    # ------------------------------------------------------------
    # Result equation
    # ------------------------------------------------------------
        result_lbl = MathTex(
        r"P(\mathrm{Heart}\mid\mathrm{Red})"
        r"=\frac{13}{26}=0.5",
        color=GOLD,
    )

        result_lbl.scale(0.7)
        result_lbl.next_to(
        formula_banner,
        UP,
        buff=0.25,
    )

        explanation = Text(
        "Seeing a red card doubles the probability.",
        font="Georgia",
        color=WHITE,
    ).scale(0.38)

        explanation.next_to(
        result_lbl,
        UP,
        buff=0.2,
    )

        self.play(Write(result_lbl))
        self.play(FadeIn(explanation))

        self.wait(2)

    # ------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------
        self.play(
        FadeOut(
            VGroup(
                bayes_box,
                heart_lbl,
                evidence_lbl,
                formula_banner,
                bars,
                prior_num,
                posterior_num,
                result_lbl,
                explanation,
            )
        )
    )
        
    #  ACT 7 — Summary
    # =========================================================================
    def _act7_summary(self):
        self._section_banner("Summary")
        self.move_camera(phi=0 * DEGREES, theta=-90 * DEGREES, run_time=1.2)

        rows = [
            ("Sample space",             "52 equally-likely outcomes",      WHITE),
            ("P(Heart)",                 "13/52 = 0.25",                    RED_SUIT),
            ("P(Face | Red)",            "6/26 = 3/13 ≈ 0.231",            GOLD),
            ("Independence",             "Color ⊥ Face-card status",        TEAL),
            ("Binomial (replacement)",   "n=10, p=1/4  →  E[X]=2.5",       PURPLE_ACC),
            ("Hypergeometric (no repl)", "N=52, K=13, n=5  →  E[X]=1.25",  TEAL),
            ("Bayes update",             "P(Heart | Red) = 0.50",           GOLD),
        ]

        title = Text("Key Results", font="Georgia", color=GOLD)
        title.scale(0.9).to_edge(UP, buff=0.3)
        self.play(Write(title))

        row_mobs = VGroup()
        for i, (key, val, col) in enumerate(rows):
            k = Text(f"• {key}:", font="Georgia", color=col).scale(0.48)
            v = Text(val, font="Georgia", color=WHITE).scale(0.48)
            v.next_to(k, RIGHT, buff=0.3)
            row = VGroup(k, v)
            row.move_to(UP * (2.2 - i * 0.65))
            row_mobs.add(row)

        self.play(
            LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in row_mobs],
                        lag_ratio=0.18),
            run_time=2.5,
        )
        self.wait(3)

        final = Text("Probability — the mathematics of uncertainty.",
                     font="Georgia", color=GOLD).scale(0.55)
        final.to_edge(DOWN, buff=0.4)
        self.play(Write(final))
        self.wait(2)
        self.play(FadeOut(VGroup(title, row_mobs, final)))

    # =========================================================================
    #  Private helpers
    # =========================================================================
    def _section_banner(self, text: str):
        """Flash a section title then clear it."""
        banner = Text(text, font="Georgia", color=GOLD)
        banner.scale(0.85).move_to(ORIGIN)
        self.play(FadeIn(banner, scale=1.1), run_time=0.6)
        self.wait(0.8)
        self.play(FadeOut(banner, shift=UP * 0.3), run_time=0.4)

    def _tiny_deck_icon(self) -> VGroup:
        """A stacked set of coloured rectangles that looks like a card deck."""
        cards = VGroup()
        colors = [RED_SUIT, BLACK_SUIT, TEAL, GOLD, PURPLE_ACC]
        for i, col in enumerate(colors):
            rect = RoundedRectangle(
                corner_radius=0.06,
                width=0.9, height=1.25,
                color=col, fill_color=col, fill_opacity=0.9,
            )
            rect.move_to(RIGHT * i * 0.12 + UP * i * 0.08)
            cards.add(rect)
        return cards