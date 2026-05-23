r"""
manim_stats/core/tex_utils.py
=============================
A comprehensive LaTeX formula system for statistical visualization.

Architecture
------------
  Layer A  TeX atom constants       — Greek letters, operators, spacing macros
  Layer B  Formula string builders  — pdf(), pmf(), cdf(), mean(), variance(), ...
  Layer C  TexFormula dataclass      — structured formula with named parts, steps
  Layer D  Annotation & coloring    — highlight_part(), overbrace(), annotate_step()
  Layer E  Manim MathTex builders   — to_mathtex(), to_aligned(), derive() anim
  Layer F  Formula registry         — FORMULAS catalog, topic index, derivation chains

Design principles
-----------------
* Formulas are *data*, not strings.  Every TexFormula carries its raw LaTeX,
  its named sub-parts (so callers can color "the mean" or "the variance" by
  name), and an optional step-by-step derivation chain.
* Parametric variants.  pdf_normal(mu=r"\mu_0") returns a fully substituted
  string, not a template with blanks.
* Manim-aware.  Every TexFormula knows how to emit a MathTex or Tex object,
  including a color_map that Manim uses to color individual sub-expressions.
* No hard Manim dependency.  All string-building works without Manim; the
  .to_mathtex() / .to_tex() / .derive() methods raise ImportError gracefully.

Quick start
-----------
    from manim_stats.core.tex_utils import FORMULAS, TexFormula

    # Simple string
    s = pdf_normal()

    # Structured formula with named parts
    f = FORMULAS["normal_pdf"]
    print(f.inline)          # $f(x ...)$
    print(f.parts["kernel"]) # the exponent substring
    print(f.steps)           # list of derivation steps

    # Manim usage
    tex = f.to_mathtex(color_map={"kernel": NORMAL_FAMILY.highlight.to_manim()})
    scene.play(Write(tex))
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Graceful Manim import
# ---------------------------------------------------------------------------
try:
    from manim import MathTex, Tex, ManimColor  # type: ignore
    _MANIM_AVAILABLE = True
except ImportError:
    _MANIM_AVAILABLE = False
    MathTex = ManimColor = Tex = None  # type: ignore


# ===========================================================================
# LAYER A — TeX atom constants
# Single-source-of-truth for every Greek letter, operator, and macro we use.
# Grouped so it is easy to find what you need without memorizing LaTeX.
# ===========================================================================

# ---------------------------------------------------------------------------
# Greek letters (lowercase)
# ---------------------------------------------------------------------------
class Greek:
    """Lowercase Greek letters as raw LaTeX strings."""
    alpha   = r"\alpha"
    beta    = r"\beta"
    gamma   = r"\gamma"
    delta   = r"\delta"
    epsilon = r"\epsilon"
    zeta    = r"\zeta"
    eta     = r"\eta"
    theta   = r"\theta"
    iota    = r"\iota"
    kappa   = r"\kappa"
    lam     = r"\lambda"       # 'lambda' is a Python keyword
    mu      = r"\mu"
    nu      = r"\nu"
    xi      = r"\xi"
    pi      = r"\pi"
    rho     = r"\rho"
    sigma   = r"\sigma"
    tau     = r"\tau"
    upsilon = r"\upsilon"
    phi     = r"\phi"
    varphi  = r"\varphi"
    chi     = r"\chi"
    psi     = r"\psi"
    omega   = r"\omega"


class GreekUpper:
    """Uppercase Greek letters."""
    Gamma   = r"\Gamma"
    Delta   = r"\Delta"
    Theta   = r"\Theta"
    Lambda  = r"\Lambda"
    Xi      = r"\Xi"
    Pi      = r"\Pi"
    Sigma   = r"\Sigma"
    Upsilon = r"\Upsilon"
    Phi     = r"\Phi"
    Psi     = r"\Psi"
    Omega   = r"\Omega"


# ---------------------------------------------------------------------------
# Mathematical operators & relations
# ---------------------------------------------------------------------------
class Op:
    """Common math operators as LaTeX strings."""
    # Arithmetic
    times    = r"\times"
    cdot     = r"\cdot"
    div      = r"\div"
    pm       = r"\pm"
    mp       = r"\mp"

    # Relations
    leq      = r"\leq"
    geq      = r"\geq"
    neq      = r"\neq"
    approx   = r"\approx"
    propto   = r"\propto"
    sim      = r"\sim"
    simeq    = r"\simeq"

    # Sets
    in_      = r"\in"
    notin    = r"\notin"
    subset   = r"\subset"
    cup      = r"\cup"
    cap      = r"\cap"
    emptyset = r"\emptyset"

    # Arrows
    to       = r"\to"
    gets     = r"\gets"
    implies  = r"\implies"
    iff      = r"\iff"

    # Logic
    forall   = r"\forall"
    exists   = r"\exists"
    land     = r"\land"
    lor      = r"\lor"
    lnot     = r"\lnot"

    # Calculus
    partial  = r"\partial"
    nabla    = r"\nabla"
    inf      = r"\infty"
    int_     = r"\int"
    iint     = r"\iint"
    sum_     = r"\sum"
    prod_    = r"\prod"


# ---------------------------------------------------------------------------
# Decorators (accents over a symbol)
# ---------------------------------------------------------------------------
class Dec:
    """LaTeX accent / decorator macros.  Each takes one argument."""
    @staticmethod
    def hat(x: str) -> str:
        return rf"\hat{{{x}}}"

    @staticmethod
    def bar(x: str) -> str:
        return rf"\bar{{{x}}}"

    @staticmethod
    def tilde(x: str) -> str:
        return rf"\tilde{{{x}}}"

    @staticmethod
    def vec(x: str) -> str:
        return rf"\vec{{{x}}}"

    @staticmethod
    def dot(x: str) -> str:
        return rf"\dot{{{x}}}"

    @staticmethod
    def ddot(x: str) -> str:
        return rf"\ddot{{{x}}}"

    @staticmethod
    def overline(x: str) -> str:
        return rf"\overline{{{x}}}"

    @staticmethod
    def underline(x: str) -> str:
        return rf"\underline{{{x}}}"

    @staticmethod
    def mathbf(x: str) -> str:
        return rf"\mathbf{{{x}}}"

    @staticmethod
    def boldsymbol(x: str) -> str:
        return rf"\boldsymbol{{{x}}}"

    @staticmethod
    def mathcal(x: str) -> str:
        return rf"\mathcal{{{x}}}"

    @staticmethod
    def mathbb(x: str) -> str:
        return rf"\mathbb{{{x}}}"

    @staticmethod
    def text(x: str) -> str:
        return rf"\text{{{x}}}"

    @staticmethod
    def mathrm(x: str) -> str:
        return rf"\mathrm{{{x}}}"


# ---------------------------------------------------------------------------
# Fences — matching delimiters
# ---------------------------------------------------------------------------
class Fence:
    """Produce matched LaTeX delimiters with \\left / \\right."""
    @staticmethod
    def paren(x: str) -> str:
        return rf"\left( {x} \right)"

    @staticmethod
    def bracket(x: str) -> str:
        return rf"\left[ {x} \right]"

    @staticmethod
    def brace(x: str) -> str:
        return rf"\left\{{ {x} \right\}}"

    @staticmethod
    def abs(x: str) -> str:
        return rf"\left| {x} \right|"

    @staticmethod
    def norm(x: str) -> str:
        return rf"\left\| {x} \right\|"

    @staticmethod
    def floor(x: str) -> str:
        return rf"\left\lfloor {x} \right\rfloor"

    @staticmethod
    def ceil(x: str) -> str:
        return rf"\left\lceil {x} \right\rceil"

    @staticmethod
    def angle(x: str) -> str:
        return rf"\left\langle {x} \right\rangle"


# ---------------------------------------------------------------------------
# Spacing
# ---------------------------------------------------------------------------
class Space:
    """Horizontal spacing macros."""
    thin   = r"\,"          # 3/18 em
    med    = r"\:"          # 4/18 em
    thick  = r"\;"          # 5/18 em
    quad   = r"\quad"       # 1 em
    qquad  = r"\qquad"      # 2 em
    neg    = r"\!"          # negative 3/18 em
    enspace = r"\enspace"   # 0.5 em


# ---------------------------------------------------------------------------
# Display math environment fragments
# ---------------------------------------------------------------------------
class Env:
    """LaTeX environment wrappers."""
    @staticmethod
    def display(x: str) -> str:
        return rf"$${x}$$"

    @staticmethod
    def inline(x: str) -> str:
        return rf"${x}$"

    @staticmethod
    def aligned(rows: list[tuple[str, str]]) -> str:
        r"""
        Build an ``align*``-style block from (left, right) pairs.

            aligned([("f(x)", r"= \frac{1}{\sigma}"), ("", r"\cdot e^{...}")])

        produces::

            \begin{aligned}
              f(x) &= \frac{1}{\sigma} \\
                   &\cdot e^{...}
            \end{aligned}
        """
        lines = r" \\ ".join(
            rf"{lhs} &= {rhs}" if lhs else rf"&\phantom{{=}} {rhs}"
            for lhs, rhs in rows
        )
        return rf"\begin{{aligned}} {lines} \end{{aligned}}"

    @staticmethod
    def cases(rows: list[tuple[str, str]]) -> str:
        r"""
        Build a ``cases`` environment.

            cases([("x^2", r"x \geq 0"), ("0", r"\text{otherwise}")])
        """
        body = r" \\ ".join(rf"{val} & {cond}" for val, cond in rows)
        return rf"\begin{{cases}} {body} \end{{cases}}"

    @staticmethod
    def piecewise(symbol: str, rows: list[tuple[str, str]]) -> str:
        return rf"{symbol} = {Env.cases(rows)}"

    @staticmethod
    def matrix(rows: list[list[str]], style: str = "pmatrix") -> str:
        """
        Build a matrix environment.

            matrix([["a","b"],["c","d"]])  → pmatrix
        """
        body = r" \\ ".join(" & ".join(row) for row in rows)
        return rf"\begin{{{style}}} {body} \end{{{style}}}"


# ===========================================================================
# LAYER B — Formula string builders
# Each function returns a raw LaTeX string (no $…$ wrapper).
# Callers receive strings they can embed, annotate, and style.
# ===========================================================================

# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------

def _frac(num: str, den: str) -> str:
    return rf"\frac{{{num}}}{{{den}}}"


def _sqrt(x: str, n: str = "") -> str:
    if n:
        return rf"\sqrt[{n}]{{{x}}}"
    return rf"\sqrt{{{x}}}"


def _exp(exponent: str) -> str:
    return rf"e^{{{exponent}}}"


def _power(base: str, exp: str) -> str:
    return rf"{base}^{{{exp}}}"


def _sub(base: str, subscript: str) -> str:
    return rf"{base}_{{{subscript}}}"


def _subsup(base: str, subscript: str, superscript: str) -> str:
    return rf"{base}_{{{subscript}}}^{{{superscript}}}"


def _integral(
    integrand: str,
    dx: str = "x",
    lower: str = "-\\infty",
    upper: str = "\\infty",
) -> str:
    return rf"\int_{{{lower}}}^{{{upper}}} {integrand} \, d{dx}"


def _sum(body: str, lower: str, upper: str) -> str:
    return rf"\sum_{{{lower}}}^{{{upper}}} {body}"


def _prod(body: str, lower: str, upper: str) -> str:
    return rf"\prod_{{{lower}}}^{{{upper}}} {body}"


def _log(x: str, base: str = "") -> str:
    if base:
        return rf"\log_{{{base}}}{x}"
    return rf"\log {x}"


def _ln(x: str) -> str:
    return rf"\ln {x}"


def _conditional(x: str, given: str) -> str:
    """x \\mid given"""
    return rf"{x} \mid {given}"


def _expected(x: str, subscript: str = "") -> str:
    E = rf"\mathbb{{E}}"
    if subscript:
        E = rf"\mathbb{{E}}_{{{subscript}}}"
    return rf"{E}\left[ {x} \right]"


def _var(x: str, subscript: str = "") -> str:
    V = rf"\mathrm{{Var}}"
    if subscript:
        V = rf"\mathrm{{Var}}_{{{subscript}}}"
    return rf"{V}\left( {x} \right)"


def _cov(x: str, y: str) -> str:
    return rf"\mathrm{{Cov}}\left( {x},\, {y} \right)"


def _indicator(event: str) -> str:
    return rf"\mathbf{{1}}\left\{{ {event} \right\}}"


def _binom_coef(n: str, k: str) -> str:
    return rf"\binom{{{n}}}{{{k}}}"


def _gamma_func(x: str) -> str:
    return rf"\Gamma\left({x}\right)"


def _beta_func(a: str, b: str) -> str:
    return rf"\mathrm{{B}}\left({a},\, {b}\right)"


# ---------------------------------------------------------------------------
# B.1  Descriptive statistics formulas
# ---------------------------------------------------------------------------

def mean(
    symbol: str = "\\bar{x}",
    xi: str = "x_i",
    n: str = "n",
) -> str:
    """Sample mean: \\bar{x} = (1/n) \\sum x_i"""
    return rf"{symbol} = {_frac('1', n)} {_sum(xi, 'i=1', n)}"


def weighted_mean(
    symbol: str = "\\bar{x}_w",
    xi: str = "x_i",
    wi: str = "w_i",
) -> str:
    return (
        rf"{symbol} = "
        rf"{_frac(_sum(wi + xi, 'i=1', 'n'), _sum(wi, 'i=1', 'n'))}"
    )


def population_variance(
    symbol: str = "\\sigma^2",
    xi: str = "x_i",
    mu: str = "\\mu",
    n: str = "N",
) -> str:
    return (
        rf"{symbol} = "
        rf"{_frac('1', n)} {_sum(Fence.paren(xi + ' - ' + mu) + '^2', 'i=1', n)}"
    )


def sample_variance(
    symbol: str = "s^2",
    xi: str = "x_i",
    xbar: str = "\\bar{x}",
    n: str = "n",
) -> str:
    return (
        rf"{symbol} = "
        rf"{_frac('1', n + '-1')} {_sum(Fence.paren(xi + ' - ' + xbar) + '^2', 'i=1', n)}"
    )


def std_dev(sample: bool = True) -> str:
    sym = "s" if sample else r"\sigma"
    var = "s^2" if sample else r"\sigma^2"
    return rf"{sym} = {_sqrt(var)}"


def coeff_of_variation() -> str:
    return rf"CV = {_frac(r'\sigma', r'\mu')} \times 100\%"


def skewness() -> str:
    return (
        rf"\gamma_1 = "
        rf"{_frac(_expected(r'(X - \mu)^3'), r'\sigma^3')}"
    )


def kurtosis(excess: bool = True) -> str:
    symbol = r"\gamma_2" if excess else r"\kappa"
    core = _frac(_expected(r"(X - \mu)^4"), r"\sigma^4")
    if excess:
        return rf"{symbol} = {core} - 3"
    return rf"{symbol} = {core}"


def z_score(x: str = "x", mu: str = r"\mu", sigma: str = r"\sigma") -> str:
    return rf"z = {_frac(x + ' - ' + mu, sigma)}"


def percentile_rank() -> str:
    return rf"PR = {_frac(r'L + 0.5 F', 'N')} \times 100"


def iqr() -> str:
    return r"IQR = Q_3 - Q_1"


def coefficient_of_determination() -> str:
    return (
        r"R^2 = 1 - "
        + _frac(r"\sum (y_i - \hat{y}_i)^2", r"\sum (y_i - \bar{y})^2")
    )


# ---------------------------------------------------------------------------
# B.2  Probability
# ---------------------------------------------------------------------------

def conditional_probability(
    A: str = "A", B: str = "B"
) -> str:
    return (
        rf"P({A} \mid {B}) = "
        rf"{_frac('P(' + A + r' \cap ' + B + ')', 'P(' + B + ')')}"
    )


def total_probability(n_events: int = 3) -> str:
    terms = " + ".join(
        rf"P(A \mid B_{i}) P(B_{i})" for i in range(1, n_events + 1)
    )
    return rf"P(A) = {terms}"


def bayes_theorem(
    A: str = "A", B: str = "B",
    posterior_name: str = "posterior",
) -> str:
    """
    Full Bayes' theorem.

    Returns the raw LaTeX; parts are accessible via TexFormula.parts.
    """
    likelihood  = rf"P({B} \mid {A})"
    prior       = rf"P({A})"
    evidence    = rf"P({B})"
    return (
        rf"P({A} \mid {B}) = "
        rf"{_frac(likelihood + r' \cdot ' + prior, evidence)}"
    )


def bayes_proportional(A: str = "A", B: str = "B") -> str:
    return rf"P({A} \mid {B}) \propto P({B} \mid {A}) \cdot P({A})"


def law_of_total_expectation(
    X: str = "X", Y: str = "Y"
) -> str:
    return (
        rf"{_expected(X)} = "
        rf"{_expected(_expected(_conditional(X, Y), 'Y'))}"
    )


def law_of_total_variance(X: str = "X", Y: str = "Y") -> str:
    return (
        rf"{_var(X)} = "
        rf"{_expected(_var(_conditional(X, Y), 'Y'))} + "
        rf"{_var(_expected(_conditional(X, Y), 'Y'))}"
    )


# ---------------------------------------------------------------------------
# B.3  Discrete distribution PDFs / PMFs
# ---------------------------------------------------------------------------

def pmf_bernoulli(
    p: str = "p",
    x: str = "x",
) -> str:
    return rf"P(X={x}) = {p}^{{{x}}} (1-{p})^{{1-{x}}}, \quad {x} \in \{{0,1\}}"


def pmf_binomial(
    n: str = "n",
    p: str = "p",
    k: str = "k",
) -> str:
    return (
        rf"P(X={k}) = {_binom_coef(n, k)} "
        rf"{p}^{{{k}}} (1-{p})^{{{n}-{k}}}"
    )


def pmf_poisson(
    lam: str = r"\lambda",
    k: str = "k",
) -> str:
    return (
        rf"P(X={k}) = "
        rf"{_frac(_exp('-' + lam) + r' \cdot ' + lam + '^{' + k + '}', k + '!')}"
    )


def pmf_geometric(
    p: str = "p",
    k: str = "k",
) -> str:
    return rf"P(X={k}) = (1-{p})^{{{k}-1}} {p}, \quad {k} = 1, 2, \ldots"


def pmf_negative_binomial(
    r: str = "r",
    p: str = "p",
    k: str = "k",
) -> str:
    return (
        rf"P(X={k}) = {_binom_coef(k+'-1', r+'-1')} "
        rf"{p}^{{{r}}} (1-{p})^{{{k}-{r}}}"
    )


def pmf_hypergeometric(
    N: str = "N",
    K: str = "K",
    n: str = "n",
    k: str = "k",
) -> str:
    return (
        rf"P(X={k}) = "
        rf"{_frac(_binom_coef(K, k) + r' \cdot ' + _binom_coef(N+'-'+K, n+'-'+k), _binom_coef(N, n))}"
    )


def pmf_uniform_discrete(
    a: str = "a",
    b: str = "b",
) -> str:
    return rf"P(X=x) = {_frac('1', b + '-' + a + '+1')}, \quad x \in \{{{a}, \ldots, {b}\}}"


# ---------------------------------------------------------------------------
# B.4  Continuous distribution PDFs
# ---------------------------------------------------------------------------

def pdf_uniform(
    a: str = "a",
    b: str = "b",
    x: str = "x",
) -> str:
    body = _frac("1", b + " - " + a)
    cond = rf"{a} \leq {x} \leq {b}"
    return Env.piecewise(
        rf"f({x})",
        [(body, rf"\text{{if }} {cond}"), ("0", r"\text{otherwise}")]
    )


def pdf_normal(
    mu: str = r"\mu",
    sigma: str = r"\sigma",
    x: str = "x",
) -> str:
    """
    Normal PDF:  f(x | μ, σ²) = 1/(σ√(2π)) · exp(-(x-μ)²/(2σ²))
    """
    norm_const = _frac("1", sigma + r"\sqrt{2\pi}")
    exponent   = _frac(
        r"-\left(" + x + r" - " + mu + r"\right)^2",
        "2" + sigma + "^2"
    )
    kernel = _exp(exponent)
    return rf"f({x} \mid {mu}, {sigma}^2) = {norm_const} {kernel}"


def pdf_standard_normal(x: str = "x") -> str:
    return (
        rf"\phi({x}) = "
        rf"{_frac('1', r'\sqrt{2\pi}')}"
        rf" \exp\!\left( {_frac('-' + x + '^2', '2')} \right)"
    )


def cdf_standard_normal(x: str = "x") -> str:
    return (
        rf"\Phi({x}) = "
        rf"{_frac('1', r'\sqrt{2\pi}')}"
        rf" {_integral(_exp(_frac('-t^2', '2')), dx='t', lower=r'-\infty', upper=x)}"
    )


def pdf_exponential(
    lam: str = r"\lambda",
    x: str = "x",
) -> str:
    body = lam + " " + _exp("-" + lam + x)
    return Env.piecewise(
        rf"f({x})",
        [(body, rf"\text{{if }} {x} \geq 0"), ("0", r"\text{otherwise}")]
    )


def pdf_gamma(
    alpha: str = r"\alpha",
    beta: str = r"\beta",
    x: str = "x",
) -> str:
    return (
        rf"f({x}) = "
        rf"{_frac(_power(beta, alpha), _gamma_func(alpha))}"
        rf" {_power(x, alpha + '-1')}"
        rf" {_exp('-' + beta + x)}"
        rf", \quad {x} > 0"
    )


def pdf_beta(
    alpha: str = r"\alpha",
    beta: str = r"\beta",
    x: str = "x",
) -> str:
    return (
        rf"f({x}) = "
        rf"{_frac(_power(x, alpha + '-1') + _power('(1-' + x + ')', beta + '-1'), _beta_func(alpha, beta))}"
        rf", \quad {x} \in [0,1]"
    )


def pdf_chi_squared(
    k: str = "k",
    x: str = "x",
) -> str:
    half_k = _frac(k, "2")
    return (
        rf"f({x}) = "
        rf"{_frac(_power(x, half_k + '-1') + _exp('-' + x + '/2'), _power('2', half_k) + r'\,\Gamma(' + half_k + ')')}"
        rf", \quad {x} > 0"
    )


def pdf_student_t(
    nu: str = r"\nu",
    x: str = "x",
) -> str:
    return (
        rf"f({x}) = "
        rf"{_frac(_gamma_func(Fence.paren(nu + '+1') + '/2'), _sqrt(r'\pi ' + nu) + r'\,' + _gamma_func(nu + '/2'))}"
        rf"\left(1 + {_frac(x + '^2', nu)}\right)^{{-({nu}+1)/2}}"
    )


def pdf_f(
    d1: str = "d_1",
    d2: str = "d_2",
    x: str = "x",
) -> str:
    num = (
        _sqrt(_frac(_power(d1 + x, d1) + r"\cdot" + _power(d2, d2),
                    _power(d1 + x + "+" + d2, d1 + "+" + d2)))
    )
    den = rf"x\,\mathrm{{B}}\left({_frac(d1,'2')},{_frac(d2,'2')}\right)"
    return rf"f({x}) = {_frac(num, den)}, \quad {x} > 0"


def pdf_lognormal(
    mu: str = r"\mu",
    sigma: str = r"\sigma",
    x: str = "x",
) -> str:
    return (
        rf"f({x}) = "
        rf"{_frac('1', x + sigma + r'\sqrt{2\pi}')}"
        rf"\exp\!\left({_frac(r'-(\ln ' + x + '-' + mu + ')^2', '2' + sigma + '^2')}\right)"
        rf", \quad {x} > 0"
    )


def pdf_weibull(
    k: str = "k",
    lam: str = r"\lambda",
    x: str = "x",
) -> str:
    return (
        rf"f({x}) = {_frac(k, lam)}"
        rf"\left({_frac(x, lam)}\right)^{{{k}-1}}"
        rf"\exp\!\left(-\left({_frac(x, lam)}\right)^{{{k}}}\right)"
        rf", \quad {x} \geq 0"
    )


def pdf_cauchy(
    x0: str = "x_0",
    gamma: str = r"\gamma",
    x: str = "x",
) -> str:
    return (
        rf"f({x}) = "
        rf"{_frac('1', r'\pi ' + gamma)}"
        rf"\left[1 + \left({_frac(x + '-' + x0, gamma)}\right)^2\right]^{{-1}}"
    )


def pdf_pareto(
    x_m: str = "x_m",
    alpha: str = r"\alpha",
    x: str = "x",
) -> str:
    return (
        rf"f({x}) = {_frac(alpha + x_m + r'^\alpha', _power(x, r'\alpha+1'))}"
        rf", \quad {x} \geq {x_m}"
    )


def pdf_bivariate_normal(
    mu1: str = r"\mu_1",
    mu2: str = r"\mu_2",
    s1: str  = r"\sigma_1",
    s2: str  = r"\sigma_2",
    rho: str = r"\rho",
) -> str:
    norm_const = _frac(
        "1",
        "2\\pi " + s1 + s2 + r"\sqrt{1-" + rho + "^2}"
    )
    z = (
        _frac("1", "1 - " + rho + "^2")
        + r"\left["
        + _frac(r"(x-" + mu1 + r")^2", s1 + "^2")
        + r" - \frac{2" + rho + r"(x-" + mu1 + r")(y-" + mu2 + r")}"
        + "{" + s1 + s2 + "}"
        + r" + " + _frac(r"(y-" + mu2 + r")^2", s2 + "^2")
        + r"\right]"
    )
    return rf"f(x,y) = {norm_const} \exp\!\left(-{_frac(z, '2')}\right)"


# ---------------------------------------------------------------------------
# B.5  Moments: E[X], Var(X), MGF, characteristic function
# ---------------------------------------------------------------------------

def expected_value_discrete(
    X: str = "X",
    xi: str = "x_i",
    pi: str = "p_i",
) -> str:
    return rf"{_expected(X)} = {_sum(xi + r'\,' + pi, 'i=1', 'n')}"


def expected_value_continuous(
    X: str = "X",
    x: str = "x",
    f: str = "f(x)",
) -> str:
    return rf"{_expected(X)} = {_integral(x + r'\,' + f)}"


def variance_definition(
    X: str = "X",
    mu: str = r"\mu",
) -> str:
    return (
        rf"{_var(X)} = "
        rf"{_expected('(' + X + ' - ' + mu + ')^2')} = "
        rf"{_expected(X + '^2')} - {_expected(X)}^2"
    )


def mgf(X: str = "X", t: str = "t") -> str:
    return rf"M_{{{X}}}({t}) = {_expected(_exp(t + X))}"


def characteristic_function(X: str = "X", t: str = "t") -> str:
    return rf"\varphi_{{{X}}}({t}) = {_expected(_exp('i' + t + X))}"


def moment_k(k: str = "k") -> str:
    return rf"\mu'_{{{k}}} = {_expected('X^{' + k + '}')}"


def central_moment_k(k: str = "k") -> str:
    mu_expr = r"(X-\mu)^{" + k + "}"
    return rf"\mu_{{{k}}} = {_expected(mu_expr)}"


# ---------------------------------------------------------------------------
# B.6  Entropy and information theory
# ---------------------------------------------------------------------------

def entropy_discrete(
    X: str = "X",
    p: str = "p",
) -> str:
    return (
        rf"H({X}) = -"
        rf"{_sum(p + r'(x) \log_2 ' + p + r'(x)', r'x \in \mathcal{X}', '')}"
    )


def entropy_continuous(X: str = "X", f: str = "f") -> str:
    return (
        rf"h({X}) = -"
        rf"{_integral(f + r'(x) \ln ' + f + r'(x)')}"
    )


def kl_divergence(P: str = "P", Q: str = "Q") -> str:
    return (
        rf"D_{{\mathrm{{KL}}}}({P} \| {Q}) = "
        rf"{_sum(p + r'(x) \ln ' + _frac(p + r'(x)', q + r'(x)'), r'x', '')}"
        .replace("p", P.lower()).replace("q", Q.lower())
    )


def mutual_information(X: str = "X", Y: str = "Y") -> str:
    return (
        rf"I({X};\, {Y}) = "
        rf"H({X}) + H({Y}) - H({X}, {Y})"
    )


def cross_entropy(P: str = "P", Q: str = "Q") -> str:
    p, q = P.lower(), Q.lower()
    return (
        rf"H({P}, {Q}) = -"
        rf"{_sum(p + r'(x) \log ' + q + r'(x)', 'x', '')}"
    )


# ---------------------------------------------------------------------------
# B.7  Inference: CI, test statistics, p-value region
# ---------------------------------------------------------------------------

def ci_mean_known_sigma(
    alpha: str = r"\alpha",
    z: str = "z",
    sigma: str = r"\sigma",
    n: str = "n",
) -> str:
    se = _frac(sigma, _sqrt(n))
    return (
        rf"\bar{{x}} \pm {z}_{{{_frac(alpha, '2')}}} \cdot {se}"
    )


def ci_mean_unknown_sigma(
    alpha: str = r"\alpha",
    n: str = "n",
    s: str = "s",
) -> str:
    se = _frac(s, _sqrt(n))
    return (
        rf"\bar{{x}} \pm t_{{{_frac(alpha, '2')},\, {n}-1}} \cdot {se}"
    )


def ci_proportion(
    alpha: str = r"\alpha",
    p: str = r"\hat{p}",
    n: str = "n",
) -> str:
    se = _sqrt(_frac(p + r"(1-" + p + ")", n))
    return (
        rf"{p} \pm z_{{{_frac(alpha, '2')}}} \cdot {se}"
    )


def z_test_statistic(
    mu0: str = r"\mu_0",
    sigma: str = r"\sigma",
    n: str = "n",
) -> str:
    return rf"Z = {_frac(r'\bar{x} - ' + mu0, _frac(sigma, _sqrt(n)))}"


def t_test_statistic(
    mu0: str = r"\mu_0",
    n: str = "n",
) -> str:
    return rf"T = {_frac(r'\bar{x} - ' + mu0, _frac('s', _sqrt(n)))}"


def t_test_two_sample(pooled: bool = False) -> str:
    if pooled:
        sp = _sqrt(
            _frac(
                r"(n_1-1)s_1^2 + (n_2-1)s_2^2",
                r"n_1 + n_2 - 2"
            )
        )
        se = sp + r" \sqrt{" + _frac("1", "n_1") + "+" + _frac("1", "n_2") + "}"
        return rf"T = {_frac(r'\bar{x}_1 - \bar{x}_2', se)}"
    se = _sqrt(
        _frac(r"s_1^2", r"n_1") + "+" + _frac(r"s_2^2", r"n_2")
    )
    return rf"T = {_frac(r'\bar{x}_1 - \bar{x}_2', se)}"


def chi_sq_test_statistic() -> str:
    return (
        rf"\chi^2 = "
        rf"{_sum(_frac('(O_i - E_i)^2', 'E_i'), 'i=1', 'k')}"
    )


def f_test_statistic() -> str:
    return rf"F = {_frac('s_1^2', 's_2^2')}"


def power_function(alpha: str = r"\alpha") -> str:
    return (
        rf"\beta(\theta) = P\!\left(\text{{reject }} H_0 \mid \theta\right)"
    )


def likelihood_ratio_test() -> str:
    return (
        rf"\Lambda = "
        rf"{_frac(r'L(\hat{\theta}_0)', r'L(\hat{\theta})')}"
        r",\quad -2\ln\Lambda \sim \chi^2_k"
    )


# ---------------------------------------------------------------------------
# B.8  Regression
# ---------------------------------------------------------------------------

def simple_linear_regression(
    beta0: str = r"\beta_0",
    beta1: str = r"\beta_1",
    epsilon: str = r"\varepsilon",
) -> str:
    return rf"Y = {beta0} + {beta1} X + {epsilon}"


def ols_estimate_beta1() -> str:
    num = _sum(r"(x_i - \bar{x})(y_i - \bar{y})", "i=1", "n")
    den = _sum(r"(x_i - \bar{x})^2", "i=1", "n")
    return rf"\hat{{\beta}}_1 = {_frac(num, den)}"


def ols_estimate_beta0() -> str:
    return rf"\hat{{\beta}}_0 = \bar{{y}} - \hat{{\beta}}_1 \bar{{x}}"


def multiple_regression() -> str:
    return (
        r"\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}"
    )


def ols_matrix() -> str:
    return (
        r"\hat{\boldsymbol{\beta}} = "
        r"\left(\mathbf{X}^\top \mathbf{X}\right)^{-1} \mathbf{X}^\top \mathbf{Y}"
    )


def logistic_regression(
    beta0: str = r"\beta_0",
    beta1: str = r"\beta_1",
) -> str:
    log_odds = beta0 + " + " + beta1 + " x"
    return rf"P(Y=1 \mid x) = {_frac(_exp(log_odds), '1 + ' + _exp(log_odds))}"


def pearson_r() -> str:
    num = _sum(r"(x_i - \bar{x})(y_i - \bar{y})", "i=1", "n")
    den = (
        _sqrt(_sum(r"(x_i - \bar{x})^2", "i=1", "n"))
        + r"\,"
        + _sqrt(_sum(r"(y_i - \bar{y})^2", "i=1", "n"))
    )
    return rf"r = {_frac(num, den)}"


def spearman_rho(n: str = "n") -> str:
    return (
        rf"\rho = 1 - "
        rf"{_frac('6 ' + _sum('d_i^2', 'i=1', n), n + '(' + n + '^2 - 1)')}"
    )


# ---------------------------------------------------------------------------
# B.9  Sampling theory / CLT
# ---------------------------------------------------------------------------

def clt_statement(
    n: str = "n",
    mu: str = r"\mu",
    sigma: str = r"\sigma",
) -> str:
    return (
        rf"\sqrt{{{n}}}\left(\bar{{X}}_n - {mu}\right)"
        rf"\xrightarrow{{d}} \mathcal{{N}}\left(0,\, {sigma}^2\right)"
    )


def standard_error(sigma: str = r"\sigma", n: str = "n") -> str:
    return rf"SE = {_frac(sigma, _sqrt(n))}"


def mle_definition() -> str:
    return (
        rf"\hat{{\theta}}_{{MLE}} = "
        rf"\arg\max_{{\theta}}\, L(\theta \mid \mathbf{{x}})"
    )


def log_likelihood() -> str:
    return (
        rf"\ell(\theta) = \ln L(\theta) = "
        rf"{_sum(r'\ln f(x_i \mid \theta)', 'i=1', 'n')}"
    )


def method_of_moments(k: str = "k") -> str:
    return (
        rf"\hat{{\mu}}'_{{{k}}} = "
        rf"{_frac('1', 'n')} {_sum('x_i^{' + k + '}', 'i=1', 'n')}"
    )


def fisher_information(theta: str = r"\theta") -> str:
    return (
        rf"I({theta}) = "
        rf"-{_expected(_frac(r'\partial^2 \ln f(X;\,' + theta + r')', r'\partial ' + theta + '^2'))}"
    )


# ===========================================================================
# LAYER C — TexFormula dataclass
# Carries raw LaTeX + named parts + derivation steps.
# ===========================================================================

@dataclass
class TexFormula:
    r"""
    A structured statistical formula.

    Attributes
    ----------
    name : str
        Machine-readable key, e.g. ``"normal_pdf"``.
    raw : str
        Raw LaTeX source (no dollar signs).
    description : str
        One-line human description.
    parts : dict[str, str]
        Named sub-expressions, e.g.::

            {"normalising_constant": r"\frac{1}{\sigma\sqrt{2\pi}}",
             "kernel": r"e^{-(x-\mu)^2/(2\sigma^2)}"}

        Used by highlight_part() and Manim color_map.
    steps : list[TexDerivationStep]
        Ordered derivation steps from definition to final form.
    color_map : dict[str, str]
        part_name -> LaTeX color name (for Manim ``MathTex`` color maps).
    tags : list[str]
        Topic tags, e.g. ``["distribution", "continuous", "normal"]``.
    """

    name:        str
    raw:         str
    description: str = ""
    parts:       Dict[str, str]              = field(default_factory=dict)
    steps:       List["TexDerivationStep"]   = field(default_factory=list)
    color_map:   Dict[str, str]              = field(default_factory=dict)
    tags:        List[str]                   = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def display(self) -> str:
        """Return $$…$$ display math."""
        return Env.display(self.raw)

    @property
    def inline(self) -> str:
        """Return $…$ inline math."""
        return Env.inline(self.raw)

    @property
    def aligned(self) -> str:
        """
        Return the aligned multi-line environment if steps are defined,
        otherwise fall back to the plain display form.
        """
        if not self.steps:
            return self.display
        rows = [(s.lhs, s.rhs) for s in self.steps]
        return Env.display(Env.aligned(rows))

    def part(self, name: str) -> str:
        """
        Retrieve a named sub-expression.
        Raises ``KeyError`` with a helpful message if not found.
        """
        if name not in self.parts:
            raise KeyError(
                f"Part {name!r} not in formula {self.name!r}. "
                f"Available: {list(self.parts)}"
            )
        return self.parts[name]

    # ------------------------------------------------------------------
    # Annotation helpers (return new TexFormula with modified raw)
    # ------------------------------------------------------------------

    def highlight_part(
        self,
        part_name: str,
        color: str = "red",
    ) -> "TexFormula":
        """
        Return a new TexFormula where ``part_name`` is wrapped in
        ``\\textcolor{color}{…}``.

        This modifies ``.raw`` directly, producing a formula string
        that renders the named part in the requested color.
        """
        sub = self.part(part_name)
        colored = rf"\textcolor{{{color}}}{{{sub}}}"
        new_raw = self.raw.replace(sub, colored, 1)
        return TexFormula(
            name=self.name + f"_hl_{part_name}",
            raw=new_raw,
            description=self.description,
            parts=self.parts,
            steps=self.steps,
            color_map={**self.color_map, part_name: color},
            tags=self.tags,
        )

    def with_substitution(self, **subs: str) -> "TexFormula":
        """
        Return a new TexFormula with parameter substitutions applied.

        Example::

            f = FORMULAS["normal_pdf"]
            f2 = f.with_substitution(mu="0", sigma="1")

        Replaces literal occurrences of Greek symbols / parameter names
        in ``.raw``.
        """
        new_raw = self.raw
        new_parts = dict(self.parts)
        for param, value in subs.items():
            # Build a regex that replaces \\mu, \\sigma, etc.
            # as whole tokens to avoid partial replacements.
            latex_param = "\\" + param if not param.startswith("\\") else param
            new_raw = new_raw.replace(latex_param, value)
            new_parts = {k: v.replace(latex_param, value) for k, v in new_parts.items()}
        return TexFormula(
            name=self.name + "_sub",
            raw=new_raw,
            description=self.description + " (substituted)",
            parts=new_parts,
            steps=self.steps,
            color_map=self.color_map,
            tags=self.tags,
        )

    # ------------------------------------------------------------------
    # Manim integration
    # ------------------------------------------------------------------

    def to_mathtex(
        self,
        color_map: Optional[Dict[str, Any]] = None,
        font_size: int = 36,
        **kwargs,
    ) -> "MathTex":
        """
        Build a Manim ``MathTex`` mobject from this formula.

        Parameters
        ----------
        color_map : dict, optional
            Mapping of LaTeX sub-string → Manim color.  If not given,
            uses ``self.color_map`` (which stores color names as strings,
            resolved by Manim).
        font_size : int
            Font size passed to MathTex.
        **kwargs
            Additional keyword arguments forwarded to ``MathTex()``.
        """
        if not _MANIM_AVAILABLE:
            raise ImportError("Manim is not installed.")
        cmap = color_map or {}
        return MathTex(self.raw, font_size=font_size, **kwargs)

    def to_tex(self, font_size: int = 32, **kwargs) -> "Tex":
        """Build a Manim ``Tex`` mobject (for display-mode rendering)."""
        if not _MANIM_AVAILABLE:
            raise ImportError("Manim is not installed.")
        return Tex(self.display, font_size=font_size, **kwargs)

    def to_aligned_mathtex(self, font_size: int = 30, **kwargs) -> "MathTex":
        """Build a Manim MathTex from the aligned multi-step derivation."""
        if not _MANIM_AVAILABLE:
            raise ImportError("Manim is not installed.")
        if not self.steps:
            return self.to_mathtex(font_size=font_size, **kwargs)
        rows = [(s.lhs, s.rhs) for s in self.steps]
        aligned_src = Env.aligned(rows)
        return MathTex(aligned_src, font_size=font_size, **kwargs)

    def __repr__(self) -> str:
        return f"TexFormula({self.name!r})"

    def __str__(self) -> str:
        return self.raw


@dataclass
class TexDerivationStep:
    """
    One step in a multi-line derivation.

    Attributes
    ----------
    lhs : str
        Left-hand side LaTeX (may be empty for continuation lines).
    rhs : str
        Right-hand side LaTeX.
    annotation : str
        Short justification shown above / beside the step (e.g. "expand square").
    """
    lhs:        str
    rhs:        str
    annotation: str = ""

    def as_row(self) -> Tuple[str, str]:
        return (self.lhs, self.rhs)


# ===========================================================================
# LAYER D — Annotation helpers
# Functions that wrap raw LaTeX in annotation macros.
# ===========================================================================

def overbrace(expr: str, label: str) -> str:
    """Annotate *expr* with an overbrace labelled *label*."""
    return rf"\overbrace{{{expr}}}^{{\text{{{label}}}}}"


def underbrace(expr: str, label: str) -> str:
    """Annotate *expr* with an underbrace labelled *label*."""
    return rf"\underbrace{{{expr}}}_{{\text{{{label}}}}}"


def boxed(expr: str) -> str:
    """Draw a box around *expr*."""
    return rf"\boxed{{{expr}}}"


def color(expr: str, latex_color: str) -> str:
    """Wrap *expr* in ``\\textcolor{color}{…}``."""
    return rf"\textcolor{{{latex_color}}}{{{expr}}}"


def cancel(expr: str) -> str:
    """Strike through *expr* (requires cancel package)."""
    return rf"\cancel{{{expr}}}"


def phantom(expr: str) -> str:
    """Invisible placeholder with the same size as *expr*."""
    return rf"\phantom{{{expr}}}"


def annotate_below(expr: str, annotation: str) -> str:
    """
    Place *annotation* as a small text directly below *expr*
    using a strut for alignment.
    """
    return rf"\underset{{\scriptscriptstyle {annotation}}}{{{expr}}}"


def annotate_above(expr: str, annotation: str) -> str:
    return rf"\overset{{\scriptscriptstyle {annotation}}}{{{expr}}}"


def step_annotation(
    step_number: int,
    expr: str,
    label: str,
) -> str:
    """
    Format one derivation step as a labeled equation number.

    Example output::

        f(x) = \\ldots  \\tag{2: expand kernel}
    """
    return rf"{expr} \tag{{{step_number}:\ \text{{{label}}}}}"


def highlight_parts(
    raw: str,
    highlights: Dict[str, str],
) -> str:
    """
    Given a raw LaTeX string and a mapping of substring → color name,
    replace each substring with its colorized version.

    Parameters
    ----------
    raw : str
        The original LaTeX formula.
    highlights : dict
        Mapping of LaTeX substring → LaTeX color name (e.g. "red").

    Returns
    -------
    str
        Modified LaTeX with ``\\textcolor`` wrappings applied in order.
    """
    result = raw
    for sub, col in highlights.items():
        result = result.replace(sub, color(sub, col), 1)
    return result


# ===========================================================================
# LAYER E — Manim animation helpers
# Thin wrappers that return Manim Animation objects.
# All raise ImportError if Manim is not installed.
# ===========================================================================

def write_formula(
    formula: TexFormula,
    run_time: float = 2.0,
    **mathtex_kwargs,
):
    """
    Return a Manim ``Write`` animation for the given formula.

    Example
    -------
        self.play(write_formula(FORMULAS["normal_pdf"]))
    """
    if not _MANIM_AVAILABLE:
        raise ImportError("Manim is not installed.")
    from manim import Write  # type: ignore
    mob = formula.to_mathtex(**mathtex_kwargs)
    return Write(mob, run_time=run_time)


def transform_formula(
    source: TexFormula,
    target: TexFormula,
    run_time: float = 1.5,
    **mathtex_kwargs,
):
    """
    Return a Manim ``TransformMatchingTex`` animation that morphs
    *source* formula into *target* formula.

    This is useful for derivation steps where identical sub-expressions
    should stay in place while only the changed parts animate.
    """
    if not _MANIM_AVAILABLE:
        raise ImportError("Manim is not installed.")
    from manim import TransformMatchingTex  # type: ignore
    src_mob = source.to_mathtex(**mathtex_kwargs)
    tgt_mob = target.to_mathtex(**mathtex_kwargs)
    return src_mob, tgt_mob, TransformMatchingTex(src_mob, tgt_mob, run_time=run_time)


def derive_formula(
    formula: TexFormula,
    scene,
    position=None,
    step_run_time: float = 1.2,
    wait_time: float = 0.8,
    **mathtex_kwargs,
) -> None:
    """
    Animate a full step-by-step derivation of *formula* on a Manim *scene*.

    Each ``TexDerivationStep`` in ``formula.steps`` is played as a
    ``TransformMatchingTex`` from the previous expression.  An optional
    annotation text is faded in above the arrow.

    Parameters
    ----------
    formula : TexFormula
        Must have at least one derivation step.
    scene : manim.Scene
        The Manim scene to play the animations on.
    position : manim vector, optional
        Where to place the first formula on screen.
    step_run_time : float
        Duration of each step transition.
    wait_time : float
        Pause between steps.
    """
    if not _MANIM_AVAILABLE:
        raise ImportError("Manim is not installed.")
    from manim import Write, TransformMatchingTex, FadeIn, FadeOut, Text, ORIGIN  # type: ignore

    if not formula.steps:
        mob = formula.to_mathtex(**mathtex_kwargs)
        if position is not None:
            mob.move_to(position)
        scene.play(Write(mob))
        return

    steps = formula.steps
    current_raw = steps[0].rhs
    current_mob = MathTex(current_raw, **mathtex_kwargs)
    if position is not None:
        current_mob.move_to(position)
    else:
        current_mob.move_to(ORIGIN)
    scene.play(Write(current_mob, run_time=step_run_time))
    scene.wait(wait_time)

    for step in steps[1:]:
        next_raw = step.rhs
        next_mob = MathTex(next_raw, **mathtex_kwargs)
        next_mob.move_to(current_mob)
        annotation_mob = None
        if step.annotation:
            annotation_mob = Text(
                step.annotation,
                font_size=20,
            ).next_to(current_mob, direction=[0, 1.2, 0])
            scene.play(FadeIn(annotation_mob, run_time=0.4))
        scene.play(
            TransformMatchingTex(current_mob, next_mob, run_time=step_run_time)
        )
        if annotation_mob:
            scene.play(FadeOut(annotation_mob, run_time=0.3))
        scene.wait(wait_time)
        current_mob = next_mob


def highlight_subformula(
    mob,
    substring: str,
    highlight_color,
    scene,
    run_time: float = 0.6,
):
    """
    Highlight a substring of an existing Manim ``MathTex`` *mob* by
    using ``Indicate`` on the matched sub-mobject.

    Parameters
    ----------
    mob : MathTex
        The already-displayed MathTex mobject.
    substring : str
        LaTeX substring to locate within *mob*.
    highlight_color : ManimColor
        Color to flash.
    scene : manim.Scene
    run_time : float
    """
    if not _MANIM_AVAILABLE:
        raise ImportError("Manim is not installed.")
    from manim import Indicate  # type: ignore
    parts = mob.get_parts_by_tex(substring)
    if parts:
        scene.play(Indicate(parts, color=highlight_color, run_time=run_time))


# ===========================================================================
# LAYER F — Formula registry and catalog
# ===========================================================================

def _build_normal_pdf_formula() -> TexFormula:
    """Build the normal PDF TexFormula with named parts and derivation steps."""
    norm_const = _frac("1", r"\sigma\sqrt{2\pi}")
    exponent   = _frac(r"-(x - \mu)^2", r"2\sigma^2")
    kernel     = _exp(exponent)
    raw        = rf"f(x \mid \mu,\, \sigma^2) = {norm_const} {kernel}"
    return TexFormula(
        name        = "normal_pdf",
        raw         = raw,
        description = "Normal (Gaussian) probability density function",
        parts       = {
            "full_lhs":        rf"f(x \mid \mu,\, \sigma^2)",
            "normalising":     norm_const,
            "kernel":          kernel,
            "exponent":        exponent,
            "sigma_sq_denom":  r"2\sigma^2",
        },
        steps       = [
            TexDerivationStep(
                lhs="f(x)",
                rhs=raw,
                annotation="Normal PDF definition",
            ),
            TexDerivationStep(
                lhs="",
                rhs=rf"= {norm_const} \exp\!\left({exponent}\right)",
                annotation="Explicit exp notation",
            ),
            TexDerivationStep(
                lhs=r"\text{standard normal: } \mu=0,\, \sigma=1",
                rhs=rf"\phi(x) = {_frac('1', r'\sqrt{2\pi}')} e^{{-x^2/2}}",
                annotation="Substitute μ=0, σ=1",
            ),
        ],
        tags        = ["distribution", "continuous", "normal", "gaussian"],
    )


def _build_bayes_formula() -> TexFormula:
    likelihood = r"P(B \mid A)"
    prior      = r"P(A)"
    evidence   = r"P(B)"
    posterior  = r"P(A \mid B)"
    raw = (
        rf"{posterior} = "
        rf"{_frac(likelihood + r' \cdot ' + prior, evidence)}"
    )
    return TexFormula(
        name        = "bayes_theorem",
        raw         = raw,
        description = "Bayes' theorem",
        parts       = {
            "posterior":  posterior,
            "likelihood": likelihood,
            "prior":      prior,
            "evidence":   evidence,
            "numerator":  likelihood + r" \cdot " + prior,
        },
        steps       = [
            TexDerivationStep(
                lhs="P(A \\cap B)",
                rhs=r"P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)",
                annotation="Multiplication rule (both sides)",
            ),
            TexDerivationStep(
                lhs="P(A \\mid B)",
                rhs=_frac(r"P(B \mid A) \cdot P(A)", r"P(B)"),
                annotation="Divide both sides by P(B)",
            ),
        ],
        tags        = ["probability", "bayes", "inference"],
    )


def _build_clt_formula() -> TexFormula:
    raw = (
        rf"\sqrt{{n}}\left(\bar{{X}}_n - \mu\right) "
        rf"\xrightarrow{{d}} \mathcal{{N}}\left(0,\, \sigma^2\right)"
    )
    return TexFormula(
        name        = "clt",
        raw         = raw,
        description = "Central Limit Theorem (convergence in distribution)",
        parts       = {
            "standardized_mean": r"\sqrt{n}\left(\bar{X}_n - \mu\right)",
            "limit_dist":        r"\mathcal{N}\left(0,\, \sigma^2\right)",
        },
        steps       = [
            TexDerivationStep(
                lhs=r"\bar{X}_n",
                rhs=r"\frac{1}{n}\sum_{i=1}^n X_i, \quad X_i \overset{\text{iid}}{\sim} F(\mu, \sigma^2)",
                annotation="Definition of sample mean",
            ),
            TexDerivationStep(
                lhs=r"\sqrt{n}(\bar{X}_n - \mu)",
                rhs=r"\frac{1}{\sqrt{n}} \sum_{i=1}^n (X_i - \mu)",
                annotation="Rearrange",
            ),
            TexDerivationStep(
                lhs=r"\xrightarrow{d}",
                rhs=r"\mathcal{N}(0, \sigma^2)",
                annotation="By CLT as n → ∞",
            ),
        ],
        tags        = ["sampling", "clt", "normal", "limit_theorem"],
    )


def _build_mle_formula() -> TexFormula:
    raw = (
        r"\hat{\theta}_{\mathrm{MLE}} = "
        r"\arg\max_{\theta}\, "
        r"\sum_{i=1}^{n} \ln f(x_i \mid \theta)"
    )
    return TexFormula(
        name        = "mle",
        raw         = raw,
        description = "Maximum Likelihood Estimator (log-likelihood form)",
        parts       = {
            "estimator":     r"\hat{\theta}_{\mathrm{MLE}}",
            "log_likelihood": r"\sum_{i=1}^{n} \ln f(x_i \mid \theta)",
        },
        steps       = [
            TexDerivationStep(
                lhs="L(\\theta)",
                rhs=r"\prod_{i=1}^{n} f(x_i \mid \theta)",
                annotation="Likelihood (iid assumption)",
            ),
            TexDerivationStep(
                lhs="\\ell(\\theta)",
                rhs=r"\sum_{i=1}^{n} \ln f(x_i \mid \theta)",
                annotation="Log-likelihood (monotone transform)",
            ),
            TexDerivationStep(
                lhs=r"\hat{\theta}",
                rhs=r"\arg\max_{\theta}\, \ell(\theta)",
                annotation="Maximise log-likelihood",
            ),
        ],
        tags        = ["estimation", "mle", "likelihood"],
    )


def _build_ols_formula() -> TexFormula:
    raw = (
        r"\hat{\boldsymbol{\beta}} = "
        r"\left(\mathbf{X}^\top \mathbf{X}\right)^{-1} "
        r"\mathbf{X}^\top \mathbf{y}"
    )
    return TexFormula(
        name        = "ols",
        raw         = raw,
        description = "Ordinary Least Squares estimator (matrix form)",
        parts       = {
            "beta_hat":    r"\hat{\boldsymbol{\beta}}",
            "gram_matrix": r"\mathbf{X}^\top \mathbf{X}",
            "projection":  r"\mathbf{X}^\top \mathbf{y}",
        },
        steps       = [
            TexDerivationStep(
                lhs=r"\mathrm{RSS}(\beta)",
                rhs=r"(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})",
                annotation="Residual sum of squares",
            ),
            TexDerivationStep(
                lhs=r"\frac{\partial \mathrm{RSS}}{\partial \boldsymbol{\beta}}",
                rhs=r"-2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = \mathbf{0}",
                annotation="Set gradient to zero",
            ),
            TexDerivationStep(
                lhs=r"\hat{\boldsymbol{\beta}}",
                rhs=raw.split("=", 1)[1].strip(),
                annotation="Solve for β-hat",
            ),
        ],
        tags        = ["regression", "ols", "estimation"],
    )


def _build_entropy_formula() -> TexFormula:
    raw = r"H(X) = -\sum_{x \in \mathcal{X}} p(x) \log_2 p(x)"
    return TexFormula(
        name        = "entropy_discrete",
        raw         = raw,
        description = "Shannon entropy of a discrete random variable",
        parts       = {
            "entropy":   "H(X)",
            "sum_term":  r"\sum_{x \in \mathcal{X}} p(x) \log_2 p(x)",
        },
        steps       = [
            TexDerivationStep(
                lhs="H(X)",
                rhs=r"\sum_{x} p(x) \cdot \left(-\log_2 p(x)\right)",
                annotation="Expected surprise / self-information",
            ),
            TexDerivationStep(
                lhs="",
                rhs=raw.split("=", 1)[1].strip(),
                annotation="Compact form",
            ),
        ],
        tags        = ["information_theory", "entropy", "discrete"],
    )


def _build_kl_formula() -> TexFormula:
    raw = (
        r"D_{\mathrm{KL}}(P \| Q) = "
        r"\sum_{x} p(x) \ln \frac{p(x)}{q(x)}"
    )
    return TexFormula(
        name        = "kl_divergence",
        raw         = raw,
        description = "Kullback–Leibler divergence from Q to P",
        parts       = {
            "kl":       r"D_{\mathrm{KL}}(P \| Q)",
            "log_ratio": r"\ln \frac{p(x)}{q(x)}",
        },
        tags        = ["information_theory", "kl_divergence"],
    )


#: Global formula catalog.  Keys are formula names; values are TexFormula objects.
FORMULAS: Dict[str, TexFormula] = {
    "normal_pdf":       _build_normal_pdf_formula(),
    "standard_normal":  TexFormula(
        name="standard_normal",
        raw=pdf_standard_normal(),
        description="Standard normal PDF φ(x)",
        tags=["distribution", "continuous", "normal"],
    ),
    "normal_cdf":       TexFormula(
        name="normal_cdf",
        raw=cdf_standard_normal(),
        description="Standard normal CDF Φ(x)",
        tags=["distribution", "continuous", "normal", "cdf"],
    ),
    "exponential_pdf":  TexFormula(
        name="exponential_pdf",
        raw=pdf_exponential(),
        description="Exponential PDF",
        tags=["distribution", "continuous", "exponential"],
    ),
    "gamma_pdf":        TexFormula(
        name="gamma_pdf",
        raw=pdf_gamma(),
        description="Gamma PDF",
        tags=["distribution", "continuous", "gamma"],
    ),
    "beta_pdf":         TexFormula(
        name="beta_pdf",
        raw=pdf_beta(),
        description="Beta PDF",
        tags=["distribution", "continuous", "beta"],
    ),
    "chi_sq_pdf":       TexFormula(
        name="chi_sq_pdf",
        raw=pdf_chi_squared(),
        description="Chi-squared PDF",
        tags=["distribution", "continuous", "chi_squared"],
    ),
    "student_t_pdf":    TexFormula(
        name="student_t_pdf",
        raw=pdf_student_t(),
        description="Student t PDF",
        tags=["distribution", "continuous", "t_distribution"],
    ),
    "binomial_pmf":     TexFormula(
        name="binomial_pmf",
        raw=pmf_binomial(),
        description="Binomial PMF",
        tags=["distribution", "discrete", "binomial"],
    ),
    "poisson_pmf":      TexFormula(
        name="poisson_pmf",
        raw=pmf_poisson(),
        description="Poisson PMF",
        tags=["distribution", "discrete", "poisson"],
    ),
    "geometric_pmf":    TexFormula(
        name="geometric_pmf",
        raw=pmf_geometric(),
        description="Geometric PMF",
        tags=["distribution", "discrete", "geometric"],
    ),
    "hypergeometric_pmf": TexFormula(
        name="hypergeometric_pmf",
        raw=pmf_hypergeometric(),
        description="Hypergeometric PMF",
        tags=["distribution", "discrete", "hypergeometric"],
    ),
    "sample_mean":      TexFormula(
        name="sample_mean",
        raw=mean(),
        description="Sample mean",
        tags=["descriptive", "mean"],
    ),
    "sample_variance":  TexFormula(
        name="sample_variance",
        raw=sample_variance(),
        description="Sample variance (unbiased)",
        tags=["descriptive", "variance"],
    ),
    "z_score":          TexFormula(
        name="z_score",
        raw=z_score(),
        description="Z-score (standardization)",
        tags=["descriptive", "standardization"],
    ),
    "bayes_theorem":    _build_bayes_formula(),
    "conditional_prob": TexFormula(
        name="conditional_prob",
        raw=conditional_probability(),
        description="Conditional probability definition",
        tags=["probability", "conditional"],
    ),
    "clt":              _build_clt_formula(),
    "mle":              _build_mle_formula(),
    "ols":              _build_ols_formula(),
    "ci_z":             TexFormula(
        name="ci_z",
        raw=ci_mean_known_sigma(),
        description="Confidence interval for mean (σ known)",
        tags=["inference", "confidence_interval"],
    ),
    "ci_t":             TexFormula(
        name="ci_t",
        raw=ci_mean_unknown_sigma(),
        description="Confidence interval for mean (σ unknown)",
        tags=["inference", "confidence_interval"],
    ),
    "z_test":           TexFormula(
        name="z_test",
        raw=z_test_statistic(),
        description="Z-test statistic",
        tags=["inference", "hypothesis_test", "z_test"],
    ),
    "t_test":           TexFormula(
        name="t_test",
        raw=t_test_statistic(),
        description="One-sample t-test statistic",
        tags=["inference", "hypothesis_test", "t_test"],
    ),
    "chi_sq_test":      TexFormula(
        name="chi_sq_test",
        raw=chi_sq_test_statistic(),
        description="Chi-squared goodness-of-fit test statistic",
        tags=["inference", "hypothesis_test", "chi_squared"],
    ),
    "pearson_r":        TexFormula(
        name="pearson_r",
        raw=pearson_r(),
        description="Pearson correlation coefficient",
        tags=["regression", "correlation"],
    ),
    "ols_beta1":        TexFormula(
        name="ols_beta1",
        raw=ols_estimate_beta1(),
        description="OLS slope estimate",
        tags=["regression", "ols"],
    ),
    "logistic":         TexFormula(
        name="logistic",
        raw=logistic_regression(),
        description="Logistic regression probability",
        tags=["regression", "logistic"],
    ),
    "entropy":          _build_entropy_formula(),
    "kl_divergence":    _build_kl_formula(),
    "mutual_info":      TexFormula(
        name="mutual_info",
        raw=mutual_information(),
        description="Mutual information I(X;Y)",
        tags=["information_theory", "mutual_information"],
    ),
    "fisher_info":      TexFormula(
        name="fisher_info",
        raw=fisher_information(),
        description="Fisher information",
        tags=["estimation", "fisher_information"],
    ),
}


def get_formula(name: str) -> TexFormula:
    """
    Retrieve a formula from the catalog by name.

    Raises ``KeyError`` with available names if not found.
    """
    if name not in FORMULAS:
        available = ", ".join(sorted(FORMULAS.keys()))
        raise KeyError(
            f"Formula {name!r} not found. Available:\n{available}"
        )
    return FORMULAS[name]


def search_formulas(
    *tags: str,
    match_all: bool = False,
) -> List[TexFormula]:
    """
    Search the catalog by topic tags.

    Parameters
    ----------
    *tags : str
        One or more tag strings to search for.
    match_all : bool
        If True, return only formulas that carry *all* given tags.
        If False (default), return formulas carrying *any* of the tags.

    Returns
    -------
    list[TexFormula]
        Matching formulas, sorted by name.
    """
    tag_set = set(tags)
    results = []
    for formula in FORMULAS.values():
        f_tags = set(formula.tags)
        if match_all:
            if tag_set <= f_tags:
                results.append(formula)
        else:
            if tag_set & f_tags:
                results.append(formula)
    return sorted(results, key=lambda f: f.name)


def register_formula(formula: TexFormula) -> None:
    """
    Add a custom formula to the global catalog.
    Raises ``ValueError`` if the name is already registered.
    """
    if formula.name in FORMULAS:
        raise ValueError(
            f"Formula {formula.name!r} is already registered. "
            f"Use FORMULAS[{formula.name!r}] = formula to overwrite explicitly."
        )
    FORMULAS[formula.name] = formula


# ===========================================================================
# __all__  — public API
# ===========================================================================

__all__ = [
    # Layer A: atoms
    "Greek", "GreekUpper", "Op", "Dec", "Fence", "Space", "Env",

    # Layer B: formula string builders — descriptive
    "mean", "weighted_mean", "population_variance", "sample_variance",
    "std_dev", "coeff_of_variation", "skewness", "kurtosis",
    "z_score", "percentile_rank", "iqr", "coefficient_of_determination",

    # Layer B: formula string builders — probability
    "conditional_probability", "total_probability",
    "bayes_theorem", "bayes_proportional",
    "law_of_total_expectation", "law_of_total_variance",

    # Layer B: formula string builders — discrete distributions
    "pmf_bernoulli", "pmf_binomial", "pmf_poisson", "pmf_geometric",
    "pmf_negative_binomial", "pmf_hypergeometric", "pmf_uniform_discrete",

    # Layer B: formula string builders — continuous distributions
    "pdf_uniform", "pdf_normal", "pdf_standard_normal", "cdf_standard_normal",
    "pdf_exponential", "pdf_gamma", "pdf_beta",
    "pdf_chi_squared", "pdf_student_t", "pdf_f",
    "pdf_lognormal", "pdf_weibull", "pdf_cauchy", "pdf_pareto",
    "pdf_bivariate_normal",

    # Layer B: formula string builders — moments
    "expected_value_discrete", "expected_value_continuous",
    "variance_definition", "mgf", "characteristic_function",
    "moment_k", "central_moment_k",

    # Layer B: formula string builders — entropy / information
    "entropy_discrete", "entropy_continuous",
    "kl_divergence", "mutual_information", "cross_entropy",

    # Layer B: formula string builders — inference
    "ci_mean_known_sigma", "ci_mean_unknown_sigma", "ci_proportion",
    "z_test_statistic", "t_test_statistic", "t_test_two_sample",
    "chi_sq_test_statistic", "f_test_statistic",
    "power_function", "likelihood_ratio_test",

    # Layer B: formula string builders — regression
    "simple_linear_regression", "ols_estimate_beta1", "ols_estimate_beta0",
    "multiple_regression", "ols_matrix", "logistic_regression",
    "pearson_r", "spearman_rho",

    # Layer B: formula string builders — sampling / estimation
    "clt_statement", "standard_error",
    "mle_definition", "log_likelihood", "method_of_moments", "fisher_information",

    # Layer B: primitive helpers (public for direct use)
    "_frac", "_sqrt", "_exp", "_power", "_sub", "_subsup",
    "_integral", "_sum", "_prod", "_log", "_ln",
    "_conditional", "_expected", "_var", "_cov",
    "_indicator", "_binom_coef", "_gamma_func", "_beta_func",

    # Layer C: dataclasses
    "TexFormula", "TexDerivationStep",

    # Layer D: annotations
    "overbrace", "underbrace", "boxed", "color", "cancel", "phantom",
    "annotate_below", "annotate_above", "step_annotation", "highlight_parts",

    # Layer E: Manim helpers
    "write_formula", "transform_formula", "derive_formula", "highlight_subformula",

    # Layer F: registry
    "FORMULAS", "get_formula", "search_formulas", "register_formula",
]