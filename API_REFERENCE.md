# API Reference


## __init__.py

## animations/__init__.py

## animations/clt_demo.py
  class CLTConfig: 
  class _SampleParticle3D(VGroup): __init__, animate_fall
  class _VarianceAnnotation3D(VGroup): __init__, _se, _build, update_n
  class PopulationDistribution3D(VGroup): __init__, _build_pdf_curve, _build_title, _build_x_axis, animate_build, get_bar_top_position
  class SampleMeanHistogram3D(VGroup): __init__, _data_to_scene_x, _bin_index, _bar_color, _build_bar_mob, _update_bar, _build_title, _build_mean_line, add_sample_mean, reset, build_normal_overlay, n_samples, empirical_mean, empirical_std
  class NormalConvergenceOverlay3D(VGroup): __init__, _make_curve_points, build_for_n, animate_narrow
  class CLTDemo: __init__, _pre_simulate, phase_population, phase_single_draw, phase_accumulate, phase_normal_overlay, phase_n_sweep, build_formula_panel, run
  class CLTUniformScene(ThreeDScene): construct
  class CLTExponentialScene(ThreeDScene): construct
  class CLTBimodalScene(ThreeDScene): construct
  class CLTSweepScene(ThreeDScene): construct
  class CLTComparisonScene(ThreeDScene): construct
  def _with_opacity(color: ManimColor, opacity: float)
  def _darken(color: ManimColor, factor: float=0.65)
  def _lighten(color: ManimColor, factor: float=1.35)
  def _lerp_color(a: ManimColor, b: ManimColor, t: float)
  def _normal_pdf(x: np.ndarray, mu: float, sigma: float)
  def _make_sampler(source: str, params: Dict, rng: np.random.Generator)
  def _population_range(source: str, params: Dict)
  def _theoretical_mean_std(source: str, params: Dict, n: int)

## animations/sampling.py
  class SamplingConfig: 
  class _PopDot3D(VGroup): __init__, select, deselect, move_to_sample_zone
  class PopulationCloud3D(VGroup): __init__, _generate_positions, animate_appear, color_by_value, get_indices_by_stratum, get_indices_by_cluster
  class _SampleZone3D(VGroup): __init__, grid_positions
  class _SweepLine3D(VGroup): __init__, animate_sweep
  class _StrataRegion3D(VGroup): __init__
  class _ClusterRegion3D(VGroup): __init__
  class _SampleStatAnnotation3D(VGroup): __init__
  class SampleSelector: __init__, build_sample_zone, animate_highlight, animate_extract, animate_fade, animate
  class SimpleRandomSampling3D: __init__, phase_population, phase_sample, phase_statistics, run
  class StratifiedSampling3D: __init__, _reposition_by_stratum, phase_population, phase_show_strata, phase_sample, phase_statistics, run
  class ClusterSampling3D: __init__, _cluster_grid_centers, _reposition_by_cluster, phase_population, phase_show_clusters, phase_select_clusters, phase_sample, run
  class SystematicSampling3D: __init__, phase_population, phase_sweep, phase_sample, run
  class BootstrapSampling3D: __init__, _init_bootstrap_hist, _boot_x_to_scene, _build_boot_bar, _update_boot_bar, phase_original_sample, phase_bootstrap_draw, phase_accumulate, phase_ci, run
  class SamplingDistributionBuilder(VGroup): __init__, _val_to_scene_x, _build_true_line, _build_axis, _build_bar, _update_bar, add_trial, run_accumulation
  class SRSScene(ThreeDScene): construct
  class StratifiedScene(ThreeDScene): construct
  class ClusterScene(ThreeDScene): construct
  class SystematicScene(ThreeDScene): construct
  class BootstrapScene(ThreeDScene): construct
  def _with_opacity(color: ManimColor, opacity: float)
  def _darken(color: ManimColor, factor: float=0.65)
  def _lighten(color: ManimColor, factor: float=1.35)
  def _lerp_color(a: ManimColor, b: ManimColor, t: float)
  def _normal_pdf(x: np.ndarray, mu: float, sigma: float)

## animations/transitions.py
  class DistMorph3D: __init__, build
  class HistMorph3D: __init__, _infer_geometry, build
  class CDFBuild3D: __init__, build, build_with_scene
  class HistToCurve3D: __init__, build
  class ParameterSweep3D: __init__, run, build_single_step
  class CurtainReveal3D: __init__, build
  class SceneWipe3D: __init__, build
  class FocusZoom3D: __init__, run
  class OrbitTransition: __init__, run
  class CIBuild3D: __init__, build
  class RippleUpdate3D: __init__, build
  class CollapseToMean3D: __init__, build, build_flash_at_mean
  class ScatterToRegression3D: __init__, build_residual_lines, build_collapse, run
  def _with_opacity(color: ManimColor, opacity: float)
  def _darken(color: ManimColor, factor: float=0.65)
  def _lighten(color: ManimColor, factor: float=1.35)
  def _lerp_color(a: ManimColor, b: ManimColor, t: float)
  def _resample_vmobject_points(mob: VMobject, n: int)
  def _build_normal_curve_points(mu: float, sigma: float, x_range: Tuple[(float, float)], n_pts: int, y_pos: float, scale: float)
  def _build_kde_curve_points(data: np.ndarray, x_range: Tuple[(float, float)], n_pts: int, y_pos: float, scale: float, bandwidth: Optional[float]=None)
  def _bar_heights_from_group(bar_group: VGroup)
  def dist_morph(source: VMobject, target: VMobject, run_time: float=1.5, n_interp_pts: int=300, color_start: Optional[ManimColor]=None, color_end: Optional[ManimColor]=None, rate_func: Callable=rate_functions.ease_in_out_cubic)
  def hist_morph(bar_group: VGroup, new_heights: Sequence[float], run_time: float=1.2, color_map: Optional[List[ManimColor]]=None, rate_func: Callable=rate_functions.ease_in_out_cubic)
  def cdf_build(pdf_curve: VMobject, cdf_curve: VMobject, area_fill: Optional[VMobject]=None, x_range: Tuple[(float, float)]=(-4.0, 4.0), run_time: float=2.0)
  def parameter_sweep(scene: ThreeDScene, func: Callable[([float, float], float)], x_range: Tuple[(float, float, float)], param_range: Tuple[(float, float, float)], param_name: str, existing_curve: VMobject, run_time_total: float, **kwargs=3.0)
  def curtain_reveal(target: VGroup, x_start: float, x_end: float=-4.0, run_time: float=4.0, **kwargs=1.5)
  def ci_build(mean_pos: np.ndarray, half_width: float, run_time: float, **kwargs=1.2)
  def collapse_to_mean(dots: List, mean_position: np.ndarray, run_time: float, lag: float=1.5, **kwargs=0.02)

## axes/__init__.py

## axes/axes3d.py
  class AxisScaleMode(Enum): 
  class TickStyle(Enum): 
  class GridStyle(Enum): 
  class AxisID(Enum): 
  class AxisConfig: 
  class GridConfig: 
  class AxesConfig: 
  class ScaleTransform: forward, inverse
  class AxisSpine3D(VGroup): __init__, set_spine_color
  class TickSystem3D(VGroup): __init__, _resolve_range, _make_tick
  class TickLabelSystem3D(VGroup): __init__, _resolve_range, _make_label
  class AxisLabel3D(VGroup): __init__, get_label_mob
  class GridPlane3D(VGroup): __init__, _resolve, _coord, _make_line, _plane_normal
  class GridSystem3D(VGroup): __init__
  class OriginDecoration3D(VGroup): __init__
  class BoundingBox3D(VGroup): __init__
  class ReferenceLineConfig: 
  class ReferenceLineSystem(VGroup): __init__, add_line, remove_line, update_line_value, add_span, add_bracket
  class PointAnnotationConfig: 
  class AnnotationSystem3D(VGroup): __init__, add_point, remove_annotation, add_region_label
  class ZeroLines3D(VGroup): __init__
  class StatsAxes3D(StatsObject3D): __init__, _build_geometry, c2p, p2c, data_to_scene_length, add_h_line, add_v_line, add_span, add_bracket, annotate_point, animate_build, animate_update, animate_update_range, animate_highlight, animate_exit, set_x_scale, set_y_scale, set_z_scale, for_distribution, for_scatter, for_histogram, for_3d_surface, for_correlation, for_hypothesis, for_time_series, x_range, y_range, z_range, config, __repr__

## axes/grid3d.py
  class GridConfig: 
  class GridPlane3D(VGroup): __init__, _uv_to_3d, _edge_fade_opacity, _make_line, _build_fill, _build_grid_lines, _add_lines_along_axis, _build_tick_labels, animate_build, set_opacity, highlight_line, get_snap_position
  class GridBoundingBox3D(VGroup): __init__, animate_build
  class FloatingGrid3D(GridPlane3D): __init__
  class FullGrid3D(VGroup): __init__, animate_build, set_floor_opacity, highlight_axes
  class BillboardLabel3D(VGroup): __init__
  class GridSnapHelper: __init__, _snap, snap, data_to_grid, axes_ticks
  def _with_opacity(color: ManimColor, opacity: float)
  def _fade_color(color: ManimColor, base_opacity: float, fade_factor: float)
  def make_stats_grid(x_range: Tuple[(float, float, float)]=(-4, 4, 1), y_range: Tuple[(float, float, float)]=(-4, 4, 1), z_range: Tuple[(float, float, float)]=(0, 6, 1), theme: str='stats', minor: int=4, label_axes: bool=False, show_bounding_box: bool=False)

## axes/number_plane3d.py
  class PlaneFace(Enum): 
  class PlaneMaterial(Enum): 
  class PlaneGridConfig: 
  class PlaneAxisConfig: 
  class PlaneConfig: 
  class PlaneGeometry(VGroup): __init__
  class PlaneGrid(VGroup): __init__, _make_line
  class PlaneAxes(VGroup): __init__, _add_spine
  class PlaneTickSystem(VGroup): __init__, _fmt_label
  class PlaneOriginMark(VGroup): __init__
  class PlaneDrawingLayer(VGroup): __init__, draw_curve, draw_region, draw_scatter, draw_vector, draw_label, remove_object, clear_all, get_object
  class NumberPlane3D(StatsObject3D): __init__, plane_c2p, plane_p2c, project_point, _build_geometry, animate_build, animate_update, animate_highlight, animate_exit, normal, basis, origin_scene
  class FaceNumberPlane3D(NumberPlane3D): __init__, plane_c2p
  class FloatingPlane3D(NumberPlane3D): __init__, _update_slice_position, animate_slice, slice_value
  class ShadowConfig: 
  class ShadowProjection3D(VGroup): __init__, animate_project
  class NumberPlane3DSystem(VGroup): __init__, add_plane, get_plane, animate_build_all, animate_exit_all, project_to_all, animate_project_to_all, for_axes
  def _normalize(v: np.ndarray)
  def _dim_color(hex_color: str, factor: float=0.6)
  def make_plane_for_distribution(u_range: Tuple[(float, float)]=(-4.0, 4.0), v_range: Tuple[(float, float)]=(0.0, 0.45), origin: Optional[np.ndarray]=None, facing: Literal[('front', 'right', 'top')]='front', theme: Optional[StatsColorPalette]=None)

## charts/__init__.py

## charts/bar_chart3d.py
  class BarColorPalette: sequential_ramp, value_mapped
  class BarConfig: 
  class _ValueLabel3D(VGroup): __init__
  class Bar3D(VGroup): __init__, _corners, _build_faces, _build_edges, _build_gloss, _build_shadow, _build_value_label, _build_category_label, animate_grow, set_height, get_top_center, set_bar_color, highlight, unhighlight
  class BarChart3D(VGroup): __init__, _build_bars, animate_grow, animate_update, highlight_bar, unhighlight_all, highlight_max, highlight_min, sorted_values, add_mean_line, add_value_labels, apply_value_coloring, apply_sequential_coloring
  class GroupedBarChart3D(VGroup): __init__, _add_category_labels, animate_grow, highlight_series
  class StackedBarChart3D(VGroup): __init__, animate_grow, highlight_layer
  def _with_opacity(color: ManimColor, opacity: float)
  def _darken(color: ManimColor, factor: float=0.6)
  def _lighten(color: ManimColor, factor: float=1.4)
  def _lerp_color(a: ManimColor, b: ManimColor, t: float)
  def bar_chart_from_dict(data: Dict[(str, float)], config: Optional[BarConfig], palette: str=None, **kwargs='categorical')
  def grouped_from_dataframe(df, series_colors: Optional[Sequence[ManimColor]], config: Optional[BarConfig]=None, **kwargs=None)

## charts/box_plot3d.py
  class BoxColorPalette: ramp
  class FiveNumberSummary: iqr, lower_fence, upper_fence, box_height, lower_whisker_length, upper_whisker_length, from_data, from_precomputed
  class BoxConfig: 
  class BoxBody3D(VGroup): __init__, _corners, _build_faces, _build_notched_front, _build_edges, _build_gloss, _build_shadow, animate_grow
  class MedianLine3D(VGroup): __init__
  class Whisker3D(VGroup): __init__, _build_line, _build_capped, _build_tapered
  class MeanMarker3D(VGroup): __init__
  class OutlierMarkers3D(VGroup): __init__, animate_scatter
  class SignificanceBracket3D(VGroup): __init__, animate_draw
  class BoxPlot3D(VGroup): __init__, _build_stats_label, animate_build, highlight, unhighlight, set_box_color, top_z, bottom_z, from_data
  class BoxPlotGroup3D(VGroup): __init__, from_data, from_precomputed, animate_build, animate_update, highlight_box, unhighlight_all, add_significance_bracket, apply_value_coloring
  class NotchedBoxPlot3D(BoxPlotGroup3D): __init__
  def _with_opacity(color: ManimColor, opacity: float)
  def _darken(color: ManimColor, factor: float=0.65)
  def _lighten(color: ManimColor, factor: float=1.35)
  def _lerp_color(a: ManimColor, b: ManimColor, t: float)
  def box_from_normal(mean: float, std: float=0.0, n: int=1.0, label: str=200, seed: int='', **kwargs=42)
  def comparison_chart(means: Sequence[float], stds: Sequence[float], labels: Sequence[str], n: int, **kwargs=100)

## charts/heat_map3d.py
  class HeatMapConfig: 
  class _ColorMapper: __init__, __call__
  class _HeatCell(VGroup): __init__, top_center, set_colors
  class _MaskedCell(VGroup): __init__
  class _HighlightRect(VGroup): __init__
  class _Dendrogram(VGroup): __init__
  class _ColorBar(VGroup): __init__
  class _MarginalBars(VGroup): __init__
  class _FloorGrid(VGroup): __init__
  class HeatMap3D(VGroup): __init__, _resolve_palette, _build, _cell_rect, highlight_row, highlight_col, highlight_cell, animate_grow, animate_sweep_row, animate_sweep_col, animate_highlight_row, animate_highlight_col, animate_highlight_cell, animate_morph_values, animate_palette_morph, animate_reveal_colorbar, animate_reveal_dendro, from_correlation, from_confusion, random_demo
  def _palette_color(t: float, stops: list[ManimColor])
  def _perceived_luminance(color: ManimColor)
  def _contrast_text_color(bg: ManimColor)
  def _darken(c: ManimColor, f: float)
  def _lighten(c: ManimColor, f: float)

## charts/histogram3d.py
  class HistogramConfig: 
  class _BarPrism(VGroup): __init__, top_center, set_bar_height
  class _FloorGrid(VGroup): __init__
  class _KDE3D(VMobject): __init__
  class _StatPlane(VGroup): __init__
  class _SigmaBand(VGroup): __init__
  class _AxisTicks(VGroup): __init__
  class Histogram3D(VGroup): __init__, _build, animate_grow, animate_sweep_bins, animate_highlight_bin, animate_morph_to, animate_reveal_stats, from_normal, from_exponential, from_bimodal, from_uniform
  def _darken(color: ManimColor, factor: float)
  def _lighten(color: ManimColor, factor: float)
  def _freq_color(value: float, vmin: float, vmax: float, palette: Sequence[ManimColor] | None=None)

## charts/line_plot3d.py
  class LineColorPalette: ramp
  class LineConfig: 
  class LineMarker3D(VGroup): __init__
  class _DropLines3D(VGroup): __init__
  class AreaFill3D(VGroup): __init__
  class LineSeries3D(VGroup): __init__, _build_layers, _build_point_labels, animate_draw, animate_trace, animate_fill, animate_rise, animate_update, highlight_range, add_horizontal_marker, get_value_at
  class MultiLinePlot3D(VGroup): __init__, animate_reveal, animate_update, highlight_series, unhighlight_all, add_legend
  class CDFLine3D(VGroup): __init__, _build_step_points, shade_tail_left, shade_tail_right, shade_between, add_critical_marker, animate_draw
  class TimeSeriesPlot3D(VGroup): __init__, _build_tick_labels, add_rolling_window, add_event_marker, add_anomaly_markers, animate_draw
  class ParametricLine3D(VGroup): __init__, animate_draw, animate_trace, get_tangent
  def _with_opacity(color: ManimColor, opacity: float)
  def _darken(color: ManimColor, factor: float=0.65)
  def _lighten(color: ManimColor, factor: float=1.35)
  def _lerp_color(a: ManimColor, b: ManimColor, t: float)
  def _catmull_rom_chain(points: np.ndarray, resolution: int=16)
  def _build_polyline(points: np.ndarray, config: LineConfig)
  def _build_glow(points: np.ndarray, config: LineConfig)
  def _build_area_fill(curve_pts: np.ndarray, config: LineConfig, y_depth: float=0.0)
  def line_from_function(func: Callable[([float], float)], x_range: Tuple[(float, float, float)]=(0, 5, 0.1), y_position: float=0.0, config: Optional[LineConfig]=None, scene: Optional[ThreeDScene]=None)
  def parametric_helix(radius: float=1.0, height: float=3.0, turns: float=3.0, color: ManimColor=ManimColor('#4A90D9'), scene: Optional[ThreeDScene]=None)

## charts/scatter_plot3d.py
  class ScatterSeries: __post_init__, xyz, from_arrays
  class ScatterConfig: 
  class _CoordMapper: __init__, __call__, x_world, y_world, z_world
  class _BackGridPlane(VGroup): __init__
  class _RegressionPlane(VGroup): __init__, predict
  class _ResidualArrows(VGroup): __init__
  class _CovarianceEllipsoid(VGroup): __init__
  class _MarginalHistogram(VGroup): __init__
  class _ScatterAxisTicks(VGroup): __init__
  class _CorrelationBadge(VGroup): __init__
  class ScatterPlot3D(VGroup): __init__, _build, _build_legend, animate_plot_points, animate_fit_plane, animate_draw_residuals, animate_ellipsoid, animate_highlight_point, animate_morph_series, animate_reveal_stats, animate_color_by_residual, from_xy, from_linear, from_quadratic, from_clusters, from_bubble
  def _darken(c: ManimColor, f: float)
  def _lighten(c: ManimColor, f: float)
  def _freq_color(t: float, cold: ManimColor, hot: ManimColor)
  def _make_point_glyph(world_pos: np.ndarray, color: ManimColor, radius: float, opacity: float, stroke_width: float, use_sphere: bool)

## charts/violin_plot3d.py
  class ViolinConfig: 
  class ViolinGroup: __post_init__, q1, q3, median, mean, iqr, whisker_bounds, outliers
  class _YMapper: __init__, __call__, inv
  class _IQRBox(VGroup): __init__
  class _MeanDiamond(Polygon): __init__
  class _SigBracket(VGroup): __init__
  class _ViolinSurface(VGroup): __init__, get_zone_surfaces
  class _SingleViolin(VGroup): __init__
  class _FloorGrid(VGroup): __init__
  class ViolinPlot3D(VGroup): __init__, _build, _build_significance_brackets, animate_grow, animate_grow_all, animate_reveal_boxplot, animate_reveal_boxplots, animate_drop_jitter, animate_drop_jitter_all, animate_compare_groups, animate_highlight_group, animate_split_reveal, animate_morph_bandwidth, single, from_normal_groups, from_skewed_groups, split_pair
  def _darken(c: ManimColor, f: float)
  def _lighten(c: ManimColor, f: float)
  def _compute_kde(data: np.ndarray, bandwidth: float | str, n_points: int=256, padding: float=0.05)

## core/__init__.py

## core/base.py
  class ThemeMode(Enum): 
  class MaterialStyle(Enum): 
  class LabelAnchor(Enum): 
  class HighlightStyle(Enum): 
  class BuildStyle(Enum): 
  class DataUpdateMode(Enum): 
  class StatsColorPalette: dist_color
  class StatsTheme: set, set_custom, current, mode, on_change, get_gradient, get_diverging
  class MaterialConfig: matte, glass, metallic, emissive, holographic, wireframe
  class MaterialApplicator: apply, add_holographic_updater, add_shadow
  class LabelConfig: 
  class LabelAttachment: __init__, add, remove, update_text, show, hide, get_mob, group, reposition_all, _build_label_mob, _position_label
  class BoundData: subscribe, update, as_array, n, mean, std, min, max
  class AnimationConfig: 
  class AnimationProtocol(ABC): animate_build, animate_update, animate_highlight, animate_exit, _default_build, _default_exit, _cascade_build
  class HighlightSystem: pulse, glow, color_shift, shake, outline, flash
  class CompositionHelper: align_left, align_right, align_top, align_bottom, center_on, distribute_horizontally, distribute_vertically, arrange_in_grid, snap_to_grid, combined_bounding_box
  class CameraAwareness: register_camera, get_camera_direction, face_camera, add_billboard_updater
  class StatsObject3D(VGroup, AnimationProtocol, CameraAwareness): __init__, _build_geometry, animate_build, animate_update, animate_highlight, animate_exit, bind_data, get_data, _on_data_update, add_label, remove_label, update_label, show_label, hide_label, set_material, composition, select, deselect, _on_theme_change, _refresh_colors, _unfold_build, __del__
  class StatsSurface3D(StatsObject3D): __init__, _make_surface, shade_region
  class StatsChart3D(StatsObject3D): __init__, _build_axes, c2p, add_axis_labels
  class StatsProp3D(StatsObject3D): __init__, throw
  def interpolate(a: float, b: float, t: float)

## core/colors.py
  class StatColor: from_hex, from_rgb8, from_rgb_float, from_hsl, rgb8, hex, hsl, luminance, contrast_ratio, is_accessible, best_label_color, lighten, darken, desaturate, saturate, shift_hue, with_alpha, mix, to_manim, __repr__, __str__
  class StatColorAlpha: rgba8, hex_alpha, to_manim
  class ColorFamily: gradient, diverging, shade, with_opacity, to_matplotlib_cmap, __repr__
  class ColorFamilyAlpha: __getattr__
  class DistributionPalette: 
  class ThemeMode(Enum): 
  class StatsTheme: apply, axes_kwargs, number_plane_kwargs, text_style, register_custom_palette, palette, __repr__
  class SemanticRole(Enum): 
  def _srgb_to_linear(c: float)
  def _linear_to_srgb(c: float)
  def gradient_ramp(colors: Sequence[StatColor], n: int=256, weights: Optional[Sequence[float]]=None)
  def diverging_map(low: StatColor, high: StatColor, n: int=256, mid: Optional[StatColor]=None)
  def sequential_map(family: ColorFamily, n: int=256)
  def qualitative_palette(families: Sequence[ColorFamily], stop: str='base')
  def heatmap_colormap(low: StatColor, high: StatColor, n: int=256, gamma: float=1.0)
  def interpolate_colors(c1: StatColor, c2: StatColor, steps: int=10)
  def _make_dark_theme(palettes: dict)
  def _make_light_theme(palettes: dict)
  def _make_paper_theme(palettes: dict)
  def _make_neon_theme(palettes: dict)
  def _make_pastel_theme(palettes: dict)
  def _make_monochrome_theme(palettes: dict)
  def get_theme(name: str)
  def register_theme(name: str, theme: StatsTheme)
  def resolve_color(role: SemanticRole, palette_key: str='normal', theme: Optional[StatsTheme]=None, stop: str='base')

## core/math_utils.py
  class DistributionResult: 
  class DistributionFunction: __init__, evaluate, pdf, cdf, ppf, sf, sample, normal, student_t, chi_squared, f_distribution, exponential, gamma, beta, uniform_continuous, log_normal, weibull, cauchy, pareto, laplace, logistic, bernoulli, binomial, poisson, geometric, negative_binomial, hypergeometric, uniform_discrete, gaussian_mixture
  class DescriptiveStats: 
  class KDEResult: 
  class OLSResult: 
  class CorrelationResult: 
  class TestResult: 
  class ProbTreeNode: 
  def compute_descriptive(data: ArrayLike, weights: Optional[ArrayLike]=None, trim: float=0.1, pcts: Sequence[float]=(1, 5, 10, 25, 50, 75, 90, 95, 99))
  def compute_running_stats(data: ArrayLike)
  def _kernel_func(name: KernelName)
  def bandwidth_silverman(x: FloatArray)
  def bandwidth_scott(x: FloatArray)
  def bandwidth_cv(x: FloatArray, bandwidths: Optional[FloatArray]=None, cv_folds: int=5)
  def compute_kde(data: ArrayLike, x_eval: Optional[ArrayLike]=None, kernel: KernelName='gaussian', bandwidth: Union[(float, Literal['silverman', 'scott', 'cv'])]='silverman', n_points: int=256, x_padding: float=0.15)
  def compute_ols(x: ArrayLike, y: ArrayLike, add_intercept: bool=True, weights: Optional[ArrayLike]=None, x_grid_n: int=200, alpha: float=0.05)
  def compute_vif(X: ArrayLike)
  def pearson_correlation(x: ArrayLike, y: ArrayLike, alpha: float=0.05)
  def spearman_correlation(x: ArrayLike, y: ArrayLike)
  def kendall_tau(x: ArrayLike, y: ArrayLike)
  def correlation_matrix(data: ArrayLike, method: Literal[('pearson', 'spearman')]='pearson')
  def partial_correlation(x: ArrayLike, y: ArrayLike, controls: ArrayLike)
  def distance_correlation(x: ArrayLike, y: ArrayLike)
  def z_test_one_sample(x: ArrayLike, mu0: float, sigma: float, alpha: float=0.05, tail: Literal[('two', 'left', 'right')]='two')
  def t_test_one_sample(x: ArrayLike, mu0: float, alpha: float=0.05, tail: Literal[('two', 'left', 'right')]='two')
  def t_test_two_sample(x1: ArrayLike, x2: ArrayLike, equal_var: bool=True, alpha: float=0.05, tail: Literal[('two', 'left', 'right')]='two')
  def chi_square_gof(observed: ArrayLike, expected: Optional[ArrayLike]=None, alpha: float=0.05)
  def chi_square_independence(contingency: ArrayLike, alpha: float=0.05)
  def f_test_anova(groups: Sequence[ArrayLike], alpha: float=0.05)
  def compute_power(effect_size: float, n: int, alpha: float=0.05, test: Literal[('z', 't', 'chi2')]='t', df: Optional[int]=None)
  def sample_size_for_power(effect_size: float, power: float=0.8, alpha: float=0.05, test: Literal[('z', 't')]='t', max_n: int=10000)
  def build_probability_tree(branch_probs: Sequence[Sequence[float]], branch_labels: Optional[Sequence[Sequence[str]]]=None, h_spacing: float=2.5, v_spacing: float=1.2)
  def bayes_update(prior: ArrayLike, likelihood: ArrayLike)
  def generate_sample_space_grid(outcomes: Sequence[Any], events: Optional[Dict[(str, Callable[[Any], bool])]]=None)
  def combinatorics(n: int, r: int, with_replacement: bool=False, ordered: bool=False)
  def bootstrap_sample(data: ArrayLike, n_boot: int=1000, statistic: Callable[([FloatArray], float)]=np.mean, seed: Optional[int]=None, alpha: float=0.05)
  def permutation_test(x: ArrayLike, y: ArrayLike, statistic: Callable[([FloatArray, FloatArray], float)]=lambda a, b: a.mean() - b.mean(), n_perm: int=5000, tail: Literal[('two', 'left', 'right')]='two', seed: Optional[int]=None)
  def stratified_sample_indices(strata: ArrayLike, n_per_stratum: int=10, seed: Optional[int]=None)
  def systematic_sample_indices(n: int, k: int)
  def cluster_sample_indices(cluster_ids: ArrayLike, n_clusters: int, seed: Optional[int]=None)
  def bivariate_normal_mesh(mu: ArrayLike=(0.0, 0.0), sigma: ArrayLike=((1.0, 0.0), (0.0, 1.0)), x_range: Tuple[(float, float)]=(-3.5, 3.5), y_range: Tuple[(float, float)]=(-3.5, 3.5), resolution: int=80, z_scale: float=1.0)
  def make_surface_func(X: FloatArray, Y: FloatArray, Z: FloatArray)
  def contour_levels(Z: FloatArray, n_levels: int=8)
  def histogram_mesh(data: ArrayLike, bins: Union[(int, ArrayLike)]=20, normalize: bool=True, z_depth: float=0.3)
  def scatter_point_cloud(x: ArrayLike, y: ArrayLike, z: Optional[ArrayLike]=None, color_by: Optional[ArrayLike]=None, jitter: float=0.0, seed: Optional[int]=None)
  def smooth_curve(x: ArrayLike, y: ArrayLike, n_out: int=300, kind: Literal[('cubic', 'quadratic', 'linear', 'akima')]='cubic')
  def lowess_smooth(x: ArrayLike, y: ArrayLike, frac: float=0.3, n_out: int=300)
  def bezier_points(control_pts: ArrayLike, n: int=100)
  def moving_average(x: ArrayLike, window: int, mode: Literal[('simple', 'exponential', 'weighted')]='simple', alpha: float=0.3)
  def entropy(p: ArrayLike, base: float=math.e)
  def kl_divergence(p: ArrayLike, q: ArrayLike, base: float=math.e)
  def js_divergence(p: ArrayLike, q: ArrayLike)
  def mutual_information(joint: ArrayLike, base: float=math.e)
  def cross_entropy(p: ArrayLike, q: ArrayLike, base: float=math.e)
  def differential_entropy(dist: 'DistributionFunction', n_points: int=1000)
  def covariance_matrix(data: ArrayLike, ddof: int=1, weights: Optional[ArrayLike]=None)
  def pca(data: ArrayLike, n_components: Optional[int]=None)
  def confidence_ellipse_params(x: ArrayLike, y: ArrayLike, n_std: float=2.0)
  def cholesky_sample(mu: ArrayLike, sigma: ArrayLike, n: int, seed: Optional[int]=None)
  def make_positive_semidefinite(M: ArrayLike)
  def area_under_curve(f: Callable[([FloatArray], FloatArray)], a: float, b: float, method: Literal[('quad', 'trapz', 'simpson', 'montecarlo')]='quad', n: int=1000, seed: Optional[int]=None)
  def cdf_from_pdf(pdf_vals: FloatArray, x: FloatArray)
  def pdf_from_samples(samples: ArrayLike, x_eval: ArrayLike, bandwidth: Union[(float, str)]='silverman')
  def nice_number(x: float, round_: bool=False)
  def auto_range(data: ArrayLike, padding: float=0.05, n_ticks: int=5)
  def generate_ticks(lo: float, hi: float, step: float, minor_per_major: int=4)
  def format_stat_value(value: float, decimals: int=4, sci_threshold: float=0.001)

## distributions/__init__.py

## distributions/base_dist.py
  class RepresentationMode(Enum): 
  class ShadeFillStyle(Enum): 
  class MomentMarkerStyle(Enum): 
  class DistributionCurveConfig: 
  class ShadeRegionConfig: 
  class MomentMarkerConfig: 
  class StatsAnnotationConfig: 
  class ProbeConfig: 
  class ParameterTrackerSystem: __init__, register, on_any_change, _fire_callbacks, get, set, tracker, all_values, all_trackers, validate
  class FillRegionSystem(VGroup): __init__, add_region, remove_region, update_region, get_region, clear_all, _build_region, _solid_polygon, _gradient_strips, _stripe_overlay, _compute_prob, _build_prob_label, _select_y
  class MomentMarkerSystem(VGroup): __init__, rebuild, _build, _make_marker, _make_sigma_band
  class StatsAnnotationPanel(VGroup): __init__, rebuild, _build, _collect_rows
  class PercentileProbe3D(VGroup): __init__, _build, animate_build
  class ComparisonConfig: 
  class ComparisonOverlay(VGroup): __init__
  class BaseDistribution3D(StatsObject3D): __init__, _build_body, _param_constraints, _build_geometry, _evaluate, _rebuild_dist_fn, _on_param_change, _rebuild, set_mode, animate_mode_switch, animate_param, animate_params, shade_region, shade_tail_left, shade_tail_right, shade_central, shade_by_sigma, remove_shade, clear_shades, probe_at, probe_quantile, remove_probe, show_confidence_interval, add_comparison, animate_build, _animate_body_build, animate_update, animate_highlight, animate_exit, animate_cdf_build, result, mode, mean, std, __repr__
  class ContinuousDistribution3D(BaseDistribution3D): _build_body, _animate_body_build, _param_constraints
  class BarConfig: 
  class DiscreteDistribution3D(BaseDistribution3D): __init__, _build_body, _make_bar, _animate_body_build, _param_constraints, animate_morph_to_continuous, highlight_bar

## distributions/cdf_viz.py
  class CDFDisplayMode(Enum): 
  class StepJumpStyle(Enum): 
  class CDFCurveConfig: 
  class StepConfig: 
  class ReadoutConfig: 
  class ComparisonConfig: 
  class QuantileFnConfig: 
  class ECDFConfig: 
  class CDFCurveLayer(VGroup): __init__, _build, _select_y
  class StepFunction3D(VGroup): __init__, _build
  class PLevelConfig: 
  class CDFAnnotationSystem(VGroup): __init__, add_p_level, remove_p_level, clear_all, _build_p_level
  class ProbabilityReadout3D(VGroup): __init__, _make_probe_from_x, sweep_x, sweep_p
  class CDFPair: 
  class CDFComparisonLayer(VGroup): __init__, _add_ks_layer
  class QuantileFunction3D(VGroup): __init__, _build
  class CDFDecompositionPanel(VGroup): __init__
  class ECDFLayer(VGroup): __init__
  class SurvivalFunctionLayer(VGroup): __init__
  class CDFViz3D(StatsObject3D): __init__, _build_geometry, add_comparison, add_ecdf, add_p_level, mark_critical_value, mark_ci_bounds, animate_build, _animate_curve_build, animate_update, animate_highlight, animate_exit, set_mode, animate_mode_switch, result, mode, __repr__

## distributions/continuous_dists.py
  class NormalDist3D(ContinuousDistribution3D): __init__, _rebuild_dist_fn, _param_constraints, empirical_rule, show_standard_normal, animate_sigma_sweep, animate_mu_shift, animate_standardize, show_formula, z_score_annotation, standard, wide, narrow, shifted
  class StudentTDist3D(ContinuousDistribution3D): __init__, _rebuild_dist_fn, _param_constraints, animate_df_to_normal, highlight_heavy_tails, critical_values, show_formula, df1, df5, df30
  class ChiSquaredDist3D(ContinuousDistribution3D): __init__, _rebuild_dist_fn, _param_constraints, animate_df_convergence, shade_critical_region, show_normal_approx, show_formula, df1, df5, df10
  class FDist3D(ContinuousDistribution3D): __init__, _rebuild_dist_fn, _param_constraints, shade_critical_region, animate_dfn_sweep, animate_dfd_sweep, show_formula, anova
  class ExponentialDist3D(ContinuousDistribution3D): __init__, _rebuild_dist_fn, _param_constraints, memoryless_demo, show_half_life, animate_rate_sweep, show_formula, rate1, rate2, rate_half
  class GammaDist3D(ContinuousDistribution3D): __init__, _rebuild_dist_fn, _param_constraints, animate_shape_sweep, show_special_cases, show_formula, exponential_special, chi_squared_special
  class BetaDist3D(ContinuousDistribution3D): __init__, _rebuild_dist_fn, _param_constraints, bayesian_update, animate_a_sweep, animate_b_sweep, show_mode_marker, show_formula, uniform, symmetric, u_shaped, j_shaped, informative_prior
  class UniformContDist3D(ContinuousDistribution3D): __init__, _rebuild_dist_fn, _param_constraints, show_mean_variance, animate_expand, show_formula, unit, symmetric
  class LogNormalDist3D(ContinuousDistribution3D): __init__, _rebuild_dist_fn, _param_constraints, show_natural_params, show_log_transform, animate_sigma_sweep, show_formula
  class WeibullDist3D(ContinuousDistribution3D): __init__, _rebuild_dist_fn, _param_constraints, hazard_rate_demo, animate_shape_sweep, show_bathtub_label, show_formula, exponential_special, rayleigh
  class CauchyDist3D(ContinuousDistribution3D): __init__, _rebuild_dist_fn, _param_constraints, show_undefined_moments, compare_to_normal, animate_scale_sweep, show_formula, standard
  class ParetoDist3D(ContinuousDistribution3D): __init__, _rebuild_dist_fn, _param_constraints, show_8020_rule, show_power_law, animate_tail_sweep, show_formula
  class LaplaceDist3D(ContinuousDistribution3D): __init__, _rebuild_dist_fn, _param_constraints, compare_to_normal, animate_b_sweep, show_formula, standard
  class LogisticDist3D(ContinuousDistribution3D): __init__, _rebuild_dist_fn, _param_constraints, show_sigmoid_cdf, show_logit_transform, compare_to_normal, animate_s_sweep, show_formula, standard
  def _formula_panel(latex: str, theme: StatsColorPalette, position: np.ndarray, font_size: float=28, title: str='')
  def _annotation_arrow(axes: StatsAxes3D, x_data: float, y_data: float, label: str, color: str, offset: np.ndarray, font_size: float=22, is_math: bool=True)

## distributions/discrete_dists.py
  class BernoulliDistribution3D(DiscreteDistribution3D): __init__, _rebuild_dist_fn, _param_constraints, _make_bar_labels, _make_p_annotation, animate_p_sweep, prob_at, cumulative_at, tail_at, from_data, from_moments
  class BinomialDistribution3D(DiscreteDistribution3D): __init__, _rebuild_dist_fn, _param_constraints, _make_mean_var_labels, overlay_normal_approx, overlay_poisson_approx, highlight_mode, shade_at_least, shade_at_most, shade_exact, prob_at, cumulative_at, tail_at, animate_n_increase, animate_p_sweep, animate_to_normal, animate_cdf_build, from_data, from_moments
  class PoissonDistribution3D(DiscreteDistribution3D): __init__, _rebuild_dist_fn, _param_constraints, _make_mean_var_label, _make_rare_event_annotation, overlay_normal_approx, shade_at_least, shade_at_most, shade_exact, prob_at, cumulative_at, tail_at, highlight_mode, animate_lambda_sweep, animate_cdf_build, from_data, from_moments
  class GeometricDistribution3D(DiscreteDistribution3D): __init__, _rebuild_dist_fn, _param_constraints, _make_memoryless_annotation, _make_mean_label, shade_first_success, shade_within, prob_at, cumulative_at, tail_at, animate_p_sweep, animate_cdf_build, from_data, from_moments
  class NegativeBinomialDistribution3D(DiscreteDistribution3D): __init__, _rebuild_dist_fn, _param_constraints, _make_overdispersion_label, shade_at_most, shade_at_least, prob_at, cumulative_at, tail_at, animate_r_sweep, animate_p_sweep, from_data, from_moments
  class HypergeometricDistribution3D(DiscreteDistribution3D): __init__, _rebuild_dist_fn, _param_constraints, _make_urn_annotation, _make_mean_var_label, compare_to_binomial, shade_at_most, shade_at_least, prob_at, cumulative_at, tail_at, animate_N_sweep, animate_n_sweep, from_data
  class DiscreteUniformDistribution3D(DiscreteDistribution3D): __init__, _rebuild_dist_fn, _param_constraints, _make_entropy_label, _make_height_label, shade_range, prob_at, cumulative_at, tail_at, animate_range_expand, animate_range_shift, from_data
  class MultinomialDistribution3D(DiscreteDistribution3D): __init__, _rebuild_dist_fn, _param_constraints, _make_sum_constraint_label, show_marginals, shade_category, animate_p_shift, from_data
  def _formula_panel(tex: str, position: np.ndarray, theme: StatsColorPalette, font_size: float=26)
  def _stat_label(text: str, position: np.ndarray, theme: StatsColorPalette, color: Optional[str]=None, font_size: float=20, is_math: bool=True)
  def _overlay_continuous(axes: StatsAxes3D, dist_fn: DistributionFunction, x_vals: np.ndarray, color: str, opacity: float=0.65, width: float=0.009, label: str='')

## distributions/normal3d.py
  class BivariateNormalConfig: 
  class NormalCurve3D(VGroup): __init__, _build_curve, _build_fill, _build_moments, animate_rule_68_95_99, _build_rule_zone, add_zscore_ruler, add_title, add_clt_overlay, shade_region, animate_standardise, animate_curve, animate_fill, animate_moments, full_reveal
  class BivariateNormal3D(VGroup): __init__, _face_normal, _shade, _z_to_color, _build_surface, _build_marginal_x, _build_marginal_y, _build_contours, _build_axes, add_conditional_slice, animate_rise, animate_contours, animate_marginals, animate_correlation_sweep, full_reveal
  class NormalApproximation3D(VGroup): __init__, animate_bars, animate_curve, full_reveal
  class QQPlot3D(VGroup): __init__, animate_reveal
  class NormalCurveScene(ThreeDScene): construct
  class BivariateNormalScene(ThreeDScene): construct
  class StandardisationScene(ThreeDScene): construct
  class NormalApproximationScene(ThreeDScene): construct
  class QQPlotScene(ThreeDScene): construct
  def _with_opacity(color: ManimColor, opacity: float)
  def _darken(color: ManimColor, factor: float=0.65)
  def _lighten(color: ManimColor, factor: float=1.35)
  def _lerp_color(a: ManimColor, b: ManimColor, t: float)
  def _erf(z: float)
  def _normal_pdf(x: np.ndarray, mu: float=0.0, sigma: float=1.0)
  def _normal_cdf(x: float, mu: float=0.0, sigma: float=1.0)
  def _bivariate_normal_pdf(x: np.ndarray, y: np.ndarray, mu_x: float, mu_y: float, sigma_x: float, sigma_y: float, rho: float)
  def _catmull_rom(points: np.ndarray, resolution: int=16)
  def _gradient_fill(curve_pts: np.ndarray, floor_z: float, y_pos: float, color: ManimColor, opacity: float, n_strips: int=12)

## distributions/pdf_viz.py
  class PDFDistribution(ABC): pdf, cdf, ppf, name, param_string, default_x_range, mean, variance, std, pdf_array
  class NormalDist(PDFDistribution): __init__, pdf, cdf, name, param_string, default_x_range, mean, variance
  class StudentTDist(PDFDistribution): __init__, pdf, cdf, name, param_string, default_x_range, mean, variance
  class ChiSquaredDist(PDFDistribution): __init__, pdf, cdf, name, param_string, default_x_range, mean, variance
  class GammaDist(PDFDistribution): __init__, pdf, cdf, name, param_string, default_x_range, mean, variance
  class BetaDist(PDFDistribution): __init__, pdf, cdf, name, param_string, default_x_range, mean, variance
  class ExponentialDist(PDFDistribution): __init__, pdf, cdf, name, param_string, default_x_range, mean, variance
  class CauchyDist(PDFDistribution): __init__, pdf, cdf, name, param_string, default_x_range, mean, variance
  class LaplaceDist(PDFDistribution): __init__, pdf, cdf, name, param_string, default_x_range, mean, variance
  class LogNormalDist(PDFDistribution): __init__, pdf, cdf, name, param_string, default_x_range, mean, variance
  class WeibullDist(PDFDistribution): __init__, pdf, cdf, name, param_string, default_x_range, mean, variance
  class UniformDist(PDFDistribution): __init__, pdf, cdf, name, param_string, default_x_range, mean, variance
  class PDFConfig: 
  class PDFVisualizer3D(VGroup): __init__, _build_curve, _build_fill, _build_moments, _build_title, shade_region, shade_tail_left, shade_tail_right, shade_two_tails, shade_between_sigmas, add_percentile_marker, add_percentile_markers, add_kde_overlay, add_critical_value, morph_to_distribution, animate_curve, animate_fill, animate_moments, animate_title, animate_percentiles, animate_regions, full_reveal
  class MultiplePDFComparison3D(VGroup): __init__, _build_divergence, _build_legend, animate_reveal, animate_fills, highlight_distribution, morph_parameter
  class NormalPDFScene(ThreeDScene): construct
  class TDistScene(ThreeDScene): construct
  class GammaPDFScene(ThreeDScene): construct
  class PDFComparisonScene(ThreeDScene): construct
  class PDFParameterSweepScene(ThreeDScene): construct
  def _with_opacity(color: ManimColor, opacity: float)
  def _darken(color: ManimColor, factor: float=0.65)
  def _lighten(color: ManimColor, factor: float=1.35)
  def _lerp_color(a: ManimColor, b: ManimColor, t: float)
  def _catmull_rom(points: np.ndarray, resolution: int=20)
  def _build_fill_polygon(curve_pts: np.ndarray, floor_z: float, y_pos: float, n_gradient_strips: int=10, color: ManimColor=ManimColor('#4A90D9'), opacity: float=0.22, gradient: bool=True)
  def _erf(z: float)
  def _lgamma(x: float)
  def _reg_lower_gamma(a: float, x: float, n_terms: int=150)
  def _reg_inc_beta(a: float, b: float, x: float, n_terms: int=200)
  def _resample_equal(pts: np.ndarray, n: int)

## distributions/pmf_viz.py
  class PMFDistribution(ABC): pmf, cdf, support, name, param_string, mean, variance, std, mode, pmf_array
  class BernoulliDist(PMFDistribution): __init__, pmf, cdf, support, name, param_string, mean, variance
  class BinomialDist(PMFDistribution): __init__, pmf, cdf, support, name, param_string, mean, variance
  class GeometricDist(PMFDistribution): __init__, pmf, cdf, support, name, param_string, mean, variance
  class NegBinomialDist(PMFDistribution): __init__, pmf, support, name, param_string, mean, variance
  class PoissonDist(PMFDistribution): __init__, pmf, cdf, support, name, param_string, mean, variance
  class HypergeometricDist(PMFDistribution): __init__, pmf, support, name, param_string, mean, variance
  class DiscreteUniformDist(PMFDistribution): __init__, pmf, cdf, support, name, param_string, mean, variance
  class PMFConfig: 
  class _PMFBar3D(VGroup): __init__, set_height_instant, animate_grow, highlight, restore_color
  class PMFVisualizer3D(VGroup): __init__, _build_bars, _build_spine, _build_mean_mode_markers, _add_float_marker, _add_vertical_marker, _build_k_labels, _build_prob_labels, _build_title, shade_region, shade_tail_left, shade_tail_right, shade_two_tails, restore_colors, build_cdf_overlay, morph_to_distribution, animate_bars, animate_mean_mode, animate_cdf, animate_title, animate_regions, full_reveal
  class MultiplePMFComparison3D(VGroup): __init__, _build_legend, animate_reveal
  class BinomialPMFScene(ThreeDScene): construct
  class PoissonPMFScene(ThreeDScene): construct
  class GeometricPMFScene(ThreeDScene): construct
  class PMFComparisonScene(ThreeDScene): construct
  class BinomialNSweepScene(ThreeDScene): construct
  def _with_opacity(color: ManimColor, opacity: float)
  def _darken(color: ManimColor, factor: float=0.65)
  def _lighten(color: ManimColor, factor: float=1.35)
  def _lerp_color(a: ManimColor, b: ManimColor, t: float)
  def _lgamma(x: float)
  def _log_comb(n: int, k: int)

## inference/__init__.py

## inference/confidence_interval.py
  class _CIBar(VGroup): __init__, _make_fog
  class _EndCap(VGroup): __init__
  class _PointEstimateMarker(VGroup): __init__
  class _CIBadge(VGroup): __init__
  class _TrueParamLine(VGroup): __init__, set_status
  class _NumberLineAxis(VGroup): __init__, to_px
  class ConfidenceInterval3D(VGroup): __init__, center, lower, upper, half_width, captures, unit_scale, reveal_capture_status, get_capture_color, rescale_to_axis
  class CIStack(VGroup): __init__, _make_counter_badge, from_simulation
  class BuildCI(Animation): __init__, interpolate_mobject
  class RevealCapture(Succession): __init__
  class SweepCI(Animation): __init__, interpolate_mobject
  class NarrowCI(Animation): __init__, interpolate_mobject
  class StackCIs(Succession): __init__
  class CIDemo(Scene): construct
  def _z_half_width(std_err: float, confidence: float=0.95)
  def _t_half_width(std_err: float, df: int, confidence: float=0.95)
  def _captures(lower: float, upper: float, true_param: float)
  def make_ci_comparison(centers: list[float], half_widths: list[float], confidence_levels: list[float], true_param: Optional[float]=None, x_range: tuple=(60.0, 100.0, 5.0), axis_length: float=9.0, y_spacing: float=0.9, themes: Optional[list[str]]=None)
  def make_clt_ci_stack(true_mean: float, true_std: float=80.0, sample_size: int=10.0, n_samples: int=30, confidence: float=20, x_range: tuple=0.95, seed: int=(60.0, 100.0, 5.0), **stack_kwargs=42)

## inference/error_types.py
  class _NormalCurve(VGroup): __init__
  class _ErrorRegion(VGroup): __init__
  class _CriticalLine(VGroup): __init__
  class _EffectArrow(VGroup): __init__
  class _DecisionTable(VGroup): __init__
  class TypeITypeII(VGroup): __init__, _build, rebuild, z_test, t_test
  class BuildDistributions(Animation): __init__, interpolate_mobject
  class RevealAlpha(Animation): __init__, interpolate_mobject
  class RevealBeta(Animation): __init__, interpolate_mobject
  class RevealPower(Animation): __init__, interpolate_mobject
  class RevealAll(Succession): __init__
  class ShiftH1(Animation): __init__, interpolate_mobject
  class NarrowCurves(Animation): __init__, interpolate_mobject
  class SweepAlpha(Animation): __init__, interpolate_mobject
  class FlashDecision(AnimationGroup): __init__
  class BuildDecisionTable(Animation): __init__, interpolate_mobject
  class ErrorTypesDemo(Scene): construct
  def _norm_pdf(x: np.ndarray, mu: float, sigma: float)
  def _norm_cdf(x: float, mu: float, sigma: float)
  def _norm_ppf(p: float, mu: float, sigma: float)
  def _norm_pdf(x, mu, sigma)
  def _norm_cdf(x, mu, sigma)
  def _norm_ppf(p, mu, sigma)
  def _curve_polygon_verts(x_vals: np.ndarray, y_vals: np.ndarray, baseline_y: float=0.0)
  def _clipped_area_verts(x_vals: np.ndarray, y_vals: np.ndarray, x_lo: float, x_hi: float, baseline_y: float=0.0)
  def _hatch_lines(x_lo: float, x_hi: float, y_lo: float, y_hi: float, angle: float=PI / 4, spacing: float=0.12, stroke_color: str=WHITE, stroke_width: float=0.8, stroke_opacity: float=0.55, z: float=0.004)
  def _clip_line_to_box(x0: float, y0: float, x1: float, y1: float, xmin: float, xmax: float, ymin: float, ymax: float)
  def _world_to_px(val: float, x_center: float, unit: float)
  def _draw_cell(parent: VGroup, x: float, y: float, w: float, h: float, color: str, text: str, z: float)
  def _draw_label(parent: VGroup, x: float, y: float, text: str, fs: int, color: str, z: float)
  def make_power_analysis_grid(mu0: float=0.0, mu1_values: list[float]=None, n_values: list[int]=None, sigma: float=1.0, alpha: float=0.05, cell_scale: float=0.38)

## inference/hypothesis.py
  class _DistributionCurve(VGroup): __init__
  class _RejectionRegion(VGroup): __init__
  class _ObservedStatLine(VGroup): __init__
  class _PValueRegion(VGroup): __init__
  class _HypothesisLabel(VGroup): __init__
  class _InfoPanel(VGroup): __init__
  class _DecisionBadge(VGroup): __init__
  class HypothesisTest3D(VGroup): __init__, _get_x_range, _add_axis_ticks, z_test, one_sample_t, two_sample_t, chi_square, f_test, rebuild
  class BuildTest(Animation): __init__, interpolate_mobject
  class DropStatistic(Animation): __init__, interpolate_mobject
  class RevealPValue(Animation): __init__, interpolate_mobject
  class RevealDecision(Animation): __init__, interpolate_mobject
  class SweepStatistic(Animation): __init__, interpolate_mobject
  class ChangeAlpha(Animation): __init__, interpolate_mobject
  class BuildInfoPanel(Succession): __init__
  class HypothesisDemo(Scene): construct
  def _pdf(dist: str, x: np.ndarray, **kw)
  def _cdf(dist: str, x: float, **kw)
  def _ppf(dist: str, p: float, **kw)
  def _pdf(dist, x, **kw)
  def _cdf(dist, x, **kw)
  def _ppf(dist, p, **kw)
  def _clip_line(x0, y0, x1, y1, xmin, xmax, ymin, ymax)
  def _hatch(x_lo, x_hi, y_lo, y_hi, angle=PI / 4, spacing=0.1, color=WHITE, sw=0.9, opacity=0.5, z=0.004)
  def _area_verts(x_vals, y_vals, x_lo, x_hi, baseline=0.0)
  def compare_tests(ht1: HypothesisTest3D, ht2: HypothesisTest3D, spacing: float=0.5, scale: float=0.72)
  def make_full_sequence(ht: HypothesisTest3D)

## inference/sampling_dist.py
  class _PopulationPanel(VGroup): __init__
  class _SampleStrip(VGroup): __init__, raw_to_px, add_dot, make_mean_dot, clear_dots
  class _HistogramPanel(VGroup): __init__, _bar_left_px, _bar_height, _make_bar, update_bar, add_count, bin_top_position, set_theory_curve
  class _SEArrow(VGroup): __init__
  class _StatsPanel(VGroup): __init__
  class SamplingDistribution3D(VGroup): __init__, _hist_px, draw_new_sample, commit_mean, update_stats_panel, normal, uniform, exponential, chi_squared, bimodal, bernoulli
  class BuildPopulation(Animation): __init__, interpolate_mobject
  class DrawSample(Succession): __init__
  class ExtractMean(Animation): __init__, interpolate_mobject
  class AddToHistogram(Animation): __init__, interpolate_mobject, clean_up_from_scene
  class RunCLT(Succession): __init__
  class FlashBin(AnimationGroup): __init__
  class NarrowSE(Animation): __init__, interpolate_mobject
  class ConvergenceRace(Succession): __init__
  class SamplingDistDemo(Scene): construct
  def _norm_pdf(x, mu, sigma)
  def _norm_ppf(p, mu, sigma)
  def _sample_pop(pop_type, params, n, rng)
  def _norm_pdf(x, mu, sigma)
  def _norm_ppf(p, mu, sigma)
  def _sample_pop(pop_type, params, n, rng)
  def _pop_pdf(pop_type: str, params: dict, x: np.ndarray)
  def _pop_stats(pop_type: str, params: dict)
  def _pop_x_range(pop_type: str, params: dict)
  def _curve_verts(x_px, y_px, baseline=0.0)
  def _map_x(raw_x, x_lo, x_hi, plot_w)
  def make_clt_comparison(pop_specs: list[tuple[(str, dict)]], n: int=10, n_bins: int=16, k_samples: int=60, panel_scale: float=0.52, seed: int=42)
  def make_n_effect_row(pop_type: str='exponential', params: dict=None, n_values: list[int]=None, k_samples: int=80, panel_scale: float=0.55, seed: int=0)

## probability/__init__.py

## probability/bayes.py
  class BayesConfig: 
  class _ShadeBox(VGroup): __init__, top_center, floor_center
  class _PosteriorBracket(VGroup): __init__
  class BayesBox3D(VGroup): __init__, _build, animate_build_box, animate_highlight_cell, animate_reveal_posterior, animate_sweep_evidence
  class PriorPosteriorBar3D(VGroup): __init__, _build, animate_build, animate_update
  class LikelihoodPanel3D(VGroup): __init__, _build, animate_build, animate_highlight_lr
  class NaturalFrequencyTree3D(VGroup): __init__, _build, _add_count_label, _build_icon_array, animate_grow_tree
  class SequentialBayesUpdater3D(VGroup): __init__, _build, _prob_to_x, _make_marker, animate_axis, animate_add_evidence, animate_reset
  class BayesFormulaBanner(VGroup): __init__, animate_appear, animate_highlight_term, animate_highlight_all_terms
  def _dk(c: ManimColor, f: float)
  def _lt(c: ManimColor, f: float)
  def _clamp(v: float, lo: float=0.0, hi: float=1.0)
  def bayes_update(prior: float, lr: float)
  def compute_joint_probs(p_h: float, p_e_h: float, p_e_nh: float)
  def build_bayes_scene(p_h: float, p_e_h: float, p_e_nh: float, config: BayesConfig | None=None)

## probability/prob_tree.py
  class ProbTreeNode: __post_init__, is_leaf, iter_nodes, iter_paths, path_probability, depth, nodes_at_depth
  class ProbTreeConfig: 
  class _LayoutEngine: __init__, layout, _compute_widths, _assign_positions
  class _NodeGlyph(VGroup): __init__
  class _EdgeGlyph(VGroup): __init__
  class _PathLabel(VGroup): __init__
  class _ConditionalTable(VGroup): __init__
  class ProbabilityTree3D(VGroup): __init__, _build, _find_node, _paths_through, _ancestor_chain, animate_grow_level, animate_grow_tree, animate_trace_path, animate_highlight_event, animate_prune, animate_label_paths, animate_reveal_table, animate_morph_probs, animate_restore, from_dict, from_bayes, from_bernoulli, from_conditional_table
  def _dk(c: ManimColor, f: float)
  def _lt(c: ManimColor, f: float)
  def _node_color(p: float)

## probability/sample_space.py
  class SampleSpaceConfig: 
  class _SampleSpaceBox(VGroup): __init__
  class _EventRegion3D(VGroup): __init__, top_center, floor_center
  class _IntersectionRegion(_EventRegion3D): __init__
  class _ComplementRegion(VGroup): __init__
  class _ProbabilityAxis(VGroup): __init__, add_event_marker
  class _OutcomeGrid(VGroup): __init__, get_glyph
  class _VennZone(_EventRegion3D): __init__
  class SampleSpace3D(VGroup): __init__, add_rect_event, add_ellipse_event, add_intersection, add_complement, add_outcome_grid, add_probability_axis, animate_build_space, animate_add_event, animate_intersection, animate_complement, animate_show_union, animate_sweep_probability, animate_highlight_outcome, animate_venn_build, animate_conditional, animate_show_operation, two_event_venn, three_event_venn, from_dice, from_cards, conditional_highlight
  def _dk(c: ManimColor, f: float)
  def _lt(c: ManimColor, f: float)

## probability/venn3d.py
  class VennData2: __post_init__, _validate, p_a_only, p_b_only, p_aub, p_neither, zone_probs, is_independent, is_disjoint, p_b_given_a, p_a_given_b
  class VennData3: __post_init__, _validate, p_a_only, p_b_only, p_c_only, p_ab_only, p_ac_only, p_bc_only, p_abc_zone, p_aub_uc, p_neither, zone_probs
  class _CircleArcGeometry: __init__, _compute, arc_for_zone
  class _ZoneSurface(VGroup): __init__, floor_center
  class _VennCircleOutline(VGroup): __init__
  class _InclusionExclusionBracket(VGroup): __init__
  class _IndependenceAnnotation(VGroup): __init__
  class VennConfig: 
  class VennDiagram3D(VGroup): __init__, _build_omega_box, _add_zone, _build_2set, _build_3set, animate_grow_circles, animate_fill_zones, animate_highlight_zone, animate_inclusion_exclusion, animate_condition_on, animate_restore, animate_independence_test, animate_morph_to, animate_add_set_c, two_set, three_set, from_counts, disjoint, subset, independent
  def _dk(c: ManimColor, f: float)
  def _lt(c: ManimColor, f: float)
  def _clamp(v: float, lo: float=0.0, hi: float=1.0)
  def _circle_centres_2(cfg: VennConfig)
  def _circle_centres_3(cfg: VennConfig)
  def _zone_arcs_2(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float, geo: _CircleArcGeometry)

## props/__init__.py

## props/card.py
  class CardSuit(Enum): symbol, is_red, latex_symbol, color_name, manim_color
  class CardValue(Enum): rank, pip_count, is_face_card, is_ace, rank_str, full_name, blackjack_value, poker_rank
  class CardFacing(Enum): 
  class CardFace: symbol, rank_str, full_name, is_red, pip_count, is_face_card, latex_label, sort_key, __lt__, __repr__, __str__
  class CardGeometry: aspect_ratio, half_w, half_h, pip_font_size, corner_label_font_size, corner_suit_font_size, face_card_monogram_font_size, pip_world_positions, pip_rotations, corner_label_offset, back_diamond_size
  class CardColorScheme: 
  class Card3D(VGroup if _MANIM_AVAILABLE else object): __init__, _build_body, _build_back_face, _build_front_face, _build_pips, _build_corner_labels, _build_face_card_monogram, _build_highlight_ring, facing, face_up, face_down, suit, value, is_red, rank_str, full_name, set_facing, _show_front, _show_back, _show_hidden, highlight, unhighlight, flip_to_face_up, flip_to_face_down, flip, reveal_anim, deal_anim, discard_anim, hover_anim, unhover_anim, clone, __repr__
  class Deck3D(VGroup if _MANIM_AVAILABLE else object): __init__, remaining, is_empty, top_card, is_shuffled, deal_one, deal_n, reset, fan_out, spread_face_up, collect_anim, shuffle, shuffle_anim, _riffle_anim, _overhand_anim, reveal_top, cut_anim, __len__, __iter__, __getitem__, __repr__
  def _require_manim(method_name: str)
  def _make_standard_deck(shuffle: bool=False, seed: int=None)
  def standard_deck(shuffle: bool=False, seed: int=None)
  def suit_subset(suit: CardSuit)
  def value_subset(value: CardValue)
  def face_cards_only()
  def number_cards_only()
  def prob_card_event(card: 'Card3D', event_desc: str, probability: float, position: np.ndarray=None, color: str='#FFD700', font_size: int=20)
  def _float_to_fraction_str(p: float, max_denom: int=52)
  def sample_without_replacement(deck: 'Deck3D', n: int, targets: List[np.ndarray], run_time: float=0.4, stagger: float=0.08, flip: bool=True)
  def hypergeometric_demo(deck: 'Deck3D', event_suit: CardSuit, sample_size: int, targets: List[np.ndarray], run_time: float=0.4)
  def birthday_problem_deck(n_people: int=23)
  def conditional_prob_demo(deck: 'Deck3D', condition: CardSuit, event: CardValue)

## props/coin.py
  class _CoinEdge(VGroup): __init__
  class _CoinFace(VGroup): __init__, _build_crown, _build_shield
  class Coin3D(VGroup): __init__, outcome, outcome, radius, thickness, _apply_outcome, show_heads_up, show_tails_up, set_outcome, get_face, copy_with_outcome
  class FlipCoin(Animation): __init__, interpolate_mobject
  class TumbleCoin(Animation): __init__, interpolate_mobject
  class LandCoin(Succession): __init__
  class SpinCoin(Animation): __init__, interpolate_mobject
  class RollCoin(Animation): __init__, interpolate_mobject
  class CoinDemo(ThreeDScene): construct
  def _arc_text(text: str, radius: float, start_angle: float, span_angle: float, font_size: float=14, color: ManimColor=WHITE, font: str='serif')
  def _star_polygon(n_points: int=5, outer_r: float=0.1, inner_r: float=0.04, color: ManimColor=WHITE, fill_opacity: float=1.0)
  def make_coin_row(n: int, outcomes: list[Literal[('heads', 'tails')]], spacing: float=None, **coin_kwargs=0.3)
  def make_coin_grid(rows: int, cols: int, outcomes: list[list[Literal[('heads', 'tails')]]], h_spacing: float=None, v_spacing: float=0.25, **coin_kwargs=0.25)

## props/die.py
  class _PipWell(VGroup): __init__
  class _D6Face(VGroup): __init__
  class _RoundedCubeBody(VGroup): __init__
  class _PolyFace(VGroup): __init__
  class Die3D(VGroup): __init__, _build_d6, _build_poly, _build_poly_convex_hull, _orient_to_outcome, outcome, outcome, die_type, n_faces, get_face_object
  class RollDie(Animation): __init__, interpolate_mobject
  class ThrowDie(Animation): __init__, interpolate_mobject
  class SpinDie(Animation): __init__, interpolate_mobject
  class BounceDie(Succession): __init__
  class ShakeDie(Animation): __init__, interpolate_mobject
  class DieDemo(ThreeDScene): construct
  def _regular_polygon_verts(n: int, r: float=1.0)
  def _d4_data(edge: float=1.0)
  def _d8_data(edge: float=1.0)
  def _d12_data(edge: float=1.0)
  def _d20_data(edge: float=1.0)
  def _angle_between(a: np.ndarray, b: np.ndarray)
  def _safe_cross(a: np.ndarray, b: np.ndarray)
  def make_die_set(die_types: list[str], spacing: float=None, color_schemes: list[str]=0.4, **shared_kwargs=None)
  def make_outcome_distribution(die_type: str='D6', outcomes: list[int]=None, spacing: float=0.3, color_scheme: str='ivory', size: float=0.6)

## props/spinner.py
  class _SpinnerBoard(VGroup): __init__
  class _SpinnerSectors(VGroup): __init__, _make_label_str
  class _SpinnerHub(VGroup): __init__
  class _SpinnerNeedle(VGroup): __init__
  class Spinner3D(VGroup): __init__, weights, n_sectors, needle_angle, needle_angle, sector_mid_angles, angle_to_sector, sector_target_angle, random_spin_outcome, uniform, from_probs, named
  class SpinToOutcome(Animation): __init__, interpolate_mobject
  class FreeSpinDecay(Animation): __init__, interpolate_mobject
  class FlickSpin(Animation): __init__, interpolate_mobject
  class RicochetSpin(Succession): __init__
  class SpinSequence(Succession): __init__
  class SpinnerDemo(ThreeDScene): construct
  def _wedge_polygon(r_inner: float, r_outer: float, start_angle: float, end_angle: float, n_arc_pts: int=32, z: float=0.0)
  def _pie_polygon(radius: float, start_angle: float, end_angle: float, n_arc_pts: int=48, z: float=0.0)
  def _sector_label_position(r_mid: float, mid_angle: float)
  def _arc_text_centered(text: str, radius: float, mid_angle: float, font_size: float, color: str, font: str='sans-serif')
  def _perceived_luminance(hex_color: str)
  def _exponential_decay_angle(alpha: float, start_angle: float, total_extra_angle: float, decay_k: float=4.0)
  def _wobble_offset(alpha: float, amplitude: float, n_wobbles: int=3)
  def make_bernoulli_spinner(p: float, labels: tuple[(str, str)]=0.5, colors: tuple[(str, str)]=('Success', 'Failure'), **spinner_kwargs=('#2A9D8F', '#E63946'))
  def make_die_spinner(n: int, **spinner_kwargs=6)
  def make_markov_spinner(state_names: Sequence[str], transition_row: Sequence[float], current_state: int, **spinner_kwargs=0)

## props/urn.py
  class Ball3D(VGroup): __init__, get_color_hex, make_copy_at
  class _UrnLatheBody(VGroup): __init__
  class _MeanderBand(VGroup): __init__
  class _UrnHandle(VGroup): __init__
  class _UrnLid(VGroup): __init__
  class Urn3D(VGroup): __init__, belly_r, lip_y, inner_bottom_y, n_balls, add_ball, remove_ball, get_ball_positions, get_draw_exit_point, count_by_color
  class FillUrn(Succession): __init__
  class DrawBall(Animation): __init__, begin, interpolate_mobject
  class ReplaceBall(Animation): __init__, begin, interpolate_mobject
  class ShakeUrn(Animation): __init__, interpolate_mobject
  class PourUrn(Succession): __init__
  class SwapBalls(Animation): __init__, interpolate_mobject
  class UrnDemo(ThreeDScene): construct
  def _rotation_matrix_y(theta: float)
  def _lathe_profile_points(belly_r: float, neck_r: float, total_h: float, n_segments: int=40)
  def _meander_key_unit(w: float, h: float)
  def get_packed_positions(n_balls: int, urn_belly_r: float, urn_inner_bottom_y: float, ball_radius: float, max_layers: int=6)
  def make_labeled_urn(n_red: int, n_blue: int, urn_kwargs: Optional[dict]=None, ball_kwargs: Optional[dict]=None)
  def make_two_urn_setup(urn1_contents: list[tuple[(str, int)]], urn2_contents: list[tuple[(str, int)]], spacing: float=4.5, scheme1: str='terracotta', scheme2: str='cobalt', ball_radius: float=0.18)

## regression/__init__.py

## regression/correlation.py
  class CorrelationMethod(Enum): 
  class CorrelationResult: abs_r, stars, significant, effect_size_label, ci_str, summary, to_formula, __repr__
  class PartialCorrelationResult(CorrelationResult): unique_variance, summary
  class RegressionResult: rss, tss, mse, rmse, sigma_hat, hat_matrix_diag, coef_table, to_formula, __repr__
  class InfluenceMeasures: high_leverage_mask, outlier_mask, influential_mask, concerning_obs, __repr__
  class CorrelationEllipse3D(VGroup if _MANIM_AVAILABLE else object): __init__, _ellipse_points, _build, _add_principal_axes, animate_grow, morph_to
  class CorrelationMatrix3D(VGroup if _MANIM_AVAILABLE else object): __init__, _r_to_color, _build, animate_build, highlight_cell, morph_data
  class RegressionLine3D(VGroup if _MANIM_AVAILABLE else object): __init__, _build_ci_band, _build_equation, animate_draw, reveal_ci_band, morph_to_fit
  class ResidualArrows3D(VGroup if _MANIM_AVAILABLE else object): __init__, _build, animate_appear, flash_outliers, update_fit
  def _require_manim(name: str)
  def _require_scipy(name: str)
  def pearson(x: np.ndarray, y: np.ndarray, compute_pvalue: bool=True, alpha: float=0.05)
  def spearman(x: np.ndarray, y: np.ndarray, compute_pvalue: bool=True)
  def kendall(x: np.ndarray, y: np.ndarray, compute_pvalue: bool=True)
  def _kendall_tau_b_numpy(x: np.ndarray, y: np.ndarray)
  def point_biserial(continuous: np.ndarray, binary: np.ndarray, compute_pvalue: bool=True)
  def phi_coefficient(table: np.ndarray, compute_pvalue: bool=True)
  def cramers_v(table: np.ndarray, compute_pvalue: bool=True, bias_corrected: bool=True)
  def partial_correlation(x: np.ndarray, y: np.ndarray, controls: np.ndarray, x_name: str='X', y_name: str='Y', control_names: Optional[List[str]]=None, compute_pvalue: bool=True, semi_partial: bool=False)
  def correlation_matrix(data: np.ndarray, method: CorrelationMethod=CorrelationMethod.PEARSON, compute_pvalue: bool=True, var_names: Optional[List[str]]=None)
  def ols_fit(x: np.ndarray, y: np.ndarray, fit_intercept: bool=True, feature_names: Optional[List[str]]=None, compute_pvalue: bool=True)
  def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float=1.0, fit_intercept: bool=True, feature_names: Optional[List[str]]=None)
  def influence_measures(result: RegressionResult)
  def build_scatter_to_line(axes: 'mn.Axes', result: RegressionResult, dot_color=None, run_time: float=3.0)
  def morph_r_value(ellipse: CorrelationEllipse3D, r_values: Sequence[float], run_time_each: float=1.2)
  def reveal_residuals(scene, result: RegressionResult, reg_line: RegressionLine3D, run_time: float=2.0)
  def animate_influence_removal(scene, result: RegressionResult, obs_index: int, axes: 'mn.Axes', dot_mob, run_time: float=2.0)
  def _build_pearson_formula()
  def _build_spearman_formula()
  def _build_ols_derivation_formula()
  def _build_partial_r_formula()
  def _build_cooks_d_formula()
  def _clean_paired(x: np.ndarray, y: np.ndarray)
  def _rankdata(x: np.ndarray)

## regression/regression_plane.py
  class SurfaceColouring(Enum): 
  class PlaneGeometry: axes_kwargs
  class RegressionPlane3D(VGroup if _MANIM_AVAILABLE else object): __init__, _build_surface, _build_equation_label, beta, r_squared, tilt_degrees, animate_grow, animate_tilt, flash_plane, update_coefficients, __repr__
  class ScatterCloud3D(VGroup if _MANIM_AVAILABLE else object): __init__, _data_to_world, _compute_colors, animate_appear, highlight_obs, dim_all_except, undim_all
  class PlaneResiduals3D(VGroup if _MANIM_AVAILABLE else object): __init__, _build, animate_appear, flash_outliers, fade_to_zero, update_fit
  class CIShell3D(VGroup if _MANIM_AVAILABLE else object): __init__, _make_surface, animate_appear, morph_level
  class PIShell3D(CIShell3D if _MANIM_AVAILABLE else object): __init__
  class ProjectionArrow3D(VGroup if _MANIM_AVAILABLE else object): __init__, animate_draw, flash
  def _require_manim(name: str)
  def _require_corr(name: str)
  def plane_predict(beta: np.ndarray, x1: Union[(float, np.ndarray)], x2: Union[(float, np.ndarray)])
  def plane_grid(beta: np.ndarray, x1_range: Tuple[(float, float)], x2_range: Tuple[(float, float)], resolution: int=30, padding: float=0.15)
  def ci_surface(beta: np.ndarray, XtX_inv: np.ndarray, s: float, n: int, x1_range: Tuple[(float, float)], x2_range: Tuple[(float, float)], resolution: int=25, level: float=0.95, padding: float=0.15)
  def pi_surface(beta: np.ndarray, XtX_inv: np.ndarray, s: float, n: int, x1_range: Tuple[(float, float)], x2_range: Tuple[(float, float)], resolution: int=25, level: float=0.95, padding: float=0.15)
  def plane_normal(beta: np.ndarray)
  def tilt_angle_deg(beta: np.ndarray)
  def leverage_grid(XtX_inv: np.ndarray, x1_range: Tuple[(float, float)], x2_range: Tuple[(float, float)], resolution: int=30, padding: float=0.15)
  def standardise_grid(Z: np.ndarray)
  def build_plane_scene(result: 'RegressionResult', x1: np.ndarray, x2: np.ndarray, y: np.ndarray, x1_name: str='x_1', x2_name: str='x_2', y_name: str='y', geometry: PlaneGeometry=DEFAULT_GEOMETRY, colouring: Union[(SurfaceColouring, str)]=SurfaceColouring.FITTED, show_ci: bool=True, show_residuals: bool=True, cloud_colouring: str='residual')
  def morph_beta(plane: RegressionPlane3D, scene, beta_sequence: Sequence[np.ndarray], run_time_each: float=1.2, wait_each: float=0.5)
  def sweep_x1(plane: RegressionPlane3D, scene, x1_vals: Sequence[float], x2_fixed: float, marker: Optional['ProjectionArrow3D']=None, run_time: float=2.5)
  def rotate_view(scene, phi: float=70 * (3.14159 / 180), theta: float=-45 * (3.14159 / 180), run_time: float=2.0)
  def _build_multiple_regression_formula()
  def _build_plane_ols_formula()
  def _build_plane_ci_formula()
  def _build_plane_pi_formula()
  def _build_rsq_decomp_formula()

## regression/residuals.py
  class NormalityResult: passes, summary_line, verdict
  class HomoscedasticityResult: passes, fan_direction, summary_line, verdict
  class AutocorrResult: dw_conclusion, significant_lags, passes, summary_line, verdict
  class LinearityResult: passes, summary_line, verdict
  class ResidualDiagnostics: all_pass, failed_assumptions, summary_table, __repr__
  class ResidualVsFittedPlot(VGroup if _MANIM_AVAILABLE else object): __init__, _build, animate_appear, highlight_band, morph_to
  class QQPlot3D(VGroup if _MANIM_AVAILABLE else object): __init__, _build, animate_build
  class ScaleLocationPlot(VGroup if _MANIM_AVAILABLE else object): __init__, animate_appear
  class InfluencePlot3D(VGroup if _MANIM_AVAILABLE else object): __init__, _build, animate_appear, highlight_influential
  class ACFPlot3D(VGroup if _MANIM_AVAILABLE else object): __init__, animate_appear, flash_significant
  class DiagnosticPanel(VGroup if _MANIM_AVAILABLE else object): __init__, animate_build_sequence, highlight_assumption, summary_text_panel
  def _require_manim(name: str)
  def _require_scipy(name: str)
  def _require_corr(name: str)
  def ordinary_residuals(result: 'RegressionResult')
  def standardised_residuals(result: 'RegressionResult')
  def externally_studentized_residuals(result: 'RegressionResult')
  def press_residuals(result: 'RegressionResult')
  def press_statistic(result: 'RegressionResult')
  def jackknife_betas(result: 'RegressionResult')
  def acf_values(e: np.ndarray, max_lag: int=20)
  def pacf_values(e: np.ndarray, max_lag: int=20)
  def lowess_smooth(x: np.ndarray, y: np.ndarray, frac: float=0.4, n_eval: int=60)
  def qq_coordinates(residuals: np.ndarray, distribution: str='norm')
  def _hat_diag(result: 'RegressionResult')
  def test_normality(result: 'RegressionResult', use_studentized: bool=True)
  def test_homoscedasticity(result: 'RegressionResult')
  def test_autocorrelation(result: 'RegressionResult', max_lag: int=10)
  def test_linearity(result: 'RegressionResult', reset_order: int=2)
  def diagnose(result: 'RegressionResult', max_lag: int=10, lowess_frac: float=0.4, compute_influence: bool=True)
  def build_diagnostic_scene(result: 'RegressionResult', scene, layout: str='panel', run_time: float=8.0, alpha: float=0.05)
  def animate_assumption_violation(scene, result_good: 'RegressionResult', result_bad: 'RegressionResult', assumption: str, run_time: float=4.0)
  def animate_fix_heteroscedasticity(scene, result: 'RegressionResult', log_y: bool=True, run_time: float=5.0)
  def _build_gauss_markov_formula()
  def _build_durbin_watson_formula()
  def _build_breusch_pagan_formula()
  def _build_press_formula()
  def _build_cooks_d_full_formula()

## scenes/__init__.py

## scenes/demo_bayes.py
  class _UrnSchematic(VGroup): __init__, _make_prob_badge, update_prior_badge
  class _ProbBar(VGroup): __init__
  class _LikelihoodTable(VGroup): __init__, _add_cell
  class _BayesFormula(VGroup): __init__
  class _BetaCurve(VGroup): __init__
  class _PosteriorTrail(VGroup): __init__
  class _SummaryPanel(VGroup): __init__
  class _CountUp(Animation): __init__, interpolate_mobject
  class BayesDemo(Scene): construct, _act0_cold_open, _act1_setup, _act2_prior, _act3_draw, _act4_likelihood, _act5_bayes_formula, _act6_posterior_bar, _act7_sequential, _act8_beta_curve, _act9_decision
  def _title_card(main: str, subtitle: str='', main_fs: int=44, sub_fs: int=22)
  def _section_label(text: str, color: str=None)
  def _morph_prob_bar(scene: Scene, old_bar: _ProbBar, new_p_a: float, run_time: float, **kwargs=1.2)
  def _drawn_ball_display(color: str, label: str='RED', radius: float=0.45)
  def _perceived_lum(hex_col: str)

## scenes/demo_clt.py
  class _MiniPopCurve(VGroup): __init__, _verts, _pdf, _stats, _x_range
  class _MiniHistogram(VGroup): __init__
  class _QQPlot(VGroup): __init__
  class _LiveCounter(VGroup): __init__
  class _CLTStatement(VGroup): __init__
  class _SEPanel(VGroup): __init__
  class CLTDemo(Scene): construct, _act0_title, _act1_population_zoo, _act2_what_is_sample_mean, _act3_build_histogram_slow, _act4_convergence_speedup, _act5_n_effect, _act6_population_independence, _act7_qq_plot, _act8_clt_statement
  def _title_card(main: str, sub: str='', main_fs: int=44, sub_fs: int=22)
  def _section_label(text: str)
  def _badge(text: str, color: str=None)
  def _norm_pdf_scene(x: np.ndarray, mu: float, sigma: float)

## scenes/demo_distributions.py
  class _StatsPanel(VGroup): __init__
  class _AppTag(VGroup): __init__
  class _PMFPanel(VGroup): __init__
  class _PDFPanel(VGroup): __init__
  class _DistTree(VGroup): __init__
  class _TailComparison(VGroup): __init__
  class _RelArrow(VGroup): __init__
  class _ComparisonOverlay(VGroup): __init__
  class DistributionsDemo(Scene): construct, _setup_bg, _section, _fade_scene, _act0_family_tree, _act1_discrete_gallery, _act2_bell_curves, _act3_skewed_bounded, _act4_heavy_tails, _act5_relationships, _act6_param_sweep, _act7_comparison_closing
  def _pdf(dist: str, x: np.ndarray, **kw)
  def _cdf(dist: str, x: np.ndarray, **kw)
  def _pmf(dist: str, k: np.ndarray, **kw)
  def _curve_poly(x_px, y_px, baseline=0.0)
  def _spine(x_px, y_px, color, sw=1.8, z=0.005)
  def _axis_line(x_lo, x_hi, y, color=None, sw=1.3)
  def _ticks(vals, y, lo_raw, hi_raw, px_w, fmt='{:.1f}', fs=10, y_off=-0.22)

## scenes/demo_hypothesis.py
  class _ScalesIcon(VGroup): __init__
  class _HypothesisSetupPanel(VGroup): __init__
  class _CalcPanel(VGroup): __init__, _make_line
  class _PValueGauge(VGroup): __init__
  class _DataDotStrip(VGroup): __init__
  class _ChiSquareBarChart(VGroup): __init__
  class _PowerCurve(VGroup): __init__
  class _DecisionFlowchart(VGroup): __init__
  class HypothesisDemo(Scene): construct, _act0_courtroom, _act1_z_test, _act2_t_test, _act3_two_sample_t, _act4_chi_square, _act5_power, _act6_framework
  def _section_label(text: str)
  def _fade(*mobs)
  def _badge(text: str, color: str, fs: int=17)

## ui/__init__.py

## ui/labels.py
  class LabelConfig: 
  class _CardPrism(VGroup): __init__, top_center
  class StatLabel3D(VGroup): __init__, animate_appear, animate_update_value, animate_pulse, mean_label, median_label, std_label, pval_label, corr_label, sample_size_label
  class AnnotationArrow3D(VGroup): __init__, animate_draw, animate_pulse, animate_redirect
  class FormulaPanel3D(VGroup): __init__, _rebuild_card, add_line, animate_write_lines, animate_collapse, animate_expand
  class LegendPanel3D(VGroup): __init__, animate_appear, from_series
  class DataCallout3D(VGroup): __init__, animate_appear
  class Ticker3D(VGroup): __init__, _layout, animate_count, animate_flash_on_change
  class AxisLabel3D(VGroup): __init__, animate_appear
  class TooltipBadge3D(VGroup): __init__, animate_show, animate_hide
  class StatSummaryBox3D(VGroup): __init__, animate_reveal
  class HighlightRing3D(VGroup): __init__, _make_ring, animate_pulse, animate_orbit, animate_appear, set_target
  def _dk(c: ManimColor, f: float)
  def _lt(c: ManimColor, f: float)
  def _auto_contrast_color(bg: ManimColor)
  def _significance_stars(p: float)
  def _place_clear_of(desired: np.ndarray, existing: list[np.ndarray], min_dist: float=0.4, max_tries: int=8)
  def _format_value(v: float, decimals: int=3, fmt: str='f', sci_thresh: float=0.001)
  def _stat_color(stat_type: str)

## ui/panels.py
  class PanelConfig: 
  class _PanelBackground(VGroup): __init__, _make_prism
  class _SectionDivider(VGroup): __init__
  class _StepBadge(VGroup): __init__, set_state
  class _ValueCell(VGroup): __init__, top_center, floor_center
  class _DecisionBanner(VGroup): __init__
  class InfoPanel3D(VGroup): __init__, add_section, _rebuild_background, _rebuild_footer, animate_build, animate_update_section, docked
  class StepPanel3D(VGroup): __init__, animate_activate_step, animate_complete_step, animate_walk_steps, animate_reveal_all
  class ComparisonPanel3D(VGroup): __init__, animate_build_columns, animate_highlight_winner, animate_reveal_all
  class DistributionInfoPanel3D(VGroup): __init__, animate_reveal
  class HypothesisPanel3D(VGroup): __init__, animate_build, animate_reveal_decision
  class FormulaDerivationPanel3D(VGroup): __init__, animate_derive, animate_full_derivation, animate_reveal_all
  class MatrixPanel3D(VGroup): __init__, animate_reveal_by_row, animate_reveal_by_col, animate_highlight_cell
  def _dk(c: ManimColor, f: float)
  def _lt(c: ManimColor, f: float)
  def _palette_t(t: float, cold: ManimColor, hot: ManimColor)
  def _format_v(v: float, d: int=3)
  def _sig_stars(p: float)

## ui/table3d.py
  class TableConfig: 
  class TableColumn: effective_align
  class _TableCell(VGroup): __init__, top_center, floor_center
  class _HeaderCell(VGroup): __init__
  class _SparklineCell(VGroup): __init__
  class _BadgeCell(VGroup): __init__
  class TableSorter: __init__, sort_by
  class TableFilter: __init__, visible_mask, filter_by, clear, pop_last
  class TableSelection: __init__, select_row, deselect_row, select_col, select_cell, clear, is_selected
  class DataTable3D(VGroup): __init__, _compute_layout, _build, _get_page_rows, _build_grid_lines, _build_summary_row, _compute_aggregate, animate_build, animate_reveal_header, animate_sort, animate_filter, animate_clear_filter, animate_select_row, animate_select_col, animate_highlight_cell, animate_update_cell, animate_next_page, animate_prev_page, _rebuild_body, animate_reveal_summary, from_dataframe, from_numpy, stats_summary, comparison_results, frequency_table
  def _dk(c: ManimColor, f: float)
  def _lt(c: ManimColor, f: float)
  def _contrast(bg: ManimColor)
  def _fmt(v: float, decimals: int=2, col_type: str='numeric')

## ui/ticker.py
  class TickerFormat(Enum): 
  class ThresholdMap: __init__, tier, is_significant
  class TickerStyle: label_font_size, unit_font_size, delta_font_size
  class TickerColors: resolve
  class Ticker3D(VGroup if _MANIM_AVAILABLE else object): __init__, _build, current_value, label_text, set_value, set_label, set_unit, highlight, unhighlight, show_delta_indicator, _update_threshold_color, _refresh_delta, count_to, odometer_to, flash_change, pulse, write_in, fade_out, update_from_stat, clone, __repr__
  class PValueTicker3D(Ticker3D): __init__, set_value, flash_significant, set_alpha_level, count_to
  class StatsCounter3D(Ticker3D): __init__, _clamp, increment, decrement, set_int
  class CorrelationTicker3D(Ticker3D): __init__, _recolour_for_r, _build_ci_mob, set_value, morph_r, show_ci_annotation, hide_ci_annotation
  class TickerGroup3D(VGroup if _MANIM_AVAILABLE else object): __init__, _arrange, _by_name, register_name, update_all, update_by_name, flash_significant, highlight_stat, write_in_sequence, align_labels, from_regression_result, from_correlation_result, __repr__
  def _require_manim(name: str)
  def auto_precision(value: float, n_sig: int=4)
  def format_value(value: float, fmt: TickerFormat=TickerFormat.AUTO, precision: int=4)
  def significance_stars(p_value: float)
  def delta_string(old: float, new: float, precision: int=4)
  def _ease_linear(t: float)
  def _ease_out_cubic(t: float)
  def _ease_in_out_cubic(t: float)
  def _ease_out_elastic(t: float)
  def _ease_out_back(t: float)
  def precision_for_format(fmt: TickerFormat, precision: int, value: float)
  def regression_dashboard(result, position=None, style: TickerStyle=None, layout: str='grid', n_cols: int=4, spacing: float=0.3)
  def correlation_dashboard(corr_result, position=None, style: TickerStyle=None, spacing: float=0.28)
  def hypothesis_dashboard(test_stat: float, p_value: float, df: int, alpha: float=0.05, stat_label: str='t =', style: TickerStyle=None, spacing: float=0.28)
  def distribution_dashboard(mean: float, std: float, skewness: float=0.0, kurtosis: float=0.0, n: int=0, style: TickerStyle=None, spacing: float=0.28)