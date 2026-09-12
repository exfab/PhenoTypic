"""Heavy imports stay deferred to the functions that use them, site by site.

``test_startup_imports.py`` guards what each entry point loads. This module guards every
deferral site on its own, so a module-level import cannot creep back into a module no
entry-point guard reaches, and a moved import cannot go missing from a function no other
test calls.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "src" / "phenotypic"

#: Module (relative to ``src/phenotypic``) -> name bound by a deferred import -> the
#: functions that must import it locally. An empty tuple means the name may only appear
#: under ``TYPE_CHECKING`` (or nowhere) at module level.
DEFERRED_SITES: dict[str, dict[str, tuple[str, ...]]] = {
    "_core/_image_parts/_grid_image_handler.py": {
        "CenteredAutoGridFinder": ("__init__",),
        "MeasureBounds": ("_draw_section_boxes_on_overlay",),
        "plt": ("_draw_section_boxes_on_overlay",),
    },
    "_core/_image_parts/_image_io_handler.py": {"h5py": ("_load_hdf5_for_migration",)},
    "_core/_image_parts/accessor_abstracts/_image_accessor_base_parents/_accessor_dash_handler.py": {
        "px": ("_plotly_imshow",),
        "go": (),
        "_pio": (),
    },
    "_core/_image_parts/accessor_abstracts/_image_accessor_base_parents/_accessor_mpl_handler.py": {
        "plt": ("_add_section_boxes", "_mpl_plot", "histogram"),
        "Rectangle": ("_add_section_boxes",),
    },
    "_core/_image_parts/accessors/_grid_accessor.py": {
        "plt": ("_build_section_box_shapes", "show_column_overlay", "show_row_overlay"),
    },
    "_core/_image_parts/accessors/_objmap_accessor.py": {"plt": ("show",)},
    "_core/_image_parts/accessors/_objmask_accessor.py": {"plt": ("show",)},
    "_core/_image_parts/color_space_accessors/_chromaticity_xy_accessor.py": {"colour": ("_subject_arr",)},
    "_core/_image_parts/color_space_accessors/_cielab_accessor.py": {"colour": ("_subject_arr",)},
    "_core/_image_parts/color_space_accessors/_hsv_accessor.py": {"plt": ("histogram", "show", "show_objects")},
    "_core/_image_parts/color_space_accessors/_xyz_conversion.py": {
        "colour": ("rgb_to_xyz",),
        "sRGB_D50": ("rgb_to_xyz",),
    },
    "_core/_image_parts/color_space_accessors/_xyz_d65_accessor.py": {"colour": ("_subject_arr",)},
    "_core/_image_parts/plot_accessor/_base_plotter.py": {
        "plt": ("_cleanup_figure", "_create_colormap", "_validate_cmap"),
    },
    "_core/_image_parts/plot_accessor/_detect_modes_plotter.py": {
        "make_subplots": ("detect_modes",),
        "go": (),
    },
    "_core/_image_parts/plot_accessor/_diagnostics_plotter.py": {
        "plt": (
            "_diagnostics_matplotlib",
            "_plot_background_estimate",
            "_plot_gradient_magnitude",
            "_plot_local_contrast_map",
            "_plot_local_variance_map",
            "_plot_noise_autocorrelation",
            "_plot_orientation_coherence",
        ),
        "go": (
            "_empty_plotly_figure",
            "fig_background_estimate",
            "fig_contrast_metrics",
            "fig_detection_matrix",
            "fig_gradient_magnitude",
            "fig_intensity_histogram",
            "fig_local_contrast_map",
            "fig_local_variance",
            "fig_noise_autocorrelation",
            "fig_orientation_coherence",
            "fig_power_spectral_density",
            "fig_quality_summary",
            "fig_ridge_response",
        ),
        "NAVY": (
            "_empty_plotly_figure",
            "fig_intensity_histogram",
            "fig_power_spectral_density",
            "fig_quality_summary",
            "fig_ridge_response",
        ),
        "OKABE_ITO": ("fig_intensity_histogram", "fig_power_spectral_density", "fig_ridge_response"),
    },
    "analysis/abc_/_model_fitter.py": {"matplotlib": ("_build_plotly_figure", "show"), "plt": ("show",)},
    "analysis/edge/_edge_correction.py": {"plt": ("_show_collapsed", "_show_individual"), "Figure": ()},
    "analysis/filter/_mad_outlier.py": {"plt": ("_show_collapsed", "_show_individual")},
    "analysis/filter/_tukey_outlier.py": {"plt": ("_show_collapsed", "_show_individual")},
    "analysis/qc/_expected_vs_detected.py": {"go": ("inspect",)},
    "analysis/qc/_grid_occupancy.py": {"go": ("inspect",)},
    "analysis/qc/_replicate_agreement.py": {"go": ("inspect",)},
    "correction/_color_correction/_color_corrector.py": {"colour": ("_operate",)},
    "correction/_color_denoise.py": {
        "bm3d": ("_build_profile", "_denoise_gat", "_denoise_plain"),
        "decode_srgb": ("_operate",),
        "encode_srgb": ("_operate",),
    },
    "correction/_denoise_block_match.py": {
        "bm3d": ("_denoise_channel",),
        "BM3DStages": ("_convert_stage_arg",),
        "decode_srgb": ("_denoise_channel",),
        "encode_srgb": ("_denoise_channel",),
    },
    "enhance/_enhance_block_match.py": {"bm3d": ("_denoise_detect_mat",), "BM3DStages": ("_convert_stage_arg",)},
    "enhance/_flatten_illumination.py": {"cv2": ("_filter",)},
    "enhance/_subtract_opening.py": {"cv2": ("_operate",)},
    "refine/_extract_colony_core.py": {"cv2": ("_build_ellipse_kernel", "_extract_single_core")},
    "measure/_measure_texture.py": {"mahotas": ("_mahotas",), "mh": ()},
    "detect/_filamentous_fungi_detector.py": {
        "ReconnectConfig": ("_reconnect_config",),
        "build_reconnect_cost": ("_operate",),
        "compute_full_image_app2_gi_cost": ("_operate",),
        "filter_mask_by_overlap": ("_operate",),
        "markers_from_centroids": ("_operate",),
        "partition_by_grid_voronoi": ("_operate",),
        "reconnect_fragments_tiled": ("_operate",),
        "select_reconnect_fragments": ("_operate",),
    },
    "detect/_two_k_filamentous_detector.py": {
        "ReconnectConfig": ("_reconnect_config",),
        "build_reconnect_cost": ("_operate",),
        "filter_mask_by_overlap": ("_operate",),
        "markers_from_centroids": ("_operate",),
        "partition_by_grid_voronoi": ("_operate",),
        "reconnect_fragments_tiled": ("_operate",),
        "select_reconnect_fragments": ("_operate",),
    },
    "sdk_/orientation_fields/_plots.py": {
        "Circle": ("plot_literal_crossing_map",),
        "Normalize": (
            "plot_literal_crossing_map",
            "plot_literal_crossing_outward_profile",
            "plot_literal_crossing_population",
        ),
        "Axes": (),
        "Colormap": (),
        "PathCollection": (),
        "Quiver": (),
    },
    "util/_robust_color_stats.py": {"colour": ("lab_to_srgb_hex", "medoid_ciede2000")},
}


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _module_level_bindings(body: list[ast.stmt]) -> dict[str, int]:
    """Names bound by runtime module-level imports (``TYPE_CHECKING`` blocks excluded) -> line."""
    bound: dict[str, int] = {}
    for node in body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound[alias.asname or alias.name.split(".")[0]] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound[alias.asname or alias.name] = node.lineno
        elif isinstance(node, ast.If):
            if not _is_type_checking(node.test):
                bound.update(_module_level_bindings(node.body))
            bound.update(_module_level_bindings(node.orelse))
        elif isinstance(node, ast.Try):
            bound.update(_module_level_bindings(node.body))
            for handler in node.handlers:
                bound.update(_module_level_bindings(handler.body))
            bound.update(_module_level_bindings(node.orelse))
            bound.update(_module_level_bindings(node.finalbody))
    return bound


def _locally_imported_names(function: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def _parse(relative_path: str) -> ast.Module:
    return ast.parse((PACKAGE_ROOT / relative_path).read_text(encoding="utf-8"))


@pytest.mark.parametrize("relative_path", sorted(DEFERRED_SITES))
def test_deferred_names_are_not_imported_at_module_level(relative_path: str) -> None:
    bound = _module_level_bindings(_parse(relative_path).body)
    leaked = {name: bound[name] for name in DEFERRED_SITES[relative_path] if name in bound}
    assert leaked == {}, f"{relative_path}: runtime module-level import of deferred names (name: line) {leaked}"


def _has_future_annotations(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom) and node.module == "__future__" and any(a.name == "annotations" for a in node.names)
        for node in tree.body
    )


def _annotation_node_ids(tree: ast.Module) -> set[int]:
    """``id()`` of every node inside an annotation.

    Only sound when the module has ``from __future__ import annotations``: without it an
    annotation is evaluated -- a function's at ``def`` time, i.e. at import -- so a deferred
    name there is a real module-level use. The caller asserts the future import first.
    """
    marked: set[int] = set()
    for node in ast.walk(tree):
        annotations: list[ast.expr | None] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args
            annotations.append(node.returns)
            annotations.extend(
                argument.annotation
                for argument in (
                    *arguments.posonlyargs,
                    *arguments.args,
                    *arguments.kwonlyargs,
                    arguments.vararg,
                    arguments.kwarg,
                )
                if argument is not None
            )
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
        for annotation in annotations:
            if annotation is not None:
                marked.update(id(child) for child in ast.walk(annotation))
    return marked


def _locally_bound_names(function: ast.AST) -> set[str]:
    """Names a function binds itself, by import **or** by assignment.

    Assignment counts because the cached-loader pattern binds through one:
    ``mh = _mahotas()`` in ``measure/_measure_texture.py`` imports mahotas once, behind a
    ``functools.cache``, and every later ``mh.`` use is as safe as a local import.
    """
    names = _locally_imported_names(function)
    names.update(
        node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    )
    return names


def _parents(tree: ast.Module) -> dict[int, ast.AST]:
    return {id(child): parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


@pytest.mark.parametrize("relative_path", sorted(DEFERRED_SITES))
def test_no_runtime_use_of_a_deferred_name_escapes_its_local_import(relative_path: str) -> None:
    """The reverse of the test below: every *use* must sit under a function that binds it.

    The table lists the functions known to need each name, so a use added to a function the
    table does not list is invisible to every other guard here -- nothing leaked to module
    level, and the listed functions still import it. The result is a ``NameError`` for the
    first user who reaches that code path, which no test in this repo would have caught.
    """
    tree = _parse(relative_path)
    if any(functions == () for functions in DEFERRED_SITES[relative_path].values()):
        # A TYPE_CHECKING-only name is only safe to skip in annotations if annotations are
        # strings. Pin the premise rather than assume it.
        assert _has_future_annotations(tree), f"{relative_path}: TYPE_CHECKING-only name needs `from __future__ import annotations`"
    annotated = _annotation_node_ids(tree) if _has_future_annotations(tree) else set()
    parents = _parents(tree)
    escaped = []
    for name in DEFERRED_SITES[relative_path]:
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)):
                continue
            if id(node) in annotated:
                continue
            current: ast.AST | None = node
            covered = False
            while current is not None and id(current) in parents:
                current = parents[id(current)]
                if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)) and name in _locally_bound_names(current):
                    covered = True
                    break
            if not covered:
                escaped.append(f"{name} used at line {node.lineno} with no enclosing local import")
    assert escaped == [], f"{relative_path}: {escaped}"


@pytest.mark.parametrize("relative_path", sorted(DEFERRED_SITES))
def test_each_user_of_a_deferred_name_imports_it_locally(relative_path: str) -> None:
    tree = _parse(relative_path)
    missing = []
    for name, function_names in DEFERRED_SITES[relative_path].items():
        for function_name in function_names:
            candidates = [
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
            ]
            assert candidates, f"{relative_path}: no function named {function_name!r}"
            if not any(name in _locally_imported_names(node) for node in candidates):
                missing.append(f"{function_name} does not import {name}")
    assert missing == [], f"{relative_path}: {missing}"
