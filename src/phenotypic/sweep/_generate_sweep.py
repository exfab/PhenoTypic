"""Generate and load parameter sweep manifests from Sweep configurations."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ._sweep import Sweep


def generate_sweep_manifest(
    configs: Union[List[Sweep], Dict[str, List[Sweep]]],
    *,
    meas: Optional[list] = None,
    filepath: Optional[Union[str, Path]] = None,
    desc: Optional[str] = None,
) -> dict:
    """Generate a parameter sweep manifest from Sweep configurations.

    Creates all cartesian-product pipeline combinations for the given sweep
    specifications, serialises them as JSON-compatible dicts, and optionally
    writes the manifest to a file.

    Args:
        configs: Either a single ``List[Sweep]`` (auto-named *Pipeline*) or a
            ``Dict[str, List[Sweep]]`` mapping config names to sweep lists.
        meas: Optional list of ``MeasureFeatures`` instances to attach to
            every generated pipeline.
        filepath: If provided, write the manifest JSON to this path.
        desc: Optional human-readable description stored in the manifest.

    Returns:
        A dict with the manifest structure::

            {
                "version": "0.13.0",
                "description": "...",
                "total_pipelines": 4,
                "configs": { "<name>": { "n_combinations": 2, "pipelines": {...} } }
            }

    Raises:
        TypeError: If ``configs`` is not a list or dict.
        TypeError: If a Sweep's ``operation_class`` is an instance.
        ValueError: If a Sweep references invalid parameter names.

    Examples:
        >>> from phenotypic.sweep import Sweep, generate_sweep_manifest
        >>> from phenotypic.enhance import GaussianBlur
        >>> from phenotypic.detect import OtsuDetector
        >>> config = [
        ...     Sweep(GaussianBlur, sigma=(1.0, 2.0), truncate=4.0),
        ...     Sweep(OtsuDetector, ignore_zeros=(True, False)),
        ... ]
        >>> manifest = generate_sweep_manifest(config)
        >>> manifest['total_pipelines']
        4
    """
    # Lazy imports to avoid circular import through phenotypic.__init__
    from phenotypic import ImagePipeline
    import phenotypic

    # Normalise input
    if isinstance(configs, list):
        named_configs: Dict[str, List[Sweep]] = {"Pipeline": configs}
    elif isinstance(configs, dict):
        named_configs = configs
    else:
        raise TypeError(
            f"configs must be a list or dict, got {type(configs).__name__}"
        )

    # Validate all Sweep objects
    for cfg_name, sweep_list in named_configs.items():
        _validate_sweep_config(sweep_list, cfg_name)

    # Build manifest
    manifest: Dict[str, Any] = {
        "version": phenotypic.__version__,
        "description": desc,
        "total_pipelines": 0,
        "configs": {},
    }

    for cfg_name, sweep_list in named_configs.items():
        combinations = _generate_param_combinations(sweep_list)
        pipelines_dict: Dict[str, Any] = {}

        for idx, combo in enumerate(combinations):
            pipe_name = f"{cfg_name}_{idx}"
            ops = []
            for sweep_obj, param_set in zip(sweep_list, combo):
                merged = {**sweep_obj.fixed_params, **param_set}
                ops.append(sweep_obj.operation_class(**merged))

            pipe = ImagePipeline(ops=ops, name=pipe_name)

            if meas is not None:
                pipe.set_meas(meas)

            # Parse the JSON string to a dict so the manifest is a pure dict
            pipelines_dict[pipe_name] = json.loads(pipe.to_json_str())

        manifest["configs"][cfg_name] = {
            "n_combinations": len(combinations),
            "pipelines": pipelines_dict,
        }
        manifest["total_pipelines"] += len(combinations)

    if filepath is not None:
        Path(filepath).write_text(json.dumps(manifest, indent=2))

    return manifest


def load_sweep_manifest(
    filepath: Union[str, Path],
) -> Dict[str, Dict[str, Any]]:
    """Load a sweep manifest and reconstruct ``ImagePipeline`` objects.

    Args:
        filepath: Path to a JSON manifest file previously created by
            :func:`generate_sweep_manifest`.

    Returns:
        Nested dict ``{config_name: {pipeline_name: ImagePipeline, ...}, ...}``.

    Raises:
        FileNotFoundError: If *filepath* does not exist.

    Examples:
        >>> from phenotypic.sweep import load_sweep_manifest
        >>> pipelines = load_sweep_manifest("sweep.json")
        >>> for cfg_name, pipes in pipelines.items():
        ...     for pipe_name, pipe in pipes.items():
        ...         print(pipe_name, [op.__class__.__name__ for op in pipe._ops.values()])
    """
    from phenotypic import ImagePipeline

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {filepath}")

    manifest = json.loads(path.read_text())
    result: Dict[str, Dict[str, Any]] = {}

    for cfg_name, cfg_data in manifest.get("configs", {}).items():
        result[cfg_name] = {}
        for pipe_name, pipe_dict in cfg_data.get("pipelines", {}).items():
            pipe = ImagePipeline.from_json(json.dumps(pipe_dict))
            result[cfg_name][pipe_name] = pipe

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_sweep_config(sweep_list: List[Sweep], cfg_name: str) -> None:
    """Validate a list of Sweep objects.

    Args:
        sweep_list: Sweep objects to validate.
        cfg_name: Name of the config (for error messages).

    Raises:
        TypeError: If any element is not a Sweep instance.
    """
    if not isinstance(sweep_list, list):
        raise TypeError(
            f"Config '{cfg_name}' must be a list of Sweep objects, "
            f"got {type(sweep_list).__name__}"
        )
    for idx, item in enumerate(sweep_list):
        if not isinstance(item, Sweep):
            raise TypeError(
                f"Config '{cfg_name}' item {idx} must be a Sweep instance, "
                f"got {type(item).__name__}"
            )


def _generate_param_combinations(
    sweep_list: List[Sweep],
) -> List[tuple]:
    """Generate cartesian product of all swept parameters.

    For each ``Sweep`` object, produces dicts of swept-param combinations.
    Returns a list of tuples, one tuple per pipeline, where each element is a
    dict of swept param values for the corresponding ``Sweep``.

    Args:
        sweep_list: List of Sweep specifications.

    Returns:
        List of tuples of dicts.  ``len(result)`` equals the total number of
        pipeline combinations.
    """
    per_op_combos: List[List[Dict[str, Any]]] = []

    for sweep_obj in sweep_list:
        if not sweep_obj.sweep_params:
            # No swept params → single empty dict (operation uses fixed only)
            per_op_combos.append([{}])
        else:
            keys = list(sweep_obj.sweep_params.keys())
            values = list(sweep_obj.sweep_params.values())
            combos = [dict(zip(keys, vals)) for vals in product(*values)]
            per_op_combos.append(combos)

    return list(product(*per_op_combos))
