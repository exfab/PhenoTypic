"""Apply each mutation to ngff_.py, run the ngff suite, revert, report."""

import pathlib
import subprocess
import sys

SRC = pathlib.Path(
    "/bigdata/exfab/anguy344/PhenoTypic/.worktrees/worktree-ome-zarr-image-store"
    "/src/phenotypic/sdk_/ngff_.py"
)
ROOT = pathlib.Path(
    "/bigdata/exfab/anguy344/PhenoTypic/.worktrees/worktree-ome-zarr-image-store"
)

MUTANTS = [
    (
        "S1a build_pyramid always mean",
        '    reduce = downsample_image if kind == "image" else downsample_label',
        "    reduce = downsample_image",
    ),
    (
        "S1b build_pyramid inverted ternary",
        '    reduce = downsample_image if kind == "image" else downsample_label',
        '    reduce = downsample_label if kind == "image" else downsample_image',
    ),
    (
        "S2 transposed block reshape",
        "    blocks = array.astype(np.float64).reshape(*lead, ph // 2, 2, pw // 2, 2)",
        "    blocks = array.astype(np.float64).reshape(*lead, 2, ph // 2, 2, pw // 2)",
    ),
    (
        "S3 drop np.rint",
        "        return np.rint(reduced).astype(array.dtype)",
        "        return reduced.astype(array.dtype)",
    ),
    (
        "S4 drop shards from array_create_kwargs",
        '        "shards": shard_shape_for(shape),\n',
        "",
    ),
    (
        "S5a swap illuminant/gamma",
        "        PhenotypicAttr.ILLUMINANT: illuminant,\n"
        "        PhenotypicAttr.GAMMA: gamma,",
        "        PhenotypicAttr.ILLUMINANT: gamma,\n"
        "        PhenotypicAttr.GAMMA: illuminant,",
    ),
    (
        "S5b hard-code image_class",
        "        PhenotypicAttr.IMAGE_CLASS: image_class,",
        '        PhenotypicAttr.IMAGE_CLASS: "Image",',
    ),
    (
        "S5c drop grid",
        "    if grid is not None:\n        block[PhenotypicAttr.GRID] = grid\n",
        "",
    ),
    (
        "S5d drop detect_mode",
        "        PhenotypicAttr.DETECT_MODE: detect_mode,\n",
        "",
    ),
    (
        "S5e drop illuminant and gamma",
        "        PhenotypicAttr.ILLUMINANT: illuminant,\n"
        "        PhenotypicAttr.GAMMA: gamma,\n",
        "",
    ),
    (
        "S5f drop the imported metadata section",
        "            PhenotypicAttr.IMPORTED: dict(\n"
        "                metadata_sections.get(PhenotypicAttr.IMPORTED, {})\n"
        "            ),\n",
        "",
    ),
    (
        "S5g drop phenotypic_version",
        "        PhenotypicAttr.PHENOTYPIC_VERSION: (\n"
        "            phenotypic_version or phenotypic.__version__\n"
        "        ),\n",
        "",
    ),
    (
        "S6a size_c hard-coded to 1",
        "        size_c = shape[0] if len(shape) == 3 else 1",
        "        size_c = 1",
    ),
    (
        "S6b swap size_x/size_y",
        "        size_y, size_x = shape[-2], shape[-1]",
        "        size_y, size_x = shape[-1], shape[-2]",
    ),
    (
        "S7 drop escape() on the annotation value",
        "            f\"{escape(_xml_text(value))}</M>\"",
        "            f\"{_xml_text(value)}</M>\"",
    ),
    (
        "S8 build_multiscales kind hard-coded to image",
        '    kind = "label" if series == OBJMAP_LABEL else "image"',
        '    kind = "image"',
    ),
    (
        "S9a drop multiscales type",
        '        "type": DOWNSAMPLE_METHODS[kind][0],\n',
        "",
    ),
    (
        "S9b drop multiscales metadata",
        '        "metadata": {"description": DOWNSAMPLE_METHODS[kind][1]},\n',
        "",
    ),
    (
        "S9c drop multiscales name",
        '    if name is not None:\n        multiscale["name"] = name\n',
        "",
    ),
    (
        "S10 tighten _XML_FORBIDDEN to strip DEL/C1/#xFFFD",
        '    "[^\\u0009\\u000A\\u000D\\u0020-\\uD7FF\\uE000-\\uFFFD'
        '\\U00010000-\\U0010FFFF]"',
        '    "[^\\u0009\\u000A\\u000D\\u0020-\\u007E\\u00A0-\\uD7FF\\uE000-\\uFFFC'
        '\\U00010000-\\U0010FFFF]"',
    ),
    (
        "D1 remove AttributeError from the except tuple",
        "    except (AttributeError, OSError, KeyError, TypeError, ValueError):",
        "    except (OSError, KeyError, TypeError, ValueError):",
    ),
]


def run_suite() -> tuple[int, str]:
    proc = subprocess.run(
        [
            "uv", "run", "pytest", "tests/unit/sdk_/test_ngff_array_policy.py",
            "tests/unit/sdk_/test_ngff_attributes.py",
            "tests/unit/sdk_/test_ngff_geometry.py",
            "tests/unit/sdk_/test_ngff_projection.py",
            "tests/unit/sdk_/test_ngff_promote.py",
            "tests/unit/sdk_/test_ngff_validity.py",
            "-q", "--no-header", "-p", "no:randomly", "-o", "addopts=",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout


original = SRC.read_text(encoding="utf-8")
only = sys.argv[1] if len(sys.argv) > 1 else None

for name, old, new in MUTANTS:
    if only and only not in name:
        continue
    if original.count(old) != 1:
        print(f"### {name}\nANCHOR NOT UNIQUE ({original.count(old)}) -- SKIPPED\n")
        continue
    SRC.write_text(original.replace(old, new), encoding="utf-8")
    code, out = run_suite()
    SRC.write_text(original, encoding="utf-8")
    killers = [ln for ln in out.splitlines() if ln.startswith("FAILED ")]
    summary = [ln for ln in out.splitlines() if " passed" in ln or " failed" in ln]
    print(f"### {name}")
    print(f"exit={code}  {summary[-1] if summary else out[-300:]}")
    for k in killers:
        print("  " + k)
    print()
