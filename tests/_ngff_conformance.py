"""Validate a written store against the vendored NGFF 0.5 JSON schemas.

Conformance is checked with ``jsonschema`` rather than ``ome-zarr-models``,
which pins ``pydantic<2.13``; pydantic 2.13 has already shipped, so adopting it
would hold the project a release behind today. There is no ``[tool.uv]
conflicts`` block, so even a dev-group-only cap would bind the whole locked
environment.

A failure here fails the suite. It is never downgraded to a warning and never
skipped: a check that cannot run must fail.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import TYPE_CHECKING

import jsonschema
import referencing
import referencing.jsonschema

if TYPE_CHECKING:  # `xmlschema` is imported lazily inside the functions below,
    # so the quoted return annotation has no runtime binding to resolve against
    # and ruff reports F821. Guarding the import here keeps the annotation
    # exactly as specified while leaving the import cost where it was.
    import xmlschema

SCHEMA_DIR = Path(__file__).resolve().parent / "fixtures" / "ngff" / "0.5"


SCHEMA_NAMES = ("image.schema", "label.schema", "ome.schema", "_version.schema")


def _schema(name: str) -> dict:
    path = SCHEMA_DIR / name
    if not path.is_file():
        raise AssertionError(
            f"vendored NGFF schema missing: {path}. A conformance check that "
            "cannot run must fail, never skip."
        )
    return json.loads(path.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def _registry() -> referencing.Registry:
    """Resolve every ``$ref`` from the vendored copies, never over the network.

    All three schemas reference ``_version.schema`` by absolute URL. jsonschema
    >= 4.18 does not fetch remote refs; it raises
    ``referencing.exceptions.Unresolvable``, which is not a ``ValidationError``
    -- so without this the suite errors instead of failing, and offline CI has
    no fallback at all.
    """
    registry = referencing.Registry()
    for name in SCHEMA_NAMES:
        payload = _schema(name)
        resource = referencing.Resource.from_contents(
            payload, default_specification=referencing.jsonschema.DRAFT202012
        )
        registry = resource @ registry
    return registry


def _attributes(group_dir: Path) -> dict:
    """Return the group's whole ``attributes`` mapping.

    NOT ``attributes["ome"]``: each vendored schema is rooted at the attributes
    object itself (``"description": "The zarr.json attributes key"``,
    ``"required": ["ome"]``). Passing the inner block fails with
    ``'ome' is a required property`` on every store ever written.
    """
    payload = json.loads((group_dir / "zarr.json").read_text(encoding="utf-8"))
    return payload["attributes"]


def _validate(attributes: dict, schema_name: str, where: Path) -> None:
    schema = _schema(schema_name)
    validator = jsonschema.Draft202012Validator(schema, registry=_registry())
    try:
        validator.validate(attributes)
    except jsonschema.ValidationError as exc:
        raise AssertionError(
            f"{where} does not conform to {schema_name}: {exc.message} "
            f"(at {list(exc.absolute_path)})"
        ) from exc


@functools.lru_cache(maxsize=1)
def _ome_xsd() -> "xmlschema.XMLSchema":
    """Load the vendored OME-XML schema.

    NGFF 2.2.3 makes the document a conditional MUST, and JSON-schema
    validation says nothing about it.
    """
    import xmlschema

    path = Path(__file__).resolve().parent / "fixtures" / "ome" / "2016-06" / "ome.xsd"
    if not path.is_file():
        raise AssertionError(
            f"vendored OME schema missing: {path}. A conformance check that "
            "cannot run must fail, never skip."
        )
    return xmlschema.XMLSchema(str(path))


def assert_ome_xml_valid(xml: str) -> None:
    """Validate an OME-XML document against the vendored `ome.xsd`.

    Catches ``XMLSchemaException``, not ``XMLSchemaValidationError``: the
    narrower class does not cover a well-formedness failure, which raises
    ``XMLResourceParseError`` -- and that is the most likely real failure, since
    a control character in an imported EXIF tag breaks well-formedness rather
    than schema conformance (ledger **ALGO-R2B-11**).

    Args:
        xml: The document as a string.

    Raises:
        AssertionError: On any schema violation OR malformed document.
    """
    import xmlschema

    try:
        _ome_xsd().validate(xml)
    except xmlschema.XMLSchemaException as exc:
        # Name the exception type: the widened catch also takes
        # XMLSchemaOSError/XMLResourceOSError, and a disk error reported as
        # "not valid against ome.xsd" would be misread as a conformance bug.
        raise AssertionError(
            f"OME-XML failed validation against ome.xsd "
            f"[{type(exc).__name__}]: {exc}"
        ) from exc


def assert_store_conforms(store_path: Path) -> None:
    """Validate every conformance surface of one written store.

    Args:
        store_path: Path to a promoted ``*.ome.zarr`` directory.

    Raises:
        AssertionError: On any schema violation, missing schema fixture, or
            structurally absent group.
    """
    from phenotypic.sdk_.ngff_ import (
        OME_GROUP,
        OME_XML_NAME,
        PhenotypicAttr,
        read_phenotypic_attributes,
    )

    store = Path(store_path)
    block = read_phenotypic_attributes(store)

    for member in block[PhenotypicAttr.SERIES].values():
        _validate(_attributes(store / member), "image.schema", store / member)

    # A label-less store is valid (a builder preview of a node that changed no
    # labels), and `.values()` on an omitted-or-empty mapping is correctly a
    # no-op -- but only because Task 1.3 now omits the key rather than pointing
    # it at a group that was never written. Ledger C3.
    for member in block.get(PhenotypicAttr.LABELS, {}).values():
        _validate(_attributes(store / member), "label.schema", store / member)

    ome_group = store / OME_GROUP
    # An ASSERTION, not a guard (ledger ALGO-3 / ALGO-13a). The group is
    # unconditional now, so `if ome_group.is_dir():` would let a regression that
    # stopped writing it pass every conformance test in the suite.
    # NOT "mandatory under layout 3" -- 2.2.3 makes the OME metadata a SHOULD
    # ("SHOULD have OME metadata ... in a file named OME/METADATA.ome.xml") and
    # the series attribute a MAY. What makes the group mandatory for THIS store
    # is the named-series layout: 2.2.3 says "If the 'series' attribute does not
    # exist and no 'plate' is present: separate 'multiscales' images MUST be
    # stored in consecutively numbered groups starting from 0". This writer
    # emits rgb/gray/detect_mat, not 0/1/2, so `series` is load-bearing -- and
    # OME/ is the only place 2.2.3 puts it. Ledger ALGO-16: an earlier message
    # cited layout 3, which a reader checking 2.2.3 would find overreaching, and
    # would then soften back to `if ome_group.is_dir():` -- reinstating the
    # silently-skipped surface ALGO-3/ALGO-13(a) closed.
    assert ome_group.is_dir(), (
        f"OME/ group is mandatory for a NAMED-series store: NGFF 2.2.3 requires "
        f"consecutively numbered groups when 'series' is absent, and 'series' "
        f"lives here: {store}"
    )
    _validate(_attributes(ome_group), "ome.schema", ome_group)
    # The XML is a separate conformance surface from the JSON: 2.2.3's MUST
    # is about ome.xsd, which no JSON schema covers.
    ome_xml = (ome_group / OME_XML_NAME).read_text(encoding="utf-8")
    assert_ome_xml_valid(ome_xml)

    _assert_reader_level_musts(store, block, ome_xml)


def _group_ome(group_dir: Path) -> dict:
    payload = json.loads((group_dir / "zarr.json").read_text(encoding="utf-8"))
    return payload["attributes"]["ome"]


def _assert_reader_level_musts(store: Path, block: dict, ome_xml: str) -> None:
    """Assert the NGFF MUSTs that no vendored JSON schema encodes.

    These are properties of a *written store* as a third-party reader resolves
    it -- not of a builder function -- so they cannot live in Phase 1's unit
    tests. Uses ``zarr`` only; the ``ome-zarr`` ban holds.

    Args:
        store: The promoted ``*.ome.zarr`` directory.
        block: The store's ``attributes.phenotypic`` block.
        ome_xml: The already-read ``OME/METADATA.ome.xml`` document.

    Raises:
        AssertionError: On any violation.
    """
    import re

    import numpy as np
    import zarr

    from phenotypic.sdk_.ngff_ import (
        LABELS_GROUP,
        OBJMAP_LABEL,
        OME_GROUP,
        PhenotypicAttr,
        primary_series,
    )

    series_names = list(block[PhenotypicAttr.SERIES].values())
    primary = primary_series(series_names)
    # Read the path the STORE DECLARES, rather than re-deriving it with
    # `objmap_path`: a re-derived path cannot fail, whereas this turns the loop
    # below into a real check that the declared label path resolves (ledger
    # ALGO-20). `.get` because a label-less store omits the key (ledger C3).
    labels = block.get(PhenotypicAttr.LABELS, {})
    label_member = labels.get(OBJMAP_LABEL) if labels else None
    # The LABEL GROUP IS IN THIS LOOP, deliberately (ledger ALGO-R2B-14).
    # `label.schema` declares only `image-label` and `version` under
    # `properties.ome` and sets no `additionalProperties: false`, so it says
    # NOTHING about the label's multiscales block -- while 2.6 requires that
    # "the zarr.json file for the label image MUST implement the multiscales
    # specification". Validated by neither schema nor any other check unless it
    # is iterated here.

    # 2.2.3: "'series' MUST be a list of string objects, each of which is a
    # path to an image group. The order of the paths MUST match the order of the
    # 'Image' elements in 'OME/METADATA.ome.xml' IF PROVIDED." (The "if
    # provided" clause was elided in an earlier draft -- ledger ALGO-17.)
    #
    # TWO assertions, because the published rule has two halves and the name
    # comparison only covers one:
    #
    #  (a) COUNT. 2.2.3 also says "Every 'multiscales' group MUST represent
    #      exactly one OME-XML 'Image'". `Name` is use="optional" in ome.xsd, so
    #      a name-only scrape cannot see an unnamed <Image> -- a document with
    #      three named Images matching `declared` plus a fourth unnamed one
    #      would pass the order check while violating the 1:1 MUST.
    #  (b) ORDER. Comparing Image/@Name to the path list is STRONGER than the
    #      spec demands: the MUST is positional, and @Name has no spec-defined
    #      relationship to the group path. It works because `build_ome_xml` sets
    #      Name={quoteattr(series)}. Keep it -- it catches a reordering AND a
    #      naming drift -- but it is THIS WRITER'S CONVENTION, not the rule.
    #
    # Both quote styles are matched: `quoteattr` switches to single quotes when
    # the value contains a double quote. (The document is already parsed by
    # xmlschema one line above; we scrape rather than re-parse because a second
    # stdlib XML parse over user-derived EXIF text has no billion-laughs guard.)
    # The OME GROUP, not the root. Task 2.2 writes the root as
    # {"version", "bioformats2raw.layout"} and puts `series` on OME/ -- which is
    # also what ome.schema requires and what test_missing_series_is_rejected
    # deletes. Reading it off the root raised KeyError on EVERY call, i.e. every
    # store-writing test in Phases 2, 3 and 5. Ledger GEN-33.
    declared = _group_ome(store / OME_GROUP)["series"]
    n_images = len(re.findall(r"<Image\b", ome_xml))
    assert n_images == len(declared), (
        f"2.2.3: every multiscales group MUST represent exactly one Image -- "
        f"{n_images} Image elements vs {len(declared)} series"
    )
    xml_order = [
        double or single
        for double, single in re.findall(
            r"""<Image\b[^>]*?\bName=(?:"([^"]*)"|'([^']*)')""", ome_xml
        )
    ]
    assert xml_order == declared, f"series order != Image order: {xml_order} vs {declared}"

    level_counts: dict[str, int] = {}
    # A label-less store is VALID and must stay tolerated (ledger GEN-33): the
    # old harness iterated `block[LABELS].values()`, a no-op on an empty
    # mapping, and `save_intermediate_zarr(layers=("gray",))` (Task 2.4) writes
    # exactly such a store. Folding the label into this loop unguarded turned
    # "tolerated" into FileNotFoundError.
    members = [*series_names, label_member] if label_member else list(series_names)
    for name in members:
        multiscale = _group_ome(store / name)["multiscales"][0]
        expected_axes = [axis["name"] for axis in multiscale["axes"]]
        level_counts[name] = len(multiscale["datasets"])
        for dataset in multiscale["datasets"]:
            level = store / name / dataset["path"]
            # A reader follows datasets[].path; a dangling one is a broken store.
            array = zarr.open_array(store=str(level), mode="r")
            # 2.4: "MUST have the same number of dimensions ... The number of
            # dimensions and order MUST correspond to number and order of
            # 'axes'." `assert array.shape` was a tautology -- every zarr array
            # has a non-empty shape tuple (ledger ALGO-18). The real check that
            # `datasets[].path` resolves is that `open_array` did not raise.
            assert len(array.shape) == len(expected_axes), (
                name,
                dataset["path"],
                array.shape,
                expected_axes,
            )
            # 2.1 MUST -- dimension_names matches the declared axes. The only
            # other assertion of this anywhere checks the kwargs builder, not a
            # written array.
            meta = json.loads((level / "zarr.json").read_text(encoding="utf-8"))
            # `.get`, not `[]` (ledger ALGO-18). 2.1: "The 'dimension_names'
            # attribute MUST be included in the zarr.json of the Zarr array of a
            # multiscale level" -- but Zarr v3 lists it as OPTIONAL array
            # metadata, so an array missing it is a valid Zarr array and an NGFF
            # violation. That is exactly this assertion's case, and it must
            # surface as AssertionError, not KeyError, or the `Raises:` contract
            # is false for the failure it exists to catch.
            names = meta.get("dimension_names")
            assert names is not None and list(names) == expected_axes, (
                name,
                dataset["path"],
                names,
                expected_axes,
            )

    # 2.6 -- the label is reachable through the labels array, and its pixels are
    # an integer dtype. Skipped entirely for a label-less store.
    if label_member is not None:
        listed = _group_ome(store / primary / LABELS_GROUP)["labels"]
        assert listed == [OBJMAP_LABEL], listed
        label_array = zarr.open_array(
            store=str(store / label_member / "0"), mode="r"
        )
        assert np.issubdtype(label_array.dtype, np.integer), (
            "2.6: label pixels MUST be an integer dtype"
        )
        # 2.6 MUST -- the label's datasets array has the same number of levels
        # as the image it labels. The referent is the PRIMARY series: 2.6 says
        # "the original unlabeled image", and the label labels the primary.
        assert level_counts[label_member] == level_counts[primary], (
            "2.6: label level count MUST match the unlabeled image"
        )

    # DESIGN SPEC 1.4 (not NGFF -- neither NGFF nor Zarr v3 imposes cross-array
    # separator uniformity; this is a PhenoTypic rule, and the bare section
    # numbers elsewhere in this function are NGFF ones). ONE check, not two
    # (ledger SIMP-23): the declared separator is direct, authoritative, and
    # fires even on an array with no chunks written yet. An earlier draft paired
    # it with an `rglob("*/0")` no-subdirectory walk, which observes far less --
    # it only inspects level-0 directories, passes vacuously on an empty array,
    # and under the sharding codec the chunks live inside shard files. The
    # separator assertion subsumes it -- and it is COMPLETE, not a narrowing:
    # under Zarr v3's `sharding_indexed` codec the inner chunks are addressed by
    # BYTE OFFSETS in the shard index, not by keys, so the array's top-level
    # `chunk_key_encoding` is the only thing in the whole store that turns
    # coordinates into a path segment. There is no second separator to check
    # (ledger ALGO-19).
    #
    # Gated on `node_type == "array"`, so an ARRAY that omits
    # `chunk_key_encoding` -- mandatory array metadata in Zarr v3 -- fails here
    # instead of being skipped along with the group documents.
    for meta_path in store.rglob("zarr.json"):
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if payload.get("node_type") != "array":
            continue
        encoding = payload.get("chunk_key_encoding")
        assert encoding is not None, f"{meta_path}: array has no chunk_key_encoding"
        separator = encoding.get("configuration", {}).get("separator")
        assert separator == ".", f"{meta_path}: separator is {separator!r}"
