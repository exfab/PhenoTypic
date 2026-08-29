from pathlib import Path

from phenotypic.gui.browse import _callbacks as cb


def _walk_components(node: object) -> list[object]:
    nodes = [node]
    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            nodes.extend(_walk_components(child))
    elif children is not None:
        nodes.extend(_walk_components(children))
    return nodes


def test_dataset_options_sorted_labels():
    datasets = {".": ["a.png"], "plates/b7": ["A1.png"]}
    opts = cb.dataset_options(datasets)
    assert opts == [
        {"label": "(root)", "value": "."},
        {"label": "plates/b7", "value": "plates/b7"},
    ]


def test_image_options_for_selected_dataset():
    datasets = {"plates/b7": ["A1.png", "A2.png"]}
    assert cb.image_options(datasets, "plates/b7") == [
        {"label": "A1.png", "value": "A1.png"},
        {"label": "A2.png", "value": "A2.png"},
    ]
    assert cb.image_options(datasets, "missing") == []


def test_dataset_row_hidden_when_flat():
    assert cb.dataset_row_hidden({".": ["a.png"]}) is True
    assert cb.dataset_row_hidden({"plates": ["a.png"]}) is False
    assert cb.dataset_row_hidden({}) is True
    # Mixed: root files + nested siblings → shown (realistic nested source).
    assert cb.dataset_row_hidden({".": ["a.png"], "plates": ["A1.png"]}) is False


def test_sandbox_rel_joins_src_dataset_filename():
    assert cb.sandbox_rel("plates/b7", "day3", "A1.png") == "plates/b7/day3/A1.png"
    assert cb.sandbox_rel("plates/b7", ".", "A1.png") == "plates/b7/A1.png"
    assert cb.sandbox_rel(".", ".", "A1.png") == "A1.png"
    # Empty src_root_rel (source IS the sandbox root) joins cleanly.
    assert cb.sandbox_rel("", "", "A1.png") == "A1.png"
    assert cb.sandbox_rel("", "plates", "A1.png") == "plates/A1.png"


def test_current_image_payload_round_trips_token():
    from phenotypic.gui.browse._source_render import decode_token

    payload = cb.current_image_payload("plates/b7", ".", "A1.png")
    assert decode_token(payload["token"]) == "plates/b7/A1.png"
    assert payload["label"] == "plates/b7/A1.png"


def test_current_image_payload_flat_source_round_trips():
    from phenotypic.gui.browse._source_render import decode_token

    # All-"." flat case: no leading slashes/dots in the token's decoded path.
    payload = cb.current_image_payload(".", ".", "A1.png")
    assert decode_token(payload["token"]) == "A1.png"
    assert payload["label"] == "A1.png"
    # No neighbours given → no prefetch key (client treats absence as no-op).
    assert "prefetch" not in payload


def test_current_image_payload_direct_store_does_not_double_join():
    from phenotypic.gui.browse._source_render import decode_token

    payload = cb.current_image_payload(
        "inputs/p01.ome.zarr", ".", "p01.ome.zarr"
    )

    assert decode_token(payload["token"]) == "inputs/p01.ome.zarr"
    assert payload["label"] == "inputs/p01.ome.zarr"


def test_neighbor_filenames_three_each_side_clamped():
    files = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    # Interior: 3 before + 3 after, current excluded, nav order preserved.
    assert cb.neighbor_filenames(files, "e") == ["b", "c", "d", "f", "g", "h"]
    # Clamp at the left edge (fewer before).
    assert cb.neighbor_filenames(files, "a") == ["b", "c", "d"]
    assert cb.neighbor_filenames(files, "b") == ["a", "c", "d", "e"]
    # Clamp at the right edge (fewer after).
    assert cb.neighbor_filenames(files, "i") == ["f", "g", "h"]
    # Unknown current → empty (no prefetch).
    assert cb.neighbor_filenames(files, "zzz") == []
    # Custom radius.
    assert cb.neighbor_filenames(files, "e", radius=1) == ["d", "f"]


def test_current_image_payload_prefetch_tokens_round_trip():
    from phenotypic.gui.browse._source_render import decode_token

    payload = cb.current_image_payload(
        "plates/b7", "day3", "A2.png", neighbor_files=["A1.png", "A3.png"]
    )
    assert decode_token(payload["token"]) == "plates/b7/day3/A2.png"
    assert [decode_token(t) for t in payload["prefetch"]] == [
        "plates/b7/day3/A1.png",
        "plates/b7/day3/A3.png",
    ]


def test_csv_metadata_panel_message_for_unset_metadata():
    panel = cb.render_csv_metadata_panel(
        cb.CsvMetadataPanelModel(state="unset", image_stem="plate_a", rows=[])
    )

    assert "No metadata CSV selected" in str(panel)


def test_csv_metadata_panel_renders_matched_values_without_image_name():
    panel = cb.render_csv_metadata_panel(
        cb.CsvMetadataPanelModel(
            state="matched",
            image_stem="plate_a",
            rows=[{"Treatment": "control", "Replicate": "1"}],
        )
    )

    text = str(panel)
    assert "Treatment" in text
    assert "control" in text
    assert "Replicate" in text
    assert "Metadata_ImageName" not in text


def test_csv_metadata_panel_renders_multiple_rows_with_count():
    panel = cb.render_csv_metadata_panel(
        cb.CsvMetadataPanelModel(
            state="matched",
            image_stem="plate_a",
            rows=[
                {"Colony": "A01", "Treatment": "control"},
                {"Colony": "A02", "Treatment": "stress"},
            ],
        )
    )

    text = str(panel)
    assert "2 metadata rows for plate_a" in text
    assert "A01" in text
    assert "A02" in text


def test_csv_metadata_panel_reports_conflicting_identity_columns():
    panel = cb.render_csv_metadata_panel(
        cb.CsvMetadataPanelModel(
            state="ambiguous_image_name",
            image_stem="plate_a",
            rows=[],
        )
    )

    assert "conflicting image-name columns" in str(panel)


def test_timeline_image_column_defaults_to_recognized_legacy_header():
    columns = ["Treatment", "Metadata_ImageFileName", "Time"]
    rows = [
        {
            "Treatment": "control",
            "Metadata_ImageFileName": "plate_a.tif",
            "Time": "0",
        }
    ]

    options, default = cb.csv_column_options_and_image_default(columns, rows)

    assert options == [
        {"label": "Treatment", "value": "Treatment"},
        {
            "label": "Metadata_ImageFileName",
            "value": "Metadata_ImageFileName",
        },
        {"label": "Time", "value": "Time"},
    ]
    assert default == "Metadata_ImageFileName"


def test_timeline_image_column_has_no_default_for_ambiguous_aliases():
    columns = ["Metadata_ImageName", "Metadata_ImageFileName"]
    rows = [
        {
            "Metadata_ImageName": "plate_a",
            "Metadata_ImageFileName": "plate_b.tif",
        }
    ]

    _options, default = cb.csv_column_options_and_image_default(columns, rows)

    assert default is None


def test_timeline_image_column_has_no_default_for_complementary_aliases():
    columns = ["Metadata_ImageName", "Metadata_ImageFileName"]
    rows = [
        {
            "Metadata_ImageName": "plate_a",
            "Metadata_ImageFileName": "",
        },
        {
            "Metadata_ImageName": "",
            "Metadata_ImageFileName": "plate_b.tif",
        },
    ]

    _options, default = cb.csv_column_options_and_image_default(columns, rows)

    assert default is None


def test_csv_metadata_panel_table_is_bounded_and_horizontally_scrollable():
    panel = cb.render_csv_metadata_panel(
        cb.CsvMetadataPanelModel(
            state="matched",
            image_stem="plate_a",
            rows=[
                {
                    "VeryLongMetadataColumnName01": "control",
                    "VeryLongMetadataColumnName02": "batch-a",
                    "VeryLongMetadataColumnName03": "edge",
                }
            ],
        )
    )

    classes = {
        getattr(component, "className", "")
        for component in _walk_components(panel)
        if getattr(component, "className", "")
    }
    assert "browse-csv-metadata-scroll" in classes
    assert "table table-sm mb-0 browse-csv-metadata-table" in classes

    css = (
        Path(__file__).parents[3] / "src/phenotypic/gui/browse/_assets/browse.css"
    ).read_text(encoding="utf-8")
    assert ".browse-csv-metadata-panel" in css
    assert ".browse-csv-metadata-scroll" in css
    assert "max-width: 100%;" in css
    assert "overflow-x: auto;" in css
    assert "width: max-content;" in css
