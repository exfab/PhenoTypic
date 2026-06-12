from phenotypic.gui.browse import _callbacks as cb


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
