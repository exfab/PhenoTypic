import pandas as pd
from phenotypic.sdk_._rembi_manifest import build_rembi_manifest


def _df():
    return pd.DataFrame({
        "Metadata_Strain": ["BY4741", "by4742"],
        "Metadata_Media": ["YPD", "YPD"],
        "Metadata_Temperature": [30, 30],
        "Metadata_CustomTag": ["x", "x"],
        "Size_Area": [10, 12],
        "Shape_Circularity": [0.9, 0.8],
    })


def _imgmeta():
    return [{"ImageName": "p1", "UUID": "u1", "BitDepth": 8, "ImageType": "rgb"}]


def test_scalar_vs_list_collapse():
    m = build_rembi_manifest(_df(), _imgmeta())
    assert sorted(m["biosample"]["Strain"]) == ["BY4741", "by4742"]  # >1 -> list
    assert m["specimen_preparation"]["Media"] == "YPD"               # 1 -> scalar


def test_unknown_metadata_goes_uncategorized():
    m = build_rembi_manifest(_df(), _imgmeta())
    assert "CustomTag" in m["uncategorized"]


def test_analyzed_data_lists_features_grouped():
    m = build_rembi_manifest(_df(), _imgmeta())
    assert m["analyzed_data"]["features"]["Size"] == ["Area"]
    assert m["analyzed_data"]["features"]["Shape"] == ["Circularity"]


def test_image_data_always_present_even_empty():
    m = build_rembi_manifest(pd.DataFrame(), _imgmeta())
    assert m["image_data"]["n_images"] == 1
    assert m["image_data"]["files"][0]["uuid"] == "u1"
    assert "biosample" not in m  # empty sections omitted


def test_study_config_overrides_csv_constant():
    df = pd.DataFrame({"Metadata_Title": ["from_csv", "from_csv"]})
    m = build_rembi_manifest(df, _imgmeta(), study_config={"Title": "from_file"})
    assert m["study"]["Title"] == "from_file"


def test_study_ambiguity_collapses_to_list():
    df = pd.DataFrame({"Metadata_Title": ["a", "b"]})
    m = build_rembi_manifest(df, _imgmeta())
    assert sorted(m["study"]["Title"]) == ["a", "b"]
