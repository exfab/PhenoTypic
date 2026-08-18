import pandas as pd
import yaml
from phenotypic.sdk_._rembi_manifest import write_rembi_manifest
from phenotypic.sdk_._io_constants import rembi_manifest_path


def test_writes_parseable_yaml(tmp_path):
    (tmp_path / "deliverables").mkdir()
    df = pd.DataFrame({"Metadata_Strain": ["BY4741"]})
    p = write_rembi_manifest(tmp_path, df, [{"ImageName": "p1", "UUID": "u1",
                                             "BitDepth": 8, "ImageType": "rgb"}])
    assert p == rembi_manifest_path(tmp_path)
    data = yaml.safe_load(p.read_text())
    assert data["image_data"]["n_images"] == 1
    assert data["biosample"]["Strain"] == "BY4741"


def test_write_never_raises(tmp_path):
    # deliverables dir missing -> best-effort returns None, no exception
    result = write_rembi_manifest(tmp_path, pd.DataFrame(), [])
    assert result is None or result.exists()
