"""Edge-section add/remove/preview wiring over the shared _filters dict."""
from phenotypic.analysis import EdgeCorrector, TukeyOutlierRemover
from phenotypic import ImagePipeline
from phenotypic.gui.analysis._callbacks import _resolve_preview_node


class _Recipe:
    def __init__(self, pipeline):
        self.pipeline = pipeline


def test_resolve_preview_node_partitions_edge_and_filter():
    p = ImagePipeline()
    p.set_filters({
        "t0": TukeyOutlierRemover(on="Shape_Area", groupby=["Metadata_Strain"]),
        "e0": EdgeCorrector(on="Shape_Area", groupby=["Metadata_Strain"]),
    })
    recipe = _Recipe(p)
    edge = _resolve_preview_node(recipe, "edge", 0)
    filt = _resolve_preview_node(recipe, "filter", 0)
    assert isinstance(edge, EdgeCorrector)
    assert isinstance(filt, TukeyOutlierRemover)
