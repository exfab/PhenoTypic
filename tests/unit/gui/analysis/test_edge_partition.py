"""The filter/edge GUI partition over the shared pipeline _filters dict."""
from phenotypic.analysis import EdgeCorrector, TukeyOutlierRemover
from phenotypic import ImagePipeline
from phenotypic.gui.analysis import _ids as ids
from phenotypic.gui.analysis._layout import filter_items_for_kind


def _pipeline_with_both():
    p = ImagePipeline()
    p.set_filters({
        "t0": TukeyOutlierRemover(on="Shape_Area", groupby=["Metadata_Strain"]),
        "e0": EdgeCorrector(on="Shape_Area", groupby=["Metadata_Strain"]),
        "t1": TukeyOutlierRemover(on="Shape_Area", groupby=["Metadata_Strain"]),
    })
    return p


def test_partition_splits_by_category():
    p = _pipeline_with_both()
    filt = filter_items_for_kind(p, "filter")
    edge = filter_items_for_kind(p, "edge")
    assert [k for k, _ in filt] == ["t0", "t1"]
    assert [k for k, _ in edge] == ["e0"]


def test_edge_ids_exist():
    assert ids.ANALYSIS_EDGE_STACK == "analysis-edge-stack"
    assert ids.ANALYSIS_EDGE_ADD_DROPDOWN == "analysis-edge-add-dropdown"
    assert ids.edge_section_id(2)["type"] == "analysis-edge-section"


class _Recipe:
    """Minimal stand-in: build_section_stack only reads recipe.pipeline."""
    def __init__(self, pipeline):
        self.pipeline = pipeline


def test_build_section_stack_edge_vs_filter():
    from phenotypic.gui.analysis._layout import build_section_stack

    recipe = _Recipe(_pipeline_with_both())
    edge_cards = build_section_stack(ids.ANALYSIS_EDGE_STACK, "edge", recipe)
    filter_cards = build_section_stack(ids.ANALYSIS_FILTER_STACK, "filter", recipe)
    assert len(edge_cards) == 1
    assert len(filter_cards) == 2
