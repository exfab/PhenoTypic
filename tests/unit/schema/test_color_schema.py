from phenotypic.schema import ColorLab, ColorHSV


def test_colorlab_has_robust_headers_and_no_legacy_suite():
    headers = ColorLab.get_headers()
    # New robust columns present (prefixed with category "ColorLab_").
    for expected in [
        "ColorLab_L*GeoMedian", "ColorLab_a*GeoMedian", "ColorLab_b*GeoMedian",
        "ColorLab_L*Medoid", "ColorLab_a*Medoid", "ColorLab_b*Medoid",
        "ColorLab_DeltaE2000MedianFromMedoid",
        "ColorLab_DeltaE2000MeanFromMedoid",
        "ColorLab_DeltaE2000P95FromMedoid",
        "ColorLab_LabTotalVariance",
        "ColorLab_MedoidColorHex",
    ]:
        assert expected in headers
    # Legacy per-channel suite (L*/a*/b* x 8 stats) + chroma columns gone.
    legacy_suffixes = ("Min", "Q1", "Mean", "Median", "Q3", "Max", "StdDev", "CoeffVar")
    for chan in ("L*", "a*", "b*"):
        assert not any(h == f"ColorLab_{chan}{suf}" for suf in legacy_suffixes for h in headers)
    assert not any("ChromaEstimated" in h for h in headers)
    assert len(ColorLab.robust_headers()) == 11
