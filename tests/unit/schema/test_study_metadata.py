from phenotypic.schema import STUDY_METADATA, REMBI_MODULE


def test_study_members_present():
    labels = STUDY_METADATA.get_labels()
    assert labels == [
        "Title", "Description", "PrivateUntilDate", "Keywords", "Author",
        "License", "Funding", "Publications", "Links", "Acknowledgements",
    ]


def test_study_module_and_namespace():
    assert STUDY_METADATA.TITLE.resolved_rembi_module is REMBI_MODULE.STUDY
    assert STUDY_METADATA.category().startswith("Metadata")
    assert STUDY_METADATA.TITLE.value.endswith("_Title")


def test_study_bio_desc_unset():
    # human-authored guardrail: agents leave bio_desc empty
    assert all(m.bio_desc == "" for m in STUDY_METADATA)
