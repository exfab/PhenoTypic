from phenotypic.schema import REMBI_MODULE, STUDY


def test_study_members_present():
    labels = STUDY.get_labels()
    assert labels == [
        "Title", "Description", "PrivateUntilDate", "Keywords", "Author",
        "License", "Funding", "Publications", "Links", "Acknowledgements",
    ]


def test_study_module_and_namespace():
    assert STUDY.TITLE.resolved_rembi_module is REMBI_MODULE.STUDY
    assert STUDY.category() == "Metadata"
    assert STUDY.TITLE.value == "Metadata_Title"


def test_study_bio_desc_unset():
    # human-authored guardrail: agents leave bio_desc empty
    assert all(m.bio_desc == "" for m in STUDY)
