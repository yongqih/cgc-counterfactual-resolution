from scripts.prepare_lea_archs4_robustness import EXCLUDED


def test_unresolved_lines_are_permanently_excluded():
    assert EXCLUDED == {
        "LineNA": "EXCLUDED_UNRESOLVED_GEO_IDENTITY",
        "Line223": "EXCLUDED_UNRESOLVED_GEO_IDENTITY",
    }
