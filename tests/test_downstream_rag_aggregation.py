from src.aggregate_downstream_rag_results import insufficient_information_flags


def test_insufficient_information_flags_exact_with_case_and_punctuation():
    flags = insufficient_information_flags("  Insufficient Information. ")
    assert flags == {"exact": True, "starts_with": True, "contains": True}


def test_insufficient_information_flags_starts_with_but_not_exact():
    flags = insufficient_information_flags("Insufficient information: the contexts disagree.")
    assert flags == {"exact": False, "starts_with": True, "contains": True}


def test_insufficient_information_flags_contains_but_not_prefix():
    flags = insufficient_information_flags("The answer is Paris. Insufficient information for the date.")
    assert flags == {"exact": False, "starts_with": False, "contains": True}


def test_insufficient_information_flags_no_match_or_empty():
    assert insufficient_information_flags("The answer is Paris.") == {
        "exact": False,
        "starts_with": False,
        "contains": False,
    }
    assert insufficient_information_flags(None) == {
        "exact": False,
        "starts_with": False,
        "contains": False,
    }
