"""Phase 6 (part 1): numbers/IDs for exact matching, and IndicXlit transliteration."""

import pytest

from app.retrieval.numbers import _pattern, extract_numbers


def test_numbers_and_ids_are_extracted_with_ascii_digits():
    assert extract_numbers("सर्व्हे क्र. १४२ मधील जमीन, दि. २१.०४.२०१५") == ["142", "21.04.2015"]
    assert extract_numbers("Survey No. 142 under section 4(1), 1 item") == ["142", "4(1)"]


def test_whole_number_matching_handles_indian_grouping():
    assert _pattern("142").search("सर्व्हे क्र. 142 मधील") and not _pattern("142").search("1420 आणि 138")
    assert _pattern("300000").search("उत्पन्न ₹3,00,000 पेक्षा")


def _xlit_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache("ai4bharat/IndicXlit",
                                                 "indicxlit-en-indic-v1.0/transformer/indicxlit.pt"), str)
    except Exception:
        return False


@pytest.mark.skipif(not _xlit_cached(), reason="IndicXlit weights are not in the Hugging Face cache")
def test_indicxlit_runs_the_official_weights_without_fairseq():
    from app.ml.xlit import IndicXlit

    x = IndicXlit()
    assert x.transliterate("gharkul", "mr") == "घरकुल"
    assert x.transliterate("yojana", "mr") == "योजना"
    assert x.transliterate("142", "mr") == "142"  # not a word: left as it is
