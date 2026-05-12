"""Drug-name biasing post-processor."""

import pytest

from services.drug_bias import bias_word, bias_text, bias_result


class TestBiasWord:
    def test_exact_match_capitalized(self):
        assert bias_word("Atorvastatin") == "Atorvastatin"

    def test_exact_match_lowercase_returns_canonical(self):
        assert bias_word("atorvastatin") == "Atorvastatin"

    def test_close_typo_corrected(self):
        # 1 char off from "Metformin"
        assert bias_word("Metformn") == "Metformin"

    def test_one_insertion_away(self):
        # "Lisinopril" missing the 'i'
        assert bias_word("Lsinopril") == "Lisinopril"

    def test_too_far_unchanged(self):
        # 4+ edits away from any drug — should stay unchanged
        assert bias_word("Metoprolxxxxxxx") == "Metoprolxxxxxxx"

    def test_short_word_unchanged(self):
        # < 4 chars: skip fuzzy match entirely
        assert bias_word("the") == "the"
        assert bias_word("hi") == "hi"

    def test_empty_string(self):
        assert bias_word("") == ""

    def test_non_drug_word_unchanged(self):
        assert bias_word("hello") == "hello"
        assert bias_word("patient") == "patient"

    def test_dosage_not_collapsed_to_unit(self):
        # Regression: "10mg" should NOT fuzzy-match to "mg" (was a real bug)
        assert bias_word("10mg") == "10mg"
        assert bias_word("50mg") == "50mg"

    def test_preserves_trailing_punctuation(self):
        assert bias_word("Atorvastatin.") == "Atorvastatin."
        assert bias_word("Metformn,") == "Metformin,"

    def test_preserves_leading_paren(self):
        assert bias_word("(Atorvastatin)") == "(Atorvastatin)"


class TestBiasText:
    def test_multi_word_sentence(self):
        out = bias_text("Pt prescribed Metformn 50mg daily")
        assert "Metformin" in out
        assert "50mg" in out  # not collapsed

    def test_empty(self):
        assert bias_text("") == ""

    def test_no_drugs_unchanged(self):
        assert bias_text("the quick brown fox") == "the quick brown fox"


class TestBiasResult:
    def test_modifies_text_field(self):
        result = {"text": "Metformn", "words": []}
        bias_result(result)
        assert result["text"] == "Metformin"
        assert result["drug_bias"] is True

    def test_modifies_words(self):
        result = {
            "text": "Metformn",
            "words": [{"text": "Metformn", "confidence": 0.5}],
        }
        bias_result(result)
        assert result["words"][0]["text"] == "Metformin"

    def test_modifies_lines(self):
        result = {
            "text": "Atorvastain",
            "lines": [{"text": "Atorvastain", "confidence": 0.5}],
        }
        bias_result(result)
        assert result["lines"][0]["text"] == "Atorvastatin"

    def test_handles_missing_text(self):
        # Don't crash on results that lack the optional fields
        bias_result({"words": []})
        bias_result({"lines": []})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
