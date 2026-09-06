"""Property tests for the injection vectorisation pipeline.

Written before the implementation.

Two things are being tested and they fail differently.

**Indic text normalisation.** Devanagari has several byte sequences for the
same visible string - nukta composed or decomposed, zero-width joiners that
render identically, Devanagari digits beside ASCII ones. Two records that
look identical to a clerk must produce the same vector, or the model learns
an encoding artefact and every downstream grouping is wrong in a way nobody
can see. The tests assert equality of the *normalised form*, not of the
rendering.

**Injection vectorisation.** The generator injects defects and records them
per row. Turning that ledger into a supervision matrix is where an off-by-one
silently destroys a training run: the model trains happily against
misaligned labels and reports a plausible loss. So alignment is asserted
directly, and the multi-label shape is asserted rather than assumed - a row
can carry a missing field *and* a cost outlier, and collapsing that to one
class throws away most of the signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.stage8.injection import (
    DEVANAGARI_RANGE,
    InjectionMatrix,
    MultiTierVectorizer,
    channel_of,
    normalise_indic,
    script_profile,
    vectorise_injections,
)


# ===========================================================================
# 1. INDIC NORMALISATION
# ===========================================================================


class TestIndicNormalisation:
    """Same visible string, same normalised form."""

    def test_is_idempotent(self) -> None:
        """Normalising twice changes nothing.

        A non-idempotent normaliser means the vector depends on how many
        times the text passed through the pipeline.
        """
        for text in ["सड़क निर्माण", "Construction of CC road", "नाली NIRMAN 12"]:
            once = normalise_indic(text)
            assert normalise_indic(once) == once

    def test_nukta_forms_unify(self) -> None:
        """Precomposed and decomposed nukta are the same word.

        U+0958 and U+0915 U+093C render identically. Left alone they are two
        different tokens, so "सड़क" typed on two keyboards would never match.
        """
        precomposed = "क़"  # क़
        decomposed = "क़"  # क + nukta
        assert normalise_indic(precomposed) == normalise_indic(decomposed)

    def test_zero_width_joiners_removed(self) -> None:
        """ZWJ/ZWNJ are invisible and must not split a token."""
        assert normalise_indic("नि‍र्माण") == normalise_indic("निर्माण")
        assert normalise_indic("नि‌र्माण") == normalise_indic("निर्माण")

    def test_devanagari_digits_become_ascii(self) -> None:
        """०-९ and 0-9 are the same numbers.

        Ward numbers and years appear in both, and a model that treats them
        as unrelated cannot compare two records about the same ward.
        """
        assert normalise_indic("वार्ड १२३") == normalise_indic("वार्ड 123")

    def test_danda_is_punctuation(self) -> None:
        """Danda and double danda are sentence marks, not content."""
        assert normalise_indic("सड़क।") == normalise_indic("सड़क")
        assert normalise_indic("सड़क॥") == normalise_indic("सड़क")

    def test_case_and_whitespace_folded(self) -> None:
        assert normalise_indic("  CC   ROAD  ") == normalise_indic("cc road")

    @pytest.mark.parametrize("empty", ["", "   ", None, np.nan, "‍"])
    def test_empty_inputs_return_empty_not_crash(self, empty) -> None:
        """Missing text is empty, never an exception and never a fake token."""
        assert normalise_indic(empty) == ""

    def test_distinct_words_stay_distinct(self) -> None:
        """The precision side: normalisation must not merge real words."""
        assert normalise_indic("सड़क") != normalise_indic("नाली")
        assert normalise_indic("road") != normalise_indic("drain")


class TestScriptProfile:
    """Which script a field is written in, measured rather than assumed."""

    def test_pure_devanagari(self) -> None:
        profile = script_profile("सड़क निर्माण")
        assert profile["devanagari"] > 0.9
        assert profile["latin"] == 0.0
        assert profile["dominant"] == "devanagari"

    def test_pure_latin(self) -> None:
        profile = script_profile("Construction of road")
        assert profile["latin"] > 0.9
        assert profile["dominant"] == "latin"

    def test_code_mixed_is_flagged(self) -> None:
        """The case that actually appears: both scripts in one field.

        Flagged explicitly because it decides whether a monolingual encoder
        is usable at all on this corpus.
        """
        profile = script_profile("Nirman of सड़क ward 4")
        assert profile["dominant"] == "mixed"
        assert profile["devanagari"] > 0.0 and profile["latin"] > 0.0

    def test_empty_text_has_no_dominant_script(self) -> None:
        profile = script_profile("")
        assert profile["dominant"] == "none"
        assert profile["devanagari"] == 0.0

    def test_devanagari_range_constant_is_correct(self) -> None:
        low, high = DEVANAGARI_RANGE
        assert low <= ord("क") <= high
        assert not (low <= ord("a") <= high)


# ===========================================================================
# 2. MULTI-TIER VECTORISATION
# ===========================================================================


class TestMultiTierVectorizer:
    """Character tier must work on Devanagari without a language model."""

    @pytest.fixture()
    def texts(self) -> list:
        return [
            "construction of cc road ward 4",
            "सड़क निर्माण वार्ड 4",
            "nirman of sadak ward 4",
            "installation of hand pump",
            "हैंड पंप स्थापना",
        ] * 20

    def test_fit_transform_shape(self, texts) -> None:
        vectorizer = MultiTierVectorizer(max_features=200)
        matrix = vectorizer.fit_transform(texts)
        assert matrix.shape[0] == len(texts)
        assert matrix.shape[1] > 0

    def test_devanagari_produces_nonzero_features(self, texts) -> None:
        """The tier must not silently drop non-Latin text.

        A character vectoriser configured for ASCII returns an all-zero row
        for Devanagari, which looks like a missing value rather than a bug.
        """
        vectorizer = MultiTierVectorizer(max_features=200)
        vectorizer.fit(texts)
        devanagari = vectorizer.transform(["सड़क निर्माण"])
        assert devanagari.nnz > 0

    def test_identical_after_normalisation_gives_identical_vectors(self) -> None:
        """Encoding variants must collapse before vectorising, not after."""
        vectorizer = MultiTierVectorizer(max_features=200)
        vectorizer.fit(["सड़क निर्माण वार्ड १२"] * 40)
        left = vectorizer.transform(["सड़क निर्माण वार्ड १२"]).toarray()
        right = vectorizer.transform(["सड़क निर्माण वार्ड 12"]).toarray()
        np.testing.assert_array_almost_equal(left, right)

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            MultiTierVectorizer().transform(["x"])

    def test_empty_text_yields_a_defined_zero_row(self, texts) -> None:
        """Empty text is a defined all-zero row, and the mask says so.

        Zero and "no text" would otherwise be indistinguishable, which is the
        same undefined-is-not-zero error the rest of PARAKH guards against.
        """
        vectorizer = MultiTierVectorizer(max_features=200)
        vectorizer.fit(texts)
        matrix, defined = vectorizer.transform_with_mask(["", "road work"])
        assert defined.tolist() == [False, True]
        assert matrix[0].nnz == 0


# ===========================================================================
# 3. INJECTION VECTORISATION
# ===========================================================================


class TestInjectionVectorisation:
    """The ledger becomes a supervision matrix without losing alignment."""

    @pytest.fixture()
    def ledger(self) -> dict:
        return {
            "defects_by_row": {
                "0": ["missing:district"],
                "3": ["cost_outlier:high", "missing:vendor_name"],
                "7": ["date_order:approval_before_proposal"],
            }
        }

    def test_shape_is_rows_by_channels(self, ledger) -> None:
        result = vectorise_injections(ledger, n_rows=10)
        assert isinstance(result, InjectionMatrix)
        assert result.matrix.shape[0] == 10
        assert result.matrix.shape[1] == len(result.channels)

    def test_rows_align_with_the_ledger(self, ledger) -> None:
        """The off-by-one test. Misalignment trains happily and means nothing."""
        result = vectorise_injections(ledger, n_rows=10)
        assert result.matrix[0].sum() == 1
        assert result.matrix[3].sum() == 2
        assert result.matrix[7].sum() == 1
        for clean in (1, 2, 4, 5, 6, 8, 9):
            assert result.matrix[clean].sum() == 0

    def test_multi_label_is_preserved(self, ledger) -> None:
        """A row with two defects keeps both.

        Collapsing to a single class would discard most of the ledger: rows
        commonly carry a missing field and a value anomaly at once.
        """
        result = vectorise_injections(ledger, n_rows=10)
        position = {name: i for i, name in enumerate(result.channels)}
        assert result.matrix[3, position["cost_outlier"]] == 1
        assert result.matrix[3, position["missing"]] == 1

    def test_channel_of_splits_on_the_colon(self) -> None:
        assert channel_of("missing:district") == "missing"
        assert channel_of("cost_outlier:high") == "cost_outlier"
        assert channel_of("duplicate_work_id") == "duplicate_work_id"

    def test_clean_rows_are_derivable_and_not_a_channel(self, ledger) -> None:
        """"Clean" is the absence of every channel, never a column of its own.

        Making it a column would let a model predict "clean" directly and
        score well by ignoring every actual defect.
        """
        result = vectorise_injections(ledger, n_rows=10)
        assert "clean" not in result.channels
        assert result.clean_mask.sum() == 7

    def test_empty_ledger_gives_all_clean(self) -> None:
        result = vectorise_injections({"defects_by_row": {}}, n_rows=5)
        assert result.matrix.sum() == 0
        assert result.clean_mask.all()

    def test_row_beyond_n_rows_is_rejected(self) -> None:
        """A ledger referencing a row the frame does not have is a real error.

        Silently dropping it would mean training against a ledger that does
        not describe this corpus.
        """
        with pytest.raises(ValueError, match="out of range"):
            vectorise_injections({"defects_by_row": {"99": ["missing:x"]}}, n_rows=5)

    def test_channels_are_sorted_and_stable(self, ledger) -> None:
        """Column order must not depend on dict iteration order."""
        first = vectorise_injections(ledger, n_rows=10).channels
        second = vectorise_injections(ledger, n_rows=10).channels
        assert first == second == sorted(first)

    def test_prevalence_is_reported(self, ledger) -> None:
        """Per-channel counts, so a degenerate target is visible immediately.

        The first consistency experiment failed partly because 88% of rows
        carried a defect and nobody looked at the base rate first.
        """
        result = vectorise_injections(ledger, n_rows=10)
        assert result.prevalence["missing"] == 2
        assert result.prevalence["cost_outlier"] == 1
        assert all(0 <= v <= 10 for v in result.prevalence.values())
