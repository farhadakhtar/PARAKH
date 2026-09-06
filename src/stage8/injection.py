"""Stage 8 - injection vectorisation and Indic-aware text encoding.

Two jobs, kept in one module because the training pipeline needs both at the
same point: turn text into vectors, and turn the generator's defect ledger
into a supervision matrix aligned to those vectors.

Indic normalisation, and why it is not cosmetic
-----------------------------------------------
Devanagari admits several byte sequences for the same visible string:

* **Nukta** - ``क़`` is either U+0958 or U+0915 U+093C. Both render
  identically. Untouched, "सड़क" typed on two keyboards is two tokens.
* **Zero-width joiners** - U+200C/U+200D are invisible and split a token in
  half for any character or word vectoriser.
* **Digits** - ``१२३`` and ``123`` are the same ward number.
* **Danda** - ``।`` is a full stop, not content.

A vectoriser fed unnormalised text learns the *keyboard*, not the language.
Nothing downstream can detect that: the vectors are dense and plausible, and
two records about the same road simply never match. So normalisation runs
before vectorisation, and the tests assert equality of normalised forms
rather than of renderings.

Unicode NFC is applied first. Note that NFC does **not** recompose the
U+0958-U+095F nukta block - those are on the composition-exclusion list - so
NFC reliably yields the *decomposed* form, which is why it is a usable
canonical target here.

What normalisation cannot do
-----------------------------
Romanised Hindi. "sadak" and "सड़क" are the same word and no amount of
Unicode normalisation will unify them, because the mapping is transliteration,
not encoding. The character tier below catches some of it by accident
(shared substrings with other romanised forms) and the rest needs pretrained
weights - see ``kaggle/parakh_nlp_train.py`` and its synonymy probe. This
module does not pretend otherwise.

Injection vectorisation
-----------------------
The generator injects defects and records them per row. Converting that to a
label matrix is where an off-by-one destroys a training run silently: the
model trains against misaligned labels, the loss curve looks fine, and the
result is noise. Alignment is therefore asserted directly in the tests, and
this module raises rather than dropping a ledger row it cannot place.

Labels are **multi-hot**. A row commonly carries a missing field *and* a
value anomaly, and collapsing that to one class discards most of the ledger.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.logger import get_logger

LOGGER = get_logger(__name__)

#: The Devanagari block. Used for script detection, which decides whether a
#: monolingual encoder is usable on a given corpus at all.
DEVANAGARI_RANGE: Tuple[int, int] = (0x0900, 0x097F)

#: Devanagari digits, in order, so they map positionally onto ASCII.
_DEVANAGARI_DIGITS = "०१२३४५६७८९"

#: Invisible characters that split tokens without changing the rendering.
_ZERO_WIDTH = "​‌‍﻿"

#: Danda and double danda: Devanagari sentence punctuation.
_DANDA = "।॥"

_TRANSLATION = {ord(c): None for c in _ZERO_WIDTH}
_TRANSLATION.update({ord(c): " " for c in _DANDA})
_TRANSLATION.update(
    {ord(d): str(i) for i, d in enumerate(_DEVANAGARI_DIGITS)}
)

_NULLISH = {"", "nan", "nat", "none", "null", "<na>"}


def normalise_indic(value: Any) -> str:
    """Canonical form of a possibly-Devanagari string.

    Order matters: NFC first so nukta sequences reach a single form, then
    remove invisibles, then fold digits and punctuation, then case and
    whitespace. Folding case before NFC would leave composed characters
    unnormalised.

    Args:
        value: Raw cell contents. Missing values are permitted.

    Returns:
        The normalised string, or ``""`` when the value asserts nothing.
        Empty rather than a sentinel, so it cannot become a token.
    """
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value)
    if text.strip().lower() in _NULLISH:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.translate(_TRANSLATION)
    text = text.lower()
    return " ".join(text.split())


def script_profile(value: Any) -> Dict[str, Any]:
    """Which scripts a string is written in, as proportions of its letters.

    Reported rather than assumed because it decides encoder choice: a corpus
    that is 40% code-mixed cannot be served by a monolingual model, and that
    fact should come from a measurement rather than from a guess about Indian
    public-works registers.

    Args:
        value: Raw text.

    Returns:
        ``devanagari`` and ``latin`` proportions, plus a ``dominant`` label of
        ``devanagari``, ``latin``, ``mixed`` or ``none``.
    """
    text = normalise_indic(value)
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return {"devanagari": 0.0, "latin": 0.0, "dominant": "none", "n_letters": 0}

    low, high = DEVANAGARI_RANGE
    devanagari = sum(1 for c in letters if low <= ord(c) <= high)
    latin = sum(1 for c in letters if "a" <= c <= "z")
    total = len(letters)
    devanagari_share = devanagari / total
    latin_share = latin / total

    # "Mixed" needs a real minority presence, not one stray character: a
    # single English digit-word in a Devanagari field is not code-mixing.
    if devanagari_share >= 0.1 and latin_share >= 0.1:
        dominant = "mixed"
    elif devanagari_share > latin_share:
        dominant = "devanagari"
    elif latin_share > 0:
        dominant = "latin"
    else:
        dominant = "none"

    return {
        "devanagari": devanagari_share,
        "latin": latin_share,
        "dominant": dominant,
        "n_letters": total,
    }


def corpus_script_report(texts: Iterable[Any]) -> Dict[str, Any]:
    """Script mix across a whole column, for the data inventory."""
    profiles = [script_profile(t) for t in texts]
    counts: Dict[str, int] = {}
    for profile in profiles:
        counts[profile["dominant"]] = counts.get(profile["dominant"], 0) + 1
    total = max(1, len(profiles))
    return {
        "n": len(profiles),
        "by_dominant_script": counts,
        "share_code_mixed": counts.get("mixed", 0) / total,
        "needs_multilingual_encoder": (
            counts.get("mixed", 0) + counts.get("devanagari", 0)
        )
        / total
        > 0.05,
    }


class MultiTierVectorizer:
    """Character n-grams over normalised text, with an explicit defined mask.

    Character n-grams rather than words, deliberately. Work registers are
    misspelled, code-mixed and inconsistently spaced; a word vocabulary
    fragments on exactly the rows that need matching, while ``char_wb``
    n-grams degrade gracefully and work on Devanagari with no language model
    and no downloads.

    This is the cheap tier. It handles encoding variation and shared
    substrings. It does **not** handle transliteration - "sadak" and "सड़क"
    share no characters - and that gap is the job of the pretrained tier in
    ``kaggle/parakh_nlp_train.py``, not of this class.
    """

    def __init__(
        self,
        *,
        ngram_range: Tuple[int, int] = (2, 4),
        max_features: int = 20000,
        min_df: int = 2,
    ) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            # Normalisation happens in normalise_indic, which is Unicode-aware.
            # sklearn's own lowercasing would run before that and is not.
            lowercase=False,
            preprocessor=None,
        )
        self._fitted = False

    @staticmethod
    def _prepare(texts: Sequence[Any]) -> List[str]:
        return [normalise_indic(text) for text in texts]

    def fit(self, texts: Sequence[Any]) -> "MultiTierVectorizer":
        self._vectorizer.fit(self._prepare(texts))
        self._fitted = True
        return self

    def fit_transform(self, texts: Sequence[Any]):
        matrix = self._vectorizer.fit_transform(self._prepare(texts))
        self._fitted = True
        return matrix

    def transform(self, texts: Sequence[Any]):
        if not self._fitted:
            raise RuntimeError("MultiTierVectorizer.fit must be called first")
        return self._vectorizer.transform(self._prepare(texts))

    def transform_with_mask(self, texts: Sequence[Any]):
        """Vectors plus a mask saying which rows actually carried text.

        An empty string yields an all-zero row, which is indistinguishable
        from a row whose n-grams all fell below ``min_df``. The mask keeps
        "no text" separate from "text with no known features" - the same
        undefined-is-not-zero discipline the rest of PARAKH uses.
        """
        prepared = self._prepare(texts)
        matrix = self.transform(prepared)
        defined = np.array([bool(text.strip()) for text in prepared], dtype=bool)
        return matrix, defined

    @property
    def vocabulary_size(self) -> int:
        if not self._fitted:
            return 0
        return len(self._vectorizer.vocabulary_)


def channel_of(label: str) -> str:
    """The defect family a ledger label belongs to.

    Ledger labels are ``family:detail`` (``missing:district``) or a bare
    family (``duplicate_work_id``). The family is the supervision target;
    the detail is usually too sparse to train on.
    """
    return str(label).split(":", 1)[0].strip()


@dataclass(frozen=True)
class InjectionMatrix:
    """Multi-hot defect labels aligned to corpus rows."""

    matrix: np.ndarray
    channels: List[str]
    prevalence: Dict[str, int]
    n_rows: int

    @property
    def clean_mask(self) -> np.ndarray:
        """Rows carrying no injected defect.

        Derived rather than stored as a column: a "clean" column would let a
        model score well by predicting it and ignoring every actual defect.
        """
        return self.matrix.sum(axis=1) == 0

    def channel_series(self, channel: str) -> np.ndarray:
        """Binary target for one channel, for a per-channel evaluation."""
        if channel not in self.channels:
            raise KeyError(f"unknown channel {channel!r}; have {self.channels}")
        return self.matrix[:, self.channels.index(channel)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_channels": len(self.channels),
            "channels": list(self.channels),
            "prevalence": dict(self.prevalence),
            "n_clean": int(self.clean_mask.sum()),
            "share_defective": float(1.0 - self.clean_mask.mean()),
        }


def vectorise_injections(
    ledger: Mapping[str, Any], n_rows: int
) -> InjectionMatrix:
    """Turn a defect ledger into a multi-hot supervision matrix.

    Args:
        ledger: The generator's ledger, carrying ``defects_by_row``.
        n_rows: Rows in the corpus the ledger describes.

    Returns:
        An :class:`InjectionMatrix` whose row *i* corresponds to corpus row
        *i*.

    Raises:
        ValueError: If the ledger references a row outside the corpus. This
            is raised rather than skipped: a ledger that does not describe
            this corpus must not be used to supervise training on it, and
            dropping the row would hide the mismatch.
    """
    by_row = ledger.get("defects_by_row", {}) or {}

    labels: Dict[int, List[str]] = {}
    for key, value in by_row.items():
        row = int(key)
        if not 0 <= row < n_rows:
            raise ValueError(
                f"ledger row {row} is out of range for a corpus of {n_rows} row(s)"
            )
        labels[row] = list(value) if isinstance(value, (list, tuple)) else [value]

    channels = sorted({channel_of(l) for values in labels.values() for l in values})
    position = {name: index for index, name in enumerate(channels)}

    matrix = np.zeros((n_rows, len(channels)), dtype=np.int8)
    for row, values in labels.items():
        for label in values:
            matrix[row, position[channel_of(label)]] = 1

    prevalence = {
        name: int(matrix[:, index].sum()) for name, index in position.items()
    }

    result = InjectionMatrix(
        matrix=matrix, channels=channels, prevalence=prevalence, n_rows=n_rows
    )
    LOGGER.info(
        "Stage 8 injections: %d row(s), %d channel(s), %.1f%% defective - %s",
        n_rows,
        len(channels),
        100.0 * (1.0 - result.clean_mask.mean()),
        prevalence,
    )
    return result


def trainable_channels(
    injections: InjectionMatrix,
    *,
    min_positive: int = 50,
    max_prevalence: float = 0.5,
) -> Dict[str, Any]:
    """Which channels are worth supervising, and why the others are not.

    Two ways a channel is useless as a target, and both were learned the
    expensive way:

    * **Too rare** - below ``min_positive`` the metric is noise.
    * **Too common** - a channel present on most rows is near-degenerate.
      The first consistency experiment scored ROC-AUC 0.538 against a target
      carrying 88% positives, and the number meant nothing.

    Args:
        injections: The label matrix.
        min_positive: Fewest positives worth training on.
        max_prevalence: Above this share, the channel is a base rate.

    Returns:
        Usable and rejected channels, each with its count and the reason.
    """
    usable: List[str] = []
    rejected: Dict[str, str] = {}

    for name in injections.channels:
        count = injections.prevalence[name]
        share = count / max(1, injections.n_rows)
        if count < min_positive:
            rejected[name] = f"only {count} positive(s); below {min_positive}"
        elif share > max_prevalence:
            rejected[name] = (
                f"{share:.1%} of rows carry it; a target this common is a base "
                "rate, not a signal"
            )
        else:
            usable.append(name)

    return {
        "usable": usable,
        "rejected": rejected,
        "n_usable": len(usable),
        "_note": (
            "Channels describe INJECTED DATA-QUALITY DEFECTS in a synthetic "
            "corpus. They are a valid target for a consistency model and are "
            "never a fraud label."
        ),
    }
