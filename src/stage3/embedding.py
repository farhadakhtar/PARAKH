"""Interpretable semantic embedding of ``work_name`` (Stage3.md sec.5).

TF-IDF, not a sentence transformer
----------------------------------
Stage3.md sec.5.2 suggests ``all-MiniLM-L6-v2``. TF-IDF is used instead, and
the reason is not convenience:

* **Every dimension is a token you can name.** An explanation can say "these two
  works are similar because both are 'cc road' works" - a 384-dimensional dense
  vector cannot say anything of the kind. For a system whose entire thesis is
  auditability, that is decisive.
* Fully deterministic, with no model download and no version drift.

The cost is no cross-lingual synonymy: "sadak" and "road" will not be
recognised as the same thing. On code-mixed registers that is a real
limitation, recorded as such.

What gets stripped, and why it matters
--------------------------------------
Two stopword families are removed before vectorising:

1. **Action boilerplate** - "construction of", "repair of". A repaired road and
   a constructed road are the same *kind* of work.
2. **Geography** - every district and state name in the corpus, plus locality
   markers. This one is load-bearing rather than cosmetic: every ``work_name``
   in an MPLADS-style register ends with its district, and leaving it in makes
   TF-IDF cluster by geography. District-level anomalies would then be
   normalised out of existence before Stage 4 ever saw them. The grouping
   features must stay disjoint from the testing features.

Measured effect on the 20k corpus: vocabulary fell from 3,556 features to 728,
and mean cluster purity against generator ground truth rose from unusable to
0.976.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.core.constants import (
    STAGE3_BOILERPLATE_TOKENS,
    STAGE3_LOCALITY_DELIMITERS,
    STAGE3_TRUNCATE_AT_LOCALITY,
    STAGE3_LOCALITY_TOKENS,
    STAGE3_SEED,
    STAGE3_STRIP_GEOGRAPHY,
    SVD_COMPONENTS,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TFIDF_SUBLINEAR_TF,
)
from src.core.logger import get_logger

LOGGER = get_logger(__name__)

_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_DIGITS = re.compile(r"^\d+$")


@dataclass(frozen=True)
class TextEmbedding:
    """TF-IDF vectors plus everything needed to explain them.

    Attributes:
        normalized_text: Post-stripping text actually vectorised, per record.
        tfidf: Sparse ``(n_unique, n_terms)`` matrix over **distinct** texts.
        projection: Dense L2-normalised ``(n_unique, d)`` SVD projection.
        unique_texts: The distinct normalised texts, in a fixed order.
        inverse: Position of each record's text within ``unique_texts``.
        vocabulary: Term -> column index.
        stopwords: The exact stopword set applied, for audit.
        empty_mask: Records whose work name was absent or fully stripped.
    """

    normalized_text: pd.Series
    tfidf: sparse.csr_matrix
    projection: np.ndarray
    unique_texts: pd.Index
    inverse: np.ndarray
    vocabulary: Dict[str, int]
    stopwords: frozenset
    empty_mask: pd.Series
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_unique(self) -> int:
        """Number of distinct normalised texts."""
        return int(len(self.unique_texts))

    def record_tfidf(self) -> sparse.csr_matrix:
        """Expand the per-text matrix to one row per record."""
        return self.tfidf[self.inverse]

    def record_projection(self) -> np.ndarray:
        """Expand the projection to one row per record."""
        return self.projection[self.inverse]

    def top_terms(self, rows: Sequence[int], k: int = 5) -> Tuple[str, ...]:
        """Highest-weight TF-IDF terms across a set of **unique-text** rows.

        This is what keeps the pipeline interpretable after dimensionality
        reduction: clusters are formed in SVD space but labelled in token space.

        Args:
            rows: Positions within ``unique_texts``.
            k: How many terms to return.

        Returns:
            Terms ordered by descending mean weight.
        """
        if not len(rows) or not self.vocabulary:
            return ()
        block = self.tfidf[list(rows)]
        weights = np.asarray(block.mean(axis=0)).ravel()
        if not weights.any():
            return ()
        terms = np.empty(len(self.vocabulary), dtype=object)
        for term, index in self.vocabulary.items():
            terms[index] = term
        order = np.argsort(-weights)[:k]
        return tuple(str(terms[i]) for i in order if weights[i] > 0)


def build_stopwords(
    frame: pd.DataFrame,
    strip_geography: bool = STAGE3_STRIP_GEOGRAPHY,
    extra: Optional[Sequence[str]] = None,
) -> frozenset:
    """Assemble the stopword set from constants plus the corpus's own geography.

    Geography terms are read from the data rather than hardcoded, so the same
    code works on any register without a curated gazetteer.

    Args:
        frame: Corpus records; ``district`` and ``state`` are read if present.
        strip_geography: Whether to add district and state tokens.
        extra: Additional tokens to strip.

    Returns:
        The complete stopword set.
    """
    stopwords = set(STAGE3_BOILERPLATE_TOKENS) | set(STAGE3_LOCALITY_TOKENS)
    if strip_geography:
        for column in ("district", "state"):
            if column not in frame.columns:
                continue
            for value in frame[column].dropna().unique():
                for token in _NON_ALNUM.sub(" ", str(value).lower()).split():
                    stopwords.add(token)
    if extra:
        stopwords.update(str(token).lower() for token in extra)
    return frozenset(stopwords)


def truncate_at_locality(
    text: str, delimiters: Sequence[str] = STAGE3_LOCALITY_DELIMITERS
) -> str:
    """Keep only the part of a work name that names the kind of work.

    "construction of cc road at ward no. 7, mysuru" -> "construction of cc road"

    Everything from the first locality delimiter onward is a place. Left in,
    village names split single work types into several clusters and make
    geography a grouping feature.

    Args:
        text: Lowercased work name.
        delimiters: Ordered locality delimiters.

    Returns:
        The head of the string, or the whole string when no delimiter is
        present - a name with no locality clause loses nothing.
    """
    cut = len(text)
    for delimiter in delimiters:
        position = text.find(delimiter)
        if position != -1:
            cut = min(cut, position)
    return text[:cut]


def normalize_work_text(
    series: pd.Series,
    stopwords: frozenset,
    truncate_at_locality_clause: bool = STAGE3_TRUNCATE_AT_LOCALITY,
    keep_digits: bool = False,
) -> pd.Series:
    """Reduce a work name to the tokens naming the kind of work.

    Order matters: truncate the locality clause first, then strip
    punctuation, stopwords and bare numbers. Stripping first would destroy
    the delimiters the truncation depends on.

    Args:
        series: Raw ``work_name`` values (already lowercased by Stage 1).
        stopwords: Tokens to drop.
        truncate_at_locality_clause: Cut at the first locality delimiter.
        keep_digits: Retain bare numbers. False for clustering, where a ward
            number is noise; True for duplicate detection, where it is the
            token that tells ward 11 from ward 35 and is therefore the whole
            signal.

    Returns:
        Object Series of normalised text; empty string where nothing survives.
    """
    if len(series) == 0:
        return pd.Series([], dtype="object", index=series.index)

    def _clean(value: Any) -> str:
        if value is None or value != value:
            return ""
        text = str(value).lower()
        if truncate_at_locality_clause:
            text = truncate_at_locality(text)
        tokens = _NON_ALNUM.sub(" ", text).split()
        kept = [
            token
            for token in tokens
            if token not in stopwords
            and (keep_digits or not _DIGITS.match(token))
        ]
        return " ".join(kept)

    return series.map(_clean)


def embed_work_names(
    frame: pd.DataFrame,
    text_field: str = "work_name",
    stopwords: Optional[frozenset] = None,
    ngram_range: Tuple[int, int] = TFIDF_NGRAM_RANGE,
    min_df: int = TFIDF_MIN_DF,
    sublinear_tf: bool = TFIDF_SUBLINEAR_TF,
    n_components: int = SVD_COMPONENTS,
    seed: int = STAGE3_SEED,
    truncate_locality: bool = STAGE3_TRUNCATE_AT_LOCALITY,
    keep_digits: bool = False,
    frozen_vocabulary: Optional[Mapping[str, int]] = None,
    frozen_idf: Optional[Sequence[float]] = None,
) -> TextEmbedding:
    """Vectorise work names deterministically.

    Vectorisation and projection run over **distinct** normalised texts, then
    map back to records. Public-works names are heavily templated - 20,000
    records reduce to 879 distinct strings - so this is both far cheaper and
    exactly equivalent for the vectoriser, since identical text yields an
    identical vector by construction.

    Args:
        frame: Corpus records.
        text_field: Column to embed.
        stopwords: Override the derived stopword set.
        ngram_range: Word n-gram range.
        min_df: Minimum document frequency, over distinct texts.
        sublinear_tf: Use ``1 + log(tf)`` term frequency.
        n_components: SVD width. Zero disables the projection.
        seed: Seed for the SVD solver.
        truncate_locality: Drop the locality clause. True for clustering,
            where the locality would split one work type across places;
            False for duplicate detection, where the locality is exactly
            what distinguishes two genuinely different roads.
        keep_digits: Retain bare numbers; see :func:`normalize_work_text`.
        frozen_vocabulary: A vocabulary from an earlier corpus. Supplying it
            pins the feature space so embeddings are comparable across runs.
        frozen_idf: IDF weights matching ``frozen_vocabulary``. Freezing the
            vocabulary alone is not enough - IDF is re-estimated from
            document frequencies, so the same token would carry a different
            weight on a different corpus and the embedding would still drift.

    Returns:
        A :class:`TextEmbedding`.

    Raises:
        ValueError: If ``text_field`` is absent from ``frame``.
    """
    if text_field not in frame.columns:
        raise ValueError(f"Column {text_field!r} is absent from the frame")

    resolved_stopwords = (
        stopwords if stopwords is not None else build_stopwords(frame)
    )
    normalized = normalize_work_text(
        frame[text_field],
        resolved_stopwords,
        truncate_at_locality_clause=truncate_locality,
        keep_digits=keep_digits,
    )
    empty_mask = (normalized.fillna("") == "").rename("empty_work_text")

    unique_texts = pd.Index(pd.unique(normalized.fillna("")))
    position = {text: index for index, text in enumerate(unique_texts)}
    inverse = normalized.fillna("").map(position).to_numpy(dtype="int64")

    n_unique = len(unique_texts)
    has_content = any(bool(text) for text in unique_texts)

    if not has_content:
        # Every name was absent or fully stripped. A zero-width space is the
        # honest representation: no term carries any weight.
        LOGGER.warning(
            "No work-name content survived normalisation; embedding is empty."
        )
        tfidf = sparse.csr_matrix((n_unique, 0), dtype="float64")
        projection = np.zeros((n_unique, max(n_components, 1)), dtype="float64")
        return TextEmbedding(
            normalized_text=normalized,
            tfidf=tfidf,
            projection=projection,
            unique_texts=unique_texts,
            inverse=inverse,
            vocabulary={},
            stopwords=resolved_stopwords,
            empty_mask=empty_mask,
            diagnostics={"degenerate": True, "n_unique": n_unique, "n_terms": 0},
        )

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=1 if frozen_vocabulary is not None else min_df,
        sublinear_tf=sublinear_tf,
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
        dtype=np.float64,
        vocabulary=dict(frozen_vocabulary)
        if frozen_vocabulary is not None
        else None,
    )
    tfidf = vectorizer.fit_transform(list(unique_texts))
    if frozen_idf is not None:
        # Restore the frozen weights and re-apply them, so a term means the
        # same thing here as it did in the corpus the artefact came from.
        weights = np.asarray(list(frozen_idf), dtype="float64")
        if weights.size != tfidf.shape[1]:
            raise ValueError(
                f"frozen_idf has {weights.size} weights for "
                f"{tfidf.shape[1]} vocabulary terms"
            )
        vectorizer.idf_ = weights
        tfidf = vectorizer.transform(list(unique_texts))
    vocabulary = {str(term): int(index) for term, index in vectorizer.vocabulary_.items()}

    width = min(n_components, max(tfidf.shape[1] - 1, 1))
    if n_components <= 0 or tfidf.shape[1] <= 1 or n_unique <= 2:
        projection = normalize(np.asarray(tfidf.todense(), dtype="float64"))
        svd_width = projection.shape[1]
        explained = 1.0
    else:
        svd = TruncatedSVD(n_components=width, random_state=seed)
        projection = normalize(svd.fit_transform(tfidf))
        svd_width = width
        explained = float(svd.explained_variance_ratio_.sum())

    LOGGER.info(
        "Embedded %d record(s) as %d distinct text(s); %d term(s) -> %d dim(s), "
        "%.1f%% variance retained.",
        len(frame),
        n_unique,
        tfidf.shape[1],
        svd_width,
        100 * explained,
    )

    return TextEmbedding(
        normalized_text=normalized,
        tfidf=tfidf.tocsr(),
        projection=np.ascontiguousarray(projection, dtype="float64"),
        unique_texts=unique_texts,
        inverse=inverse,
        vocabulary=vocabulary,
        stopwords=resolved_stopwords,
        empty_mask=empty_mask,
        diagnostics={
            "degenerate": False,
            "n_unique": n_unique,
            "n_terms": int(tfidf.shape[1]),
            "n_components": int(svd_width),
            "explained_variance_ratio": round(explained, 6),
            "n_stopwords": len(resolved_stopwords),
            "idf": tuple(float(v) for v in vectorizer.idf_),
            "frozen_vocabulary": frozen_vocabulary is not None,
            "frozen_idf": frozen_idf is not None,
            "empty_text_pct": round(100.0 * float(empty_mask.mean()), 4)
            if len(frame)
            else 0.0,
        },
    )
