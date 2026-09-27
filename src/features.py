"""
Feature extraction for Greek New Testament stylometric analysis.

Turns each New Testament book into a numerical feature vector based on:

    - lexical diversity
    - vocabulary characteristics
    - part-of-speech distributions
    - verb morphology
    - noun/adjective morphology
    - grammatical case
    - grammatical number
    - grammatical gender
    - person
    - function-word frequencies
    - common lemma frequencies

These features can later be used for:

    - cosine similarity
    - distance calculations
    - hierarchical clustering
    - PCA
    - authorship/style comparison

The goal is to describe linguistic behavior rather than assume
traditional authorship classifications.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.loader import load_new_testament
from src.morphology import expand_morphology


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PAULINE_BOOKS = [
    "Romans",
    "1 Corinthians",
    "2 Corinthians",
    "Galatians",
    "Ephesians",
    "Philippians",
    "Colossians",
    "1 Thessalonians",
    "2 Thessalonians",
    "1 Timothy",
    "2 Timothy",
    "Titus",
    "Philemon",
]


# Function words and highly grammatical lemmas are useful for stylometry
# because they tend to be less dependent on subject matter than content words.
#
# We can refine this list considerably as the project develops.

FUNCTION_LEMMAS = [
    "καί",
    "δέ",
    "γάρ",
    "ἀλλά",
    "οὖν",
    "τε",
    "μέν",
    "εἰ",
    "ἐάν",
    "ὅτι",
    "ἵνα",
    "ὡς",
    "ὅταν",
    "μή",
    "οὐ",
    "οὐκ",
    "οὐχ",
    "ἐν",
    "εἰς",
    "ἐκ",
    "ἀπό",
    "διά",
    "κατά",
    "μετά",
    "περί",
    "πρός",
    "ὑπό",
    "ὑπέρ",
    "ἐπί",
    "παρά",
    "πρό",
    "σύν",
]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def safe_ratio(
    numerator: int | float,
    denominator: int | float,
) -> float:
    """
    Divide safely.

    Returns 0.0 when denominator is zero.
    """

    if denominator == 0:
        return 0.0

    return float(numerator) / float(denominator)


def normalized_counts(
    series: pd.Series,
) -> dict[str, float]:
    """
    Return normalized value counts for a Series.

    Missing values are ignored.

    Example:

        present       0.52
        aorist        0.31
        perfect       0.09
        ...
    """

    series = series.dropna()

    if series.empty:
        return {}

    counts = series.value_counts(normalize=True)

    return {
        str(key): float(value)
        for key, value in counts.items()
    }


def add_distribution_features(
    features: dict[str, float],
    prefix: str,
    series: pd.Series,
) -> None:
    """
    Add normalized categorical frequencies to a feature dictionary.

    Example:

        prefix = "tense"

    produces:

        tense_present
        tense_aorist
        tense_perfect
        ...
    """

    distribution = normalized_counts(series)

    for category, proportion in distribution.items():
        feature_name = f"{prefix}_{category}"
        features[feature_name] = proportion


# ---------------------------------------------------------------------------
# Lexical features
# ---------------------------------------------------------------------------

def lexical_features(
    book: pd.DataFrame,
) -> dict[str, float]:
    """
    Calculate basic lexical statistics for one book.
    """

    total_tokens = len(book)

    lemmas = book["lemma"].dropna()

    unique_lemmas = lemmas.nunique()

    lemma_counts = lemmas.value_counts()

    hapax = int((lemma_counts == 1).sum())
    dis_legomena = int((lemma_counts == 2).sum())

    return {
        "tokens": float(total_tokens),

        "unique_lemmas": float(unique_lemmas),

        "type_token_ratio": safe_ratio(
            unique_lemmas,
            total_tokens,
        ),

        "hapax_lemmas": float(hapax),

        "hapax_ratio": safe_ratio(
            hapax,
            unique_lemmas,
        ),

        "dis_legomena": float(dis_legomena),

        "dis_legomena_ratio": safe_ratio(
            dis_legomena,
            unique_lemmas,
        ),

        "average_lemma_length": (
            float(lemmas.str.len().mean())
            if not lemmas.empty
            else 0.0
        ),

        "average_word_length": (
            float(book["word"].dropna().str.len().mean())
            if not book["word"].dropna().empty
            else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Part-of-speech features
# ---------------------------------------------------------------------------

def pos_features(
    book: pd.DataFrame,
) -> dict[str, float]:
    """
    Calculate normalized part-of-speech distributions.
    """

    features: dict[str, float] = {}

    add_distribution_features(
        features,
        "pos",
        book["part_of_speech"],
    )

    return features


# ---------------------------------------------------------------------------
# Verb features
# ---------------------------------------------------------------------------

def verb_features(
    book: pd.DataFrame,
) -> dict[str, float]:
    """
    Calculate grammatical features for verbs.
    """

    features: dict[str, float] = {}

    verbs = book[
        book["part_of_speech"] == "verb"
    ]

    total_tokens = len(book)

    features["verb_token_ratio"] = safe_ratio(
        len(verbs),
        total_tokens,
    )

    add_distribution_features(
        features,
        "verb_tense",
        verbs["tense"],
    )

    add_distribution_features(
        features,
        "verb_voice",
        verbs["voice"],
    )

    add_distribution_features(
        features,
        "verb_mood",
        verbs["mood"],
    )

    add_distribution_features(
        features,
        "verb_person",
        verbs["person"],
    )

    return features


# ---------------------------------------------------------------------------
# Nominal features
# ---------------------------------------------------------------------------

def nominal_features(
    book: pd.DataFrame,
) -> dict[str, float]:
    """
    Analyze nouns, adjectives, articles and pronouns.
    """

    features: dict[str, float] = {}

    nominals = book[
        book["case"].notna()
    ]

    add_distribution_features(
        features,
        "case",
        nominals["case"],
    )

    add_distribution_features(
        features,
        "number",
        nominals["number"],
    )

    add_distribution_features(
        features,
        "gender",
        nominals["gender"],
    )

    return features


# ---------------------------------------------------------------------------
# Function-word features
# ---------------------------------------------------------------------------

def function_word_features(
    book: pd.DataFrame,
    lemmas: Iterable[str] = FUNCTION_LEMMAS,
) -> dict[str, float]:
    """
    Calculate frequency of selected grammatical/function lemmas.

    Frequencies are expressed per token.

    These are potentially useful stylometric features because function-word
    usage is often less topic-dependent than content vocabulary.
    """

    features: dict[str, float] = {}

    total_tokens = len(book)

    lemma_counts = book["lemma"].value_counts()

    for lemma in lemmas:
        count = int(lemma_counts.get(lemma, 0))

        features[f"function_{lemma}"] = safe_ratio(
            count,
            total_tokens,
        )

    return features


# ---------------------------------------------------------------------------
# Common lemma features
# ---------------------------------------------------------------------------

def determine_common_lemmas(
    df: pd.DataFrame,
    top_n: int = 50,
) -> list[str]:
    """
    Determine the most common content lemmas in the corpus.

    Function words are excluded because they are represented separately
    in the function-word feature family.
    """

    counts = df["lemma"].value_counts()

    counts = counts[
        ~counts.index.isin(FUNCTION_LEMMAS)
    ]

    return counts.head(top_n).index.tolist()

def common_lemma_features(
    book: pd.DataFrame,
    lemmas: Iterable[str],
) -> dict[str, float]:
    """
    Calculate normalized frequencies for common corpus lemmas.
    """

    features: dict[str, float] = {}

    total_tokens = len(book)

    lemma_counts = book["lemma"].value_counts()

    for lemma in lemmas:
        count = int(lemma_counts.get(lemma, 0))

        features[f"lemma_{lemma}"] = safe_ratio(
            count,
            total_tokens,
        )

    return features


# ---------------------------------------------------------------------------
# Structural features
# ---------------------------------------------------------------------------

def structural_features(
    book: pd.DataFrame,
) -> dict[str, float]:
    """
    Calculate simple structural statistics.

    These are not yet true syntactic features. Proper syntax will require
    a dependency/treebank dataset such as PROIEL.
    """

    verse_lengths = (
        book.groupby(["chapter", "verse"])
        .size()
    )

    chapter_lengths = (
        book.groupby("chapter")
        .size()
    )

    return {
        "chapters": float(
            book["chapter"].nunique()
        ),

        "verses": float(
            book.groupby(
                ["chapter", "verse"]
            ).ngroups
        ),

        "mean_tokens_per_verse": (
            float(verse_lengths.mean())
            if not verse_lengths.empty
            else 0.0
        ),

        "std_tokens_per_verse": (
            float(verse_lengths.std(ddof=0))
            if not verse_lengths.empty
            else 0.0
        ),

        "mean_tokens_per_chapter": (
            float(chapter_lengths.mean())
            if not chapter_lengths.empty
            else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Book feature extraction
# ---------------------------------------------------------------------------

def extract_book_features(
    book: pd.DataFrame,
    common_lemmas: Iterable[str] | None = None,
) -> dict[str, float]:
    """
    Extract all currently supported features for one book.
    """

    features: dict[str, float] = {}

    features.update(
        lexical_features(book)
    )

    features.update(
        pos_features(book)
    )

    features.update(
        verb_features(book)
    )

    features.update(
        nominal_features(book)
    )

    features.update(
        function_word_features(book)
    )

    features.update(
        structural_features(book)
    )

    if common_lemmas is not None:
        features.update(
            common_lemma_features(
                book,
                common_lemmas,
            )
        )

    return features


# ---------------------------------------------------------------------------
# Corpus feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    common_lemma_count: int = 50,
) -> pd.DataFrame:
    """
    Build one feature vector per New Testament book.

    Rows:
        books

    Columns:
        linguistic features
    """

    common_lemmas = determine_common_lemmas(
        df,
        top_n=common_lemma_count,
    )

    rows = []

    grouped = df.groupby(
        ["book_number", "book"],
        sort=True,
    )

    for (book_number, book_name), book_df in grouped:

        features = extract_book_features(
            book_df,
            common_lemmas=common_lemmas,
        )

        features["book_number"] = book_number
        features["book"] = book_name

        rows.append(features)

    matrix = pd.DataFrame(rows)

    matrix = matrix.fillna(0.0)

    matrix = matrix.sort_values(
        "book_number"
    )

    matrix = matrix.set_index("book")

    return matrix


# ---------------------------------------------------------------------------
# Analysis-ready matrix
# ---------------------------------------------------------------------------

def analysis_matrix(
    feature_matrix: pd.DataFrame,
    include_size_features: bool = False,
) -> pd.DataFrame:
    """
    Return only numerical features suitable for similarity/clustering.

    By default, raw size features are removed because book length should not
    cause two books to appear stylistically similar or different.
    """

    matrix = feature_matrix.copy()

    drop_columns = [
        "book_number",
    ]

    if not include_size_features:
        drop_columns.extend([
            "tokens",
            "unique_lemmas",
            "hapax_lemmas",
            "dis_legomena",
            "chapters",
            "verses",
            "mean_tokens_per_chapter",
        ])

    matrix = matrix.drop(
        columns=[
            column
            for column in drop_columns
            if column in matrix.columns
        ]
    )

    return matrix.select_dtypes(
        include=[np.number]
    )


# ---------------------------------------------------------------------------
# Pauline subset
# ---------------------------------------------------------------------------

def pauline_feature_matrix(
    feature_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return the traditionally Pauline corpus.

    This function only selects the books. It does not assert common authorship.
    """

    books = [
        book
        for book in PAULINE_BOOKS
        if book in feature_matrix.index
    ]

    return feature_matrix.loc[books]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("Loading Greek New Testament...")

    nt = load_new_testament()

    print(
        f"Loaded {len(nt):,} tokens."
    )

    print("Expanding morphology...")

    nt = expand_morphology(nt)

    print("Building linguistic feature matrix...")

    features = build_feature_matrix(nt)

    analysis = analysis_matrix(features)

    print()
    print("Feature Matrix")
    print("=" * 80)

    print(
        f"Books:    {len(features):,}"
    )

    print(
        f"Features: {len(analysis.columns):,}"
    )

    print()
    print("First 30 features")
    print("=" * 80)

    for column in list(analysis.columns)[:30]:
        print(column)

    print()
    print("Pauline Corpus")
    print("=" * 80)

    paul = pauline_feature_matrix(features)

    display_columns = [
        column
        for column in [
            "tokens",
            "unique_lemmas",
            "type_token_ratio",
            "hapax_ratio",
            "verb_token_ratio",
            "mean_tokens_per_verse",
        ]
        if column in paul.columns
    ]

    print(
        paul[display_columns]
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()