"""
Similarity and stylometric analysis for the Greek New Testament.

This module supports two complementary approaches:

1. Whole-book comparison
2. Fixed-token-window comparison

It also separates linguistic features into families:

    morphology
    function_words
    lexical
    part_of_speech
    structural
    combined

This makes it possible to determine whether relationships between books
remain stable across independent linguistic measurements.

Similarity is descriptive and does not establish authorship.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
)
from sklearn.preprocessing import StandardScaler

from src.loader import load_new_testament
from src.morphology import expand_morphology
from src.features import (
    analysis_matrix,
    build_feature_matrix,
    extract_book_features,
    determine_common_lemmas,
    PAULINE_BOOKS,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DEFAULT_WINDOW_SIZE = 500
DEFAULT_MIN_WINDOW_SIZE = 250


# ---------------------------------------------------------------------------
# Feature families
# ---------------------------------------------------------------------------

FEATURE_FAMILIES = {
    "morphology": (
        "verb_tense_",
        "verb_voice_",
        "verb_mood_",
        "verb_person_",
        "case_",
        "number_",
        "gender_",
        "verb_token_ratio",
    ),

    "function_words": (
        "function_",
    ),

    "lexical": (
        "lemma_",
        "type_token_ratio",
        "hapax_ratio",
        "dis_legomena_ratio",
        "average_lemma_length",
        "average_word_length",
    ),

    "part_of_speech": (
        "pos_",
    ),

    "structural": (
        "mean_tokens_per_verse",
        "std_tokens_per_verse",
    ),

    "style": (
        "verb_tense_",
        "verb_voice_",
        "verb_mood_",
        "verb_person_",
        "case_",
        "number_",
        "gender_",
        "verb_token_ratio",
        "function_",
        "pos_",
    ),
}


def select_feature_family(
    matrix: pd.DataFrame,
    family: str,
) -> pd.DataFrame:
    """
    Select a particular family of linguistic features.

    family may be:

        morphology
        function_words
        lexical
        part_of_speech
        structural
        combined
    """

    if family == "combined":
        return matrix.copy()

    if family not in FEATURE_FAMILIES:
        raise ValueError(
            f"Unknown feature family: {family}. "
            f"Available families: "
            f"{', '.join(FEATURE_FAMILIES)}, combined"
        )

    prefixes = FEATURE_FAMILIES[family]

    columns = []

    for column in matrix.columns:
        for prefix in prefixes:
            if column == prefix or column.startswith(prefix):
                columns.append(column)
                break

    if not columns:
        raise ValueError(
            f"No features found for family '{family}'."
        )

    return matrix[columns]


# ---------------------------------------------------------------------------
# Matrix preparation
# ---------------------------------------------------------------------------

def prepare_matrix(
    features: pd.DataFrame,
    family: str = "combined",
    standardize: bool = True,
) -> pd.DataFrame:
    """
    Prepare a feature matrix for analysis.
    """

    matrix = analysis_matrix(features)

    matrix = select_feature_family(
        matrix,
        family,
    )

    # Remove features with no variation.
    #
    # These contain no information for distinguishing samples and can
    # cause unnecessary noise during standardization.

    variable_columns = (
        matrix.nunique(dropna=False) > 1
    )

    matrix = matrix.loc[
        :,
        variable_columns,
    ]

    if matrix.empty:
        raise ValueError(
            f"No variable features available "
            f"for family '{family}'."
        )

    if not standardize:
        return matrix

    scaler = StandardScaler()

    values = scaler.fit_transform(matrix)

    return pd.DataFrame(
        values,
        index=matrix.index,
        columns=matrix.columns,
    )


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate cosine similarity between every pair of samples.
    """

    values = cosine_similarity(matrix)

    return pd.DataFrame(
        values,
        index=matrix.index,
        columns=matrix.index,
    )


def euclidean_distance_matrix(
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate Euclidean distance between every pair of samples.
    """

    values = euclidean_distances(matrix)

    return pd.DataFrame(
        values,
        index=matrix.index,
        columns=matrix.index,
    )


# ---------------------------------------------------------------------------
# Pair extraction
# ---------------------------------------------------------------------------

def similarity_pairs(
    similarity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert a square similarity matrix into unique sample pairs.
    """

    labels = similarity.index.tolist()

    rows = []

    for i, label_a in enumerate(labels):
        for j in range(i + 1, len(labels)):

            label_b = labels[j]

            rows.append({
                "sample_a": label_a,
                "sample_b": label_b,
                "similarity": float(
                    similarity.loc[
                        label_a,
                        label_b,
                    ]
                ),
            })

    return (
        pd.DataFrame(rows)
        .sort_values(
            "similarity",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def distance_pairs(
    distances: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert a square distance matrix into unique sample pairs.
    """

    labels = distances.index.tolist()

    rows = []

    for i, label_a in enumerate(labels):
        for j in range(i + 1, len(labels)):

            label_b = labels[j]

            rows.append({
                "sample_a": label_a,
                "sample_b": label_b,
                "distance": float(
                    distances.loc[
                        label_a,
                        label_b,
                    ]
                ),
            })

    return (
        pd.DataFrame(rows)
        .sort_values("distance")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Book ranking
# ---------------------------------------------------------------------------

def most_similar(
    similarity: pd.DataFrame,
    sample: str,
    n: int = 10,
) -> pd.Series:
    """
    Return samples most similar to a selected sample.
    """

    if sample not in similarity.index:
        raise ValueError(
            f"Unknown sample: {sample}"
        )

    return (
        similarity.loc[sample]
        .drop(labels=[sample])
        .sort_values(ascending=False)
        .head(n)
    )


# ---------------------------------------------------------------------------
# Token windows
# ---------------------------------------------------------------------------

def create_token_windows(
    df: pd.DataFrame,
    window_size: int = DEFAULT_WINDOW_SIZE,
    min_window_size: int = DEFAULT_MIN_WINDOW_SIZE,
) -> list[dict]:
    """
    Split every book into fixed-size token windows.

    Windows do not cross book boundaries.

    Example with window_size=500:

        Romans_001
        Romans_002
        Romans_003
        ...

    The final window is retained only if it contains at least
    min_window_size tokens.
    """

    windows = []

    grouped = df.groupby(
        ["book_number", "book"],
        sort=True,
    )

    for (book_number, book_name), book_df in grouped:

        book_df = book_df.reset_index(drop=True)

        total_tokens = len(book_df)

        window_number = 1

        for start in range(
            0,
            total_tokens,
            window_size,
        ):

            end = min(
                start + window_size,
                total_tokens,
            )

            window = book_df.iloc[
                start:end
            ].copy()

            if len(window) < min_window_size:
                continue

            sample_id = (
                f"{book_name}_{window_number:03d}"
            )

            windows.append({
                "sample_id": sample_id,
                "book": book_name,
                "book_number": book_number,
                "window": window_number,
                "start_token": start + 1,
                "end_token": end,
                "token_count": len(window),
                "data": window,
            })

            window_number += 1

    return windows


# ---------------------------------------------------------------------------
# Window feature matrix
# ---------------------------------------------------------------------------

def build_window_feature_matrix(
    df: pd.DataFrame,
    window_size: int = DEFAULT_WINDOW_SIZE,
    min_window_size: int = DEFAULT_MIN_WINDOW_SIZE,
    common_lemma_count: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build linguistic feature vectors for fixed-token windows.

    Returns:

        feature_matrix
        metadata
    """

    common_lemmas = determine_common_lemmas(
        df,
        top_n=common_lemma_count,
    )

    windows = create_token_windows(
        df,
        window_size=window_size,
        min_window_size=min_window_size,
    )

    feature_rows = []
    metadata_rows = []

    for window in windows:

        features = extract_book_features(
            window["data"],
            common_lemmas=common_lemmas,
        )

        features["sample_id"] = (
            window["sample_id"]
        )

        feature_rows.append(features)

        metadata_rows.append({
            "sample_id": window["sample_id"],
            "book": window["book"],
            "book_number": window["book_number"],
            "window": window["window"],
            "start_token": window["start_token"],
            "end_token": window["end_token"],
            "token_count": window["token_count"],
        })

    feature_matrix = pd.DataFrame(
        feature_rows
    )

    feature_matrix = feature_matrix.fillna(0.0)

    feature_matrix = feature_matrix.set_index(
        "sample_id"
    )

    metadata = pd.DataFrame(
        metadata_rows
    ).set_index(
        "sample_id"
    )

    return feature_matrix, metadata


# ---------------------------------------------------------------------------
# Cross-book window analysis
# ---------------------------------------------------------------------------

def cross_book_window_pairs(
    similarity: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return window similarities where the two windows belong to
    different books.

    This prevents within-book windows from dominating the top rankings.
    """

    pairs = similarity_pairs(
        similarity
    )

    book_lookup = metadata["book"].to_dict()

    pairs["book_a"] = (
        pairs["sample_a"]
        .map(book_lookup)
    )

    pairs["book_b"] = (
        pairs["sample_b"]
        .map(book_lookup)
    )

    pairs = pairs[
        pairs["book_a"]
        != pairs["book_b"]
    ]

    return pairs.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aggregate window similarity to book level
# ---------------------------------------------------------------------------

def aggregate_book_similarity(
    similarity: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate window-to-window similarities into book-to-book averages.

    For each pair of books, every window from book A is compared with every
    window from book B.

    The resulting score is the mean window similarity.
    """

    books = (
        metadata["book"]
        .drop_duplicates()
        .tolist()
    )

    result = pd.DataFrame(
        index=books,
        columns=books,
        dtype=float,
    )

    for book_a in books:

        samples_a = metadata[
            metadata["book"] == book_a
        ].index

        for book_b in books:

            samples_b = metadata[
                metadata["book"] == book_b
            ].index

            values = similarity.loc[
                samples_a,
                samples_b,
            ].to_numpy()

            if book_a == book_b:
                # Exclude each window's self-comparison of 1.0.
                mask = ~np.eye(
                    len(values),
                    dtype=bool,
                )

                values = values[mask]

            else:
                values = values.flatten()

            if len(values) == 0:
                result.loc[
                    book_a,
                    book_b,
                ] = np.nan
            else:
                result.loc[
                    book_a,
                    book_b,
                ] = float(
                    np.mean(values)
                )

    return result


# ---------------------------------------------------------------------------
# Feature-family analysis
# ---------------------------------------------------------------------------

def analyze_feature_families(
    features: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Calculate whole-book cosine similarity independently for each
    feature family.
    """

    results = {}

    families = [
        "morphology",
        "function_words",
        "lexical",
        "part_of_speech",
        "structural",
        "combined",
    ]

    for family in families:

        matrix = prepare_matrix(
            features,
            family=family,
            standardize=True,
        )

        results[family] = (
            cosine_similarity_matrix(
                matrix
            )
        )

    return results


# ---------------------------------------------------------------------------
# Consensus similarity
# ---------------------------------------------------------------------------

def consensus_similarity(
    family_results: dict[str, pd.DataFrame],
    exclude_combined: bool = True,
) -> pd.DataFrame:
    """
    Calculate mean similarity across independent feature families.

    This avoids relying on a single combined feature space.
    """

    matrices = []

    for family, matrix in family_results.items():

        if (
            exclude_combined
            and family == "combined"
        ):
            continue

        matrices.append(matrix)

    stacked = np.stack(
        [
            matrix.to_numpy()
            for matrix in matrices
        ]
    )

    mean_values = np.mean(
        stacked,
        axis=0,
    )

    reference = matrices[0]

    return pd.DataFrame(
        mean_values,
        index=reference.index,
        columns=reference.columns,
    )


# ---------------------------------------------------------------------------
# Pauline helpers
# ---------------------------------------------------------------------------

def pauline_subset(
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select the traditionally Pauline books from a square matrix.
    """

    books = [
        book
        for book in PAULINE_BOOKS
        if book in matrix.index
    ]

    return matrix.loc[
        books,
        books,
    ]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_matrix(
    matrix: pd.DataFrame,
    filename: str,
) -> None:
    """
    Save a DataFrame under outputs/.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix.to_csv(
        OUTPUT_DIR / filename
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    print()
    print("Greek New Testament Stylometric Analysis")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load corpus
    # ------------------------------------------------------------------

    print()
    print("Loading corpus...")

    nt = load_new_testament()

    print(
        f"Loaded {len(nt):,} Greek tokens."
    )

    print("Expanding morphology...")

    nt = expand_morphology(nt)

    # ------------------------------------------------------------------
    # Whole-book analysis
    # ------------------------------------------------------------------

    print()
    print("Building whole-book feature matrix...")

    book_features = build_feature_matrix(
        nt
    )

    print(
        f"{len(book_features)} books, "
        f"{len(book_features.columns)} features."
    )

    print()
    print("Analyzing independent feature families...")

    family_results = analyze_feature_families(
        book_features
    )

    for family, similarity in family_results.items():

        print(
            f"  {family:<20} "
            f"{len(similarity)} books"
        )

        save_matrix(
            similarity,
            f"book_similarity_{family}.csv",
        )

    # ------------------------------------------------------------------
    # Consensus
    # ------------------------------------------------------------------

    consensus = consensus_similarity(
        family_results
    )

    save_matrix(
        consensus,
        "book_similarity_consensus.csv",
    )

    consensus_pairs = similarity_pairs(
        consensus
    )

    save_matrix(
        consensus_pairs,
        "book_similarity_consensus_pairs.csv",
    )

    print()
    print("Top Whole-Book Consensus Similarities")
    print("=" * 80)

    print(
        consensus_pairs
        .head(20)
        .to_string(
            index=False,
            formatters={
                "similarity": lambda x: f"{x:.4f}"
            },
        )
    )

    # ------------------------------------------------------------------
    # Pauline whole-book comparison
    # ------------------------------------------------------------------

    paul = pauline_subset(
        consensus
    )

    save_matrix(
        paul,
        "pauline_similarity_consensus.csv",
    )

    print()
    print("Pauline Consensus Similarity")
    print("=" * 80)

    print(
        paul.round(3).to_string()
    )

    # ------------------------------------------------------------------
    # Token-window analysis
    # ------------------------------------------------------------------

    print()
    print(
        f"Building {DEFAULT_WINDOW_SIZE}-token windows..."
    )

    window_features, metadata = (
        build_window_feature_matrix(
            nt,
            window_size=DEFAULT_WINDOW_SIZE,
            min_window_size=DEFAULT_MIN_WINDOW_SIZE,
        )
    )

    print(
        f"Generated {len(window_features):,} "
        f"text samples."
    )

    save_matrix(
        metadata,
        "window_metadata.csv",
    )

    # ------------------------------------------------------------------
    # Analyze each feature family at window level
    # ------------------------------------------------------------------

    window_book_results = {}

    print()
    print("Window-Based Feature Families")
    print("=" * 80)

    for family in [
        "morphology",
        "function_words",
        "lexical",
        "part_of_speech",
        "combined",
    ]:

        matrix = prepare_matrix(
            window_features,
            family=family,
            standardize=True,
        )

        window_similarity = (
            cosine_similarity_matrix(
                matrix
            )
        )

        book_similarity = (
            aggregate_book_similarity(
                window_similarity,
                metadata,
            )
        )

        window_book_results[
            family
        ] = book_similarity

        save_matrix(
            book_similarity,
            f"window_book_similarity_{family}.csv",
        )

        print(
            f"{family:<20} "
            f"{matrix.shape[1]:>4} features"
        )

    # ------------------------------------------------------------------
    # Window-based consensus
    # ------------------------------------------------------------------

    window_consensus = consensus_similarity(
        window_book_results,
        exclude_combined=True,
    )

    save_matrix(
        window_consensus,
        "window_book_similarity_consensus.csv",
    )

    window_pairs = similarity_pairs(
        window_consensus
    )

    save_matrix(
        window_pairs,
        "window_book_similarity_consensus_pairs.csv",
    )

    print()
    print("Top Window-Based Book Similarities")
    print("=" * 80)

    print(
        window_pairs
        .head(20)
        .to_string(
            index=False,
            formatters={
                "similarity": lambda x: f"{x:.4f}"
            },
        )
    )

    # ------------------------------------------------------------------
    # Pauline window analysis
    # ------------------------------------------------------------------

    paul_window = pauline_subset(
        window_consensus
    )

    save_matrix(
        paul_window,
        "pauline_window_similarity.csv",
    )

    print()
    print("Pauline Window-Based Similarity")
    print("=" * 80)

    print(
        paul_window
        .round(3)
        .to_string()
    )

    # ------------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------------

    print()
    print("=" * 80)

    print(
        f"Analysis complete. Results written to:\n"
        f"    {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()