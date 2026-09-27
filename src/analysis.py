"""
Statistical style analysis for the Greek New Testament.

This module extends the exploratory clustering/PCA analysis by quantifying
relationships between books in the full multidimensional feature space.

The central question is not simply:

    "How far apart are two books?"

but:

    "How far apart are two books relative to the amount of stylistic
     variation that occurs naturally within those books?"

Analyses
--------
1. Fixed-token windows
2. Full standardized style feature space
3. Book centroid distances
4. Within-book dispersion
5. Normalized between-book separation
6. Cross-book window distances
7. Permutation tests
8. Bootstrap confidence intervals
9. Nearest-neighbor relationships
10. Multi-window-size sensitivity analysis
11. Pauline-specific analysis
12. CSV exports
13. Heatmaps

Important
---------
This is descriptive/statistical stylometry.

It does not establish authorship.

Observed differences can reflect:

    - authorship
    - genre
    - subject matter
    - audience
    - rhetorical situation
    - chronology
    - quotation patterns
    - editorial activity
    - amanuensis involvement
    - textual transmission
    - feature selection
    - window size

The analysis therefore reports linguistic similarity and separation,
not authorship conclusions.
"""

from pathlib import Path
from itertools import combinations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.spatial.distance import (
    cdist,
    pdist,
    squareform,
)

from sklearn.metrics import pairwise_distances

from src.loader import load_new_testament
from src.morphology import expand_morphology
from src.features import PAULINE_BOOKS

from src.similarity import (
    prepare_matrix,
    build_window_feature_matrix,
)


# ===========================================================================
# Configuration
# ===========================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis"
PLOT_DIR = OUTPUT_DIR / "plots"


# Main analysis window size.

DEFAULT_WINDOW_SIZE = 500
DEFAULT_MIN_WINDOW_SIZE = 250


# Sensitivity analysis.
#
# 500 is the primary analysis.
#
# Smaller windows:
#     more observations
#     noisier estimates
#
# Larger windows:
#     fewer observations
#     more stable estimates

WINDOW_SIZES = [
    250,
    500,
    750,
    1000,
]


# Minimum final partial-window size as a fraction of the requested
# window size.

MIN_WINDOW_FRACTION = 0.50


# Statistical resampling.

PERMUTATIONS = 5000
BOOTSTRAP_ITERATIONS = 2000

RANDOM_SEED = 42


# Primary feature family.

PRIMARY_FAMILY = "style"

# A book needs at least this many windows for within-book dispersion
# and normalized-separation statistics to be meaningful.

MIN_WINDOWS_FOR_DISPERSION = 2

# A pair must have valid normalized-separation measurements at this many
# different window sizes before it is treated as sufficiently supported
# for window-size robustness interpretation.

MIN_VALID_WINDOW_SIZES = 3

# ===========================================================================
# Utilities
# ===========================================================================

def ensure_output_directories() -> None:
    """
    Create analysis output directories.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def safe_filename(value: str) -> str:
    """
    Convert a string into a filesystem-friendly name.
    """

    return (
        value
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
    )


def ordered_books(
    metadata: pd.DataFrame,
) -> list[str]:
    """
    Return books in their first-occurrence order.
    """

    return list(
        dict.fromkeys(
            metadata["book"].tolist()
        )
    )


# ===========================================================================
# Window construction
# ===========================================================================

def build_windows(
    nt: pd.DataFrame,
    window_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build feature vectors for fixed-token windows.
    """

    min_window_size = max(
        1,
        int(
            window_size
            * MIN_WINDOW_FRACTION
        ),
    )

    features, metadata = (
        build_window_feature_matrix(
            nt,
            window_size=window_size,
            min_window_size=min_window_size,
        )
    )

    return features, metadata


# ===========================================================================
# Matrix preparation
# ===========================================================================

def prepare_style_matrix(
    window_features: pd.DataFrame,
    family: str = PRIMARY_FAMILY,
) -> pd.DataFrame:
    """
    Prepare the standardized multidimensional feature matrix.

    PCA is intentionally NOT applied here.

    Distances are calculated in the complete standardized feature space
    so information contained outside PC1/PC2 is retained.
    """

    return prepare_matrix(
        window_features,
        family=family,
        standardize=True,
    )


# ===========================================================================
# Book centroids
# ===========================================================================

def book_centroids(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the centroid of each book in feature space.
    """

    data = matrix.copy()

    data["book"] = metadata.loc[
        matrix.index,
        "book",
    ]

    return data.groupby(
        "book"
    ).mean()


# ===========================================================================
# Centroid distance matrix
# ===========================================================================

def centroid_distance_matrix(
    centroids: pd.DataFrame,
) -> pd.DataFrame:
    """
    Euclidean distance between book centroids.

    Because features are standardized, this represents distance in
    standardized linguistic feature space.
    """

    distances = pairwise_distances(
        centroids,
        metric="euclidean",
    )

    return pd.DataFrame(
        distances,
        index=centroids.index,
        columns=centroids.index,
    )


# ===========================================================================
# Within-book dispersion
# ===========================================================================

def within_book_dispersion(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Measure natural stylistic variation within each book.

    A book requires at least MIN_WINDOWS_FOR_DISPERSION windows for
    dispersion estimates to be considered valid.

    Books with fewer windows remain in the output but their dispersion
    statistics are reported as NaN. This prevents a one-window book from
    being treated as having zero internal stylistic variation.
    """

    records = []

    for book in ordered_books(metadata):

        indices = metadata.index[
            metadata["book"] == book
        ]

        indices = [
            index
            for index in indices
            if index in matrix.index
        ]

        subset = matrix.loc[
            indices
        ].to_numpy()

        n_windows = len(subset)

        if n_windows == 0:
            continue

        if n_windows < MIN_WINDOWS_FOR_DISPERSION:

            records.append(
                {
                    "book": book,
                    "n_windows": n_windows,
                    "dispersion_valid": False,
                    "mean_centroid_distance": np.nan,
                    "median_centroid_distance": np.nan,
                    "std_centroid_distance": np.nan,
                    "mean_pairwise_distance": np.nan,
                    "median_pairwise_distance": np.nan,
                }
            )

            continue

        centroid = subset.mean(
            axis=0
        )

        centroid_distances = np.linalg.norm(
            subset - centroid,
            axis=1,
        )

        pairwise = pdist(
            subset,
            metric="euclidean",
        )

        records.append(
            {
                "book": book,
                "n_windows": n_windows,
                "dispersion_valid": True,

                "mean_centroid_distance":
                    float(
                        centroid_distances.mean()
                    ),

                "median_centroid_distance":
                    float(
                        np.median(
                            centroid_distances
                        )
                    ),

                "std_centroid_distance":
                    float(
                        centroid_distances.std(
                            ddof=1
                        )
                    ),

                "mean_pairwise_distance":
                    float(
                        pairwise.mean()
                    ),

                "median_pairwise_distance":
                    float(
                        np.median(
                            pairwise
                        )
                    ),
            }
        )

    return (
        pd.DataFrame(records)
        .set_index("book")
    )

# ===========================================================================
# Cross-book window distances
# ===========================================================================

def cross_book_distance_statistics(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate distances between every pair of windows belonging to
    different books.

    This complements centroid distance.

    Two books may have similar centroids while having very different
    distributions, so cross-window distances provide additional information.
    """

    books = ordered_books(
        metadata
    )

    records = []

    for book_a, book_b in combinations(
        books,
        2,
    ):

        indices_a = metadata.index[
            metadata["book"] == book_a
        ]

        indices_b = metadata.index[
            metadata["book"] == book_b
        ]

        indices_a = [
            index
            for index in indices_a
            if index in matrix.index
        ]

        indices_b = [
            index
            for index in indices_b
            if index in matrix.index
        ]

        a = matrix.loc[
            indices_a
        ].to_numpy()

        b = matrix.loc[
            indices_b
        ].to_numpy()

        if len(a) == 0 or len(b) == 0:
            continue

        distances = cdist(
            a,
            b,
            metric="euclidean",
        ).ravel()

        records.append(
            {
                "book_a": book_a,
                "book_b": book_b,

                "n_windows_a": len(a),
                "n_windows_b": len(b),

                "mean_cross_distance":
                    float(
                        distances.mean()
                    ),

                "median_cross_distance":
                    float(
                        np.median(
                            distances
                        )
                    ),

                "std_cross_distance":
                    float(
                        distances.std(
                            ddof=1
                        )
                    )
                    if len(distances) > 1
                    else 0.0,
            }
        )

    return pd.DataFrame(
        records
    )


# ===========================================================================
# Normalized book separation
# ===========================================================================

def normalized_separation(
    centroid_distances: pd.DataFrame,
    dispersion: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize between-book centroid distance by natural within-book
    variation.

    Both books must have valid within-book dispersion estimates.

    If either book has too few windows, normalized separation is NaN.

    This is intentional: a centroid can still be calculated from a
    single window, but within-book variation cannot.
    """

    books = [
        book
        for book in centroid_distances.index
        if book in dispersion.index
    ]

    result = pd.DataFrame(
        np.nan,
        index=books,
        columns=books,
        dtype=float,
    )

    for book_a in books:

        for book_b in books:

            if book_a == book_b:

                if bool(
                    dispersion.loc[
                        book_a,
                        "dispersion_valid",
                    ]
                ):
                    result.loc[
                        book_a,
                        book_b,
                    ] = 0.0

                continue

            valid_a = bool(
                dispersion.loc[
                    book_a,
                    "dispersion_valid",
                ]
            )

            valid_b = bool(
                dispersion.loc[
                    book_b,
                    "dispersion_valid",
                ]
            )

            if not (
                valid_a
                and valid_b
            ):
                continue

            distance = centroid_distances.loc[
                book_a,
                book_b,
            ]

            dispersion_a = dispersion.loc[
                book_a,
                "mean_centroid_distance",
            ]

            dispersion_b = dispersion.loc[
                book_b,
                "mean_centroid_distance",
            ]

            pooled = np.sqrt(
                (
                    dispersion_a ** 2
                    + dispersion_b ** 2
                )
                / 2
            )

            if (
                not np.isfinite(pooled)
                or pooled <= 0
            ):
                continue

            result.loc[
                book_a,
                book_b,
            ] = (
                distance
                / pooled
            )

    return result

# ===========================================================================
# Pairwise long-form table
# ===========================================================================

def pairwise_book_statistics(
    centroid_distances: pd.DataFrame,
    normalized: pd.DataFrame,
    dispersion: pd.DataFrame,
    cross_distances: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine major pairwise metrics into one table.
    """

    cross_lookup = {}

    for _, row in cross_distances.iterrows():

        key = frozenset(
            [
                row["book_a"],
                row["book_b"],
            ]
        )

        cross_lookup[key] = row

    records = []

    books = list(
        centroid_distances.index
    )

    for book_a, book_b in combinations(
        books,
        2,
    ):

        key = frozenset(
            [
                book_a,
                book_b,
            ]
        )

        cross = cross_lookup.get(
            key
        )

        record = {
            "book_a": book_a,
            "book_b": book_b,

            "centroid_distance":
                centroid_distances.loc[
                    book_a,
                    book_b,
                ],

            "normalized_separation":
                normalized.loc[
                    book_a,
                    book_b,
                ],

            "within_dispersion_a":
                dispersion.loc[
                    book_a,
                    "mean_centroid_distance",
                ],

            "within_dispersion_b":
                dispersion.loc[
                    book_b,
                    "mean_centroid_distance",
                ],
        }

        if cross is not None:

            record[
                "mean_cross_distance"
            ] = cross[
                "mean_cross_distance"
            ]

            record[
                "median_cross_distance"
            ] = cross[
                "median_cross_distance"
            ]

        records.append(
            record
        )

    result = pd.DataFrame(
        records
    )

    return result.sort_values(
        "normalized_separation"
    )


# ===========================================================================
# Permutation test
# ===========================================================================

def permutation_test_pair(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    book_a: str,
    book_b: str,
    permutations: int = PERMUTATIONS,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """
    Permutation test for centroid separation between two books.

    Null hypothesis
    ---------------
    The windows from the two books are exchangeable with respect to
    book identity.

    Test statistic
    --------------
    Euclidean distance between book centroids.

    The labels are randomly permuted while preserving the original
    number of windows in each book.
    """

    rng = np.random.default_rng(
        random_seed
    )

    indices_a = metadata.index[
        metadata["book"] == book_a
    ]

    indices_b = metadata.index[
        metadata["book"] == book_b
    ]

    indices_a = [
        index
        for index in indices_a
        if index in matrix.index
    ]

    indices_b = [
        index
        for index in indices_b
        if index in matrix.index
    ]

    a = matrix.loc[
        indices_a
    ].to_numpy()

    b = matrix.loc[
        indices_b
    ].to_numpy()

    n_a = len(a)
    n_b = len(b)

    if n_a == 0 or n_b == 0:

        return {
            "book_a": book_a,
            "book_b": book_b,
            "n_a": n_a,
            "n_b": n_b,
            "observed_distance": np.nan,
            "permutation_p": np.nan,
        }

    observed = np.linalg.norm(
        a.mean(axis=0)
        - b.mean(axis=0)
    )

    combined = np.vstack(
        [
            a,
            b,
        ]
    )

    null_distances = np.empty(
        permutations,
        dtype=float,
    )

    for i in range(
        permutations
    ):

        permutation = rng.permutation(
            len(combined)
        )

        permuted_a = combined[
            permutation[:n_a]
        ]

        permuted_b = combined[
            permutation[n_a:]
        ]

        null_distances[i] = (
            np.linalg.norm(
                permuted_a.mean(axis=0)
                - permuted_b.mean(axis=0)
            )
        )

    # Add-one correction avoids reporting an impossible p=0.

    p_value = (
        (
            np.count_nonzero(
                null_distances
                >= observed
            )
            + 1
        )
        /
        (
            permutations
            + 1
        )
    )

    return {
        "book_a": book_a,
        "book_b": book_b,
        "n_a": n_a,
        "n_b": n_b,
        "observed_distance":
            float(observed),
        "permutation_p":
            float(p_value),
        "null_mean":
            float(
                null_distances.mean()
            ),
        "null_std":
            float(
                null_distances.std(
                    ddof=1
                )
            ),
    }


def permutation_tests(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    books: list[str],
    permutations: int = PERMUTATIONS,
) -> pd.DataFrame:
    """
    Run pairwise permutation tests for a collection of books.
    """

    records = []

    pairs = list(
        combinations(
            books,
            2,
        )
    )

    total = len(
        pairs
    )

    for number, (
        book_a,
        book_b,
    ) in enumerate(
        pairs,
        start=1,
    ):

        print(
            f"    permutation "
            f"{number}/{total}: "
            f"{book_a} vs {book_b}"
        )

        result = permutation_test_pair(
            matrix,
            metadata,
            book_a,
            book_b,
            permutations=permutations,
            random_seed=(
                RANDOM_SEED
                + number
            ),
        )

        records.append(
            result
        )

    results = pd.DataFrame(
        records
    )

    if results.empty:
        return results

    # Benjamini-Hochberg false-discovery-rate correction.

    results = add_bh_fdr(
        results,
        p_column="permutation_p",
    )

    return results.sort_values(
        [
            "permutation_p",
            "observed_distance",
        ],
        ascending=[
            True,
            False,
        ],
    )


# ===========================================================================
# Multiple-comparison correction
# ===========================================================================

def add_bh_fdr(
    dataframe: pd.DataFrame,
    p_column: str,
    output_column: str = "fdr_q",
) -> pd.DataFrame:
    """
    Benjamini-Hochberg FDR correction.
    """

    result = dataframe.copy()

    valid = result[
        p_column
    ].notna()

    p_values = result.loc[
        valid,
        p_column,
    ].to_numpy(
        dtype=float
    )

    if len(p_values) == 0:

        result[
            output_column
        ] = np.nan

        return result

    order = np.argsort(
        p_values
    )

    ranked = p_values[
        order
    ]

    m = len(
        ranked
    )

    adjusted = (
        ranked
        * m
        / np.arange(
            1,
            m + 1,
        )
    )

    # Enforce monotonicity.

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(
        adjusted,
        0,
        1,
    )

    restored = np.empty_like(
        adjusted
    )

    restored[
        order
    ] = adjusted

    result[
        output_column
    ] = np.nan

    result.loc[
        valid,
        output_column,
    ] = restored

    return result


# ===========================================================================
# Bootstrap confidence intervals
# ===========================================================================

def bootstrap_pair(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    book_a: str,
    book_b: str,
    iterations: int = BOOTSTRAP_ITERATIONS,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """
    Bootstrap confidence interval for centroid distance.

    Windows are resampled WITH replacement independently within each book.
    """

    rng = np.random.default_rng(
        random_seed
    )

    indices_a = metadata.index[
        metadata["book"] == book_a
    ]

    indices_b = metadata.index[
        metadata["book"] == book_b
    ]

    indices_a = [
        index
        for index in indices_a
        if index in matrix.index
    ]

    indices_b = [
        index
        for index in indices_b
        if index in matrix.index
    ]

    a = matrix.loc[
        indices_a
    ].to_numpy()

    b = matrix.loc[
        indices_b
    ].to_numpy()

    n_a = len(a)
    n_b = len(b)

    if n_a == 0 or n_b == 0:

        return {
            "book_a": book_a,
            "book_b": book_b,
            "bootstrap_mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    distances = np.empty(
        iterations,
        dtype=float,
    )

    for i in range(
        iterations
    ):

        sample_a = a[
            rng.integers(
                0,
                n_a,
                size=n_a,
            )
        ]

        sample_b = b[
            rng.integers(
                0,
                n_b,
                size=n_b,
            )
        ]

        distances[i] = (
            np.linalg.norm(
                sample_a.mean(axis=0)
                - sample_b.mean(axis=0)
            )
        )

    low, high = np.percentile(
        distances,
        [
            2.5,
            97.5,
        ],
    )

    return {
        "book_a": book_a,
        "book_b": book_b,

        "bootstrap_mean":
            float(
                distances.mean()
            ),

        "ci_low":
            float(low),

        "ci_high":
            float(high),
    }


def bootstrap_all_pairs(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    books: list[str],
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> pd.DataFrame:
    """
    Bootstrap all pairwise book centroid distances.
    """

    records = []

    pairs = list(
        combinations(
            books,
            2,
        )
    )

    total = len(
        pairs
    )

    for number, (
        book_a,
        book_b,
    ) in enumerate(
        pairs,
        start=1,
    ):

        print(
            f"    bootstrap "
            f"{number}/{total}: "
            f"{book_a} vs {book_b}"
        )

        records.append(
            bootstrap_pair(
                matrix,
                metadata,
                book_a,
                book_b,
                iterations=iterations,
                random_seed=(
                    RANDOM_SEED
                    + number
                ),
            )
        )

    return pd.DataFrame(
        records
    )


# ===========================================================================
# Nearest neighbors
# ===========================================================================

def nearest_neighbors(
    distance_matrix: pd.DataFrame,
    n: int = 5,
) -> pd.DataFrame:
    """
    Find the nearest books to each book according to centroid distance.
    """

    records = []

    for book in distance_matrix.index:

        distances = (
            distance_matrix.loc[
                book
            ]
            .drop(
                labels=[
                    book
                ]
            )
            .sort_values()
            .head(n)
        )

        for rank, (
            neighbor,
            distance,
        ) in enumerate(
            distances.items(),
            start=1,
        ):

            records.append(
                {
                    "book": book,
                    "rank": rank,
                    "neighbor": neighbor,
                    "distance":
                        float(distance),
                }
            )

    return pd.DataFrame(
        records
    )


# ===========================================================================
# Pauline subset
# ===========================================================================

def available_pauline_books(
    metadata: pd.DataFrame,
) -> list[str]:
    """
    Return Pauline books actually represented in the current windows.
    """

    available = set(
        metadata["book"]
    )

    return [
        book
        for book in PAULINE_BOOKS
        if book in available
    ]


def subset_matrix_and_metadata(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    books: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Restrict matrix and metadata to selected books.
    """

    indices = metadata.index[
        metadata["book"].isin(
            books
        )
    ]

    indices = [
        index
        for index in indices
        if index in matrix.index
    ]

    return (
        matrix.loc[
            indices
        ],
        metadata.loc[
            indices
        ],
    )


# ===========================================================================
# Heatmaps
# ===========================================================================

def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    output_file: Path,
    label: str,
) -> None:
    """
    Plot a matrix heatmap.

    No fixed color scheme is specified so matplotlib's default
    rendering is used.
    """

    if matrix.empty:
        return

    fig, ax = plt.subplots(
        figsize=(14, 12)
    )

    image = ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
    )

    ax.set_xticks(
        range(
            len(
                matrix.columns
            )
        )
    )

    ax.set_xticklabels(
        matrix.columns,
        rotation=90,
        fontsize=8,
    )

    ax.set_yticks(
        range(
            len(
                matrix.index
            )
        )
    )

    ax.set_yticklabels(
        matrix.index,
        fontsize=8,
    )

    ax.set_title(
        title
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        label
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ===========================================================================
# Single-window-size analysis
# ===========================================================================

def analyze_window_size(
    nt: pd.DataFrame,
    window_size: int,
    run_resampling: bool = False,
) -> dict:
    """
    Run complete style analysis for one token-window size.

    Expensive permutation/bootstrap tests are normally run only for the
    primary window size.
    """

    print()
    print("=" * 80)
    print(
        f"{window_size}-TOKEN STYLE ANALYSIS"
    )
    print("=" * 80)

    window_features, metadata = (
        build_windows(
            nt,
            window_size,
        )
    )

    print(
        f"Generated "
        f"{len(window_features):,} windows."
    )

    matrix = prepare_style_matrix(
        window_features,
        family=PRIMARY_FAMILY,
    )

    print(
        f"Style matrix: "
        f"{matrix.shape[0]} windows x "
        f"{matrix.shape[1]} features"
    )

    # ------------------------------------------------------------------
    # Whole NT
    # ------------------------------------------------------------------

    centroids = book_centroids(
        matrix,
        metadata,
    )

    centroid_distances = (
        centroid_distance_matrix(
            centroids
        )
    )

    dispersion = (
        within_book_dispersion(
            matrix,
            metadata,
        )
    )

    normalized = (
        normalized_separation(
            centroid_distances,
            dispersion,
        )
    )

    cross_distances = (
        cross_book_distance_statistics(
            matrix,
            metadata,
        )
    )

    pairwise = (
        pairwise_book_statistics(
            centroid_distances,
            normalized,
            dispersion,
            cross_distances,
        )
    )

    neighbors = nearest_neighbors(
        centroid_distances,
        n=5,
    )

    # ------------------------------------------------------------------
    # Save whole-NT results
    # ------------------------------------------------------------------

    prefix = (
        f"{window_size}_token"
    )

    centroids.to_csv(
        OUTPUT_DIR
        / f"{prefix}_book_centroids.csv"
    )

    centroid_distances.to_csv(
        OUTPUT_DIR
        / f"{prefix}_centroid_distances.csv"
    )

    dispersion.to_csv(
        OUTPUT_DIR
        / f"{prefix}_within_book_dispersion.csv"
    )

    normalized.to_csv(
        OUTPUT_DIR
        / f"{prefix}_normalized_separation.csv"
    )

    cross_distances.to_csv(
        OUTPUT_DIR
        / f"{prefix}_cross_book_distances.csv",
        index=False,
    )

    pairwise.to_csv(
        OUTPUT_DIR
        / f"{prefix}_pairwise_statistics.csv",
        index=False,
    )

    neighbors.to_csv(
        OUTPUT_DIR
        / f"{prefix}_nearest_neighbors.csv",
        index=False,
    )

    plot_heatmap(
        centroid_distances,
        title=(
            f"{window_size}-Token "
            "Greek NT Style Centroid Distance"
        ),
        output_file=(
            PLOT_DIR
            / f"{prefix}_centroid_distance.png"
        ),
        label="Euclidean distance",
    )

    plot_heatmap(
        normalized,
        title=(
            f"{window_size}-Token "
            "Greek NT Normalized Style Separation"
        ),
        output_file=(
            PLOT_DIR
            / f"{prefix}_normalized_separation.png"
        ),
        label="Normalized separation",
    )

    # ------------------------------------------------------------------
    # Pauline corpus
    # ------------------------------------------------------------------

    paul_books = (
        available_pauline_books(
            metadata
        )
    )

    paul_matrix, paul_metadata = (
        subset_matrix_and_metadata(
            matrix,
            metadata,
            paul_books,
        )
    )

    paul_centroids = book_centroids(
        paul_matrix,
        paul_metadata,
    )

    paul_centroid_distances = (
        centroid_distance_matrix(
            paul_centroids
        )
    )

    paul_dispersion = (
        within_book_dispersion(
            paul_matrix,
            paul_metadata,
        )
    )

    paul_normalized = (
        normalized_separation(
            paul_centroid_distances,
            paul_dispersion,
        )
    )

    paul_cross = (
        cross_book_distance_statistics(
            paul_matrix,
            paul_metadata,
        )
    )

    paul_pairwise = (
        pairwise_book_statistics(
            paul_centroid_distances,
            paul_normalized,
            paul_dispersion,
            paul_cross,
        )
    )

    paul_neighbors = nearest_neighbors(
        paul_centroid_distances,
        n=min(
            5,
            max(
                1,
                len(paul_books) - 1,
            ),
        ),
    )

    paul_centroids.to_csv(
        OUTPUT_DIR
        / f"{prefix}_pauline_centroids.csv"
    )

    paul_centroid_distances.to_csv(
        OUTPUT_DIR
        / f"{prefix}_pauline_centroid_distances.csv"
    )

    paul_dispersion.to_csv(
        OUTPUT_DIR
        / f"{prefix}_pauline_dispersion.csv"
    )

    paul_normalized.to_csv(
        OUTPUT_DIR
        / f"{prefix}_pauline_normalized_separation.csv"
    )

    paul_cross.to_csv(
        OUTPUT_DIR
        / f"{prefix}_pauline_cross_distances.csv",
        index=False,
    )

    paul_pairwise.to_csv(
        OUTPUT_DIR
        / f"{prefix}_pauline_pairwise_statistics.csv",
        index=False,
    )

    paul_neighbors.to_csv(
        OUTPUT_DIR
        / f"{prefix}_pauline_nearest_neighbors.csv",
        index=False,
    )

    plot_heatmap(
        paul_centroid_distances,
        title=(
            f"Pauline Corpus "
            f"{window_size}-Token "
            "Style Centroid Distance"
        ),
        output_file=(
            PLOT_DIR
            / f"{prefix}_pauline_centroid_distance.png"
        ),
        label="Euclidean distance",
    )

    plot_heatmap(
        paul_normalized,
        title=(
            f"Pauline Corpus "
            f"{window_size}-Token "
            "Normalized Style Separation"
        ),
        output_file=(
            PLOT_DIR
            / f"{prefix}_pauline_normalized_separation.png"
        ),
        label="Normalized separation",
    )

    # ------------------------------------------------------------------
    # Resampling statistics
    # ------------------------------------------------------------------

    permutation_results = None
    bootstrap_results = None

    if run_resampling:

        print()
        print(
            "Running Pauline permutation tests..."
        )

        permutation_results = (
            permutation_tests(
                paul_matrix,
                paul_metadata,
                paul_books,
                permutations=PERMUTATIONS,
            )
        )

        permutation_results.to_csv(
            OUTPUT_DIR
            / f"{prefix}_pauline_permutation_tests.csv",
            index=False,
        )

        print()
        print(
            "Running Pauline bootstrap intervals..."
        )

        bootstrap_results = (
            bootstrap_all_pairs(
                paul_matrix,
                paul_metadata,
                paul_books,
                iterations=BOOTSTRAP_ITERATIONS,
            )
        )

        bootstrap_results.to_csv(
            OUTPUT_DIR
            / f"{prefix}_pauline_bootstrap.csv",
            index=False,
        )

        # Merge everything into one master table.

        master = paul_pairwise.merge(
            permutation_results,
            on=[
                "book_a",
                "book_b",
            ],
            how="left",
            suffixes=(
                "",
                "_permutation",
            ),
        )

        master = master.merge(
            bootstrap_results,
            on=[
                "book_a",
                "book_b",
            ],
            how="left",
        )

        master = master.sort_values(
            "normalized_separation"
        )

        master.to_csv(
            OUTPUT_DIR
            / f"{prefix}_pauline_master_results.csv",
            index=False,
        )

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    print()
    print(
        "Closest Pauline centroid pairs:"
    )

    print(
        paul_pairwise[
            [
                "book_a",
                "book_b",
                "centroid_distance",
                "normalized_separation",
            ]
        ]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print()
    print(
        "Most separated Pauline centroid pairs:"
    )

    print(
        paul_pairwise[
            [
                "book_a",
                "book_b",
                "centroid_distance",
                "normalized_separation",
            ]
        ]
        .tail(10)
        .sort_values(
            "normalized_separation",
            ascending=False,
        )
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    return {
        "window_size":
            window_size,

        "n_windows":
            len(window_features),

        "n_features":
            matrix.shape[1],

        "pairwise":
            pairwise,

        "pauline_pairwise":
            paul_pairwise,

        "pauline_neighbors":
            paul_neighbors,

        "permutation":
            permutation_results,

        "bootstrap":
            bootstrap_results,
    }


# ===========================================================================
# Window-size sensitivity
# ===========================================================================

def build_window_sensitivity(
    analyses: list[dict],
) -> pd.DataFrame:
    """
    Combine Pauline pairwise results across window sizes.

    This allows us to ask whether a relationship persists when the
    arbitrary window-size choice changes.
    """

    frames = []

    for analysis in analyses:

        frame = (
            analysis[
                "pauline_pairwise"
            ]
            .copy()
        )

        frame[
            "window_size"
        ] = analysis[
            "window_size"
        ]

        frames.append(
            frame
        )

    if not frames:

        return pd.DataFrame()

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    return combined


def summarize_window_sensitivity(
    sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize pairwise normalized separation across window sizes.

    Only finite normalized-separation measurements contribute to the
    robustness summary.

    A pair is considered sufficiently supported for robustness
    interpretation only when it has valid measurements at
    MIN_VALID_WINDOW_SIZES or more window sizes.
    """

    if sensitivity.empty:
        return pd.DataFrame()

    records = []

    for (
        book_a,
        book_b,
    ), group in sensitivity.groupby(
        [
            "book_a",
            "book_b",
        ],
        sort=False,
    ):

        values = group[
            "normalized_separation"
        ]

        valid_mask = np.isfinite(
            values
        )

        valid = group.loc[
            valid_mask
        ].copy()

        n_tested = group[
            "window_size"
        ].nunique()

        n_valid = valid[
            "window_size"
        ].nunique()

        valid_window_sizes = (
            sorted(
                valid[
                    "window_size"
                ]
                .unique()
                .tolist()
            )
        )

        missing_window_sizes = (
            sorted(
                set(
                    group[
                        "window_size"
                    ].unique()
                )
                - set(
                    valid_window_sizes
                )
            )
        )

        sufficiently_supported = (
            n_valid
            >= MIN_VALID_WINDOW_SIZES
        )

        if n_valid > 0:

            normalized_values = valid[
                "normalized_separation"
            ]

            mean_separation = float(
                normalized_values.mean()
            )

            min_separation = float(
                normalized_values.min()
            )

            max_separation = float(
                normalized_values.max()
            )

            separation_range = (
                max_separation
                - min_separation
            )

            if n_valid >= 2:

                std_separation = float(
                    normalized_values.std(
                        ddof=1
                    )
                )

            else:

                std_separation = np.nan

            mean_centroid_distance = float(
                valid[
                    "centroid_distance"
                ].mean()
            )

            if n_valid >= 2:

                std_centroid_distance = float(
                    valid[
                        "centroid_distance"
                    ].std(
                        ddof=1
                    )
                )

            else:

                std_centroid_distance = np.nan

        else:

            mean_separation = np.nan
            std_separation = np.nan
            min_separation = np.nan
            max_separation = np.nan
            separation_range = np.nan
            mean_centroid_distance = np.nan
            std_centroid_distance = np.nan

        records.append(
            {
                "book_a":
                    book_a,

                "book_b":
                    book_b,

                "mean_centroid_distance":
                    mean_centroid_distance,

                "std_centroid_distance":
                    std_centroid_distance,

                "mean_normalized_separation":
                    mean_separation,

                "std_normalized_separation":
                    std_separation,

                "min_normalized_separation":
                    min_separation,

                "max_normalized_separation":
                    max_separation,

                "separation_range":
                    separation_range,

                "n_window_sizes_tested":
                    n_tested,

                "n_valid_window_sizes":
                    n_valid,

                "valid_window_sizes":
                    ",".join(
                        str(x)
                        for x in valid_window_sizes
                    ),

                "missing_window_sizes":
                    ",".join(
                        str(x)
                        for x in missing_window_sizes
                    ),

                "robustness_supported":
                    sufficiently_supported,
            }
        )

    result = pd.DataFrame(
        records
    )

    return result.sort_values(
        [
            "robustness_supported",
            "mean_normalized_separation",
        ],
        ascending=[
            False,
            True,
        ],
        na_position="last",
    )

# ===========================================================================
# Main
# ===========================================================================
def main() -> None:

    ensure_output_directories()

    print()
    print(
        "Greek New Testament "
        "Multidimensional Style Analysis"
    )
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load corpus
    # ------------------------------------------------------------------

    print()
    print(
        "Loading Greek New Testament..."
    )

    nt = load_new_testament()

    print(
        f"Loaded "
        f"{len(nt):,} tokens."
    )

    print(
        "Expanding morphology..."
    )

    nt = expand_morphology(
        nt
    )

    print(
        "Morphology ready."
    )

    # ------------------------------------------------------------------
    # Analyze multiple window sizes
    # ------------------------------------------------------------------

    analyses = []

    for window_size in WINDOW_SIZES:

        # Run expensive permutation/bootstrap analysis only for the
        # primary 500-token model.

        run_resampling = (
            window_size
            == DEFAULT_WINDOW_SIZE
        )

        result = analyze_window_size(
            nt,
            window_size=window_size,
            run_resampling=run_resampling,
        )

        analyses.append(
            result
        )

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "WINDOW-SIZE SENSITIVITY"
    )
    print("=" * 80)

    sensitivity = (
        build_window_sensitivity(
            analyses
        )
    )

    sensitivity.to_csv(
        OUTPUT_DIR
        / "pauline_window_size_sensitivity_raw.csv",
        index=False,
    )

    sensitivity_summary = (
        summarize_window_sensitivity(
            sensitivity
        )
    )

    sensitivity_summary.to_csv(
        OUTPUT_DIR
        / "pauline_window_size_sensitivity_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Separate sufficiently supported and insufficiently supported pairs
    # ------------------------------------------------------------------

    supported = sensitivity_summary[
        sensitivity_summary[
            "robustness_supported"
        ]
    ].copy()

    unsupported = sensitivity_summary[
        ~sensitivity_summary[
            "robustness_supported"
        ]
    ].copy()

    # ------------------------------------------------------------------
    # Most consistently similar pairs
    # ------------------------------------------------------------------

    print()
    print(
        "Most consistently similar Pauline pairs "
        f"(>= {MIN_VALID_WINDOW_SIZES} valid window sizes):"
    )

    if supported.empty:

        print(
            "    No pairs meet the robustness threshold."
        )

    else:

        print(
            supported[
                [
                    "book_a",
                    "book_b",
                    "mean_normalized_separation",
                    "std_normalized_separation",
                    "separation_range",
                    "n_valid_window_sizes",
                    "valid_window_sizes",
                ]
            ]
            .sort_values(
                "mean_normalized_separation",
                ascending=True,
            )
            .head(15)
            .to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )

    # ------------------------------------------------------------------
    # Most consistently separated pairs
    # ------------------------------------------------------------------

    print()
    print(
        "Most consistently separated Pauline pairs "
        f"(>= {MIN_VALID_WINDOW_SIZES} valid window sizes):"
    )

    if supported.empty:

        print(
            "    No pairs meet the robustness threshold."
        )

    else:

        print(
            supported[
                [
                    "book_a",
                    "book_b",
                    "mean_normalized_separation",
                    "std_normalized_separation",
                    "separation_range",
                    "n_valid_window_sizes",
                    "valid_window_sizes",
                ]
            ]
            .sort_values(
                "mean_normalized_separation",
                ascending=False,
            )
            .head(15)
            .to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )

    # ------------------------------------------------------------------
    # Pairs lacking sufficient window-size support
    # ------------------------------------------------------------------

    if not unsupported.empty:

        print()
        print(
            "Pairs excluded from robustness ranking "
            "because too few window sizes have valid "
            "within-book dispersion:"
        )

        print(
            unsupported[
                [
                    "book_a",
                    "book_b",
                    "n_valid_window_sizes",
                    "valid_window_sizes",
                    "missing_window_sizes",
                ]
            ]
            .sort_values(
                [
                    "n_valid_window_sizes",
                    "book_a",
                    "book_b",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
            )
            .to_string(
                index=False
            )
        )

    # ------------------------------------------------------------------
    # Sensitivity summary
    # ------------------------------------------------------------------

    print()
    print(
        "Window-size robustness summary:"
    )

    print(
        f"    Total Pauline pairs: "
        f"{len(sensitivity_summary)}"
    )

    print(
        f"    Robustness-supported pairs: "
        f"{len(supported)}"
    )

    print(
        f"    Insufficiently supported pairs: "
        f"{len(unsupported)}"
    )

    print(
        f"    Required valid window sizes: "
        f"{MIN_VALID_WINDOW_SIZES}"
    )

    # ------------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "Analysis complete."
    )

    print()
    print(
        "Results:"
    )

    print(
        f"    {OUTPUT_DIR}"
    )

    print()
    print(
        "Plots:"
    )

    print(
        f"    {PLOT_DIR}"
    )

    print()
    print(
        "Primary result:"
    )

    print(
        "    "
        + str(
            OUTPUT_DIR
            / "500_token_pauline_master_results.csv"
        )
    )

    print()
    print(
        "Window-size robustness:"
    )

    print(
        "    "
        + str(
            OUTPUT_DIR
            / "pauline_window_size_sensitivity_summary.csv"
        )
    )


if __name__ == "__main__":
    main()