"""
Human-readable interpretation of Greek NT stylometric analysis.

This module converts the numerical outputs produced by src.analysis into
plain-language descriptions.

IMPORTANT:
    These interpretations describe linguistic similarity under the model.
    They do NOT establish common authorship, different authorship, chronology,
    dependence, authenticity, or historical provenance.

Run with:

    python -m src.interpret

Outputs:

    outputs/interpretation/
        summary.txt
        summary.md
        pair_interpretations.csv
        book_profiles.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ===========================================================================
# Paths
# ===========================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

ANALYSIS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "interpretation"
)

SENSITIVITY_FILE = (
    ANALYSIS_DIR
    / "pauline_window_size_sensitivity_summary.csv"
)

PRIMARY_FILE = (
    ANALYSIS_DIR
    / "500_token_pauline_master_results.csv"
)


# ===========================================================================
# Settings
# ===========================================================================

TOP_RELATIONSHIPS = 10

# These are descriptive bands, not statistical significance thresholds.
#
# Normalized separation compares between-book centroid distance with
# typical within-book stylistic dispersion.
#
# < 1:
#     Between-book separation is smaller than typical internal variation.
#
# ~ 1:
#     Separation is comparable to internal variation.
#
# > 1:
#     Separation exceeds typical internal variation.

VERY_SIMILAR_MAX = 0.75
SIMILAR_MAX = 1.00
COMPARABLE_MAX = 1.25
MODERATELY_SEPARATED_MAX = 1.50


# ===========================================================================
# Utilities
# ===========================================================================


def ensure_output_directory() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def require_file(
    path: Path,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            "\nRequired analysis output was not found:\n\n"
            f"    {path}\n\n"
            "Run the analysis first:\n\n"
            "    python -m src.analysis\n"
        )


def finite(
    value,
) -> bool:

    try:

        return bool(
            np.isfinite(
                float(value)
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        return False


def format_number(
    value,
    digits: int = 3,
) -> str:

    if not finite(value):

        return "unavailable"

    return f"{float(value):.{digits}f}"


def parse_window_sizes(
    value,
) -> list[int]:

    if pd.isna(value):

        return []

    text = str(value).strip()

    if not text:

        return []

    result = []

    for part in text.split(","):

        part = part.strip()

        if not part:

            continue

        try:

            result.append(
                int(float(part))
            )

        except ValueError:

            continue

    return result


def format_window_sizes(
    value,
) -> str:

    sizes = parse_window_sizes(
        value
    )

    if not sizes:

        return "none"

    return ", ".join(
        str(size)
        for size in sizes
    )


# ===========================================================================
# Descriptive interpretation
# ===========================================================================

def separation_label(
    value: float,
) -> str:
    """
    Convert normalized separation into a descriptive category.

    These categories are explanatory conveniences, not inferential
    thresholds and not authorship classifications.
    """

    if not finite(value):
        return "not estimable"

    value = float(value)

    # Treat values effectively equal to 1.0 as comparable to
    # within-book variation. This avoids floating-point boundary
    # artifacts such as 0.9997 displaying as 1.000 while being
    # classified as "low separation."
    if np.isclose(
        value,
        1.0,
        atol=0.05,
    ):
        return (
            "between-book separation "
            "comparable to within-book variation"
        )

    if value < VERY_SIMILAR_MAX:
        return (
            "very low between-book "
            "stylistic separation"
        )

    if value < SIMILAR_MAX:
        return (
            "low between-book "
            "stylistic separation"
        )

    if value < COMPARABLE_MAX:
        return (
            "between-book separation "
            "comparable to within-book variation"
        )

    if value < MODERATELY_SEPARATED_MAX:
        return (
            "moderate between-book "
            "stylistic separation"
        )

    return (
        "relatively high between-book "
        "stylistic separation"
    )


def explain_separation(
    value: float,
) -> str:

    if not finite(value):
        return (
            "Normalized separation could not be estimated because "
            "there was insufficient within-book window variation."
        )

    value = float(value)

    # Check the approximately-equal case first so the prose agrees
    # with the displayed rounded value and separation label.
    if np.isclose(
        value,
        1.0,
        atol=0.05,
    ):
        return (
            "The distance between the two book centroids is roughly "
            "comparable to the stylistic variation observed within "
            "the books themselves."
        )

    if value < 1.0:
        return (
            "The distance between the two book centroids is smaller "
            "than the typical stylistic variation observed within "
            "the books themselves."
        )

    return (
        "The distance between the two book centroids is greater than "
        "the typical stylistic variation observed within the books "
        "themselves."
    )

def robustness_language(
    row: pd.Series,
) -> str:

    supported = bool(
        row.get(
            "robustness_supported",
            False,
        )
    )

    n_valid = int(
        row.get(
            "n_valid_window_sizes",
            0,
        )
    )

    valid_sizes = format_window_sizes(
        row.get(
            "valid_window_sizes",
            "",
        )
    )

    missing_sizes = format_window_sizes(
        row.get(
            "missing_window_sizes",
            "",
        )
    )

    if supported:

        if n_valid >= 4:

            return (
                "This relationship is supported across all four "
                f"tested window sizes ({valid_sizes} tokens), making "
                "it comparatively insensitive to the choice of "
                "window size."
            )

        return (
            f"This relationship is estimable at {n_valid} tested "
            f"window sizes ({valid_sizes} tokens) and meets the "
            "analysis threshold for window-size robustness."
        )

    if n_valid == 0:

        return (
            "This pair cannot be evaluated for window-size robustness "
            "because none of the tested window sizes provides enough "
            "within-book windows to estimate normalized separation."
        )

    return (
        f"This pair has valid normalized separation at only "
        f"{n_valid} tested window size"
        f"{'' if n_valid == 1 else 's'} ({valid_sizes} tokens). "
        f"The remaining sizes ({missing_sizes}) are not estimable, "
        "so the pair is excluded from the robustness ranking."
    )


def variability_language(
    row: pd.Series,
) -> str:

    std = row.get(
        "std_normalized_separation",
        np.nan,
    )

    spread = row.get(
        "separation_range",
        np.nan,
    )

    if not finite(std):

        return (
            "There are too few valid window-size measurements to "
            "characterize sensitivity to window size reliably."
        )

    std = float(std)

    if std < 0.25:

        stability = "quite stable"

    elif std < 0.40:

        stability = "reasonably stable"

    elif std < 0.60:

        stability = (
            "somewhat sensitive to window size"
        )

    else:

        stability = (
            "substantially sensitive to window size"
        )

    return (
        f"The normalized separation is {stability} "
        f"(SD {std:.3f}, range "
        f"{format_number(spread)} across valid window sizes)."
    )


# ===========================================================================
# Pair interpretation
# ===========================================================================


def interpret_pair(
    row: pd.Series,
) -> str:

    book_a = str(
        row["book_a"]
    )

    book_b = str(
        row["book_b"]
    )

    separation = row.get(
        "mean_normalized_separation",
        np.nan,
    )

    label = separation_label(
        separation
    )

    if finite(separation):

        opening = (
            f"{book_a} and {book_b} show {label} under the "
            f"style-feature model, with a mean normalized separation "
            f"of {float(separation):.3f}."
        )

    else:

        opening = (
            f"The normalized stylistic separation between {book_a} "
            f"and {book_b} cannot be estimated reliably."
        )

    return " ".join(
        [
            opening,
            explain_separation(
                separation
            ),
            robustness_language(
                row
            ),
            variability_language(
                row
            ),
            (
                "This describes similarity in the measured linguistic "
                "features and should not by itself be interpreted as "
                "evidence for or against common authorship."
            ),
        ]
    )


# ===========================================================================
# Book profiles
# ===========================================================================


def build_book_profiles(
    supported: pd.DataFrame,
) -> pd.DataFrame:

    books = sorted(
        set(
            supported["book_a"]
        )
        | set(
            supported["book_b"]
        )
    )

    records = []

    for book in books:

        relationships = supported[
            (
                supported["book_a"]
                == book
            )
            |
            (
                supported["book_b"]
                == book
            )
        ].copy()

        if relationships.empty:

            continue

        relationships = (
            relationships
            .sort_values(
                "mean_normalized_separation"
            )
        )

        closest = (
            relationships.iloc[0]
        )

        most_distant = (
            relationships.iloc[-1]
        )

        def other_book(
            row: pd.Series,
        ) -> str:

            if row["book_a"] == book:

                return str(
                    row["book_b"]
                )

            return str(
                row["book_a"]
            )

        closest_book = other_book(
            closest
        )

        distant_book = other_book(
            most_distant
        )

        mean_relationship = (
            relationships[
                "mean_normalized_separation"
            ].mean()
        )

        records.append(
            {
                "book":
                    book,

                "n_supported_comparisons":
                    len(
                        relationships
                    ),

                "closest_book":
                    closest_book,

                "closest_normalized_separation":
                    closest[
                        "mean_normalized_separation"
                    ],

                "most_distant_book":
                    distant_book,

                "most_distant_normalized_separation":
                    most_distant[
                        "mean_normalized_separation"
                    ],

                "mean_normalized_separation":
                    mean_relationship,

                "interpretation":
                    (
                        f"Among robustness-supported comparisons, "
                        f"{book}'s closest measured stylistic neighbor "
                        f"is {closest_book} "
                        f"({closest['mean_normalized_separation']:.3f}), "
                        f"while its most separated measured comparison "
                        f"is {distant_book} "
                        f"({most_distant['mean_normalized_separation']:.3f}). "
                        f"These are relative positions within this "
                        f"feature model, not authorship judgments."
                    ),
            }
        )

    return (
        pd.DataFrame(
            records
        )
        .sort_values(
            "book"
        )
        .reset_index(
            drop=True
        )
    )


# ===========================================================================
# Report generation
# ===========================================================================


def report_header() -> str:

    return """
# Greek New Testament Stylometric Interpretation

## What this report measures

This report summarizes computational stylistic relationships among the
Pauline corpus using the multidimensional style features generated by the
analysis pipeline.

The principal quantity is **normalized separation**:

    between-book centroid distance
    --------------------------------
       pooled within-book dispersion

A value below 1 means that the measured distance between two books is
smaller than the typical stylistic variation observed within those books.
A value around 1 means between-book separation is comparable to internal
variation. Values above 1 indicate progressively greater separation
relative to internal variation.

These values are descriptive. They are not authorship probabilities,
confidence scores, or historical authenticity tests.

The window-size robustness analysis repeats the comparison using
250-, 500-, 750-, and 1000-token windows. A pair is included in the
robustness ranking only when normalized separation can be estimated at
at least three window sizes.

## Important limitation

Linguistic similarity does not establish common authorship, and linguistic
difference does not establish different authorship. Genre, subject matter,
audience, rhetorical situation, chronology, quotation, editorial activity,
amanuensis practices, and other factors may affect the measured style.

"""


def generate_markdown_report(
    sensitivity: pd.DataFrame,
    profiles: pd.DataFrame,
) -> str:

    supported = sensitivity[
        sensitivity[
            "robustness_supported"
        ].astype(bool)
    ].copy()

    unsupported = sensitivity[
        ~sensitivity[
            "robustness_supported"
        ].astype(bool)
    ].copy()

    supported = (
        supported
        .sort_values(
            "mean_normalized_separation"
        )
    )

    sections = [
        report_header()
    ]

    # ------------------------------------------------------------------
    # Overview
    # ------------------------------------------------------------------

    sections.append(
        "\n## Dataset-level summary\n\n"
    )

    sections.append(
        f"The analysis contains **{len(sensitivity)} Pauline "
        f"book-pair comparisons**. Of these, "
        f"**{len(supported)} pairs** have enough information to meet "
        f"the window-size robustness criterion, while "
        f"**{len(unsupported)} pairs** are excluded from robustness "
        f"ranking because too few tested window sizes permit reliable "
        f"estimation of within-book dispersion.\n"
    )

    # ------------------------------------------------------------------
    # Similar relationships
    # ------------------------------------------------------------------

    sections.append(
        "\n## Strongest robustness-supported stylistic similarities\n\n"
    )

    closest = supported.head(
        TOP_RELATIONSHIPS
    )

    for rank, (_, row) in enumerate(
        closest.iterrows(),
        start=1,
    ):

        sections.append(
            f"### {rank}. "
            f"{row['book_a']} — {row['book_b']}\n\n"
        )

        sections.append(
            interpret_pair(
                row
            )
        )

        sections.append(
            "\n\n"
        )

    # ------------------------------------------------------------------
    # Separated relationships
    # ------------------------------------------------------------------

    sections.append(
        "\n## Strongest robustness-supported stylistic separations\n\n"
    )

    separated = (
        supported
        .sort_values(
            "mean_normalized_separation",
            ascending=False,
        )
        .head(
            TOP_RELATIONSHIPS
        )
    )

    for rank, (_, row) in enumerate(
        separated.iterrows(),
        start=1,
    ):

        sections.append(
            f"### {rank}. "
            f"{row['book_a']} — {row['book_b']}\n\n"
        )

        sections.append(
            interpret_pair(
                row
            )
        )

        sections.append(
            "\n\n"
        )

    # ------------------------------------------------------------------
    # Individual books
    # ------------------------------------------------------------------

    sections.append(
        "\n## Book-by-book profile\n\n"
    )

    for _, row in profiles.iterrows():

        sections.append(
            f"### {row['book']}\n\n"
        )

        sections.append(
            str(
                row["interpretation"]
            )
        )

        sections.append(
            "\n\n"
        )

    # ------------------------------------------------------------------
    # Unsupported comparisons
    # ------------------------------------------------------------------

    sections.append(
        "\n## Comparisons with insufficient robustness support\n\n"
    )

    sections.append(
        "The following comparisons are not included in the principal "
        "robustness ranking. This does **not** mean the books are similar "
        "or dissimilar. It means the available text does not produce "
        "enough independent fixed-size windows to estimate normalized "
        "separation across at least three tested window sizes.\n\n"
    )

    for _, row in unsupported.iterrows():

        sections.append(
            f"- **{row['book_a']} — {row['book_b']}**: "
            f"{robustness_language(row)}\n"
        )

    # ------------------------------------------------------------------
    # Conclusions
    # ------------------------------------------------------------------

    sections.append(
        """

## How to read these results

The safest interpretation is relational rather than categorical.

When two books repeatedly have low normalized separation across several
window sizes, the measured style features place them relatively close to
one another compared with the amount of stylistic variation occurring
inside the books themselves.

When two books repeatedly have high normalized separation, their measured
style profiles are farther apart relative to their internal variation.

Neither result answers the historical authorship question by itself.
Authorship is a broader historical inference that must integrate textual,
linguistic, chronological, literary, manuscript, and historical evidence.

"""
    )

    return "".join(
        sections
    )


# ===========================================================================
# Console summary
# ===========================================================================


def print_console_summary(
    sensitivity: pd.DataFrame,
) -> None:

    supported = sensitivity[
        sensitivity[
            "robustness_supported"
        ].astype(bool)
    ].copy()

    unsupported = sensitivity[
        ~sensitivity[
            "robustness_supported"
        ].astype(bool)
    ].copy()

    supported = (
        supported
        .sort_values(
            "mean_normalized_separation"
        )
    )

    print()
    print("=" * 80)
    print(
        "HUMAN-LANGUAGE INTERPRETATION"
    )
    print("=" * 80)

    print()
    print(
        f"Robustness-supported comparisons: "
        f"{len(supported)}"
    )

    print(
        f"Insufficiently supported comparisons: "
        f"{len(unsupported)}"
    )

    print()
    print(
        "Strongest supported similarities:"
    )

    for _, row in supported.head(5).iterrows():

        print()
        print(
            f"  {row['book_a']} ↔ {row['book_b']}"
        )

        print(
            f"    Mean normalized separation: "
            f"{row['mean_normalized_separation']:.3f}"
        )

        print(
            "    "
            + explain_separation(
                row[
                    "mean_normalized_separation"
                ]
            )
        )

    print()
    print(
        "Strongest supported separations:"
    )

    separated = (
        supported
        .sort_values(
            "mean_normalized_separation",
            ascending=False,
        )
        .head(5)
    )

    for _, row in separated.iterrows():

        print()
        print(
            f"  {row['book_a']} ↔ {row['book_b']}"
        )

        print(
            f"    Mean normalized separation: "
            f"{row['mean_normalized_separation']:.3f}"
        )

        print(
            "    "
            + explain_separation(
                row[
                    "mean_normalized_separation"
                ]
            )
        )


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:

    ensure_output_directory()

    require_file(
        SENSITIVITY_FILE
    )

    print()
    print(
        "Loading analysis results..."
    )

    sensitivity = pd.read_csv(
        SENSITIVITY_FILE
    )

    # Pandas normally reads this as bool, but normalize it in case
    # the CSV representation changes.
    if (
        sensitivity[
            "robustness_supported"
        ].dtype
        != bool
    ):

        sensitivity[
            "robustness_supported"
        ] = (
            sensitivity[
                "robustness_supported"
            ]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                }
            )
            .fillna(False)
            .astype(bool)
        )

    # ------------------------------------------------------------------
    # Pair-level interpretations
    # ------------------------------------------------------------------

    pair_interpretations = (
        sensitivity.copy()
    )

    pair_interpretations[
        "separation_description"
    ] = pair_interpretations[
        "mean_normalized_separation"
    ].apply(
        separation_label
    )

    pair_interpretations[
        "human_interpretation"
    ] = pair_interpretations.apply(
        interpret_pair,
        axis=1,
    )

    pair_interpretations.to_csv(
        OUTPUT_DIR
        / "pair_interpretations.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Book profiles
    # ------------------------------------------------------------------

    supported = sensitivity[
        sensitivity[
            "robustness_supported"
        ]
    ].copy()

    profiles = build_book_profiles(
        supported
    )

    profiles.to_csv(
        OUTPUT_DIR
        / "book_profiles.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------

    report = generate_markdown_report(
        sensitivity,
        profiles,
    )

    markdown_path = (
        OUTPUT_DIR
        / "summary.md"
    )

    markdown_path.write_text(
        report,
        encoding="utf-8",
    )

    # Also produce a plain-text version.
    text_report = (
        report
        .replace("**", "")
        .replace("# ", "")
        .replace("## ", "")
        .replace("### ", "")
    )

    text_path = (
        OUTPUT_DIR
        / "summary.txt"
    )

    text_path.write_text(
        text_report,
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Console
    # ------------------------------------------------------------------

    print_console_summary(
        sensitivity
    )

    print()
    print("=" * 80)
    print(
        "Interpretation complete."
    )

    print()
    print(
        "Human-readable report:"
    )

    print(
        f"    {markdown_path}"
    )

    print()
    print(
        "Pair-by-pair interpretations:"
    )

    print(
        "    "
        + str(
            OUTPUT_DIR
            / "pair_interpretations.csv"
        )
    )

    print()
    print(
        "Book profiles:"
    )

    print(
        "    "
        + str(
            OUTPUT_DIR
            / "book_profiles.csv"
        )
    )


if __name__ == "__main__":
    main()