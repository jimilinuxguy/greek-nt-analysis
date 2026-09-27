"""
MorphGNT morphology parser.

Expands MorphGNT parsing codes into explicit grammatical features:

    person
    tense
    voice
    mood
    case
    number
    gender
    degree

MorphGNT uses an 8-character parsing field:

    0  person
    1  tense
    2  voice
    3  mood
    4  case
    5  number
    6  gender
    7  degree

Example:

    3AAI-S--

means approximately:

    person = third
    tense  = aorist
    voice  = active
    mood   = indicative
    case   = None
    number = singular
    gender = None
    degree = None

Dataset:
    MorphGNT / SBLGNT

Expected dataset location:
    data/sblgnt/
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Morphological mappings
# ---------------------------------------------------------------------------

PERSON = {
    "1": "first",
    "2": "second",
    "3": "third",
}


TENSE = {
    "P": "present",
    "I": "imperfect",
    "F": "future",
    "A": "aorist",
    "X": "perfect",
    "Y": "pluperfect",
}


VOICE = {
    "A": "active",
    "M": "middle",
    "P": "passive",
}


MOOD = {
    "I": "indicative",
    "D": "imperative",
    "S": "subjunctive",
    "O": "optative",
    "N": "infinitive",
    "P": "participle",
}


CASE = {
    "N": "nominative",
    "G": "genitive",
    "D": "dative",
    "A": "accusative",
    "V": "vocative",
}


NUMBER = {
    "S": "singular",
    "P": "plural",
}


GENDER = {
    "M": "masculine",
    "F": "feminine",
    "N": "neuter",
}


DEGREE = {
    "C": "comparative",
    "S": "superlative",
}


# ---------------------------------------------------------------------------
# Part-of-speech mappings
# ---------------------------------------------------------------------------

POS = {
    "N-": "noun",
    "V-": "verb",
    "A-": "adjective",
    "D-": "adverb",
    "P-": "preposition",
    "C-": "conjunction",
    "RA": "article",
    "RD": "demonstrative pronoun",
    "RI": "interrogative pronoun",
    "RP": "personal pronoun",
    "RR": "relative pronoun",
    "RX": "indefinite pronoun",
    "I-": "interjection",
    "X-": "particle",
}


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def _decode(
    value: str,
    mapping: dict[str, str],
) -> str | None:
    """
    Decode one MorphGNT morphology character.

    "-" represents an unused grammatical field.
    """

    if not value or value == "-":
        return None

    return mapping.get(
        value,
        value,
    )


# ---------------------------------------------------------------------------
# Morphology parsing
# ---------------------------------------------------------------------------

def parse_morphology(
    parse: str,
) -> dict[str, str | None]:
    """
    Parse a MorphGNT 8-character morphology code.

    Positions:

        0  person
        1  tense
        2  voice
        3  mood
        4  case
        5  number
        6  gender
        7  degree

    Example:

        3AAI-S--

    becomes:

        person = third
        tense  = aorist
        voice  = active
        mood   = indicative
        case   = None
        number = singular
        gender = None
        degree = None
    """

    if pd.isna(parse):
        parse = ""

    parse = str(parse).strip()

    # MorphGNT morphology should contain exactly eight positions.
    #
    # Padding allows empty/incomplete values to be represented safely
    # while validation below catches unexpected grammatical symbols.

    parse = parse.ljust(
        8,
        "-",
    )

    return {
        "person": _decode(
            parse[0],
            PERSON,
        ),

        "tense": _decode(
            parse[1],
            TENSE,
        ),

        "voice": _decode(
            parse[2],
            VOICE,
        ),

        "mood": _decode(
            parse[3],
            MOOD,
        ),

        "case": _decode(
            parse[4],
            CASE,
        ),

        "number": _decode(
            parse[5],
            NUMBER,
        ),

        "gender": _decode(
            parse[6],
            GENDER,
        ),

        "degree": _decode(
            parse[7],
            DEGREE,
        ),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_morphology(
    df: pd.DataFrame,
) -> None:
    """
    Validate expanded morphology.

    Raises ValueError if any grammatical column contains a value that is
    not recognized by the MorphGNT mappings.

    This is intentionally strict so parsing errors cannot silently
    contaminate downstream statistical analysis.
    """

    expected = {
        "person": set(
            PERSON.values()
        ),

        "tense": set(
            TENSE.values()
        ),

        "voice": set(
            VOICE.values()
        ),

        "mood": set(
            MOOD.values()
        ),

        "case": set(
            CASE.values()
        ),

        "number": set(
            NUMBER.values()
        ),

        "gender": set(
            GENDER.values()
        ),

        "degree": set(
            DEGREE.values()
        ),
    }

    errors = []

    for column, allowed in expected.items():

        if column not in df.columns:
            errors.append(
                f"{column}: column missing"
            )
            continue

        actual = set(
            df[column]
            .dropna()
            .unique()
        )

        unexpected = (
            actual - allowed
        )

        if unexpected:

            errors.append(
                f"{column}: "
                f"{sorted(unexpected)}"
            )

    if errors:

        raise ValueError(
            "Unexpected morphology values detected.\n"
            "This may indicate that MorphGNT parsing positions "
            "are incorrect.\n\n"
            + "\n".join(errors)
        )


# ---------------------------------------------------------------------------
# POS validation
# ---------------------------------------------------------------------------

def validate_pos(
    df: pd.DataFrame,
) -> None:
    """
    Verify that every POS code in the dataset is known.
    """

    actual = set(
        df["pos"]
        .dropna()
        .unique()
    )

    expected = set(
        POS.keys()
    )

    unexpected = (
        actual - expected
    )

    if unexpected:

        raise ValueError(
            "Unexpected MorphGNT POS codes detected:\n"
            f"{sorted(unexpected)}"
        )


# ---------------------------------------------------------------------------
# DataFrame expansion
# ---------------------------------------------------------------------------

def expand_morphology(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Expand MorphGNT POS and morphology fields into explicit columns.

    Adds:

        part_of_speech
        person
        tense
        voice
        mood
        case
        number
        gender
        degree
    """

    result = df.copy()

    # -------------------------------------------------------
    # Validate POS codes before translating them.
    # -------------------------------------------------------

    validate_pos(
        result
    )

    result[
        "part_of_speech"
    ] = result[
        "pos"
    ].map(
        POS
    )

    # -------------------------------------------------------
    # Parse morphology
    # -------------------------------------------------------

    morphology = result[
        "parse"
    ].apply(
        parse_morphology
    )

    morphology_df = pd.DataFrame(
        morphology.tolist(),
        index=result.index,
    )

    for column in morphology_df.columns:

        result[column] = (
            morphology_df[column]
        )

    # -------------------------------------------------------
    # Validate expanded morphology
    # -------------------------------------------------------

    validate_morphology(
        result
    )

    return result


# ---------------------------------------------------------------------------
# Morphology summaries
# ---------------------------------------------------------------------------

def morphology_summary(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """
    Return raw counts for a grammatical category by book.

    Example:

        morphology_summary(nt, "tense")
    """

    if column not in df.columns:

        raise ValueError(
            f"Column '{column}' does not exist. "
            "Run expand_morphology() first."
        )

    filtered = df[
        df[column].notna()
    ]

    summary = (
        filtered
        .groupby(
            [
                "book",
                column,
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "count"
            }
        )
    )

    return summary


def normalized_morphology_summary(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """
    Return normalized grammatical-category frequencies by book.

    Example:

        Romans
            present    0.42
            aorist     0.31
            perfect    0.08
            ...
    """

    summary = morphology_summary(
        df,
        column,
    )

    summary[
        "proportion"
    ] = (
        summary["count"]
        / summary
        .groupby("book")["count"]
        .transform("sum")
    )

    return summary


# ---------------------------------------------------------------------------
# Diagnostic report
# ---------------------------------------------------------------------------

def morphology_diagnostics(
    df: pd.DataFrame,
) -> None:
    """
    Print the unique values found for each grammatical category.

    Useful for verifying that the parser is behaving correctly before
    running statistical analysis.
    """

    columns = [
        "part_of_speech",
        "person",
        "tense",
        "voice",
        "mood",
        "case",
        "number",
        "gender",
        "degree",
    ]

    print()
    print("Morphology Diagnostics")
    print("=" * 80)

    for column in columns:

        values = (
            df[column]
            .dropna()
            .value_counts()
        )

        print()
        print(column)
        print("-" * 40)

        if values.empty:

            print("(none)")

        else:

            print(
                values.to_string()
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:

    from src.loader import (
        load_new_testament,
    )

    print()
    print(
        "MorphGNT Morphology Validation"
    )
    print("=" * 80)

    print()
    print(
        "Loading Greek New Testament..."
    )

    nt = load_new_testament()

    print(
        f"Loaded {len(nt):,} tokens."
    )

    print(
        "Expanding morphology..."
    )

    nt = expand_morphology(
        nt
    )

    print(
        "Morphology validation passed."
    )

    # -------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------

    morphology_diagnostics(
        nt
    )

    # -------------------------------------------------------
    # Sample
    # -------------------------------------------------------

    print()
    print("Sample")
    print("=" * 120)

    columns = [
        "book",
        "chapter",
        "verse",
        "word",
        "lemma",
        "part_of_speech",
        "person",
        "tense",
        "voice",
        "mood",
        "case",
        "number",
        "gender",
        "degree",
    ]

    print(
        nt[
            columns
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    # -------------------------------------------------------
    # Verb tense distribution
    # -------------------------------------------------------

    print()
    print(
        "Verb Tense Distribution"
    )
    print("=" * 120)

    verbs = nt[
        nt["part_of_speech"]
        == "verb"
    ]

    tense_summary = (
        verbs
        .groupby(
            [
                "book",
                "tense",
            ],
            as_index=False,
        )
        .size()
        .pivot(
            index="book",
            columns="tense",
            values="size",
        )
        .fillna(0)
        .astype(int)
    )

    print(
        tense_summary.to_string()
    )

    # -------------------------------------------------------
    # Romans diagnostic
    # -------------------------------------------------------

    print()
    print(
        "Romans Verb Tenses"
    )
    print("=" * 80)

    romans = verbs[
        verbs["book"]
        == "Romans"
    ]

    romans_tenses = (
        romans["tense"]
        .value_counts()
        .rename_axis("tense")
        .reset_index(
            name="count"
        )
    )

    if not romans_tenses.empty:

        romans_tenses[
            "percent"
        ] = (
            romans_tenses["count"]
            / romans_tenses[
                "count"
            ].sum()
            * 100
        )

        print(
            romans_tenses.to_string(
                index=False,
                formatters={
                    "percent":
                        lambda x: f"{x:.2f}%"
                },
            )
        )

    print()
    print("=" * 80)
    print(
        "Morphology parser passed validation."
    )


if __name__ == "__main__":
    main()