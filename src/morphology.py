"""
MorphGNT morphology parser.

Expands MorphGNT parsing codes into explicit grammatical features such as:

    tense
    voice
    mood
    person
    case
    number
    gender
    degree

MorphGNT parse fields use fixed positions:

    0  POS
    1  person
    2  tense
    3  voice
    4  mood
    5  case
    6  number
    7  gender
    8  degree

Example:

    V-3AAI-S--

becomes approximately:

    person = third
    tense  = aorist
    voice  = active
    mood   = indicative
    number = singular

Dataset:
    https://github.com/morphgnt/sblgnt
"""

import pandas as pd


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


# Human-readable MorphGNT part-of-speech codes.
#
# MorphGNT POS codes normally include a trailing dash, for example:
#
#     N-
#     V-
#     A-
#
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


def _decode(
    value: str,
    mapping: dict[str, str],
) -> str | None:
    """
    Decode one MorphGNT grammatical character.

    '-' indicates that the grammatical feature does not apply.
    """

    if not value or value == "-":
        return None

    return mapping.get(value, value)


def parse_morphology(parse: str) -> dict[str, str | None]:
    """
    Parse a MorphGNT morphology code.

    Parameters
    ----------
    parse:
        MorphGNT parsing string such as:

            -3AAI-S--
            ----NSM--
            ----GSF--

    Returns
    -------
    dict
        Expanded grammatical features.
    """

    if pd.isna(parse):
        parse = ""

    parse = str(parse).ljust(9, "-")

    return {
        "person": _decode(parse[1], PERSON),
        "tense": _decode(parse[2], TENSE),
        "voice": _decode(parse[3], VOICE),
        "mood": _decode(parse[4], MOOD),
        "case": _decode(parse[5], CASE),
        "number": _decode(parse[6], NUMBER),
        "gender": _decode(parse[7], GENDER),
        "degree": _decode(parse[8], DEGREE),
    }


def expand_morphology(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add expanded morphology columns to a MorphGNT DataFrame.

    Existing columns are preserved.

    Added columns:

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

    result["part_of_speech"] = result["pos"].map(POS)

    morphology = result["parse"].apply(parse_morphology)

    morphology_df = pd.DataFrame(
        morphology.tolist(),
        index=result.index,
    )

    for column in morphology_df.columns:
        result[column] = morphology_df[column]

    return result


def morphology_summary(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """
    Count a morphological feature by book.

    Example:

        morphology_summary(nt, "tense")

    returns counts of present, aorist, perfect, etc. for each book.
    """

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist. "
            "Run expand_morphology() first."
        )

    filtered = df[df[column].notna()]

    summary = (
        filtered.groupby(
            ["book", column],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "count"})
    )

    return summary


def normalized_morphology_summary(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """
    Calculate proportional usage of a morphological feature by book.

    Unlike raw counts, proportions allow books of different lengths
    to be compared more meaningfully.
    """

    summary = morphology_summary(df, column)

    summary["proportion"] = (
        summary["count"]
        / summary.groupby("book")["count"].transform("sum")
    )

    return summary


def main() -> None:
    # Import here to avoid a circular dependency.
    from src.loader import load_new_testament

    nt = load_new_testament()
    nt = expand_morphology(nt)

    print()
    print("Expanded Morphology")
    print("=" * 100)

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
    ]

    print(
        nt[columns]
        .head(25)
        .to_string(index=False)
    )

    print()
    print("Verb Tense Distribution")
    print("=" * 100)

    verbs = nt[
        nt["part_of_speech"] == "verb"
    ]

    tense_summary = (
        verbs.groupby(
            ["book", "tense"],
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

    print(tense_summary)

    print()
    print("Romans Verb Tenses")
    print("=" * 100)

    romans = verbs[
        verbs["book"] == "Romans"
    ]

    romans_tenses = (
        romans["tense"]
        .value_counts()
        .rename_axis("tense")
        .reset_index(name="count")
    )

    romans_tenses["percent"] = (
        romans_tenses["count"]
        / romans_tenses["count"].sum()
        * 100
    )

    print(
        romans_tenses.to_string(
            index=False,
            formatters={
                "percent": lambda x: f"{x:.2f}%"
            },
        )
    )


if __name__ == "__main__":
    main()