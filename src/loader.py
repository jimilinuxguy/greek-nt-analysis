"""
MorphGNT / SBLGNT dataset loader.

Loads the MorphGNT Greek New Testament into a Pandas DataFrame.

Dataset:
    https://github.com/morphgnt/sblgnt

Expected location:
    data/sblgnt/
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "sblgnt"


COLUMNS = [
    "bcv",
    "pos",
    "parse",
    "text",
    "word",
    "normalized",
    "lemma",
]


BOOK_CODES = {
    "Mt": "Matthew",
    "Mk": "Mark",
    "Lk": "Luke",
    "Jn": "John",
    "Ac": "Acts",
    "Ro": "Romans",
    "1Co": "1 Corinthians",
    "2Co": "2 Corinthians",
    "Ga": "Galatians",
    "Eph": "Ephesians",
    "Php": "Philippians",
    "Col": "Colossians",
    "1Th": "1 Thessalonians",
    "2Th": "2 Thessalonians",
    "1Ti": "1 Timothy",
    "2Ti": "2 Timothy",
    "Tit": "Titus",
    "Phm": "Philemon",
    "Heb": "Hebrews",
    "Jas": "James",
    "1Pe": "1 Peter",
    "2Pe": "2 Peter",
    "1Jn": "1 John",
    "2Jn": "2 John",
    "3Jn": "3 John",
    "Jud": "Jude",
    "Re": "Revelation",
}


def parse_bcv(bcv: str) -> tuple[int, int, int]:
    """
    Parse a MorphGNT BCV identifier.

    Example:
        010101 -> book 1, chapter 1, verse 1
        060101 -> book 6, chapter 1, verse 1

    Format:
        BBCCVV
    """

    bcv = str(bcv).zfill(6)

    book = int(bcv[0:2])
    chapter = int(bcv[2:4])
    verse = int(bcv[4:6])

    return book, chapter, verse


def get_book_code(filename: str) -> str:
    """
    Extract the MorphGNT book code from a filename.

    Example:
        61-Mt-morphgnt.txt -> Mt
        67-1Co-morphgnt.txt -> 1Co
    """

    parts = filename.split("-")

    if len(parts) < 3:
        raise ValueError(f"Unexpected MorphGNT filename: {filename}")

    return parts[1]


def load_book(path: Path) -> pd.DataFrame:
    """
    Load a single MorphGNT book.
    """

    book_code = get_book_code(path.name)

    df = pd.read_csv(
        path,
        sep=r"\s+",
        names=COLUMNS,
        dtype={"bcv": str},
        encoding="utf-8",
    )

    parsed = df["bcv"].apply(parse_bcv)

    df["book_number"] = parsed.apply(lambda x: x[0])
    df["chapter"] = parsed.apply(lambda x: x[1])
    df["verse"] = parsed.apply(lambda x: x[2])

    df["book_code"] = book_code
    df["book"] = BOOK_CODES.get(book_code, book_code)

    df["source_file"] = path.name

    return df


def load_new_testament(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Load the complete MorphGNT Greek New Testament.

    Returns one row per Greek token.
    """

    if not data_dir.exists():
        raise FileNotFoundError(
            f"MorphGNT dataset not found at:\n"
            f"    {data_dir}\n\n"
            "Initialize the Git submodule with:\n"
            "    git submodule update --init --recursive"
        )

    files = sorted(data_dir.glob("*-morphgnt.txt"))

    if not files:
        raise FileNotFoundError(
            f"No MorphGNT files found in:\n"
            f"    {data_dir}"
        )

    frames = [load_book(path) for path in files]

    return pd.concat(
        frames,
        ignore_index=True,
    )


def book_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a basic summary of the loaded New Testament.
    """

    summary = (
        df.groupby(
            ["book_number", "book"],
            as_index=False,
        )
        .agg(
            tokens=("word", "size"),
            chapters=("chapter", "nunique"),
            verses=("bcv", "nunique"),
            unique_lemmas=("lemma", "nunique"),
        )
        .sort_values("book_number")
    )

    return summary


def main() -> None:
    nt = load_new_testament()

    print()
    print("Greek New Testament")
    print("=" * 70)

    print(f"Tokens:        {len(nt):,}")
    print(f"Books:         {nt['book'].nunique()}")
    print(f"Chapters:      {nt.groupby(['book', 'chapter']).ngroups:,}")
    print(f"Verses:        {nt['bcv'].nunique():,}")
    print(f"Unique lemmas: {nt['lemma'].nunique():,}")

    print()
    print("Books")
    print("=" * 70)

    summary = book_summary(nt)

    print(
        summary.to_string(
            index=False,
            columns=[
                "book",
                "chapters",
                "verses",
                "tokens",
                "unique_lemmas",
            ],
        )
    )

    print()
    print("Sample")
    print("=" * 70)

    print(
        nt[
            [
                "book",
                "chapter",
                "verse",
                "word",
                "lemma",
                "pos",
                "parse",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()