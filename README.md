# Greek New Testament Analysis

Computational analysis of the Greek New Testament using linguistic,
morphological, syntactic, and stylometric methods to compare vocabulary,
grammar, structure, and authorship patterns across books.

## Goals

This project explores linguistic similarities and differences across the
books of the Greek New Testament.

Areas of analysis include:

- Vocabulary and lemma frequency
- Lexical diversity
- Part-of-speech distributions
- Verb tense, voice, and mood
- Noun case, number, and gender
- Participial and infinitival constructions
- Function-word usage
- Sentence and clause structure
- Stylometric similarity
- Hierarchical clustering
- Principal component analysis (PCA)

One area of particular interest is comparing the Pauline epistles to determine
which letters exhibit similar linguistic and grammatical characteristics.

The analysis does not assume traditional authorship classifications. Linguistic
features are extracted from the text and compared computationally.

## Greek Text

The primary dataset is the MorphGNT edition of the SBL Greek New Testament:

https://github.com/morphgnt/sblgnt

MorphGNT provides token-level morphological annotation including:

- Part of speech
- Parsing information
- Text form
- Word form
- Normalized form
- Lemma

The dataset is included as a Git submodule under:

    data/sblgnt

## Clone

Clone the repository and its MorphGNT submodule:

    git clone --recurse-submodules https://github.com/jimilinuxguy/greek-nt-analysis.git
    cd greek-nt-analysis

If the repository was cloned without submodules:

    git submodule update --init --recursive

## Project Structure

    greek-nt-analysis/
    ├── data/
    │   └── sblgnt/
    ├── notebooks/
    ├── src/
    │   └── loader.py
    ├── .gitignore
    ├── .gitmodules
    ├── README.md
    └── requirements.txt

## Installation

Python 3.11+ is recommended.

Create a virtual environment:

    python3 -m venv .venv

Activate it:

macOS/Linux:

    source .venv/bin/activate

Windows:

    .venv\Scripts\activate

Install dependencies:

    pip install -r requirements.txt

## Loading the Dataset

Run:

    python -m src.loader

The loader reads all MorphGNT book files and combines them into a Pandas
DataFrame.

Each token includes:

- Book
- Chapter
- Verse
- Part of speech
- Morphological parse
- Text
- Word
- Normalized form
- Lemma

## Planned Analysis

The project will progressively add:

1. Morphological parsing
2. Per-book linguistic feature extraction
3. Vocabulary analysis
4. Function-word analysis
5. Grammatical distribution analysis
6. Book-to-book similarity metrics
7. Hierarchical clustering
8. PCA and other dimensionality-reduction techniques
9. Pauline epistle comparison
10. Syntactic analysis using Greek New Testament treebanks

## Methodological Considerations

Vocabulary alone is not sufficient evidence for authorship.

Differences between books may also reflect:

- Subject matter
- Audience
- Genre
- Letter length
- Quotations
- Editorial activity
- Amanuensis or scribal involvement

For this reason, the project will examine multiple independent linguistic
features, with particular attention to grammatical and function-word patterns
that may be less dependent on subject matter.

## License

This project's source code is separate from the licensing of the underlying
Greek New Testament datasets.

See the MorphGNT and SBLGNT repositories for their respective licensing and
attribution requirements.