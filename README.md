# Greek New Testament Computational Analysis

A computational stylometric analysis of the Greek New Testament using the
MorphGNT / SBLGNT dataset.

The project explores linguistic relationships among New Testament books using
Greek vocabulary, morphology, grammatical patterns, function words,
part-of-speech distributions, fixed-token windows, clustering, PCA, and
statistical distance analysis.

A particular area of interest is the traditionally Pauline corpus, but
authorship classifications are **not supplied to the statistical models**.

The goal is to measure linguistic similarity and variation in the Greek text,
not to assume or prove authorship in advance.

---

## Research Questions

The project is designed to explore questions such as:

- How linguistically similar are New Testament books?
- Which books naturally cluster together?
- How much stylistic variation exists within a single book?
- Are differences between two books larger than the variation found within
  those books?
- Do traditionally Pauline letters occupy similar regions of stylistic space?
- Do apparent relationships persist across different linguistic feature sets?
- Do relationships remain stable when token-window size changes?
- Which grammatical features contribute most strongly to observed differences?

---

## Dataset

The project uses the MorphGNT edition of the SBL Greek New Testament.

The dataset is included as a Git submodule:

```text
data/sblgnt/
```

MorphGNT provides token-level information including:

- Greek text
- normalized word forms
- lemmas
- part of speech
- morphological parsing
- book/chapter/verse information

The analysis currently loads approximately:

```text
137,554 Greek tokens
```

---

## Project Structure

```text
greek-nt-analysis/
├── data/
│   └── sblgnt/
│
├── src/
│   ├── loader.py
│   ├── morphology.py
│   ├── features.py
│   ├── similarity.py
│   ├── clustering.py
│   └── analysis.py
│
├── notebooks/
│
├── outputs/
│   ├── plots/
│   └── analysis/
│       └── plots/
│
├── requirements.txt
├── README.md
├── .gitignore
└── .gitmodules
```

---

## Analysis Pipeline

The analysis proceeds through several stages.

```text
MorphGNT / SBLGNT
        │
        ▼
Token Loader
        │
        ▼
Morphological Parsing
        │
        ▼
Linguistic Feature Extraction
        │
        ├── Morphology
        ├── Function Words
        ├── Part of Speech
        ├── Lexical Features
        └── Style Features
        │
        ▼
Fixed-Token Windows
        │
        ▼
Standardization
        │
        ├── Hierarchical Clustering
        ├── PCA
        ├── Centroid Distances
        ├── Within-Book Dispersion
        ├── Normalized Separation
        ├── Permutation Tests
        ├── Bootstrap Confidence Intervals
        └── Window-Size Sensitivity
```

---

# Feature Families

Rather than relying on one arbitrary definition of "style," the project
examines several linguistic feature families independently.

## Morphology

Morphological features include distributions of:

- verb tense
- verb voice
- verb mood
- verb person
- grammatical case
- grammatical number
- grammatical gender
- proportion of verb tokens

MorphGNT parsing codes are expanded into explicit grammatical categories before
analysis.

The morphology parser also validates the resulting categories to detect parsing
or positional errors before statistical analysis begins.

---

## Function Words

Function-word frequencies include common Greek particles, conjunctions,
prepositions, and other high-frequency grammatical words.

These are particularly useful in stylometry because they are generally less
dependent on subject matter than content vocabulary.

---

## Part of Speech

The project measures distributions of categories including:

- nouns
- verbs
- adjectives
- adverbs
- conjunctions
- particles
- prepositions
- articles
- pronouns

---

## Lexical Features

Lexical analysis includes:

- common content lemmas
- type-token ratio
- hapax ratio
- dis-legomena ratio
- average word length
- average lemma length

Function words are excluded from the common-lemma feature set because they are
analyzed separately.

Lexical results should be interpreted carefully because vocabulary can be
strongly affected by subject matter, genre, audience, and rhetorical context.

---

## Style Features

The primary stylometric feature family combines relatively topic-resistant
features:

```text
morphology
+
function words
+
part-of-speech distributions
```

Content lemmas are deliberately excluded from this feature family.

The current style representation contains approximately 71 features, depending
on which features survive preprocessing and zero-variance filtering.

This feature family is the primary representation used for multidimensional
style-distance analysis.

---

# Whole-Book Analysis

Each New Testament book can be represented by one feature vector.

The project performs:

- hierarchical clustering
- dendrogram generation
- principal component analysis
- PCA loading analysis
- feature-family-specific comparisons

Whole-book analysis provides an intuitive overview of relationships among the
27 New Testament books.

However, it provides only 27 observations and can hide substantial stylistic
variation within individual books.

For that reason, fixed-token-window analysis is also performed.

---

# Fixed-Token Windows

Books are divided into fixed-size token windows.

The primary exploratory window size is:

```text
500 tokens
```

with sufficiently large final partial windows retained.

For the current corpus this produces approximately:

```text
274 windows
```

Each window receives its own linguistic feature vector.

This allows the analysis to distinguish:

```text
variation between books
```

from:

```text
variation within books
```

That distinction is essential because two books should not be considered
strongly separated merely because their averages differ slightly if comparable
variation occurs naturally within each book.

---

# Principal Component Analysis

PCA is used as an exploratory visualization and dimensionality-reduction tool.

Plots are generated for:

- whole books
- token windows
- individual feature families
- the traditionally Pauline corpus

PCA loadings are exported so that the linguistic features driving each
principal component can be examined.

PCA plots should **not** be interpreted as complete measurements of similarity.

For example, a PC1/PC2 plot displays only the variance represented by those two
components. Information contained in later dimensions is not visible.

For this reason, the primary statistical analysis uses the complete
standardized feature space rather than only PCA coordinates.

---

# Hierarchical Clustering

Hierarchical clustering is used to examine whether books or token windows
naturally form groups based on their linguistic features.

The clustering process is unsupervised.

Authorship categories are not supplied to the clustering algorithm.

Generated outputs include:

```text
book_dendrogram_<feature-family>.png
window_dendrogram_<feature-family>.png
```

Clustering is descriptive and should not itself be treated as evidence of
authorship.

---

# Pauline Corpus Analysis

The project performs additional analysis on the traditionally Pauline corpus:

- Romans
- 1 Corinthians
- 2 Corinthians
- Galatians
- Ephesians
- Philippians
- Colossians
- 1 Thessalonians
- 2 Thessalonians
- 1 Timothy
- 2 Timothy
- Titus
- Philemon

These books are selected as the corpus being examined.

No disputed/undisputed authorship classification is supplied to PCA,
clustering, distance calculations, or statistical tests.

The analysis therefore asks how the Greek texts relate computationally before
comparing those results with historical authorship hypotheses.

---

# Multidimensional Style Analysis

`src/analysis.py` extends the exploratory PCA and clustering analysis.

Instead of asking only:

> How far apart are two books?

it asks:

> How far apart are two books relative to the amount of linguistic variation
> naturally occurring within those books?

The calculations use the complete standardized style feature space rather than
only the first two PCA dimensions.

---

## Book Centroids

Each book's token windows form a distribution in multidimensional style space.

The mean position of those windows is the book's centroid.

Distances between centroids provide a simple measure of overall stylistic
separation.

---

## Within-Book Dispersion

For each book the analysis calculates measures including:

- mean distance from its centroid
- median distance from its centroid
- standard deviation of centroid distance
- mean pairwise window distance
- median pairwise window distance

This provides a baseline for how much a book varies internally.

---

## Normalized Separation

Raw distance alone can be misleading.

The project therefore compares centroid distance with pooled within-book
dispersion.

Conceptually:

```text
                         between-book centroid distance
normalized separation = --------------------------------
                          pooled within-book dispersion
```

A value near `1` means that the separation between two book centroids is
roughly comparable to their typical internal variation.

Values below or above `1` should be interpreted descriptively rather than as
authorship thresholds.

---

## Cross-Book Window Distances

The analysis also compares every token window from one book with every token
window from another.

This helps detect situations in which two books have similar centroids but
different underlying distributions.

---

## Permutation Tests

For Pauline comparisons, permutation tests evaluate whether the observed
distance between two book centroids is unusually large under a null model in
which their window labels are exchangeable.

The primary analysis performs:

```text
5,000 permutations
```

per pair.

Multiple pairwise tests are corrected using the Benjamini-Hochberg false
discovery rate procedure.

A small p-value indicates evidence of linguistic separation under this model.

It does **not** by itself establish different authorship.

---

## Bootstrap Confidence Intervals

Book windows are independently resampled with replacement to estimate the
uncertainty surrounding pairwise centroid distances.

The primary analysis uses:

```text
2,000 bootstrap iterations
```

and reports 95% bootstrap intervals.

---

# Window-Size Sensitivity

Any fixed window size is partly arbitrary.

The statistical analysis is therefore repeated using:

```text
250 tokens
500 tokens
750 tokens
1000 tokens
```

This provides an important robustness test.

A relationship that appears only at one window size is less convincing than a
relationship that remains stable across several reasonable window sizes.

The 500-token analysis is treated as the primary model, while the other window
sizes provide sensitivity checks.

---

# Installation

Clone the repository recursively so that the SBLGNT submodule is downloaded:

```bash
git clone --recursive <repository-url>
cd greek-nt-analysis
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Analysis

## Validate Morphology

Before running statistical analysis:

```bash
python -m src.morphology
```

The parser validates the MorphGNT grammatical fields and prints diagnostic
distributions.

---

## Exploratory Clustering and PCA

Run:

```bash
python -m src.clustering
```

This generates whole-book and token-window:

- PCA coordinates
- PCA loadings
- dendrograms
- cluster assignments
- Pauline PCA results

Outputs are written under:

```text
outputs/
outputs/plots/
```

---

## Statistical Style Analysis

Run:

```bash
python -m src.analysis
```

This performs the multidimensional style analysis, including:

- centroid distances
- within-book dispersion
- normalized separation
- cross-book distances
- nearest neighbors
- Pauline permutation tests
- false-discovery-rate correction
- bootstrap confidence intervals
- window-size sensitivity analysis

Results are written under:

```text
outputs/analysis/
outputs/analysis/plots/
```

The two most useful summary files are:

```text
outputs/analysis/500_token_pauline_master_results.csv

outputs/analysis/pauline_window_size_sensitivity_summary.csv
```

---

# Interpreting the Results

No single metric should be treated as an authorship detector.

Instead, stronger evidence comes from convergence among independent analyses.

For example, a relationship becomes more interesting when it is visible in
multiple feature families, persists in the full multidimensional style space,
is large relative to within-book variation, survives permutation testing, has
reasonably stable bootstrap estimates, and remains similar across multiple
window sizes.

Conversely, a relationship that appears only in a two-dimensional PCA plot or
under one particular feature/window configuration should be treated
cautiously.

---

# Methodological Limitations

Computational similarity is not equivalent to authorship.

Observed linguistic differences can arise from many factors, including:

- genre
- subject matter
- audience
- rhetorical purpose
- chronology
- quotation and allusion
- document length
- editorial activity
- amanuensis involvement
- textual transmission
- feature selection
- statistical normalization
- window size

Short books also provide relatively few independent windows, which makes their
estimated stylistic distributions less stable.

The project therefore treats computational stylometry as one source of
evidence rather than a standalone authorship test.

---

# Reproducibility

The analysis is intended to be reproducible.

Important methodological choices are defined explicitly in the source code,
including:

- feature definitions
- morphological mappings
- window sizes
- minimum window sizes
- normalization
- clustering method
- permutation count
- bootstrap count
- random seed

Generated CSV files preserve intermediate results so analyses can be inspected
without relying solely on visualizations.

---

# Current Status

The core analytical pipeline is implemented.

Current capabilities include:

- MorphGNT/SBLGNT loading
- validated Greek morphology parsing
- linguistic feature extraction
- separate feature-family analysis
- topic-reduced style features
- whole-book comparison
- fixed-token-window comparison
- hierarchical clustering
- PCA and loading analysis
- Pauline-specific analysis
- multidimensional centroid comparison
- within-book dispersion
- normalized separation
- cross-window comparison
- permutation testing
- FDR correction
- bootstrap confidence intervals
- nearest-neighbor analysis
- multi-window-size robustness testing

The next phase is primarily **interpretation and validation of results**, not
adding additional statistical methods.

---

## Disclaimer

This project performs computational analysis of Greek textual features.

Its results describe patterns of linguistic similarity, difference, and
variation in the analyzed text.

They should not be interpreted as proof of traditional or nontraditional
authorship claims.