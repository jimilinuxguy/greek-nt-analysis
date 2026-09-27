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

## How to Read the Results

The analysis produces several statistical measurements describing how similar
or different New Testament books are in the linguistic features measured by
this project.

These measurements describe **linguistic relationships under the model**.
They are not probabilities of authorship and should not be interpreted as
proof that two books were or were not written by the same person.

### Feature

A **feature** is one measurable characteristic of the Greek text.

The style model uses features such as:

- verb tense frequencies
- verb voice frequencies
- verb mood frequencies
- grammatical person
- grammatical case
- grammatical number
- grammatical gender
- part-of-speech frequencies
- function-word frequencies

Each text window can therefore be represented as a collection of numbers.

Conceptually:

```text
Window
│
├── present tense frequency
├── aorist tense frequency
├── indicative frequency
├── participle frequency
├── nominative frequency
├── genitive frequency
├── conjunction frequencies
├── preposition frequencies
├── part-of-speech frequencies
└── ...
```

In the current style model, each window is represented by **71 features**.

---

### Feature Space

Because the style model measures 71 features, each window can be thought of
as a point in a **71-dimensional feature space**.

A simplified three-feature example might look like:

```text
                   participles
                       ↑
                       |
                       |       • Window B
                       |
                       |  • Window A
                       |
                       +----------------→ aorist verbs
                      /
                     /
                    ↙
             conjunctions
```

The real analysis works the same way, except that it uses 71 dimensions
instead of three.

Texts with similar combinations of measured features tend to occupy nearby
regions of this space.

---

### Standardization

Different features naturally occur on different numerical scales.

Before calculating distances, the features are **standardized** so that a
feature with naturally large numerical values does not dominate the analysis
simply because of its scale.

Conceptually, standardization asks:

> How high or low is this value relative to the normal variation of this
> feature across the corpus?

This allows different grammatical measurements to contribute on comparable
scales.

---

### Token Window

Rather than treating an entire biblical book as a single observation, the
analysis divides books into fixed-size **token windows**.

A token is approximately one Greek word or textual unit in the MorphGNT
dataset.

For example:

```text
Romans

tokens 1–500       → Window 1
tokens 501–1000    → Window 2
tokens 1001–1500   → Window 3
...
```

The principal analysis uses **500-token windows**.

Robustness testing also repeats the analysis using:

```text
250 tokens
500 tokens
750 tokens
1000 tokens
```

Using windows allows the project to measure not only how books differ from
one another, but also how much a single book varies internally.

---

### Centroid

A **centroid** is the mathematical center of a group of points.

Every token window from a book has a location in the multidimensional feature
space. The centroid is the average location of all of that book's windows.

Conceptually:

```text
        • Window
                  • Window

             X
          CENTROID

    • Window
                       • Window
```

If Romans has 14 usable windows, for example, the Romans centroid represents
the average stylistic position of those windows.

The centroid should therefore be understood as:

> the book's average position in the measured linguistic feature space.

It does **not** mean that every section of the book has that exact style.

---

### Centroid Distance

**Centroid distance** measures the distance between the average stylistic
positions of two books.

Conceptually:

```text
Romans centroid                    Galatians centroid

       X -------------------------------- X
              centroid distance
```

Smaller distances mean that the books' average measured profiles are closer
together.

Larger distances mean that their average profiles are farther apart.

The project uses **Euclidean distance** in the standardized multidimensional
feature space.

Centroid distance by itself does not tell us whether a distance is unusually
large, because books also contain internal stylistic variation.

That is why the project also measures within-book dispersion.

---

### Within-Book Dispersion

**Within-book dispersion** measures how much the windows of a single book
vary around that book's centroid.

A stylistically consistent book might look conceptually like:

```text
       • •
      • X •
       • •

X = centroid
```

A book with greater internal variation might look like:

```text
•                 •

         X

    •                 •
```

The second book has greater within-book dispersion.

This matters because two books should not automatically be considered
strongly separated merely because their centroids differ.

The relevant question is:

> How large is the difference between the books compared with the amount of
> variation already occurring inside the books themselves?

---

### Normalized Separation

**Normalized separation** is one of the most important measurements in this
project.

It compares the distance between two book centroids with their typical
within-book stylistic variation.

Conceptually:

```text
                         centroid distance
normalized separation = -------------------------------
                         pooled within-book dispersion
```

This makes the distance easier to interpret.

#### Normalized separation below 1

```text
normalized separation < 1
```

The difference between the two book centroids is smaller than the typical
stylistic variation occurring within the books themselves.

For example, the current analysis gives Galatians and Romans a mean normalized
separation of approximately:

```text
0.726
```

across the window-size sensitivity analysis.

This means their average measured stylistic separation is relatively small
compared with their internal variation.

#### Normalized separation around 1

```text
normalized separation ≈ 1
```

The distance between the books is approximately comparable to the amount of
variation occurring within them.

#### Normalized separation above 1

```text
normalized separation > 1
```

The difference between the book centroids is greater than the typical
within-book variation.

Increasing values indicate increasing separation relative to internal
variation.

These values are **descriptive measurements**, not authorship thresholds.

There is no rule such as:

```text
< 1 = same author
> 1 = different author
```

Such an interpretation would go beyond what the statistical model can
establish.

---

### Pooled Within-Book Dispersion

Normalized separation requires combining the internal variation of both books.

The project uses a **pooled dispersion** based on the dispersion of Book A and
Book B.

Conceptually:

```text
Book A dispersion ─┐
                   ├──→ pooled within-book dispersion
Book B dispersion ─┘
```

This prevents the comparison from being judged only against the variability
of one book.

---

### Pairwise Comparison

A **pairwise comparison** compares two books at a time.

Examples include:

```text
Romans ↔ Galatians
Romans ↔ 1 Corinthians
Ephesians ↔ Colossians
1 Timothy ↔ Romans
```

With 13 books in the Pauline corpus, there are:

```text
13 × 12
────── = 78 unique book pairs
   2
```

The current analysis therefore evaluates **78 Pauline book-pair
relationships**.

---

### Window-Size Sensitivity

The size of a token window is an analytical choice.

A relationship that appears only when using 500-token windows but disappears
with 250-, 750-, or 1000-token windows may depend heavily on that choice.

The analysis therefore repeats the comparison using:

```text
250
500
750
1000
```

token windows.

This is called **window-size sensitivity analysis**.

A relationship that remains similar across several window sizes is less
dependent on one arbitrary window-size choice.

---

### Robustness-Supported Comparison

For normalized separation to be calculated, both books need enough windows
to estimate their internal dispersion.

This becomes difficult for short books.

The project currently requires a pair to have valid normalized-separation
measurements at **at least three tested window sizes** before it is included
in the robustness ranking.

The output therefore contains:

```text
robustness_supported = True
```

or:

```text
robustness_supported = False
```

A value of `False` does **not** mean that the books failed a stylistic test.

It means:

> there is insufficient information across the tested window sizes to make
> the robustness comparison required by this analysis.

---

### Why Short Books Are Special

Very short books cannot produce many independent fixed-size windows.

For example, Philemon is too short to provide enough windows for reliable
within-book dispersion under the current windowing scheme.

The program therefore reports unavailable measurements as:

```text
NaN
```

rather than pretending that the book has zero internal variation.

`NaN` means:

> this quantity cannot be reliably estimated from the available data.

It does **not** mean zero distance, zero variation, or perfect similarity.

This distinction is particularly important for short books such as Philemon
and, at larger window sizes, Titus and 2 Thessalonians.

---

### Standard Deviation

The sensitivity output includes a **standard deviation** for normalized
separation across valid window sizes.

For example:

```text
mean normalized separation = 0.726
standard deviation          = 0.231
```

The mean describes the average result across window sizes.

The standard deviation describes how much those measurements vary around
that average.

Lower values generally indicate that the relationship changes less when the
window size changes.

Higher values indicate greater sensitivity to the window-size choice.

---

### Separation Range

The **separation range** is:

```text
largest normalized separation
-
smallest normalized separation
```

across the valid tested window sizes.

A small range means the relationship remains relatively similar as window
size changes.

A large range means the measured relationship changes substantially depending
on the window size.

Standard deviation and range therefore provide two complementary descriptions
of robustness.

---

### PCA

**Principal Component Analysis (PCA)** is a dimensionality-reduction
technique.

The style model may contain dozens of dimensions, which cannot be directly
visualized on a computer screen.

PCA creates new dimensions called **principal components** that summarize
major patterns of variation in the original features.

```text
71-dimensional feature space
            │
            ▼
           PCA
            │
            ▼
        PC1 and PC2
            │
            ▼
       2D visualization
```

PCA is primarily used in this project for exploration and visualization.

The primary multidimensional distance calculations are performed in the
standardized feature space rather than using only the two dimensions shown
in a PCA plot.

Therefore:

> Points appearing close together on a two-dimensional PCA plot are not
> necessarily equally close in the complete feature space.

---

### Principal Component

A **principal component** is a new statistical axis constructed from a
combination of the original features.

`PC1` captures the largest direction of variation found by PCA.

`PC2` captures the next largest independent direction.

Then come:

```text
PC3
PC4
PC5
...
```

A PCA plot commonly displays:

```text
x-axis = PC1
y-axis = PC2
```

This provides a useful projection of the data but does not contain all of the
information in the original feature space.

---

### Explained Variance

**Explained variance** tells us how much of the dataset's total variation is
represented by a principal component.

For example:

```text
PC1 = 17.6%
PC2 = 10.4%
```

would mean that the two-dimensional PC1/PC2 plot displays approximately:

```text
17.6% + 10.4% = 28.0%
```

of the total measured variation.

The remaining variation exists in later principal components.

This is why PCA plots should not be treated as complete maps of stylistic
distance.

---

### PCA Loading

A **PCA loading** describes how strongly an original linguistic feature
contributes to a principal component.

For example, if participle frequency has a strong positive loading on PC1,
windows with higher participle frequencies may tend to move toward the
positive side of PC1.

Loadings help answer:

> What linguistic features are causing this statistical dimension to exist?

They are therefore useful for moving from a mathematical pattern back toward
a linguistic explanation.

---

### Cluster

A **cluster** is a group of observations that the clustering algorithm places
relatively close together according to the measured features.

Depending on the analysis, an observation may represent:

```text
an entire book
```

or:

```text
a fixed-token window from a book
```

Clusters are exploratory structures.

They do not automatically represent authors.

For example, a cluster might reflect:

- similar grammar
- similar discourse style
- similar genre
- similar subject matter
- similar rhetorical situation
- common authorship
- editorial influence
- or some combination of factors

The statistical analysis alone does not determine which historical
explanation is correct.

---

### Hierarchical Clustering

The project uses **hierarchical clustering** to visualize relationships among
books or windows.

The algorithm progressively joins observations that are relatively similar,
creating a tree-like structure.

That tree is displayed as a **dendrogram**.

---

### Dendrogram

A **dendrogram** is a tree diagram showing the order in which observations or
groups are joined during hierarchical clustering.

Conceptually:

```text
Romans ─────────┐
                ├────────┐
Galatians ──────┘        │
                         ├──────
1 Corinthians ───────────┘
```

Books that join earlier are more similar according to the features and
clustering method being used.

The vertical or horizontal branch lengths represent clustering distance.

A dendrogram should be treated as an exploratory visualization, not as an
authorship family tree.

---

### Euclidean Distance

**Euclidean distance** is the multidimensional equivalent of measuring a
straight line between two points.

In two dimensions:

```text
A •
   \
    \ distance
     \
      • B
```

The project applies the same idea in a feature space containing many
dimensions.

Because the features are standardized first, the distance reflects differences
across the collection of measured linguistic features rather than being
dominated simply by features with larger numerical scales.

---

### Cosine Similarity

Some exploratory analyses also use **cosine similarity**.

Rather than primarily asking how far apart two vectors are, cosine similarity
asks how similarly they point through feature space.

Conceptually:

```text
        A
       /
      /
-----•---------------->
      \
       \
        B
```

Vectors pointing in similar directions have higher cosine similarity.

This provides another way of describing similarity among linguistic feature
profiles.

---

### Permutation Test

A **permutation test** asks whether the observed difference between two groups
would be unusual if the group labels were interchangeable.

Conceptually:

```text
Observed:

Romans windows      → Romans
Galatians windows   → Galatians

Permutation:

all windows
    │
    ├── randomly relabel
    │
    └── recalculate centroid distance
```

Repeating this process thousands of times creates a reference distribution.

The primary 500-token analysis uses **5,000 permutations**.

The resulting p-value describes how often randomized group assignments produce
a centroid separation at least as large as the observed separation under the
permutation model.

It does not give the probability that two books have the same or different
author.

---

### P-Value

A **p-value** from the permutation test measures how unusual the observed
statistic is under the particular null model used by the test.

A small p-value means that randomized reassignment of the windows rarely
produced a separation as large as the observed one.

It does **not** mean:

```text
probability that Paul wrote the book
```

or:

```text
probability that the books have different authors
```

Those are different questions that this statistical test does not calculate.

---

### Multiple Comparisons and FDR

The Pauline analysis compares many book pairs simultaneously.

When many statistical tests are performed, some small p-values can occur by
chance.

The project therefore applies **Benjamini-Hochberg False Discovery Rate
(FDR) correction**.

The corrected value helps account for the fact that many pairwise hypotheses
are being examined at once.

This reduces the risk of highlighting an apparently unusual comparison merely
because dozens of comparisons were performed.

---

### Bootstrap Confidence Interval

**Bootstrapping** estimates how much a statistic changes when the available
data are repeatedly resampled.

Conceptually:

```text
original windows
      │
      ▼
sample with replacement
      │
      ▼
calculate centroid distance
      │
      ▼
repeat many times
```

The primary analysis uses **2,000 bootstrap iterations**.

The resulting 95% bootstrap interval describes the range containing the
central 95% of the bootstrap estimates under this resampling procedure.

A narrow interval indicates greater stability under resampling.

A wide interval indicates greater uncertainty.

Because windows from the same biblical book are not necessarily statistically
independent, these intervals should be treated as model-based diagnostics
rather than definitive historical confidence intervals.

---

## A Worked Example

Suppose the analysis reports:

```text
Book A: Romans
Book B: Galatians

mean normalized separation: 0.726
standard deviation:          0.231
valid window sizes:          250, 500, 750, 1000
robustness supported:        True
```

A responsible interpretation would be:

> Romans and Galatians occupy relatively similar positions in the measured
> stylistic feature space. Their average centroid separation is smaller than
> the typical within-book stylistic variation represented by the model. The
> relationship remains measurable across all four tested window sizes, making
> it comparatively insensitive to the choice of window size.

An irresponsible interpretation would be:

> The computer proved that Paul wrote both books.

The statistical result supports the first statement.

It does not establish the second.

---

## Interpretation Cheat Sheet

| Term | Plain-English meaning |
|---|---|
| **Feature** | One measurable characteristic of the Greek text |
| **Feature space** | The multidimensional mathematical space created by all measured features |
| **Token window** | A fixed-size section of text treated as one observation |
| **Centroid** | The average stylistic position of a book's windows |
| **Centroid distance** | How far apart two books' average positions are |
| **Within-book dispersion** | How much a book's own windows vary around its centroid |
| **Normalized separation** | Between-book distance relative to normal variation inside the books |
| **Standard deviation** | How much a measurement varies around its mean |
| **Separation range** | Difference between the largest and smallest result across window sizes |
| **PCA** | A way to compress many dimensions into fewer dimensions for exploration |
| **Principal component** | A statistical direction capturing part of the variation in the data |
| **Explained variance** | How much of the total variation a principal component represents |
| **PCA loading** | How strongly an original feature contributes to a principal component |
| **Cluster** | Observations grouped because they are relatively similar |
| **Dendrogram** | Tree visualization of hierarchical clustering |
| **Euclidean distance** | Straight-line distance through multidimensional feature space |
| **Cosine similarity** | Similarity in the direction of two feature profiles |
| **Permutation test** | Tests how unusual an observed separation is under randomized labels |
| **p-value** | How unusual the observed statistic is under the test's null model |
| **FDR correction** | Adjustment for performing many statistical comparisons |
| **Bootstrap** | Repeated resampling used to examine stability and uncertainty |
| **95% bootstrap interval** | Central range of estimates produced by bootstrap resampling |
| **Window-size sensitivity** | Checks whether results change when text-window size changes |
| **Robustness supported** | The pair has enough valid window sizes for the project's robustness criterion |
| **NaN** | The quantity could not be reliably estimated from the available data |

## What the Model Can and Cannot Say

The model can support statements such as:

- two books are relatively close in the measured linguistic feature space;
- two books are relatively separated in that space;
- a book contains substantial or limited internal stylistic variation;
- a relationship persists or changes across different window sizes;
- particular linguistic features contribute strongly to observed statistical
  patterns.

The model cannot by itself establish:

- who wrote a biblical book;
- whether a traditional attribution is historically correct;
- whether two books must have the same author;
- whether two books must have different authors;
- whether an amanuensis or editor influenced the text;
- why a particular linguistic difference exists.

Those are historical questions for which computational stylometry can provide
evidence, but not a complete answer.

## Disclaimer

This project performs computational analysis of Greek textual features.

Its results describe patterns of linguistic similarity, difference, and
variation in the analyzed text.

They should not be interpreted as proof of traditional or nontraditional
authorship claims.