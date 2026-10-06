# Automatic-Aspect-Extraction-for-Russian-Aspect-Based-Sentiment-Analysis
Automatic aspect extraction for Russian ABSA with reduced reliance on aspect-level annotation, using linguistic rules, word embeddings, ML, and RuBERT.

Research code from my Master's thesis on automatic aspect extraction for aspect-based sentiment analysis (ABSA), completed in 2024 at the Department of Theoretical and Applied Linguistics, Lomonosov Moscow State University.

The project investigates how explicit aspect terms can be extracted from Russian restaurant reviews while reducing dependence on detailed aspect-level annotation. The experiments combine linguistic rules, sentiment lexicons, semantic similarity, dependency parsing, classical machine learning, and RuBERT-based category classification.

This repository presents the experimental pipeline in a consolidated and documented form for portfolio use.

## Project overview

The project explores several complementary approaches to automatic aspect extraction and filtering:

1. **Lexicon- and pattern-based extraction** using sentiment lexicons and syntactic patterns.
2. **Candidate propagation and filtering** based on co-occurrence between aspect and opinion words.
3. **Semantic expansion** with pretrained Word2Vec and FastText embeddings.
4. **Dependency-based extraction** using a Russian dependency parser.
5. **Aspect category classification** with classical machine-learning models and RuBERT.
6. **Category-based filtering** using predicted category probabilities.
7. **KL-divergence filtering** to remove candidates with insufficiently category-specific distributions.

The experiments focus on the trade-off between **precision and recall** when expanding the set of aspect candidates beyond directly annotated examples.

## Data

### SemEval 2016 Russian Restaurant Reviews

The main annotated dataset contains Russian restaurant reviews with aspect terms and aspect categories:

- **Training set:** 300 reviews, 3,654 sentences
- **Test set:** 100 reviews, 1,208 sentences

Six aspect categories are used:

`FOOD` · `SERVICE` · `RESTAURANT` · `AMBIENCE` · `DRINKS` · `LOCATION`

### SentiRuEval

An additional collection of **19,304 restaurant-review sentences** is used as a larger corpus (`large_train`) for candidate propagation and extraction experiments.

The datasets and external linguistic resources are not redistributed in this repository.

## Methodology

### 1. Lexicon- and pattern-based extraction

Aspect candidates are first extracted using local syntactic configurations involving nouns and evaluative adjectives.

Five sentiment resources are used:

- RuSentiLex
- ProductSentiRus
- Kotelnikov sentiment lexicon
- Blinov sentiment lexicon
- EmoLex / NRC Emotion Lexicon

The extraction rules cover configurations such as:

- adjective + noun
- adjective + adjective + noun
- adjective + conjunction + adjective + noun
- adjective + noun + preposition + noun
- noun + adjective
- noun + auxiliary + adjective

Candidate aspect and opinion words are subsequently filtered using their co-occurrence patterns.

### 2. Semantic expansion with Word2Vec and FastText

Pretrained Russian **Word2Vec** and **FastText** embeddings are used to expand the aspect vocabulary through semantic similarity.

Similarity thresholds from **0.50 to 0.90** are evaluated to study the precision–recall trade-off introduced by semantic expansion.

The Word2Vec pipeline uses UDPipe-based preprocessing compatible with RusVectōrēs representations.

### 3. Dependency-based aspect extraction

A second extraction strategy uses dependency relations between sentiment-bearing adjectives and nouns.

Dependency parsing is performed with a pretrained **Russian SynTagRus UD 2.5 UDPipe model**. Nouns linked to evaluative adjectives are extracted as potential aspect terms.

### 4. Aspect category classification

Sentence-level aspect category classification is evaluated using four text representations:

- CountVectorizer
- TF-IDF
- Word2Vec
- FastText

Five classical classifiers are compared:

- Support Vector Machine
- Random Forest
- Multinomial Naive Bayes
- Gradient Boosting
- Logistic Regression

The annotated training collection is split into training and validation subsets in a **4:1 ratio**. Model quality is evaluated using weighted precision, recall, and F1.

Hyperparameter tuning with `GridSearchCV` is also included for Multinomial Naive Bayes.

### 5. RuBERT category classification

`DeepPavlov/rubert-base-cased` is fine-tuned for aspect category classification.

Main training settings:

- maximum sequence length: `256`
- training batch size: `16`
- AdamW learning rate: `2e-5`
- epochs: `4`
- training/validation split: `90/10`

The model predicts six restaurant aspect categories plus a null class for sentences without an annotated aspect category.

### 6. Category-probability filtering

Predicted aspect-category probabilities are used to evaluate the reliability of extracted candidates.

Category confidence is aggregated across repeated occurrences of each candidate, and different confidence thresholds are evaluated to remove weak candidates.

### 7. KL-divergence filtering

For each candidate, its category-confidence values are normalized into a probability distribution.

KL divergence from a uniform distribution is then used as a measure of **category specificity**. Candidates with low divergence are less strongly associated with a particular aspect category and can be filtered out.

## Results

### Aspect category classification

| Representation | Classifier | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Count | SVM | 0.6903 | 0.6774 | 0.6770 |
| Count | Random Forest | 0.6992 | 0.6594 | 0.6591 |
| Count | Naive Bayes | 0.6836 | 0.6504 | 0.6526 |
| Count | Gradient Boosting | 0.6776 | 0.6504 | 0.6507 |
| Count | Logistic Regression | 0.7296 | 0.6362 | 0.6533 |
| TF-IDF | SVM | 0.7289 | 0.6474 | 0.6658 |
| TF-IDF | Random Forest | 0.7036 | 0.6422 | 0.6480 |
| TF-IDF | Naive Bayes | 0.5808 | 0.6452 | 0.6067 |
| TF-IDF | Gradient Boosting | 0.6467 | 0.6279 | 0.6279 |
| TF-IDF | Logistic Regression | 0.7468 | 0.5529 | 0.5775 |
| Word2Vec | SVM | 0.7475 | 0.6797 | 0.6903 |
| Word2Vec | Random Forest | 0.7138 | 0.4959 | 0.5154 |
| Word2Vec | Naive Bayes | 0.4991 | 0.6444 | 0.5256 |
| Word2Vec | Gradient Boosting | 0.7311 | 0.6272 | 0.6473 |
| Word2Vec | Logistic Regression | 0.6024 | 0.6002 | 0.5990 |
| FastText | SVM | **0.7378** | **0.6849** | **0.6990** |
| FastText | Random Forest | 0.6889 | 0.5071 | 0.5216 |
| FastText | Naive Bayes | 0.5301 | 0.2513 | 0.1465 |
| FastText | Gradient Boosting | 0.7209 | 0.6197 | 0.6442 |
| FastText | Logistic Regression | 0.7068 | 0.5821 | 0.6121 |

Among the classical configurations, **FastText + SVM** achieved the highest F1 score (**0.6990**).

The RuBERT category classifier achieved:

- **Precision:** 0.7206
- **Recall:** 0.7130
- **F1:** **0.7112**

### Aspect extraction and filtering

| Method | Dataset | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Syntactic propagation | Train | 0.4501 | 0.6029 | 0.5154 |
| Syntactic propagation | Large train | 0.2154 | **0.7613** | 0.3358 |
| Preliminary filtering | Train | 0.4333 | 0.5947 | **0.5013** |
| Preliminary filtering | Large train | 0.2184 | 0.7582 | 0.3392 |
| + aspect markers | Train | 0.3784 | 0.6420 | 0.4762 |
| + aspect markers | Large train | 0.2159 | **0.7613** | 0.3364 |
| + category filtering | Train | 0.3823 | 0.6317 | 0.4763 |
| + category filtering | Large train | 0.2268 | 0.7305 | 0.3461 |
| + KL-divergence filtering | Train | 0.3853 | 0.6276 | 0.4775 |
| + KL-divergence filtering | Large train | **0.2399** | 0.6914 | **0.3562** |

On the larger corpus, category-based and KL-divergence filtering increased precision from **0.2154 to 0.2399** and F1 from **0.3358 to 0.3562**, while reducing recall from **0.7613 to 0.6914**.

The experiments therefore illustrate the central trade-off of the pipeline: propagation and semantic expansion increase coverage, while category-based filtering can recover part of the lost precision.

## Repository structure

```text
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── models/
│   └── README.md
└── notebooks/
    ├── 01_lexicon_and_pattern_extraction.ipynb
    ├── 02_embeddings_and_dependency_parsing.ipynb
    ├── 03_category_classification_classical_ml.ipynb
    ├── 04_rubert_category_classification.ipynb
    └── 05_category_probability_and_kl_filtering.ipynb
```

## Notebook guide

### `01_lexicon_and_pattern_extraction.ipynb`

Lexicon-based extraction, syntactic patterns, aspect/opinion propagation, support filtering, and contextual aspect markers.

### `02_embeddings_and_dependency_parsing.ipynb`

Semantic expansion with Word2Vec and FastText, similarity-threshold experiments, and dependency-based aspect extraction with UDPipe.

### `03_category_classification_classical_ml.ipynb`

Multi-label aspect category classification with CountVectorizer, TF-IDF, Word2Vec, and FastText representations combined with classical ML classifiers.

### `04_rubert_category_classification.ipynb`

Fine-tuning and evaluation of `DeepPavlov/rubert-base-cased` for restaurant aspect category classification.

### `05_category_probability_and_kl_filtering.ipynb`

Category-confidence aggregation, confidence-based candidate filtering, and KL-divergence filtering.

## Technologies

**NLP:** spaCy, Hugging Face Transformers, RuBERT, UDPipe  
**Machine Learning:** scikit-learn, PyTorch  
**Embeddings:** Word2Vec, FastText, Gensim  
**Data processing:** pandas, NumPy  
**Language:** Python

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

External datasets, sentiment lexicons, pretrained embedding files, and parser models are not included in the repository. See `data/README.md` and `models/README.md` for the resources used in the experiments.

## Author

**Ekaterina Leonova**

Master's thesis, 2024  
Department of Theoretical and Applied Linguistics  
Lomonosov Moscow State University
