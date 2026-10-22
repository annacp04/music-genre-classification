# Lab 1 — Music Genre Classification

Anna Casanovas Poirier.

Classification of music into 10 genres using classical machine learning. Several algorithms are compared (Logistic Regression, k-NN, Decision Tree, MLP, Random Forest, Extra Trees, Gradient Boosting, AdaBoost and a Voting Ensemble), combined with different feature selection techniques. All of them are evaluated with 10-fold cross-validation and a held-out test set, and each one is checked for overfitting.

> ## Results
>
> **All the results, plots, discussion and conclusions of the lab are in the two notebooks:**
>
> | Notebook | Content |
> |---|---|
> | [`task_1.ipynb`](task_1.ipynb) | **Task 1**: models trained on the Essentia descriptors (`GenreAll.csv`), hyperparameter tuning, feature selection, overfitting analysis and final model. |
> | [`task_2.ipynb`](task_2.ipynb) | **Task 2**: models trained on `features_30_sec.csv` and `features_3_sec.csv`, plus the research questions. |
>
> The `.py` files only contain the reusable code (classes) that the notebooks call. They do not produce any results by themselves.

---

## Main results

### Task 1 — Essentia descriptors (`GenreAll.csv`)

| Model | Val. accuracy (10-fold CV) | Test accuracy | Test macro-F1 |
|---|---|---|---|
| **Voting Ensemble** (soft: LR + RF + ET) | **0.837 ± 0.039** | **0.878** | **0.879** |
| Logistic Regression + PCA (99% variance) | 0.821 ± 0.045 | 0.842 | 0.843 |
| Decision Tree (worst) | 0.603 ± 0.066 | 0.648 | 0.654 |

The full table for all 9 models and the overfitting analysis (train/validation gaps, validation curves, learning curves and the effect of feature selection) are in `task_1.ipynb`.

### Task 2 — 30-second and 3-second features

| Dataset | Best model | Val. accuracy (10-fold CV) | Test accuracy | Test macro-F1 |
|---|---|---|---|---|
| 30 s clips | Extra Trees + correlation filter | 0.789 | 0.800 | 0.797 |
| 3 s segments (CV grouped by song) | Extra Trees + correlation filter | 0.716 | 0.728 | 0.723 |

| Research question | Answer |
|---|---|
| **Q5 — Is the 3 s accuracy real or inflated?** | Inflated if segments are split at random: all 896 test songs then have other segments in the training set. Accuracy rises by **≈18 points** (CV 0.893 vs 0.716, test 0.905 vs 0.728). The real 3 s accuracy is ≈0.72–0.73. |
| **Q1 — Are 3 seconds enough to identify a genre?** | Only partly. Every algorithm loses accuracy (on average −5.4 points in CV and −8.2 on test). Timbre-defined genres (classical, jazz, metal) are still recognised; rhythm-defined genres (disco, rock, reggae, hip-hop) suffer most. |
| **Q2 — Does combining short windows beat one long window?** | No. Combining the 10 segments of a song (vote / log-sum 0.769 in CV) recovers about 70% of the gap but stays below the 30 s clip (0.789). The difference is not significant for mean, vote and log-sum (McNemar p = 0.07–0.14); max is significantly worse. |
| **Q4 — Does the position within the clip matter?** | No. Single-segment accuracy only varies between 0.69 and 0.73 across the 10 positions (Cochran's Q, p = 0.29). |

---

## Repository structure

```
music-genre-classification/
├── task_1.ipynb                  ← RESULTS of Task 1
├── task_2.ipynb                  ← RESULTS of Task 2 + research questions
├── exploration_datasets.ipynb    ← exploratory analysis of the datasets
│
├── ExploratoryDataAnalysis.py    ← class for data inspection
├── Preprocessing.py              ← class for cleaning, encoding and train/test split
├── FeatureSelection.py           ← factory of feature selection methods
├── Models.py                     ← class to build, tune and evaluate the models
├── grid_config.py                ← hyperparameter grids and grid-search helpers
│
├── GenreAll.csv                  ← dataset: Essentia descriptors (Task 1)
├── features_30_sec.csv           ← dataset: 30 s song descriptors (Task 2)
├── features_3_sec.csv            ← dataset: 3 s segment descriptors (Task 2)
│
└── results/                      ← CSV tables and trained pipelines from Task 1
```

### Notebooks

**`task_1.ipynb`** — Task 1. It loads and preprocesses `GenreAll.csv`, then defines the hyperparameter grid for every model × feature-selector combination and runs the grid search. It saves all the results to `results/`, re-evaluates the tuned model of each algorithm with 10-fold CV, and analyses overfitting with train vs validation gaps, validation curves (Logistic Regression, KNN, Decision Tree, Random Forest), learning curves (Voting Ensemble, Decision Tree). At the end it evaluates the models on the test set and selects the final model.

**`task_2.ipynb`** — Task 2. It preprocesses the 30 s and 3 s datasets so that **both use exactly the same test songs**, and it checks this with an audit function. It then runs the same grid search as Task 1 on the 30 s features and on the 3 s segments (with cross-validation grouped by song), checks overfitting with train vs validation gaps, evaluates the best models on the test set, and answers four research questions: data leakage in the 3 s dataset (Q5), whether 3 seconds are enough (Q1), whether combining the 3 s segments beats the 30 s window (Q2) and whether the position of the segment matters (Q4).

**`exploration_datasets.ipynb`** — First look at the three datasets: shapes, class balance, feature distributions and correlations. It uses `ExploratoryDataAnalysis.py`.

### Python modules

**`ExploratoryDataAnalysis.py`** — Class `ExploratoryDataAnalysis(df, y_name)`. It prints basic information about a dataset (types, statistics, class counts) and plots feature distributions.

**`Preprocessing.py`** — Class `Preprocessing`, which turns a raw dataframe into train/test matrices. Its `run()` method executes the full pipeline:

1. `clean()` — strips column names, drops identifier columns, rows without a label and duplicate rows.
2. `split_X_y()` — separates the features from the target.
3. `split_train_test()` — stratified 80/20 split. **With `group_col="filename"` the split is done by song**, so all the 3 s segments of a song end up on the same side. With `test_groups=...` you can force the test songs, which is how Task 2 uses the same test set for the 30 s and 3 s datasets.
4. `encode_categorical()` — one-hot encoding, learned on train only.
5. `remove_constant_columns()` — drops zero-variance columns.
6. `encode_target()` — label-encodes the genres.

Accessors: `get_data()`, `get_groups()` (song ID of each row) and `get_meta()` (song and segment index).

**`FeatureSelection.py`** — Factory of feature selectors that can be placed inside a scikit-learn `Pipeline`. `make(method, k)` returns one selector, and `catalog()` returns a set of them. The available methods are:

| Family | Methods |
|---|---|
| Baseline | `none` (no selection) |
| Feature extraction | `pca_var` (PCA keeping a % of variance), `lda` |
| Filters (ranking) | `anova` (F-score), `mutual_info` |
| Filters (redundancy) | `correlation` (removes features with \|r\| above a threshold) |
| Embedded | `l1_logistic` (L1-regularised logistic regression), `rf` / `et` (tree importances) |

The docstring at the top of the file explains why this subset of techniques was chosen.

**`grid_config.py`** — Hyperparameter grid of each model (`MODEL_GRIDS`), the feature selectors tried with each one (`SELECTORS_PER_MODEL`) and helpers to build the grids for `Models.grid_search()` and summarise the results.

**`Models.py`** — Class `Models`, the place where every pipeline (`scaler → selector → model`) is built, trained and evaluated. Scaling and feature selection are fitted inside each CV fold, so there is no data leakage. Main methods:

| Method | What it does |
|---|---|
| `build_models()`, `add_model()` | Create the model × selector combinations (9 algorithms available). |
| `evaluate_all()`, `best_per_model()` | 10-fold CV with train accuracy, validation accuracy, gap and macro-F1. |
| `grid_search()` | Hyperparameter search, optionally with nested CV. |
| `plot_gap()`, `validation_curve()`, `learning_curve()`, `sweep_k()` | Overfitting analysis. |
| `feature_stability()` | How often each feature is selected across folds. |
| `paired_comparison()` | Corrected paired t-test (Nadeau & Bengio) between two models. |
| `cv_by_song()`, `aggregate_by_song()`, `evaluate_on_test_by_song()` | Song-level accuracy from 3 s segment predictions. |
| `fit_best()`, `evaluate_on_test()`, `predict()` | Final training on the full train set and evaluation on the test set. |

With `groups=` (song IDs), cross-validation uses `StratifiedGroupKFold`, so segments of the same song never appear in both the training and validation folds.

### Datasets

| File | Rows | Features | Description |
|---|---|---|---|
| `GenreAll.csv` | 993 songs (980 after removing duplicates) | ≈240 | Global Essentia descriptors per song (spectral, rhythmic, tonal and high-level). Used in Task 1. |
| `features_30_sec.csv` | 1,000 songs | 58 | Descriptors computed over the whole 30 s clip (MFCCs, chroma, spectral centroid, tempo…). Used in Task 2. |
| `features_3_sec.csv` | 9,990 segments | 58 | The same descriptors computed on 3 s windows (10 per song). Used in Task 2. |

### `results/` (Task 1)

| File | Content |
|---|---|
| `cv_all_combinations.csv` | CV score of **every** combination tried in the grid search. |
| `cv_<Model>.csv` | The same results, one file per algorithm. |
| `best_per_model.csv` | Best configuration (selector + hyperparameters) of each algorithm. |
| `best_per_model_selector.csv` | Best configuration of each selector within each algorithm. |
| `test_best_per_model.csv` | CV accuracy, test accuracy and test macro-F1 of the best configuration of each algorithm. |
| `pipelines/*.joblib` | Tuned pipelines trained on the full training set, which can be loaded with `joblib.load()`. |

---

## How to run

### Google Colab

Add this cell at the top of a notebook so that the modules and datasets are available:

```python
!git clone https://github.com/annacp04/music-genre-classification.git
%cd music-genre-classification
```

Then run all the cells in order. The two task notebooks are independent.

### Locally

```bash
git clone https://github.com/annacp04/music-genre-classification.git
cd music-genre-classification
pip install numpy pandas scikit-learn scipy matplotlib seaborn joblib
jupyter notebook
```