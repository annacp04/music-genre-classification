import warnings

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import (
    SelectKBest,
    SelectFromModel,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

"""

We are aware that the course covered a wider range of feature selection techniques,
but we deliberately restricted our experiments to a representative subset,
given the characteristics of our dataset and the cost of combining every selector
with nine models and their hyperparameter grids.

We kept one baseline (no selection) plus one or two methods from each relevant family:
a correlation filter to remove redundancy, ANOVA and mutual information as complementary
ranking filters (linear and non-linear), PCA and LDA as feature extraction methods,
and L1-regularized logistic regression and random forest importances as embedded methods.

"""


# Utilities (score calculation)
def _anova_score(X, y):
    """ANOVA F-score; constant columns (NaN) are converted to 0."""
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore")
        return np.nan_to_num(f_classif(X, y)[0], nan=0.0, posinf=0.0)


def _safe_abs_corr(X):
    """Absolute Pearson correlation between columns; constant columns are set to 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(np.abs(np.corrcoef(X, rowvar=False)))


def _filter_scores(name, X, y, random_state):
    if name == "mutual_info":
        return mutual_info_classif(X, y, random_state=random_state)
    raise ValueError(f"Unknown ranking criterion: {name}")


def _l1_logistic(C, random_state):
    """L1 logistic regression (saga). In sklearn >= 1.8, L1 is specified using l1_ratio=1."""
    import sklearn

    major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
    common = dict(solver="saga", C=C, max_iter=10000, tol=1e-3, random_state=random_state)
    if (major, minor) >= (1, 8):
        return LogisticRegression(l1_ratio=1.0, **common)
    return LogisticRegression(penalty="l1", **common)


# Custom selectors
class _SupportSelector(BaseEstimator, TransformerMixin):
    """Base class: subclasses set self.support_ (a Boolean mask) during fit."""

    def get_support(self, indices=False):
        return np.flatnonzero(self.support_) if indices else self.support_

    def transform(self, X):
        return np.asarray(X)[:, self.support_]


class CorrelationFilter(_SupportSelector):
    """Removes highly correlated features (|r| > threshold), keeping the first one."""

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        corr = _safe_abs_corr(X)
        keep = np.ones(corr.shape[1], dtype=bool)
        for i in range(corr.shape[1]):
            if not keep[i]:
                continue
            dup = np.where(corr[i, i + 1:] > self.threshold)[0] + i + 1
            keep[dup] = False
        self.support_ = keep
        self.n_features_in_ = corr.shape[1]
        return self


class FilterRanker(_SupportSelector):
    """
    Ranking filter: scores each feature individually and keeps the top k features.
    score: 'mutual_info'
    """

    def __init__(self, score="mutual_info", k=50, random_state=0):
        self.score = score
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        s = np.nan_to_num(
            _filter_scores(self.score, X, np.asarray(y), self.random_state),
            nan=0.0,
        )
        order = np.argsort(-s, kind="stable")[: min(self.k, X.shape[1])]
        support = np.zeros(X.shape[1], dtype=bool)
        support[order] = True
        self.scores_ = s
        self.support_ = support
        self.n_features_in_ = X.shape[1]
        return self


class FeatureSelection:
    """
    Creates feature selectors.

        fs = FeatureSelection()
        fs.make("anova", k=50)                          -> a selector
        fs.catalog(k=50, families=["filter_rank"])      -> {name: selector}

    Any selector returned by this class can be passed to Models:
        models.random_forest(selector=fs.make("l1_logistic", C=0.1))

    Baseline
        No selection ................ "none"

    Feature extraction (not feature selection)
        PCA (k = variance fraction) . "pca_var"
        LDA (supervised) ............ "lda"

    Filters: individual ranking
        Statistical distance ........ "anova" (Fisher/F)
        Information .................. "mutual_info"

    Filters: redundancy
        Correlation (|r| > threshold) "correlation"

    Embedded methods
        Direct optimization ......... "l1_logistic"
        Feature importances .......... "rf", "et"
    """

    FAMILIES = {
        "baseline": "Baseline",
        "extraction": "Feature extraction",
        "filter_rank": "Filter (ranking)",
        "filter_subset": "Filter (redundancy)",
        "embedded": "Embedded",
    }

    METHODS = {
        "baseline": ["none"],
        "extraction": ["pca_var", "lda"],
        "filter_rank": ["anova", "mutual_info"],
        "filter_subset": ["correlation"],
        "embedded": ["l1_logistic", "rf", "et"],
    }

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.family = {}          # Catalog name -> family

    def list_methods(self):
        for fam, methods in self.METHODS.items():
            print(f"{self.FAMILIES[fam]:<38} {', '.join(methods)}")

    def make(self, method, k=50, **kwargs):
        """
        Returns a selector for the 'selector' step of a Pipeline.

        k: number of features (anova, mutual_info, rf, et) or fraction of
           variance (pca_var). Not used for correlation, l1_logistic, or lda.
        kwargs: method-specific parameters (threshold, C, n_components...).
        """
        rs = self.random_state

        if method in ("none", "passthrough"):
            return "passthrough"

        # ---- Feature extraction
        if method == "pca_var":
            return PCA(n_components=k, svd_solver="full", random_state=rs)
        if method == "lda":
            # n_components=None -> number of classes - 1 (9 with 10 genres)
            return LinearDiscriminantAnalysis(n_components=kwargs.get("n_components"))

        # ---- Ranking filters
        if method == "anova":
            return SelectKBest(_anova_score, k=k)
        if method == "mutual_info":
            return FilterRanker(score="mutual_info", k=k, random_state=rs)

        # ---- Redundancy filters
        if method == "correlation":
            return CorrelationFilter(threshold=kwargs.get("threshold", 0.95))

        # ---- Embedded methods
        if method == "l1_logistic":
            return SelectFromModel(_l1_logistic(kwargs.get("C", 0.1), rs))
        if method == "rf":
            return SelectFromModel(
                RandomForestClassifier(n_estimators=200, random_state=rs, n_jobs=1),
                max_features=k, threshold=-np.inf,
            )
        if method == "et":
            return SelectFromModel(
                ExtraTreesClassifier(n_estimators=200, random_state=rs, n_jobs=1),
                max_features=k, threshold=-np.inf,
            )

        raise ValueError(f"Unknown method: {method}. See list_methods().")

    def catalog(self, k=50, families=None):
        """
        Selected techniques grouped by family.
        Returns {name: selector}, ready for models.build_models(..., selectors=...).

        families: subset of {'baseline', 'extraction', 'filter_rank',
                  'filter_subset', 'embedded'} (all families by default).
        """
        fam = set(families) if families else set(self.FAMILIES)
        entries = []

        def add(family, name, selector):
            if family in fam:
                entries.append((family, name, selector))

        mk = self.make

        add("baseline", "No selection", mk("none"))

        add("extraction", "PCA 95% variance", mk("pca_var", 0.95))
        add("extraction", "LDA", mk("lda"))

        add("filter_rank", f"ANOVA k={k}", mk("anova", k))
        add("filter_rank", f"Mutual information k={k}", mk("mutual_info", k))

        add("filter_subset", "Correlation <0.95", mk("correlation", threshold=0.95))

        add("embedded", "L1 logistic regression", mk("l1_logistic"))
        add("embedded", f"RF importance k={k}", mk("rf", k))

        self.family.update({name: self.FAMILIES[f] for f, name, _ in entries})
        return {name: sel for _, name, sel in entries}
