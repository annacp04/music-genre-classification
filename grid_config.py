
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.model_selection import ParameterGrid
from FeatureSelection import *
import pandas as pd
# Hyperparameters of each model to try

MODEL_GRIDS = {
    "Logistic Regression": {                      
        "model__C": [0.01, 0.1, 1],
    },
    "KNN": {                                      
        "model__n_neighbors": [5, 7, 9, 13],
        "model__weights": ["uniform", "distance"],
    },
    "Decision Tree": {                            
        "model__max_depth": [None, 10],
        "model__min_samples_leaf": [1, 5],
        "model__criterion": ["gini", "entropy"],
    },
    "ANN (MLP)": {                                
        "model__hidden_layer_sizes": [(128,), (128, 64)],
        "model__alpha": [0.001, 0.01],
    },
    "Random Forest": {                            
        "model__min_samples_leaf": [1, 3],
        "model__max_features": ["sqrt", 0.2],
    },
    "Extra Trees": {                             
        "model__min_samples_leaf": [1, 3],
        "model__max_features": ["sqrt", 0.2],
    },
    "Gradient Boosting": {                       
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8],
    },
    "AdaBoost": {                                
        "model__n_estimators": [200],
        "model__learning_rate": [0.1, 0.5],
        "model__estimator__max_depth": [2, 3],
    },
    "Voting Ensemble": {                          
        "model__voting": ["soft"],
        "model__weights": [None, [2, 1, 1]],
    },
}


# Which selectors to try with each model

SELECTORS_PER_MODEL = {
    "Logistic Regression": ["none", "anova", "pca_var", "l1_logistic"],
    "KNN": ["none", "lda", "pca_var", "l1_logistic"],
    "Decision Tree": ["none", "correlation", "rf"],
    "ANN (MLP)": ["none", "pca_var"],
    "Random Forest": ["none", "correlation"],
    "Extra Trees": ["none", "correlation"],
    "Gradient Boosting": ["none", "correlation"],
    "AdaBoost": ["none"],
    "Voting Ensemble": ["none"],
}


# Search space of each selector
def selector_spaces(fs, k_values=(50, 100, 150)):
    k = list(k_values)
    return {
        "none": {"selector": ["passthrough"]},
        "correlation": {
            "selector": [fs.make("correlation")],
            "selector__threshold": [0.90],
        },
        "anova": {"selector": [fs.make("anova")], "selector__k": k},
        "mutual_info": {"selector": [fs.make("mutual_info")], "selector__k": [100]},
        "pca_var": {
            "selector": [fs.make("pca_var", 0.95)],
            "selector__n_components": [0.95, 0.99],
        },
        "lda": {"selector": [fs.make("lda")]},
        "l1_logistic": {
            "selector": [fs.make("l1_logistic")],
            "selector__estimator__C": [0.1],
        },
        "rf": {"selector": [fs.make("rf")], "selector__max_features": [50]},
    }


def build_param_grids(fs, model_grids=None, k_values=(50, 100, 150),
                      selectors_per_model=None):
    """
    {model name: [one dict per selector]} ready for models.grid_search().

    selectors_per_model: {model: [selector keys]}; defaults to SELECTORS_PER_MODEL.
    Models missing from it get all selectors.
    """
    model_grids = MODEL_GRIDS if model_grids is None else model_grids
    selectors_per_model = (
        SELECTORS_PER_MODEL if selectors_per_model is None else selectors_per_model
    )
    spaces = selector_spaces(fs, k_values)

    grids = {}
    for model, mgrid in model_grids.items():
        keys = selectors_per_model.get(model, list(spaces))
        grids[model] = [{**spaces[s], **mgrid} for s in keys]
    return grids



def print_grid_sizes(grids, inner_splits=10, outer_splits=10):
    """Number of combinations and fits per model and in total."""
    rows = []
    for name, grid in grids.items():
        n = len(ParameterGrid(grid))
        rows.append({
            "Model": name,
            "Combinaciones": n,
            "Fits (nested=False)": n * inner_splits,
            "Fits (nested=True)": n * inner_splits * (outer_splits + 1),
        })
    df = pd.DataFrame(rows)
    total = df.drop(columns="Model").sum()
    df.loc[len(df)] = ["TOTAL", *total.values]
    print(df.to_string(index=False))
    return df


def selector_label(sel):
    if isinstance(sel, str):
        return "Sin selección"
    if isinstance(sel, SelectKBest):
        return "ANOVA"
    if isinstance(sel, FilterRanker):
        return "Info mutua"
    if isinstance(sel, CorrelationFilter):
        return "Correlación"
    if isinstance(sel, PCA):
        return "PCA var"
    if isinstance(sel, LinearDiscriminantAnalysis):
        return "LDA"
    if isinstance(sel, SelectFromModel):
        return "RF importancia" if isinstance(sel.estimator, RandomForestClassifier) else "L1 logística"
    return type(sel).__name__


def best_per_selector(summary):
    """Best combination of each selector, from the `summary` of grid_search()."""
    df = summary["cv_results"].copy()
    df["Selector"] = df["params"].map(lambda p: selector_label(p["selector"]))
    df["Params"] = df["params"].map(
        lambda p: {k: v for k, v in p.items() if k != "selector"}
    )
    cols = ["Selector", "mean_test_score", "std_test_score"]
    if "mean_train_score" in df:              # only if return_train_score=True
        df["Gap"] = df["mean_train_score"] - df["mean_test_score"]
        cols += ["mean_train_score", "Gap"]
    best = (
        df.sort_values("mean_test_score", ascending=False)
        .groupby("Selector", sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    return best[cols + ["Params"]]