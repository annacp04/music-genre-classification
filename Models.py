import random
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from scipy import stats

from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
    PredefinedSplit,
    cross_val_predict,
    cross_validate,
    cross_val_score,
    train_test_split,
    learning_curve,
    validation_curve,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    get_scorer,
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)

from FeatureSelection import *

class Models:
    """
    Único sitio donde se construyen, entrenan y evalúan pipelines:

        imputer -> scaler -> selector -> modelo

    El selector viene de FeatureSelection (feature_selection.py). Puedes
    combinar varios modelos con varios selectores y Models los evalúa todos
    con la misma CV sobre el TRAIN, midiendo también el overfitting
    (accuracy en train vs validación).

    Modelos: "logistic", "knn", "tree", "ann", "random_forest", "extra_trees",
             "gradient_boosting", "adaboost", "voting"

    Flujo típico:
        fs = models.fs
        models.build_models(
            ["logistic", "random_forest"],
            selectors={"Sin selección": "passthrough",
                       "ANOVA k=50": fs.make("anova", 50)},
        )
        models.evaluate_all()          # todas las combinaciones
        models.best_per_model()        # mejor selector para cada modelo
        models.grid_search("Random Forest | ANOVA k=50",
                           {"selector__k": [30, 50, 100],
                            "model__min_samples_leaf": [1, 3, 5]})
        models.evaluate_on_test(X_test, y_test)
    """

    AVAILABLE_MODELS = {
        "logistic": ("logistic", "Logistic Regression"),
        "knn": ("knn", "KNN"),
        "tree": ("tree", "Decision Tree"),
        "ann": ("ann", "ANN (MLP)"),
        "random_forest": ("random_forest", "Random Forest"),
        "extra_trees": ("extra_trees", "Extra Trees"),
        "gradient_boosting": ("gradient_boosting", "Gradient Boosting"),
        "adaboost": ("adaboost", "AdaBoost"),
        "voting": ("voting", "Voting Ensemble"),
    }

    TABLE_COLUMNS = [
        "Name", "Train Acc", "Val Acc", "Val Acc Std", "Gap", "Val F1", "N features",
    ]

    def __init__(
        self,
        X_train,
        y_train,
        label_encoder=None,
        n_splits=10,
        random_state=42,
        n_jobs=-1,
        models=None,
        selectors=None,
        groups=None,
        cv_mode="group",
        fold_ids=None,
    ):
        """
        groups:   id de canción de cada fila (Preprocessing.get_groups()[0]). Con él,
                  la CV mantiene todos los segmentos de una canción en el mismo fold.
        cv_mode:  "group" (por canción, sin leakage) | "random" (folds por segmento,
                  CON leakage: solo para medir cuánto infla; ver leakage_check()).
        fold_ids: fold de cada fila impuesto desde fuera (ver make_song_folds) para
                  validar varios datasets con las mismas canciones.
        """
        if isinstance(X_train, pd.DataFrame):
            self.X = X_train.reset_index(drop=True).copy()
        else:
            self.X = pd.DataFrame(X_train)

        if isinstance(y_train, pd.DataFrame):
            if y_train.shape[1] != 1:
                raise ValueError("y must have a single column of labels.")
            y_train = y_train.iloc[:, 0]
        self.y = pd.Series(np.asarray(y_train).ravel())

        if len(self.X) != len(self.y):
            raise ValueError("X and y must have the same number of rows.")
        if self.y.isna().any():
            raise ValueError("y contains missing values.")
        if self.X.shape[1] == 0:
            raise ValueError("X has no features.")

        non_numeric = self.X.select_dtypes(exclude="number").columns.tolist()
        if non_numeric:
            raise ValueError(
                f"X has non-numeric columns: {non_numeric}. "
                "Run Preprocessing.encode_categorical() first."
            )
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")

        # --- grupos (segmentos de una misma canción) ----------------------
        if cv_mode not in ("group", "random"):
            raise ValueError("cv_mode must be 'group' or 'random'.")
        self.groups = (
            None if groups is None
            else pd.Series(np.asarray(groups)).reset_index(drop=True)
        )
        if self.groups is not None and len(self.groups) != len(self.y):
            raise ValueError("groups and y must have the same number of rows.")
        if fold_ids is not None and len(fold_ids) != len(self.y):
            raise ValueError("fold_ids and y must have the same number of rows.")

        # el mínimo por clase se cuenta en canciones, no en segmentos
        y_units = self.y if self.groups is None else self.y.groupby(self.groups).first()
        min_count = y_units.value_counts().min()
        if min_count < n_splits:
            raise ValueError(
                f"For {n_splits}-fold CV each class needs at least "
                f"{n_splits} samples (songs). Current minimum: {min_count}."
            )

        self.label_encoder = label_encoder
        self.n_splits = n_splits
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.cv_mode = cv_mode
        self._use_groups = self.groups is not None and cv_mode == "group"
        if fold_ids is not None:
            # folds impuestos desde fuera (p. ej. los mismos en 3 s y 30 s)
            self.cv = PredefinedSplit(np.asarray(fold_ids))
            self.n_splits = self.cv.get_n_splits()
        elif self._use_groups:
            self.cv = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
        else:
            self.cv = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
        # kwargs de groups para cross_validate / cv.split (vacío si no hacen falta)
        self._gkw = (
            {"groups": self.groups} if self._use_groups and fold_ids is None else {}
        )

        # Catálogo de selectores: models.fs.make(...), models.fs.catalog(...)
        self.fs = FeatureSelection(random_state=random_state)

        self.models = {}
        self.configs = {}
        self.results = None
        self.best_model_name = None
        self.best_model = None

        self.build_models(models, selectors)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def _resolve_selector(self, selector):
        """None / 'passthrough' / 'none' -> sin selección; texto -> fs.make(texto)."""
        if selector is None:
            return "passthrough"
        if isinstance(selector, str):
            if selector in ("passthrough", "none"):
                return "passthrough"
            return self.fs.make(selector)
        return selector

    def _make_pipeline(self, estimator, selector=None, scale=True, scaler=None):
        """
        scaler -> selector -> model

        El scaler y el selector se ajustan dentro de cada fold
        (solo con su parte de train), así que no hay data leakage.
        """
        selector = self._resolve_selector(selector)
        has_selector = not isinstance(selector, str)

        steps = []
        if scale or has_selector:
            steps.append(
                ("scaler", clone(scaler) if scaler is not None else StandardScaler())
            )
        steps.append(("selector", clone(selector) if has_selector else "passthrough"))
        steps.append(("model", estimator))
        return Pipeline(steps)

    # ------------------------------------------------------------------
    # Un método por modelo. Hiperparámetros como keywords, más:
    #   selector=...  -> selector de FeatureSelection (o su nombre, p. ej. "anova")
    #   scale / scaler
    # ------------------------------------------------------------------
    def logistic(self, selector=None, scale=True, scaler=None, **params):
        params = {"max_iter": 3000, "random_state": self.random_state, **params}
        return self._make_pipeline(LogisticRegression(**params), selector, scale, scaler)

    def knn(self, selector=None, scale=True, scaler=None, **params):
        params = {"n_neighbors": 5, "weights": "distance", **params}
        return self._make_pipeline(KNeighborsClassifier(**params), selector, scale, scaler)

    def tree(self, selector=None, scale=True, scaler=None, **params):
        params = {"class_weight": "balanced", "random_state": self.random_state, **params}
        return self._make_pipeline(DecisionTreeClassifier(**params), selector, scale, scaler)

    def ann(self, selector=None, scale=True, scaler=None, **params):
        params = {
            "hidden_layer_sizes": (128, 64),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.001,
            "learning_rate_init": 0.001,
            "max_iter": 500,
            "early_stopping": True,
            "random_state": self.random_state,
            **params,
        }
        return self._make_pipeline(MLPClassifier(**params), selector, scale, scaler)

    def random_forest(self, selector=None, scale=True, scaler=None, **params):
        params = {
            "n_estimators": 300,
            "class_weight": "balanced",
            "random_state": self.random_state,
            "n_jobs": 1,
            **params,
        }
        return self._make_pipeline(RandomForestClassifier(**params), selector, scale, scaler)

    def extra_trees(self, selector=None, scale=True, scaler=None, **params):
        params = {
            "n_estimators": 300,
            "class_weight": "balanced",
            "random_state": self.random_state,
            "n_jobs": 1,
            **params,
        }
        return self._make_pipeline(ExtraTreesClassifier(**params), selector, scale, scaler)

    def gradient_boosting(self, selector=None, scale=True, scaler=None, **params):
        params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": self.random_state,
            **params,
        }
        return self._make_pipeline(GradientBoostingClassifier(**params), selector, scale, scaler)

    def adaboost(self, selector=None, scale=True, scaler=None, base_max_depth=2, **params):
        if "estimator" not in params:
            params["estimator"] = DecisionTreeClassifier(
                max_depth=base_max_depth, random_state=self.random_state
            )
        params = {
            "n_estimators": 100,
            "learning_rate": 0.5,
            "random_state": self.random_state,
            **params,
        }
        return self._make_pipeline(AdaBoostClassifier(**params), selector, scale, scaler)

    def voting(
        self,
        members=("logistic", "random_forest", "extra_trees"),
        voting="soft",
        selector=None,
        scale=True,
        scaler=None,
        **params,
    ):
        """
        members: lista de claves o dict {clave: {hiperparámetros}}.
        El selector se aplica UNA vez antes del voting: todos los miembros
        reciben las mismas variables.
        """
        if not isinstance(members, dict):
            members = {m: {} for m in members}

        estimators = []
        for key, member_params in members.items():
            if key == "voting":
                raise ValueError("A voting ensemble cannot contain another voting.")
            method_name, _ = self._resolve_model_key(key)
            estimators.append((key, getattr(self, method_name)(**(member_params or {}))))

        params = {"voting": voting, "n_jobs": 1, **params}
        return self._make_pipeline(
            VotingClassifier(estimators=estimators, **params), selector, scale, scaler
        )

    # ------------------------------------------------------------------
    # Elegir qué se compara
    # ------------------------------------------------------------------
    def _resolve_model_key(self, key):
        if key not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model: '{key}'. Available: {list(self.AVAILABLE_MODELS)}"
            )
        return self.AVAILABLE_MODELS[key]

    def list_available_models(self):
        return list(self.AVAILABLE_MODELS)

    def build_models(self, selection=None, selectors=None, append=False):
        """
        Crea todas las combinaciones modelo x selector.

        selection:
            None                         -> todos los modelos, parámetros por defecto
            ["knn", "logistic"]          -> esos, parámetros por defecto
            {"knn": {"n_neighbors": 9}}  -> esos, con tus hiperparámetros
        selectors:
            None                         -> sin selección (nombres = solo el modelo)
            {"ANOVA k=50": fs.make("anova", 50), "Sin selección": "passthrough"}
            models.fs.catalog(k=50, families=[...])
            ["anova", "pca"]             -> nombres de método con k por defecto
        append: True añade a lo que ya hubiera en lugar de reemplazarlo.

        Nombres resultantes: "Random Forest | ANOVA k=50".
        """
        if selection is None:
            selection = {key: {} for key in self.AVAILABLE_MODELS}
        elif not isinstance(selection, dict):
            selection = {key: {} for key in selection}

        plain = selectors is None
        if plain:
            selectors = {"Sin selección": "passthrough"}
        elif not isinstance(selectors, dict):
            selectors = {s: s for s in selectors}

        if not append:
            self.models = {}
            self.configs = {}
            self.best_model_name = None
            self.best_model = None

        for key, params in selection.items():
            method_name, display = self._resolve_model_key(key)
            for sel_name, sel in selectors.items():
                name = display if plain else f"{display} | {sel_name}"
                if name in self.models:
                    raise ValueError(f"A model named '{name}' already exists.")
                self.models[name] = getattr(self, method_name)(
                    selector=sel, **(params or {})
                )
                self.configs[name] = {
                    "Model": display,
                    "Selector": sel_name,
                    "Family": self.fs.family.get(sel_name, "—"),
                }

        self.results = None
        return self

    def add_model(self, name, pipeline, model=None, selector=None):
        """
        Añade un pipeline ya construido (p. ej. otra configuración del mismo modelo):
            models.add_model("RF leaf=5 | ANOVA",
                             models.random_forest(selector="anova", min_samples_leaf=5),
                             model="Random Forest", selector="ANOVA k=50")
        """
        if name in self.models:
            raise ValueError(f"A model named '{name}' already exists.")
        self.models[name] = pipeline
        self.configs[name] = {
            "Model": model or name,
            "Selector": selector or "—",
            "Family": self.fs.family.get(selector, "—"),
        }
        self.results = None
        return self

    def remove_model(self, name):
        self.models.pop(name)
        self.configs.pop(name, None)
        self.results = None
        return self

    def _get(self, name):
        if name not in self.models:
            raise KeyError(f"'{name}' not found. Available: {list(self.models)}")
        return self.models[name]

    # ------------------------------------------------------------------
    # CV con train y validación
    # ------------------------------------------------------------------
    @staticmethod
    def _n_selected(fitted_pipe, n_total):
        sel = fitted_pipe.named_steps["selector"]
        if isinstance(sel, str):
            return n_total
        if hasattr(sel, "get_support"):
            return int(np.sum(sel.get_support()))
        if hasattr(sel, "n_components_"):
            return int(sel.n_components_)
        try:
            return int(sel.transform(np.zeros((1, n_total))).shape[1])
        except Exception:
            return n_total

    def _cv(self, pipe):
        s = cross_validate(
            pipe, self.X, self.y,
            cv=self.cv,
            scoring={"acc": "accuracy", "f1": "f1_macro"},
            return_train_score=True,
            return_estimator=True,
            n_jobs=self.n_jobs,
            error_score="raise",
            **self._gkw,
        )
        n_feat = np.mean([self._n_selected(e, self.X.shape[1]) for e in s["estimator"]])
        return {
            "Train Acc": s["train_acc"].mean(),
            "Val Acc": s["test_acc"].mean(),
            "Val Acc Std": s["test_acc"].std(),
            "Gap": s["train_acc"].mean() - s["test_acc"].mean(),
            "Val F1": s["test_f1"].mean(),
            "Val F1 Std": s["test_f1"].std(),
            "N features": n_feat,
        }

    # ------------------------------------------------------------------
    # Evaluación de todas las combinaciones
    # ------------------------------------------------------------------
    def evaluate_all(self, names=None):
        """
        Evalúa con la misma CV todos los pipelines (o solo `names`).
        Train Acc = accuracy en los folds de entrenamiento
        Val Acc   = accuracy en el fold de validación (NO es el test)
        Gap       = Train - Val  -> cuanto mayor, más overfitting
        """
        if not self.models:
            raise RuntimeError("No models. Use build_models() or add_model().")

        names = names or list(self.models)
        print(
            f"Evaluating {len(names)} pipelines with "
            f"{self.n_splits}-fold stratified CV on the training set..."
        )

        rows = []
        for name in names:
            print(f"  - {name}")
            info = self.configs.get(name, {"Model": name, "Selector": "—", "Family": "—"})
            rows.append({"Name": name, **info, **self._cv(self._get(name))})

        self.results = (
            pd.DataFrame(rows)
            .sort_values("Val Acc", ascending=False)
            .reset_index(drop=True)
        )
        self.best_model_name = self.results.loc[0, "Name"]

        print("\n" + "=" * 90)
        print("CROSS-VALIDATION RESULTS")
        print("=" * 90)
        print(
            self.results[self.TABLE_COLUMNS].to_string(
                index=False, float_format="{:.4f}".format
            )
        )
        print(f"\nBest mean validation accuracy: {self.best_model_name}")
        return self.results

    def _require_results(self):
        if self.results is None:
            raise RuntimeError("Run evaluate_all() first.")

    def best_per_model(self):
        """La mejor combinación (selector) de cada modelo, según Val Acc."""
        self._require_results()
        best = (
            self.results.sort_values("Val Acc", ascending=False)
            .groupby("Model", sort=False)
            .head(1)
            .reset_index(drop=True)
        )
        print(
            best[["Model", "Selector", "Train Acc", "Val Acc", "Gap", "N features"]]
            .to_string(index=False, float_format="{:.4f}".format)
        )
        return best

    # ------------------------------------------------------------------
    # Gráficos
    # ------------------------------------------------------------------
    def plot_results(self, metric="Val Acc"):
        """Barras horizontales de 'Val Acc' o 'Val F1' con su desviación."""
        self._require_results()
        d = self.results.sort_values(metric, ascending=True)
        std_col = f"{metric} Std"
        plt.figure(figsize=(11, 0.4 * len(d) + 2))
        plt.barh(
            d["Name"], d[metric],
            xerr=d[std_col] if std_col in d else None, capsize=3,
        )
        plt.xlabel(f"{metric} ({self.n_splits} folds)")
        plt.title(f"Comparison by {metric}")
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.show()

    def plot_f1_results(self):
        self.plot_results("Val F1")

    def plot_gap(self, top=None):
        """Train vs validación por pipeline: la distancia entre barras es el overfitting."""
        self._require_results()
        d = self.results.head(top) if top else self.results
        d = d.iloc[::-1]
        y = np.arange(len(d))
        plt.figure(figsize=(10, 0.55 * len(d) + 2))
        plt.barh(y + 0.2, d["Train Acc"], height=0.4, label="Train")
        plt.barh(y - 0.2, d["Val Acc"], height=0.4, label="Validation (CV)")
        plt.yticks(y, d["Name"])
        plt.xlim(0, 1.05)
        plt.xlabel("Accuracy")
        plt.title("Train vs validation (overfitting)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_grid(self):
        #Heatmaps selector x modelo de Val Acc y Gap.
        self._require_results()
        fig, axes = plt.subplots(
            1, 2, figsize=(14, 0.45 * self.results["Selector"].nunique() + 2)
        )
        for ax, col, cmap in zip(axes, ["Val Acc", "Gap"], ["viridis", "Reds"]):
            piv = self.results.pivot_table(
                index="Selector", columns="Model", values=col, aggfunc="first"
            )
            sns.heatmap(piv, annot=True, fmt=".3f", cmap=cmap, ax=ax)
            ax.set_title(col)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Análisis de overfitting y selección
    # ------------------------------------------------------------------
    def sweep_k(
        self,
        model_key,
        method="anova",
        k_values=(5, 10, 20, 30, 50, 75, 100, 150, 200),
        model_params=None,
        plot=True,
        **selector_kwargs,
    ):
        """
        Train y validación en función del nº de variables k para un modelo
        y un método de selección. Ej.: models.sweep_k("logistic", "anova").
        """
        method_name, display = self._resolve_model_key(model_key)
        model_params = model_params or {}
        build = getattr(self, method_name)

        rows = []
        for k in k_values:
            if k > self.X.shape[1]:
                continue
            print(f"{display} + {method} k={k}")
            pipe = build(selector=self.fs.make(method, k, **selector_kwargs), **model_params)
            rows.append({"k": k, **self._cv(pipe)})

        ref = self._cv(build(**model_params))
        df = pd.DataFrame(rows)

        if plot and len(df):
            plt.figure(figsize=(9, 5))
            plt.plot(df["k"], df["Train Acc"], "o-", label="Train")
            plt.plot(df["k"], df["Val Acc"], "o-", label="Validation (CV)")
            plt.fill_between(
                df["k"], df["Val Acc"] - df["Val Acc Std"],
                df["Val Acc"] + df["Val Acc Std"], alpha=0.2,
            )
            plt.axhline(
                ref["Val Acc"], color="gray", ls="--",
                label=f"Validation without selection ({ref['Val Acc']:.3f})",
            )
            plt.xlabel("Number of features / components (k)")
            plt.ylabel("Accuracy")
            plt.title(f"{display} — sweep of k ({method})")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        return df

    def learning_curve(self, name):
        """Si train >> validación y no converge al añadir datos, hay overfitting."""
        sizes, tr, va = learning_curve(
            self._get(name), self.X, self.y,
            cv=self.cv, train_sizes=np.linspace(0.1, 1.0, 8),
            scoring="accuracy", n_jobs=self.n_jobs, **self._gkw,
        )
        plt.figure(figsize=(8, 5))
        for vals, label in [(tr, "Train"), (va, "Validation (CV)")]:
            m, s = vals.mean(axis=1), vals.std(axis=1)
            plt.plot(sizes, m, "o-", label=label)
            plt.fill_between(sizes, m - s, m + s, alpha=0.2)
        plt.xlabel("Training samples")
        plt.ylabel("Accuracy")
        plt.title(f"Learning curve — {name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        return pd.DataFrame({
            "Train samples": sizes,
            "Train": tr.mean(axis=1),
            "Validation": va.mean(axis=1),
            "Gap": tr.mean(axis=1) - va.mean(axis=1),
        })

    def validation_curve(self, name, param_name, param_range, log_x=False):
        """
        Curva de un hiperparámetro: 'model__C', 'model__max_depth',
        'model__min_samples_leaf', 'selector__k', ...
        """
        tr, va = validation_curve(
            self._get(name), self.X, self.y,
            param_name=param_name, param_range=param_range,
            cv=self.cv, scoring="accuracy", n_jobs=self.n_jobs, **self._gkw,
        )
        x = param_range if log_x else range(len(param_range))
        plt.figure(figsize=(8, 5))
        for vals, label in [(tr, "Train"), (va, "Validation (CV)")]:
            m, s = vals.mean(axis=1), vals.std(axis=1)
            plt.plot(x, m, "o-", label=label)
            plt.fill_between(x, m - s, m + s, alpha=0.2)
        if log_x:
            plt.xscale("log")
        else:
            plt.xticks(list(x), [str(p) for p in param_range])
        plt.xlabel(param_name)
        plt.ylabel("Accuracy")
        plt.title(f"Validation curve — {name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        return pd.DataFrame({
            param_name: list(param_range),
            "Train": tr.mean(axis=1),
            "Validation": va.mean(axis=1),
            "Val Std": va.std(axis=1),
        })

    def paired_comparison(self, name_a, name_b):
        """
        ¿A es realmente mejor que B? t-test pareado corregido (Nadeau & Bengio)
        sobre los mismos folds. Con p > 0.05 la diferencia puede ser ruido.
        """
        def fold_scores(name):
            return cross_val_score(
                self._get(name), self.X, self.y,
                cv=self.cv, scoring="accuracy", n_jobs=self.n_jobs, **self._gkw,
            )

        a, b = fold_scores(name_a), fold_scores(name_b)
        d = a - b
        k, n = len(d), len(self.y)
        n_test = n / k
        var = d.var(ddof=1)

        if var == 0:
            t, p = (0.0, 1.0) if d.mean() == 0 else (np.inf, 0.0)
        else:
            t = d.mean() / np.sqrt((1.0 / k + n_test / (n - n_test)) * var)
            p = 2 * stats.t.sf(abs(t), k - 1)

        return {
            "A": name_a, "B": name_b,
            "acc_A": a.mean(), "acc_B": b.mean(),
            "mean_diff (A-B)": d.mean(), "t": t, "p_value": p,
            "folds A wins / ties / loses": (
                int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())
            ),
        }

    def feature_stability(self, selector="anova", k=50, threshold=0.8, top=30,
                          plot=True, **selector_kwargs):
        """
        Repite la selección en cada fold y cuenta cuántas veces sale cada variable.
        Las elegidas en >= threshold de los folds son estables.
        selector: nombre de método o selector ya construido. No vale para PCA / LDA.
        """
        counts = np.zeros(self.X.shape[1])

        for tr_idx, _ in self.cv.split(self.X, self.y, **self._gkw):
            sel = (
                self.fs.make(selector, k, **selector_kwargs)
                if isinstance(selector, str) else clone(selector)
            )
            if not hasattr(sel, "fit") or isinstance(sel, (PCA,)):
                raise ValueError("This selector does not select original features.")
            pre = Pipeline([
                ("scaler", StandardScaler()),
                ("selector", sel),
            ])
            pre.fit(self.X.iloc[tr_idx], self.y.iloc[tr_idx])
            fitted = pre.named_steps["selector"]
            if not hasattr(fitted, "get_support"):
                raise ValueError("This selector does not select original features.")
            counts += np.asarray(fitted.get_support()).astype(int)

        df = (
            pd.DataFrame({"Feature": self.X.columns, "Frequency": counts / self.n_splits})
            .sort_values("Frequency", ascending=False)
            .reset_index(drop=True)
        )
        n_stable = int((df["Frequency"] >= threshold).sum())
        print(f"{n_stable} features selected in >= {threshold:.0%} of the folds")

        if plot:
            d = df.head(top).iloc[::-1]
            plt.figure(figsize=(8, 0.28 * len(d) + 1.5))
            plt.barh(d["Feature"].astype(str), d["Frequency"])
            plt.axvline(threshold, color="r", ls="--")
            plt.xlabel("Selection frequency across folds")
            plt.title("Feature selection stability")
            plt.tight_layout()
            plt.show()

        return df

    # Grid search

    def grid_search(self, name, param_grid, nested=True, inner_splits=5,
                    scoring="accuracy", set_best=True):
        """
        Grid search sobre un pipeline existente (un nombre de self.models).

        param_grid con prefijos del pipeline:
            {"model__C": [0.01, 0.1, 1], "selector__k": [30, 50, 100]}
        También puedes cambiar el selector entero dentro del grid:
            [{"selector": [fs.make("anova")], "selector__k": [30, 50]},
             {"selector": ["passthrough"]}]

        nested=True: CV anidada (interna elige parámetros, externa mide).
                     `nested_mean` es el número honesto para reportar.
        nested=False: más rápido, pero best_inner_cv_score es optimista.

        set_best=True: el pipeline ganador (entrenado con todo el train)
        queda en self.best_model, listo para evaluate_on_test().
        """
        base = clone(self._get(name))
        Inner = StratifiedGroupKFold if self._use_groups else StratifiedKFold
        inner = Inner(
            n_splits=inner_splits, shuffle=True, random_state=self.random_state
        )
        gs = GridSearchCV(
            base, param_grid, cv=inner, scoring=scoring,
            n_jobs=1 if nested else self.n_jobs,
        )

        summary = {}
        if nested:
            # Bucle manual: así los grupos llegan también al GridSearchCV interno
            scorer, scores, chosen = get_scorer(scoring), [], []
            for tr, va in self.cv.split(self.X, self.y, **self._gkw):
                g = clone(gs).set_params(n_jobs=self.n_jobs)
                g.fit(self.X.iloc[tr], self.y.iloc[tr], **self._sub(tr))
                scores.append(scorer(g, self.X.iloc[va], self.y.iloc[va]))
                chosen.append(str(g.best_params_))
            summary["nested_mean"] = float(np.mean(scores))
            summary["nested_std"] = float(np.std(scores))
            summary["best_params_per_fold"] = Counter(chosen).most_common()
            print(
                f"[{name}] nested CV {scoring}: "
                f"{summary['nested_mean']:.4f} ± {summary['nested_std']:.4f}"
            )
            print("Parameters chosen per fold:")
            for p, c in summary["best_params_per_fold"]:
                print(f"  {c}x {p}")

        final = clone(gs).set_params(n_jobs=self.n_jobs)
        final.fit(self.X, self.y, **self._sub(slice(None)))

        summary["best_params"] = final.best_params_
        summary["best_inner_cv_score"] = final.best_score_
        summary["cv_results"] = (
            pd.DataFrame(final.cv_results_)
            .sort_values("rank_test_score")
            [["params", "mean_test_score", "std_test_score", "rank_test_score"]]
            .reset_index(drop=True)
        )
        print(f"Best parameters (full train): {final.best_params_}")

        if set_best:
            self.best_model = final.best_estimator_
            self.best_model_name = f"{name} (tuned)"
            print(f"best_model -> {self.best_model_name}")

        return final.best_estimator_, summary

    # ------------------------------------------------------------------
    # Probabilidades out-of-fold, leakage y métricas por canción
    # ------------------------------------------------------------------
    def _sub(self, idx):
        """groups de un subconjunto de filas, para GridSearchCV.fit(..., groups=...)."""
        return {"groups": self.groups.iloc[idx]} if self._use_groups else {}

    def oof_proba(self, name):
        """
        Probabilidad de cada segmento predicha por un modelo que NO lo vio
        (out-of-fold), más el fold de cada fila. Es la entrada de SongAnalysis.
        Devuelve (proba[n_filas, n_clases], folds[n_filas]).
        """
        proba = cross_val_predict(
            self._get(name), self.X, self.y, cv=self.cv,
            method="predict_proba", n_jobs=self.n_jobs, **self._gkw,
        )
        folds = np.empty(len(self.y), dtype=int)
        for k, (_, va) in enumerate(self.cv.split(self.X, self.y, **self._gkw)):
            folds[va] = k
        return proba, folds

    @staticmethod
    def aggregate_by_song(proba, groups, classes, how="mean"):
        """proba (n_segmentos, n_clases) -> Serie {canción: clase predicha}. how: mean|sum|max|vote."""
        P = pd.DataFrame(np.asarray(proba), columns=classes)
        g = pd.Series(np.asarray(groups))
        if how == "vote":                     # voto mayoritario de las predicciones duras
            S = pd.crosstab(g, P.idxmax(axis=1)).reindex(columns=classes, fill_value=0)
        else:                                 # "mean" | "sum" | "max" sobre las probabilidades
            S = getattr(P.groupby(g), how)()
        return S.idxmax(axis=1)

    def _song_scores(self, model, X, y, groups, hows):
        proba = model.predict_proba(X)
        y_song = pd.Series(np.asarray(y)).groupby(np.asarray(groups)).first()
        out = {"segment": accuracy_score(y, model.classes_[proba.argmax(axis=1)])}
        for how in hows:
            pred = self.aggregate_by_song(proba, groups, model.classes_, how)
            out[how] = accuracy_score(y_song.loc[pred.index], pred)
        return out

    def cv_by_song(self, name, hows=("mean", "max", "vote")):
        """CV (por canción) con accuracy por segmento y por canción (varias agregaciones)."""
        if self.groups is None:
            raise RuntimeError("cv_by_song() needs groups.")
        rows = []
        for tr, va in self.cv.split(self.X, self.y, **self._gkw):
            m = clone(self._get(name)).fit(self.X.iloc[tr], self.y.iloc[tr])
            rows.append(
                self._song_scores(
                    m, self.X.iloc[va], self.y.iloc[va], self.groups.iloc[va], hows
                )
            )
        df = pd.DataFrame(rows)
        print(df.agg(["mean", "std"]).T.round(4))
        return df

    def evaluate_on_test_by_song(self, X_test, y_test, groups_test, hows=("mean", "max", "vote")):
        """Como evaluate_on_test, pero con accuracy por segmento y por canción."""
        if self.best_model is None:
            raise RuntimeError("Run fit_best() or grid_search() first.")
        scores = self._song_scores(
            self.best_model, self._align(X_test),
            np.asarray(y_test).ravel(), groups_test, hows,
        )
        print(f"Model: {self.best_model_name}")
        print(pd.Series(scores).round(4))
        return scores

    # Final Model

    def fit_best(self, name=None):
        """
        We retrain the best model and hyperparameters with the full train set
        """
        if name is None:
            self._require_results()
            name = self.results.loc[0, "Name"]

        self.best_model_name = name
        self.best_model = clone(self._get(name))
        self.best_model.fit(self.X, self.y)
        print(f"Model trained on the full training set: {name}")
        return self.best_model

    def _align(self, X_new):
        if isinstance(X_new, pd.DataFrame):
            return X_new.reindex(columns=self.X.columns)
        return pd.DataFrame(X_new, columns=self.X.columns)

    def _class_names(self, labels):
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(labels)
        return labels

    def predict(self, X_new, decode=True):
        """
        We take the best model and do the prediction
        """
        if self.best_model is None:
            raise RuntimeError("Run fit_best() or grid_search() first.")
        
        predictions = self.best_model.predict(self._align(X_new))
        return self._class_names(predictions) if decode else predictions

    def evaluate_on_test(self, X_test, y_test):
        """
        We evaluate the best model with the final 20% test
        """
        if self.best_model is None:
            raise RuntimeError("Run fit_best() or grid_search() first.")

        y_test = np.asarray(y_test).ravel()
        predictions = self.best_model.predict(self._align(X_test))

        labels = self.best_model.classes_
        names = [str(n) for n in self._class_names(labels)]

        print(f"Model: {self.best_model_name}")
        print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}")
        print(
            "Test macro F1: "
            f"{f1_score(y_test, predictions, average='macro', zero_division=0):.4f}"
        )
        print("\nClassification report:")
        print(
            classification_report(
                y_test, predictions, labels=labels, target_names=names, zero_division=0
            )
        )

        cm = confusion_matrix(y_test, predictions, labels=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=names, yticklabels=names,
        )
        plt.xlabel("Predicted genre")
        plt.ylabel("True genre")
        plt.title("Confusion matrix — test set")
        plt.tight_layout()
        plt.show()

        return predictions