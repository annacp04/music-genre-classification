import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


class ExploratoryDataAnalysis:
    """
    Explores the dataset and provides insights into the features and target variable.
    """

    def __init__(self, df, y_name):
        if y_name not in df.columns:
            raise ValueError(f"Target column '{y_name}' not found in dataframe.")

        self.df = df.copy()
        self.y_name = y_name

    def basic_information(self):
        # General Info
        print("=== Basic information ===")
        self.df.info()

        print("\n=== First rows ===")
        print(self.df.head())

        print("\n=== Column names ===")
        print(list(self.df.columns))

        print("\n=== Numerical distributions ===")
        print(self.df.describe())

        y = self.df[self.y_name]
        print("\n=== Target column ===")
        print("Unique values:", y.unique())
        print(y.value_counts())

    def plot_variables(self, num_graphics=3, random_state=None):
        """Plots histograms for a random selection of numerical variables."""
        numeric_cols = (
            self.df.drop(columns=[self.y_name])
            .select_dtypes(include="number")
            .columns.tolist()
        )

        if not numeric_cols:
            print("No numerical columns to plot.")
            return

        rng = random.Random(random_state)
        n_cols = min(num_graphics, len(numeric_cols))
        selected = rng.sample(numeric_cols, n_cols)

        for column in selected:
            plt.figure(figsize=(10, 6))
            self.df[column].hist(bins=30)
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.show()

    def correlation_analysis(self, threshold=0.7, top_n=30, random_state=42):
        """
        1. Identifies highly correlated feature pairs (Pearson correlation).
        2. Calculates mutual information score between each feature and the target variable.
        """
        X_num = self.df.drop(columns=[self.y_name]).select_dtypes(include="number")
        y = self.df[self.y_name]

        # 1. Feature correlation matrix (unique pairs only)
        corr_X = X_num.corr(method="pearson")
        upper = corr_X.where(np.triu(np.ones(corr_X.shape), k=1).astype(bool))

        high_corr = upper.stack().rename("correlation").to_frame()
        high_corr["abs_correlation"] = high_corr["correlation"].abs()
        high_corr = high_corr[high_corr["abs_correlation"] >= threshold].sort_values(
            "abs_correlation", ascending=False
        )

        print(f"\n=== Pairs with |correlation| >= {threshold} ===")
        print(high_corr if not high_corr.empty else "No pairs above the threshold.")

        # 2. Mutual information with target (dropping incomplete rows)
        data = X_num.join(y.rename("__target__")).dropna()
        X_clean = data.drop(columns="__target__")
        y_clean = data["__target__"]

        mi = mutual_info_classif(X_clean, y_clean, random_state=random_state)
        mi_scores = pd.Series(
            mi, index=X_clean.columns, name="mutual_information"
        ).sort_values(ascending=False)

        print(f"\n=== Top {top_n} variables related to the target ===")
        print(mi_scores.head(top_n))

        mi_scores.head(top_n).sort_values().plot(
            kind="barh", figsize=(10, max(5, top_n * 0.3))
        )
        plt.title("Mutual information with target variable")
        plt.xlabel("Mutual information")
        plt.tight_layout()
        plt.show()

        return high_corr, mi_scores

    # Initial Checks (Read-only diagnostics)
    def _get_categorical_columns(self):
        return (
            self.df.drop(columns=[self.y_name])
            .select_dtypes(exclude=["number", "bool"])
            .columns.tolist()
        )

    def _check_missing_values(self):
        print("\n=== Missing values ===")
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]

        if missing.empty:
            print("No missing values found.")
        else:
            print(missing.sort_values(ascending=False))
            print("They will be imputed inside the model Pipeline.")

    def _check_duplicates(self):
        print("\n=== Duplicated rows ===")
        n_dup = self.df.duplicated().sum()
        print(f"{n_dup} duplicated rows.")

    def _check_balanced_dataset(self):
        print("\n=== Target class balance ===")
        counts = self.df[self.y_name].value_counts()
        print(counts)
        print("\nClass proportions:")
        print(self.df[self.y_name].value_counts(normalize=True))

        ratio = counts.min() / counts.max()
        verdict = "imbalanced" if ratio < 0.5 else "balanced"
        print(f"\nMin/max class ratio: {ratio:.2f} -> dataset is {verdict}.")

    def _check_categorical_columns(self):
        print("\n=== Categorical columns ===")
        categorical_columns = self._get_categorical_columns()
        print("Categorical columns:", categorical_columns)

        if categorical_columns:
            print("They will be encoded in Preprocessing.")
        else:
            print("No categorical features found.")

    def initial_checks(self):
        self._check_missing_values()
        self._check_duplicates()
        self._check_balanced_dataset()
        self._check_categorical_columns()