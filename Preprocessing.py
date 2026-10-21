import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder


def _default_group_func(filename):
    """Extract song ID from filename. e.g. 'blues.00000.3.wav' -> 'blues.00000'"""
    return ".".join(str(filename).split(".")[:2])


def _default_segment_func(filename):
    """Extract segment index from filename. e.g. 'blues.00000.3.wav' -> 3, 'blues.00000.wav' -> 0"""
    parts = str(filename).split(".")
    return int(parts[-2]) if len(parts) >= 4 else 0


class Preprocessing:
    """
    Data preprocessing pipeline for audio segment / song datasets.

    Pipeline order:
        1. clean()                  -> Clean names, remove unneeded columns, drop missing & duplicates.
        2. split_X_y()              -> Separate features (X) and target (y).
        3. split_train_test()       -> Stratified train/test split (grouped by song if group_col is provided).
        4. encode_categorical()     -> One-hot encode categorical features (learned on train).
        5. remove_constant_columns()-> Remove zero-variance columns in train.
        6. encode_target()          -> Fit LabelEncoder on y_train and transform y.

    prep = Preprocessing(df, y_name="label", drop_columns=["filename"], group_col="filename")
    prep.run()
    X_train, X_test, y_train, y_test = prep.get_data()
    g_train, g_test = prep.get_groups()
    meta_train, meta_test = prep.get_meta()
    """

    _AUX = ("__group__", "__segment__")

    def __init__(
        self,
        df,
        y_name,
        test_size=0.20,
        random_state=42,
        drop_columns=None,
        strip_names=True,
        drop_duplicates=True,
        group_col=None,
        group_func=None,
        segment_func=None,
        test_groups=None,
    ):
        self.df = df.copy()
        self.y_name = y_name
        self.test_size = test_size
        self.random_state = random_state
        self.drop_columns = list(drop_columns) if drop_columns else []
        self.strip_names = strip_names
        self.drop_duplicates = drop_duplicates

        # Grouping configuration
        self.group_col = group_col
        self.group_func = group_func or _default_group_func
        self.segment_func = segment_func or _default_segment_func
        self.test_groups = set(test_groups) if test_groups is not None else None

        # Feature and target containers
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.encoder = None

        # Metadata containers
        self.groups = None
        self.segments = None
        self.groups_train = None
        self.groups_test = None
        self.segments_train = None
        self.segments_test = None

    def _require(self, attr, step):
        """Internal validation check to ensure step dependencies are met."""
        if getattr(self, attr) is None:
            raise RuntimeError(f"Run {step}() first.")

    # 1. Cleaning
    def clean(self):
        """Clean column names, extract song/segment metadata, drop missing targets and duplicates."""
        print("=== Cleaning ===")

        if self.strip_names:
            self.df.columns = self.df.columns.str.strip()
            self.y_name = self.y_name.strip()
            self.drop_columns = [c.strip() for c in self.drop_columns]
            if self.group_col is not None:
                self.group_col = self.group_col.strip()

        if self.y_name not in self.df.columns:
            raise ValueError(f"Target column '{self.y_name}' not found.")

        # Extract song ID and segment index before dropping group_col
        if self.group_col is not None:
            if self.group_col not in self.df.columns:
                raise ValueError(f"Group column '{self.group_col}' not found.")
            
            self.df["__group__"] = self.df[self.group_col].map(self.group_func)
            self.df["__segment__"] = self.df[self.group_col].map(self.segment_func)
            
            if self.group_col not in self.drop_columns:
                self.drop_columns.append(self.group_col)
            print(f"Songs found: {self.df['__group__'].nunique()}")

        # Drop non-feature identifier columns
        to_drop = [c for c in self.drop_columns if c in self.df.columns]
        missing = [c for c in self.drop_columns if c not in self.df.columns]
        if missing:
            print(f"Warning: columns not found, not dropped: {missing}")
        if to_drop:
            self.df = self.df.drop(columns=to_drop)
            print(f"Dropped columns: {to_drop}")

        # Strip whitespace from target strings if categorical
        if not pd.api.types.is_numeric_dtype(self.df[self.y_name]):
            self.df[self.y_name] = self.df[self.y_name].astype(str).str.strip()

        # Remove rows with missing target values
        n_before = len(self.df)
        self.df = self.df.dropna(subset=[self.y_name])
        print(f"Rows without target removed: {n_before - len(self.df)}")

        # Drop duplicate rows based on features + target (ignoring auxiliary metadata)
        if self.drop_duplicates:
            n_before = len(self.df)
            feature_cols = [c for c in self.df.columns if c not in self._AUX]
            self.df = self.df.drop_duplicates(subset=feature_cols)
            print(f"Duplicated rows removed: {n_before - len(self.df)}")

        self.df = self.df.reset_index(drop=True)
        print(f"Final shape: {self.df.shape}")
        return self

    # 2. Separate Features and Target
    def split_X_y(self):
        """Split dataframe into X (features) and y (target), preserving group metadata."""
        aux_cols = [c for c in self._AUX if c in self.df.columns]
        self.X = self.df.drop(columns=[self.y_name] + aux_cols)
        self.y = self.df[self.y_name]
        
        self.groups = self.df["__group__"] if "__group__" in aux_cols else None
        self.segments = self.df["__segment__"] if "__segment__" in aux_cols else None
        
        print(f"\nX: {self.X.shape} | y: {self.y.shape}")
        return self

    # 3. Train / Test Split
    def split_train_test(self):
        """Split into train and test sets, grouping by song to prevent data leakage."""
        self._require("X", "split_X_y")

        # Standard stratified split when no group_col is defined
        if self.groups is None:
            if self.test_groups is not None:
                raise ValueError("test_groups requires group_col.")
            
            X_tr, X_te, y_tr, y_te = train_test_split(
                self.X,
                self.y,
                test_size=self.test_size,
                stratify=self.y,
                random_state=self.random_state,
            )
        # Grouped split: ensure all segments of a song belong to either train or test
        else:
            if self.test_groups is not None:
                is_test = self.groups.isin(self.test_groups).to_numpy()
                tr_idx, te_idx = np.flatnonzero(~is_test), np.flatnonzero(is_test)
            else:
                sgkf = StratifiedGroupKFold(
                    n_splits=round(1 / self.test_size),
                    shuffle=True,
                    random_state=self.random_state,
                )
                tr_idx, te_idx = next(sgkf.split(self.X, self.y, self.groups))

            X_tr, X_te = self.X.iloc[tr_idx], self.X.iloc[te_idx]
            y_tr, y_te = self.y.iloc[tr_idx], self.y.iloc[te_idx]
            
            self.groups_train = self.groups.iloc[tr_idx].reset_index(drop=True)
            self.groups_test = self.groups.iloc[te_idx].reset_index(drop=True)
            
            if self.segments is not None:
                self.segments_train = self.segments.iloc[tr_idx].reset_index(drop=True)
                self.segments_test = self.segments.iloc[te_idx].reset_index(drop=True)

            # Prevent song overlap
            overlap = set(self.groups_train) & set(self.groups_test)
            assert not overlap, f"Song leakage detected between train and test: {len(overlap)} overlapping songs."

        self.X_train = X_tr.reset_index(drop=True)
        self.X_test = X_te.reset_index(drop=True)
        self.y_train = y_tr.reset_index(drop=True)
        self.y_test = y_te.reset_index(drop=True)

        print(f"\n=== Train / test split ({1 - self.test_size:.0%} / {self.test_size:.0%}) ===")
        if self.groups is not None:
            print(
                f"Split by song -> train: {self.groups_train.nunique()} songs | "
                f"test: {self.groups_test.nunique()} songs"
            )
        print(f"Train: {self.X_train.shape} | Test: {self.X_test.shape}")
        print("\nClass proportions (train vs test):")
        print(
            pd.DataFrame({
                "train": self.y_train.value_counts(normalize=True),
                "test": self.y_test.value_counts(normalize=True),
            }).round(3)
        )
        return self

    # 4. Categorical Encoding
    def encode_categorical(self, drop_first=False):
        """Convert boolean columns to 0/1 and apply One-Hot Encoding to categorical columns."""
        self._require("X_train", "split_train_test")

        # Convert boolean flags to integers
        bool_cols = self.X_train.select_dtypes(include="bool").columns
        if len(bool_cols) > 0:
            self.X_train[bool_cols] = self.X_train[bool_cols].astype(int)
            self.X_test[bool_cols] = self.X_test[bool_cols].astype(int)

        cat_cols = self.X_train.select_dtypes(exclude="number").columns.tolist()
        print("\n=== Categorical encoding ===")

        if not cat_cols:
            print("No categorical features to encode.")
            return self

        for col in cat_cols:
            print(f"{col}: {self.X_train[col].nunique()} categories")

        self.X_train = pd.get_dummies(
            self.X_train, columns=cat_cols, drop_first=drop_first, dtype=int
        )
        self.X_test = pd.get_dummies(
            self.X_test, columns=cat_cols, drop_first=drop_first, dtype=int
        )

        # Align test columns strictly to match train columns
        self.X_test = self.X_test.reindex(columns=self.X_train.columns, fill_value=0)

        print(f"Features after encoding: {self.X_train.shape[1]}")
        return self

    # 5. Constant Columns Removal
    def remove_constant_columns(self):
        """Remove zero-variance columns (single unique value) from train and test."""
        self._require("X_train", "split_train_test")
        
        constant_cols = self.X_train.columns[self.X_train.nunique(dropna=True) <= 1].tolist()
        self.X_train = self.X_train.drop(columns=constant_cols)
        self.X_test = self.X_test.drop(columns=constant_cols)
        
        print(f"\n=== Constant columns removed ({len(constant_cols)}) ===")
        if constant_cols:
            print(constant_cols)
        return self

    # 6. Target Encoding
    def encode_target(self):
        """Fit LabelEncoder on y_train and transform both y_train and y_test."""
        self._require("y_train", "split_train_test")

        self.encoder = LabelEncoder()
        self.y_train = pd.Series(
            self.encoder.fit_transform(self.y_train), name=self.y_name
        )
        self.y_test = pd.Series(
            self.encoder.transform(self.y_test), name=self.y_name
        )

        print("\n=== Target encoding ===")
        for code, label in enumerate(self.encoder.classes_):
            print(f"{code} -> {label}")
        return self

    # Execution & Accessor Methods
    def run(self):
        """Run the complete pipeline sequentially."""
        return (
            self.clean()
            .split_X_y()
            .split_train_test()
            .encode_categorical()
            .remove_constant_columns()
            .encode_target()
        )

    def get_data(self):
        """Return (X_train, X_test, y_train, y_test)."""
        self._require("X_train", "split_train_test")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_groups(self):
        """Return (groups_train, groups_test) song IDs."""
        self._require("groups_train", "split_train_test (with group_col)")
        return self.groups_train, self.groups_test

    def get_meta(self):
        """Return DataFrames with 'song' and 'segment' columns for train and test."""
        self._require("groups_train", "split_train_test (with group_col)")

        def create_meta_df(groups, segments):
            seg_vals = 0 if segments is None else segments.values
            return pd.DataFrame({"song": groups.values, "segment": seg_vals})

        return (
            create_meta_df(self.groups_train, self.segments_train),
            create_meta_df(self.groups_test, self.segments_test),
        )

    @property
    def class_names(self):
        """Return array of target class names after target encoding."""
        self._require("encoder", "encode_target")
        return self.encoder.classes_