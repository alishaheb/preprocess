# pipeline_oop.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

# --------------------------- #
# Custom Transformers
# --------------------------- #

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols_: List[str] = []
    def fit(self, X, y=None):
        self.cols_ = list(X.columns)
        return self
    def transform(self, X):
        Xc = X.copy()
        out = pd.DataFrame(index=Xc.index)
        for c in self.cols_:
            s = pd.to_datetime(Xc[c], errors="coerce")
            out[f"{c}__year"] = s.dt.year
            out[f"{c}__month"] = s.dt.month
            out[f"{c}__day"] = s.dt.day
            out[f"{c}__dow"] = s.dt.dayofweek
        return out

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, rules: Optional[Dict[str, Dict[str, str]]] = None):
        self.rules = rules or {}
        self.cols_: List[str] = []
    def fit(self, X, y=None):
        self.cols_ = list(X.columns)
        return self
    def transform(self, X):
        Xc = X.copy()
        for col, mapping in self.rules.items():
            if col in Xc.columns:
                Xc[col] = (
                    Xc[col]
                    .astype("string")
                    .str.strip()
                    .str.lower()
                    .map(lambda v: mapping.get(v, np.nan) if v is not pd.NA else np.nan)
                    .astype("object")
                )
        return Xc[self.cols_]

class QuantileCapper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.bounds_: Dict[str, Tuple[float, float]] = {}
        self.cols_: List[str] = []
    def fit(self, X, y=None):
        Xc = pd.DataFrame(X).copy()
        self.cols_ = Xc.columns.tolist()
        for c in self.cols_:
            lo = Xc[c].quantile(self.lower_q)
            hi = Xc[c].quantile(self.upper_q)
            self.bounds_[c] = (lo, hi)
        return self
    def transform(self, X):
        Xc = pd.DataFrame(X).copy()
        for c in self.cols_:
            lo, hi = self.bounds_[c]
            Xc[c] = Xc[c].clip(lower=lo, upper=hi)
        return Xc

class HighCorrDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.97):
        self.threshold = threshold
        self.keep_cols_: Optional[List[str]] = None
    def fit(self, X, y=None):
        Xc = pd.DataFrame(X).copy()
        if Xc.shape[1] <= 1:
            self.keep_cols_ = Xc.columns.tolist()
            return self
        corr = Xc.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_cols = [c for c in upper.columns if any(upper[c] > self.threshold)]
        self.keep_cols_ = [c for c in Xc.columns if c not in drop_cols]
        return self
    def transform(self, X):
        Xc = pd.DataFrame(X).copy()
        return Xc[self.keep_cols_] if self.keep_cols_ is not None else Xc

# --------------------------- #
# Configuration
# --------------------------- #

@dataclass
class PreprocessConfig:
    scale_numeric: bool = True
    cap_outliers: bool = True
    corr_threshold: float = 0.97
    variance_threshold: float = 0.0
    categorical_ordinal: Optional[Dict[str, List[str]]] = None
    unify_rules: Optional[Dict[str, Dict[str, str]]] = None
    problem_type: str = "classification"
    handle_imbalance: str = "class_weight"  # 'none', 'class_weight', 'oversample'

# --------------------------- #
# Preprocessor
# --------------------------- #

class TabularPreprocessor:
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.num_cols_: List[str] = []
        self.cat_cols_: List[str] = []
        self.date_cols_: List[str] = []
        self.pipeline_: Optional[Pipeline] = None

    def _infer_column_types(self, df: pd.DataFrame, target: str):
        X = df.drop(columns=[target])
        self.num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.date_cols_ = X.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
        obj_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
        maybe_dates = []
        for c in obj_cols:
            try:
                pd.to_datetime(X[c], errors="raise")
                maybe_dates.append(c)
            except Exception:
                pass
        self.date_cols_ += maybe_dates
        self.date_cols_ = list(dict.fromkeys(self.date_cols_))
        self.cat_cols_ = [c for c in obj_cols if c not in self.date_cols_]

    def build(self, df: pd.DataFrame, target: str) -> Pipeline:
        self._infer_column_types(df, target)
        cat_cleaner = ("cat_cleaner", TextCleaner(self.config.unify_rules)) if self.config.unify_rules else None

        num_steps = []
        if self.config.cap_outliers:
            num_steps.append(("capper", QuantileCapper()))
        num_steps.append(("imputer", SimpleImputer(strategy="median")))
        if self.config.scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        num_pipe = Pipeline(num_steps)

        if self.config.categorical_ordinal:
            ordinal_cols = [c for c in self.cat_cols_ if c in self.config.categorical_ordinal]
            nominal_cols = [c for c in self.cat_cols_ if c not in ordinal_cols]
            ord_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(categories=[self.config.categorical_ordinal[c] for c in ordinal_cols]))
            ])
        else:
            ordinal_cols, nominal_cols = [], self.cat_cols_

        nom_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        date_pipe = Pipeline([
            ("extract", DateFeatureExtractor()),
            ("imputer", SimpleImputer(strategy="most_frequent"))
        ])

        transformers = []
        if self.num_cols_:
            transformers.append(("num", num_pipe, self.num_cols_))
        if ordinal_cols:
            transformers.append(("ord", ord_pipe, ordinal_cols))
        if nominal_cols:
            if cat_cleaner:
                transformers.append(("nom", Pipeline([cat_cleaner, ("nominal", nom_pipe)]), nominal_cols))
            else:
                transformers.append(("nom", nom_pipe, nominal_cols))
        if self.date_cols_:
            transformers.append(("date", date_pipe, self.date_cols_))

        coltf = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
        steps = [
            ("columns", coltf),
            ("var", VarianceThreshold(threshold=self.config.variance_threshold)),
            ("corr", HighCorrDropper(threshold=self.config.corr_threshold)),
        ]
        self.pipeline_ = Pipeline(steps)
        return self.pipeline_

# --------------------------- #
# Oversampling
# --------------------------- #

def simple_random_oversample(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y
    max_count = counts.max()
    X_res, y_res = [X], [y]
    for cls, cnt in zip(classes, counts):
        if cnt < max_count:
            idx = np.where(y == cls)[0]
            add = rng.choice(idx, size=max_count - cnt, replace=True)
            X_res.append(X[add]); y_res.append(y[add])
    X_new = np.vstack(X_res); y_new = np.concatenate(y_res)
    perm = rng.permutation(len(y_new))
    return X_new[perm], y_new[perm]

# --------------------------- #
# Model Trainer
# --------------------------- #

class ModelTrainer:
    def __init__(self, pre: TabularPreprocessor, target: str, pos_label: Optional[str]=None, random_state: int = 42):
        self.pre = pre
        self.target = target
        self.pos_label = pos_label
        self.random_state = random_state
        self.model_ = None

    def _make_model(self):
        if self.pre.config.problem_type == "classification":
            if self.pre.config.handle_imbalance == "class_weight":
                return LogisticRegression(max_iter=200, class_weight="balanced", random_state=self.random_state)
            else:
                return LogisticRegression(max_iter=200, random_state=self.random_state)
        else:
            raise NotImplementedError("Only classification demo is implemented.")

    def _prepare_target(self, y_raw: pd.Series) -> np.ndarray:
        if pd.api.types.is_numeric_dtype(y_raw):
            return y_raw.astype(int).values
        if self.pos_label is None:
            vc = y_raw.value_counts()
            self.pos_label = vc.idxmin()
        return (y_raw == self.pos_label).astype(int).values

    def fit_eval(self, df: pd.DataFrame, test_size: float = 0.2, cv_folds: int = 5) -> Dict[str, Any]:
        X = df.drop(columns=[self.target])
        y_raw = df[self.target]
        y = self._prepare_target(y_raw)

        pipe = self.pre.build(pd.concat([X, pd.Series(y, name=self.target)], axis=1), self.target)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=self.random_state)

        X_train_p = pipe.fit_transform(X_train, y_train)
        X_test_p = pipe.transform(X_test)

        if self.pre.config.handle_imbalance == "oversample":
            X_train_p, y_train = simple_random_oversample(X_train_p, y_train, random_state=self.random_state)

        self.model_ = self._make_model()
        self.model_.fit(X_train_p, y_train)

        y_pred = self.model_.predict(X_test_p)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }
        try:
            if hasattr(self.model_, "predict_proba"):
                y_prob = self.model_.predict_proba(X_test_p)[:,1]
            else:
                y_prob = self.model_.decision_function(X_test_p)
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
        except Exception:
            pass

        full_pipe = Pipeline([("prep", pipe), ("clf", self._make_model())])
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(full_pipe, X, y, cv=skf, scoring="f1")
        metrics["cv_f1_mean"] = float(np.mean(cv_scores))
        metrics["cv_f1_std"] = float(np.std(cv_scores))
        return metrics

# --------------------------- #
# CLI
# --------------------------- #

def demo(csv_path: str, target: str = "target", pos_label: Optional[str] = None):
    df = pd.read_csv(csv_path)
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            df[c] = pd.to_datetime(df[c], errors="coerce")

    config = PreprocessConfig(
        scale_numeric=True, cap_outliers=True, corr_threshold=0.98,
        variance_threshold=0.0, categorical_ordinal=None,
        unify_rules=None, problem_type="classification",
        handle_imbalance="class_weight"
    )
    pre = TabularPreprocessor(config)
    trainer = ModelTrainer(pre, target=target, pos_label=pos_label, random_state=42)
    return trainer.fit_eval(df, test_size=0.2, cv_folds=5)

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--pos_label", default=None, help="Positive class label for non-numeric targets")
    args = ap.parse_args()
    m = demo(args.csv, args.target, args.pos_label)
    print(json.dumps(m, indent=2))
