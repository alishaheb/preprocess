import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def split_columns(df: pd.DataFrame, target: str, drop_columns=None):
    drop_columns = drop_columns or []
    X = df.drop(columns=[target] + [c for c in drop_columns if c in df.columns])
    y = df[target]

    # Detect types
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    # If all columns end up categorical/numeric, that's fine.
    return X, y, num_cols, cat_cols


def make_preprocessors(num_cols, cat_cols):
    # For trees/forests: impute, one-hot, NO scaling on numerics
    num_tree = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_common = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor_tree = ColumnTransformer(
        transformers=[
            ("num", num_tree, num_cols),
            ("cat", cat_common, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    # For MLP: impute + scale numerics, same cats
    num_mlp = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])

    preprocessor_mlp = ColumnTransformer(
        transformers=[
            ("num", num_mlp, num_cols),
            ("cat", cat_common, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return preprocessor_tree, preprocessor_mlp


def build_models(preprocessor_tree, preprocessor_mlp, random_state=42):
    models = {
        "DecisionTree": Pipeline(steps=[
            ("prep", preprocessor_tree),
            ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=random_state))
        ]),
        "RandomForest": Pipeline(steps=[
            ("prep", preprocessor_tree),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                n_jobs=-1,
                class_weight="balanced_subsample",
                random_state=random_state
            ))
        ]),
        "MLP": Pipeline(steps=[
            ("prep", preprocessor_mlp),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                max_iter=200,
                random_state=random_state
            ))
        ])
    }
    return models


def safe_metrics(y_true, y_pred, y_proba=None):
    """Works for binary or multi-class. y_proba optional (needed for ROC AUC)."""
    labels = np.unique(y_true)
    average = "binary" if len(labels) == 2 else "macro"

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }

    # ROC AUC only if we have probabilities and >1 class
    if y_proba is not None and len(labels) > 1:
        try:
            if len(labels) == 2:
                # use positive class column
                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    proba_pos = y_proba[:, 1]
                else:
                    # Some classifiers return 1-D probs for the positive class
                    proba_pos = y_proba
                metrics["roc_auc"] = float(roc_auc_score(y_true, proba_pos))
            else:
                metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
        except Exception:
            # Skip ROC AUC if something odd happens (e.g., no variance)
            pass

    return metrics


def evaluate_models(models, X_train, X_test, y_train, y_test, cv_folds=5):
    results = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Try predict_proba; if not available, skip ROC AUC
        y_proba = None
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            try:
                y_proba = pipe.predict_proba(X_test)
            except Exception:
                y_proba = None

        res = safe_metrics(y_test, y_pred, y_proba)
        # Add cross-validated accuracy for a more stable view
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_acc = cross_val_score(pipe, np.asarray(X_train), np.asarray(y_train), cv=cv, scoring="accuracy")
        res["cv_accuracy_mean"] = float(cv_acc.mean())
        res["cv_accuracy_std"] = float(cv_acc.std())

        results[name] = res
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--drop", default="", help="Comma-separated columns to drop")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found. Columns: {list(df.columns)}")

    drop_cols = [c.strip() for c in args.drop.split(",")] if args.drop else []
    X, y, num_cols, cat_cols = split_columns(df, args.target, drop_cols)
    pre_tree, pre_mlp = make_preprocessors(num_cols, cat_cols)
    models = build_models(pre_tree, pre_mlp, random_state=args.random_state)

    # Train/test split (stratified if possible)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y if y.nunique() > 1 else None
    )

    results = evaluate_models(models, X_train, X_test, y_train, y_test, cv_folds=5)

    # Helpful extras
    out = {
        "dataset": Path(args.csv).name,
        "n_samples": int(len(df)),
        "n_features_before": int(df.drop(columns=[args.target]).shape[1]),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "results": results,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))

    # Also print a concise table
    print("\nSummary:")
    for name, r in results.items():
        print(f"- {name}: "
              f"acc={r['accuracy']:.3f}, f1={r['f1']:.3f}, "
              f"cv_acc={r['cv_accuracy_mean']:.3f}±{r['cv_accuracy_std']:.3f}")


if __name__ == "__main__":
    main()
