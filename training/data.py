from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.preprocessing import NORMAL_LABEL, Schema, normalize_attack_labels

DATASET_URL = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"
DEFAULT_TRAIN = "data/raw/UNSW_NB15_training-set.csv"
DEFAULT_TEST = "data/raw/UNSW_NB15_testing-set.csv"


def load_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV (or Parquet) file and tidy column names and class labels."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {p}\n"
            f"Download UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv from the official page\n"
            f"  {DATASET_URL}\n"
            f"(link 'HERE' on that page) and place them in data/raw/. See README.md."
        )
    df = (
        pd.read_parquet(p)
        if p.suffix.lower() == ".parquet"
        else pd.read_csv(p, low_memory=False)
    )
    df.columns = [str(c).strip() for c in df.columns]
    if "attack_cat" not in df.columns:
        raise ValueError(
            f"{p.name} has no 'attack_cat' column; this project needs the attack category to train a "
            f"multi-class model. Columns found: {list(df.columns)[:12]} ..."
        )
    df["attack_cat"] = normalize_attack_labels(df["attack_cat"])
    return df


def feature_hash(df: pd.DataFrame, features: list[str]) -> pd.Series:
    """One 64-bit hash per row computed from the feature values only."""
    return pd.util.hash_pandas_object(df[features].astype(str), index=False)


def drop_duplicates(df: pd.DataFrame, schema: Schema) -> tuple[pd.DataFrame, int]:
    """Drop rows identical in features AND class; returns (clean table, rows removed)."""
    cols = schema.features + ["attack_cat"]
    keep = ~pd.util.hash_pandas_object(df[cols].astype(str), index=False).duplicated()
    return df.loc[keep].reset_index(drop=True), int((~keep).sum())


def audit(
    train: pd.DataFrame, test: pd.DataFrame, schema: Schema
) -> tuple[dict[str, Any], np.ndarray]:
    """Quantify data-quality problems that can inflate results.

    Returns (audit dict, boolean mask over test rows: True = identical features also appear in train).
    """
    feats = schema.features
    missing_in_test = [f for f in feats if f not in test.columns]
    if missing_in_test:
        raise ValueError(
            f"Test file lacks feature columns present in train: {missing_in_test}"
        )

    tr_h, te_h = feature_hash(train, feats), feature_hash(test, feats)
    in_train = te_h.isin(set(tr_h.tolist())).to_numpy()

    tmp = pd.DataFrame({"h": tr_h, "y": train["attack_cat"]})
    conflicts = int((tmp.groupby("h")["y"].nunique() > 1).sum())

    label_mismatch = None
    if "label" in train.columns:
        label_mismatch = int(
            (
                (pd.to_numeric(train["label"], errors="coerce") == 1)
                != (train["attack_cat"] != NORMAL_LABEL)
            ).sum()
        )

    report = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "n_features": len(feats),
        "numeric_features": len(schema.numeric_features),
        "categorical_features": schema.categorical_features,
        "class_counts_train": train["attack_cat"].value_counts().to_dict(),
        "class_counts_test": test["attack_cat"].value_counts().to_dict(),
        "duplicate_rows_train_features": int(tr_h.duplicated().sum()),
        "duplicate_rows_test_features": int(te_h.duplicated().sum()),
        "test_rows_with_features_also_in_train": int(in_train.sum()),
        "test_rows_with_features_also_in_train_pct": float(100.0 * in_train.mean()),
        "feature_vectors_with_conflicting_labels_in_train": conflicts,
        "missing_values_train": int(train[feats].isna().sum().sum()),
        "label_vs_attack_cat_mismatch_train": label_mismatch,
    }
    return report, in_train
