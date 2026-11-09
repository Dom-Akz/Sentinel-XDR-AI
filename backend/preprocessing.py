from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Columns that describe the answer (or are just row identifiers): never features.
NON_FEATURE_COLUMNS = {"id", "attack_cat", "label"}
NORMAL_LABEL = "Normal"
UNKNOWN_CATEGORY = "-"

_CANONICAL_LABELS = {
    "normal": "Normal",
    "generic": "Generic",
    "exploits": "Exploits",
    "fuzzers": "Fuzzers",
    "dos": "DoS",
    "reconnaissance": "Reconnaissance",
    "analysis": "Analysis",
    "backdoor": "Backdoor",
    "backdoors": "Backdoor",  # some distributions use the plural
    "shellcode": "Shellcode",
    "worms": "Worms",
}


class InputValidationError(ValueError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


def normalize_attack_labels(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    mapped = s.str.lower().map(_CANONICAL_LABELS)
    out = mapped.where(mapped.notna(), s)
    out = out.fillna(NORMAL_LABEL)
    out = out.replace("", NORMAL_LABEL)
    return out.astype(object)


@dataclass
class Schema:
    numeric_features: list[str]
    categorical_features: list[str]
    numeric_medians: dict[str, float]
    categories: dict[str, list[str]]

    @property
    def features(self) -> list[str]:
        return list(self.numeric_features) + list(self.categorical_features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "numeric_medians": self.numeric_medians,
            "categories": self.categories,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Schema":
        return cls(
            numeric_features=list(d["numeric_features"]),
            categorical_features=list(d["categorical_features"]),
            numeric_medians={k: float(v) for k, v in d["numeric_medians"].items()},
            categories={k: list(v) for k, v in d["categories"].items()},
        )


def build_schema(df: pd.DataFrame) -> Schema:
    features = [
        c for c in df.columns if str(c).strip().lower() not in NON_FEATURE_COLUMNS
    ]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    medians = {}
    for c in numeric:
        m = df[c].astype(float).median()
        medians[c] = float(m) if np.isfinite(m) else 0.0
    categories = {
        c: sorted(
            df[c]
            .astype("string")
            .str.strip()
            .fillna(UNKNOWN_CATEGORY)
            .unique()
            .tolist()
        )
        for c in categorical
    }
    return Schema(numeric, categorical, medians, categories)


def validate_and_prepare(
    df: pd.DataFrame, schema: Schema
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if df is None or df.shape[0] == 0:
        raise InputValidationError("No flows were provided.")

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Tolerate different capitalisation of column names.
    lower_map = {c.lower(): c for c in df.columns}
    rename = {
        lower_map[f.lower()]: f
        for f in schema.features
        if f not in df.columns and f.lower() in lower_map
    }
    if rename:
        df = df.rename(columns=rename)

    missing = [f for f in schema.features if f not in df.columns]
    if missing:
        raise InputValidationError(
            f"Missing {len(missing)} required column(s): {', '.join(missing[:10])}"
            + (" ..." if len(missing) > 10 else ""),
            {"missing_columns": missing, "expected_columns": schema.features},
        )

    out = df[schema.features].reset_index(drop=True)

    bad_columns: dict[str, dict[str, Any]] = {}
    imputed: dict[str, int] = {}
    for c in schema.numeric_features:
        raw = out[c]
        conv = pd.to_numeric(raw, errors="coerce").astype(float)
        raw_txt = raw.astype("string").str.strip()
        bad = (
            (conv.isna() & raw_txt.notna() & (raw_txt != "")).fillna(False).astype(bool)
        )
        if bad.any():
            bad_columns[c] = {
                "count": int(bad.sum()),
                "examples": raw_txt[bad].head(3).tolist(),
                "first_rows": [int(i) for i in np.flatnonzero(bad.to_numpy())[:3]],
            }
            continue
        conv = conv.replace([np.inf, -np.inf], np.nan)
        n_missing = int(conv.isna().sum())
        if n_missing:
            imputed[c] = n_missing
            conv = conv.fillna(schema.numeric_medians[c])
        out[c] = conv
    if bad_columns:
        raise InputValidationError(
            "Non-numeric values found in numeric column(s): " + ", ".join(bad_columns),
            {"invalid_numeric_columns": bad_columns},
        )
    if imputed:
        total = sum(imputed.values())
        warnings.append(
            f"{total} missing or non-finite numeric value(s) in column(s) "
            f"{', '.join(sorted(imputed))} were replaced by the training median."
        )

    for c in schema.categorical_features:
        s = out[c].astype("string").str.strip().fillna(UNKNOWN_CATEGORY)
        s = s.replace("", UNKNOWN_CATEGORY)
        unknown = ~s.isin(schema.categories[c])
        if unknown.any():
            examples = sorted(set(s[unknown].tolist()))[:3]
            warnings.append(
                f"{int(unknown.sum())} value(s) in column '{c}' were not seen during training "
                f"(e.g. {', '.join(examples)}); they are treated as 'unknown'."
            )
        out[c] = s.astype(object)

    return out, warnings
