from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from backend.model import METRICS_FILE, MODEL_FILE
from backend.preprocessing import (
    NORMAL_LABEL,
    Schema,
    build_schema,
    validate_and_prepare,
)
from training import data as D
from training import metrics as M

DEPLOYED = "random_forest"
DEFAULT_THRESHOLD = 0.5


def _signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def tree_preprocessor(schema: Schema) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", "passthrough", schema.numeric_features),
            (
                "cat",
                OrdinalEncoder(
                    categories=[
                        schema.categories[c] for c in schema.categorical_features
                    ],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.float64,
                ),
                schema.categorical_features,
            ),
        ],
        verbose_feature_names_out=False,
    )


def linear_preprocessor(schema: Schema) -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("log", FunctionTransformer(_signed_log1p)),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, schema.numeric_features),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore", min_frequency=20, sparse_output=False
                ),
                schema.categorical_features,
            ),
        ]
    )


def candidates(
    schema: Schema,
    seed: int,
    quick: bool,
    n_estimators: int,
    max_depth: int,
    class_weight: str = "none",
):
    if quick:
        n_estimators, max_depth, hgb_iter, lr_iter = (
            min(n_estimators, 40),
            min(max_depth, 12),
            60,
            150,
        )
    else:
        hgb_iter, lr_iter = 200, 300
    cw = None if class_weight == "none" else "balanced"
    rf_cw = None if class_weight == "none" else "balanced_subsample"
    return {
        "logistic_regression": (
            "baseline",
            lambda: Pipeline(
                [
                    ("prep", linear_preprocessor(schema)),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=lr_iter, class_weight=cw, random_state=seed
                        ),
                    ),
                ]
            ),
        ),
        DEPLOYED: (
            "deployed",
            lambda: Pipeline(
                [
                    ("prep", tree_preprocessor(schema)),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_leaf=3,
                            class_weight=rf_cw,
                            n_jobs=-1,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
        "hist_gradient_boosting": (
            "alternative",
            lambda: Pipeline(
                [
                    ("prep", tree_preprocessor(schema)),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            max_iter=hgb_iter,
                            learning_rate=0.1,
                            class_weight=cw,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
    }


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not serialisable: {type(o)}")


def _latency_ms_per_1000(pipe, X: pd.DataFrame) -> float:
    n = min(len(X), 1000)
    batch = X.iloc[:n]
    best = min(_timed(lambda: pipe.predict_proba(batch)) for _ in range(3))
    return float(best / n * 1000 * 1000)


def _timed(fn) -> float:
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def _safe_eval(classes, y, proba, mask=None):
    if mask is not None:
        y, proba = y[mask], proba[mask]
    yb = np.asarray(y) != NORMAL_LABEL
    if len(y) == 0 or yb.all() or (~yb).all():
        return None
    return M.evaluate(classes, y, proba, threshold=DEFAULT_THRESHOLD)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--train", default=D.DEFAULT_TRAIN)
    ap.add_argument("--test", default=D.DEFAULT_TEST)
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--max-depth", type=int, default=18)
    ap.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="subsample the training set (speed)",
    )
    ap.add_argument(
        "--quick", action="store_true", help="smaller models for a fast smoke test"
    )
    ap.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="'balanced' raises rare-class recall but inflates false alarms (default: none)",
    )
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    t_start = time.perf_counter()

    train_path = Path(args.train)
    is_synthetic = (train_path.parent / "SYNTHETIC.txt").exists()
    if is_synthetic:
        print(
            "!! SYNTHETIC data detected: the resulting model is for plumbing tests only. !!"
        )

    print("[1/6] Loading data")
    train_raw, test_raw = D.load_table(args.train), D.load_table(args.test)
    schema = build_schema(train_raw)
    print(
        f"      train {train_raw.shape}, test {test_raw.shape}, "
        f"{len(schema.numeric_features)} numeric + {len(schema.categorical_features)} categorical features"
    )

    print("[2/6] Auditing data quality (duplicates, overlap, imbalance)")
    audit, test_in_train = D.audit(train_raw, test_raw, schema)
    train_clean, n_dropped = D.drop_duplicates(train_raw, schema)
    audit["train_rows_after_dedup"] = int(len(train_clean))
    audit["train_duplicates_removed"] = n_dropped
    print(
        f"      removed {n_dropped} duplicate training rows; "
        f"{audit['test_rows_with_features_also_in_train']} test rows "
        f"({audit['test_rows_with_features_also_in_train_pct']:.1f}%) have features identical to a training row"
    )
    if args.max_train_rows and len(train_clean) > args.max_train_rows:
        train_clean = train_clean.sample(
            args.max_train_rows, random_state=args.seed
        ).reset_index(drop=True)
        audit["train_rows_used"] = int(len(train_clean))
        print(f"      subsampled training set to {len(train_clean)} rows")

    X_train, _ = validate_and_prepare(train_clean, schema)
    y_train = train_clean["attack_cat"].to_numpy()
    X_test, _ = validate_and_prepare(test_raw, schema)
    y_test = test_raw["attack_cat"].to_numpy()

    counts = pd.Series(y_train).value_counts()
    stratify = y_train if counts.min() >= 2 else None
    if stratify is None:
        print(
            "      (a class has <2 training rows: validation split is not stratified)"
        )
    idx_fit, idx_val = train_test_split(
        np.arange(len(X_train)),
        test_size=args.val_size,
        stratify=stratify,
        random_state=args.seed,
    )
    X_fit, y_fit = X_train.iloc[idx_fit], y_train[idx_fit]
    X_val, y_val = X_train.iloc[idx_val], y_train[idx_val]

    print("[3/6] Training and evaluating candidate models")
    models_report: dict[str, dict] = {}
    deployed_pipe = None
    sweep = None
    specs = candidates(
        schema,
        args.seed,
        args.quick,
        args.n_estimators,
        args.max_depth,
        args.class_weight,
    )
    for name, (role, factory) in specs.items():
        print(f"      - {name} ({role})")
        pipe = factory()
        pipe.fit(X_fit, y_fit)
        val_proba = pipe.predict_proba(X_val)
        val_eval = M.evaluate(
            pipe.classes_, y_val, val_proba, threshold=DEFAULT_THRESHOLD
        )
        if name == DEPLOYED:
            p_att = 1.0 - val_proba[:, list(pipe.classes_).index(NORMAL_LABEL)]
            sweep = M.threshold_sweep(y_val != NORMAL_LABEL, p_att)

        final = factory()
        t0 = time.perf_counter()
        final.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0
        proba = final.predict_proba(X_test)
        test_eval = M.evaluate(
            final.classes_, y_test, proba, threshold=DEFAULT_THRESHOLD
        )
        unseen_eval = _safe_eval(final.classes_, y_test, proba, mask=~test_in_train)
        models_report[name] = {
            "role": role,
            "fit_seconds": fit_seconds,
            "latency_ms_per_1000_flows": _latency_ms_per_1000(final, X_test),
            "validation": val_eval,
            "test": test_eval,
            "test_excluding_rows_seen_in_train": unseen_eval,
            "test_excluding_rows_seen_in_train_n": int((~test_in_train).sum()),
        }
        if name == DEPLOYED:
            deployed_pipe = final
        b = test_eval["binary"]
        print(
            f"        test: macro-F1 {test_eval['multiclass']['macro_f1']:.3f} | "
            f"recall {b['recall']:.3f} precision {b['precision']:.3f} "
            f"FN {b['fn']} FP {b['fp']} | fit {fit_seconds:.1f}s"
        )

    print("[4/6] Saving the deployed model")
    model_dir, reports_dir = Path(args.model_dir), Path(args.reports_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    prep, clf = deployed_pipe.named_steps["prep"], deployed_pipe.named_steps["clf"]
    metadata = {
        "model_name": "Random Forest",
        "dataset_name": "UNSW-NB15"
        + (" (SYNTHETIC test data)" if is_synthetic else ""),
        "dataset_url": D.DATASET_URL,
        "is_synthetic": is_synthetic,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_rows": int(len(train_clean)),
        "test_rows": int(len(test_raw)),
        "seed": args.seed,
        "class_weight": args.class_weight,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "hyperparameters": clf.get_params(),
    }
    artifact = {
        "preprocessor": prep,
        "classifier": clf,
        "schema": schema.to_dict(),
        "classes": [str(c) for c in clf.classes_],
        "feature_names": schema.numeric_features + schema.categorical_features,
        "default_threshold": DEFAULT_THRESHOLD,
        "metadata": metadata,
    }
    model_path = model_dir / MODEL_FILE
    joblib.dump(artifact, model_path, compress=3)
    size_mb = model_path.stat().st_size / 1024 / 1024
    print(f"      {model_path} ({size_mb:.1f} MB)")

    print("[5/6] Writing metrics and reports")
    metrics = {
        "created_at": metadata["trained_at"],
        "is_synthetic": is_synthetic,
        "deployed_model": DEPLOYED,
        "deployed_model_size_mb": size_mb,
        "decision_threshold": DEFAULT_THRESHOLD,
        "protocol": (
            "Official train file -> de-duplicated -> 80/20 fit/validation split. Models are fitted on 'fit' and "
            "scored on validation; then refitted on the full de-duplicated training set and scored once on the "
            "official test file. Multi-class metrics use the arg-max class; binary metrics (Normal vs Attack) use "
            "P(attack) >= threshold. Nothing is tuned on the test file."
        ),
        "class_weight": args.class_weight,
        "audit": audit,
        "threshold_sweep_validation": sweep,
        "models": models_report,
        "limitations": [
            "UNSW-NB15 is lab-generated 2015 traffic with synthetic attacks; scores do not predict real-world performance.",
            "Published analyses report class overlap and labelling artefacts in this dataset.",
            "Rare classes (e.g. Worms, Shellcode) have very few examples: their per-class scores are unstable.",
            "Only flow statistics are used (no payload); unseen attack types are not covered.",
        ],
    }
    (model_dir / METRICS_FILE).write_text(
        json.dumps(metrics, indent=2, default=_json_default)
    )
    (model_dir / "schema.json").write_text(json.dumps(schema.to_dict(), indent=2))

    rows = []
    for name, r in models_report.items():
        b, mc = r["test"]["binary"], r["test"]["multiclass"]
        u = r["test_excluding_rows_seen_in_train"]
        rows.append(
            {
                "model": name,
                "role": r["role"],
                "macro_f1": mc["macro_f1"],
                "weighted_f1": mc["weighted_f1"],
                "accuracy": mc["accuracy"],
                "binary_precision": b["precision"],
                "binary_recall": b["recall"],
                "binary_f1": b["f1"],
                "binary_fpr": b["fpr"],
                "binary_fnr": b["fnr"],
                "false_negatives": b["fn"],
                "false_positives": b["fp"],
                "pr_auc": b["pr_auc"],
                "recall_on_unseen_rows": u["binary"]["recall"] if u else None,
                "fpr_on_unseen_rows": u["binary"]["fpr"] if u else None,
                "latency_ms_per_1000": r["latency_ms_per_1000_flows"],
                "fit_seconds": r["fit_seconds"],
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
    cm = models_report[DEPLOYED]["test"]["multiclass"]["confusion_matrix"]
    pd.DataFrame(
        cm["matrix"],
        index=[f"true_{l}" for l in cm["labels"]],
        columns=[f"pred_{l}" for l in cm["labels"]],
    ).to_csv(reports_dir / "confusion_matrix_deployed.csv")
    pd.DataFrame(models_report[DEPLOYED]["test"]["multiclass"]["per_class"]).T.to_csv(
        reports_dir / "per_class_deployed.csv", index_label="class"
    )
    (reports_dir / "data_audit.json").write_text(
        json.dumps(audit, indent=2, default=_json_default)
    )

    print("[6/6] Summary (official test set)")
    show = comparison[
        [
            "model",
            "role",
            "macro_f1",
            "binary_recall",
            "binary_precision",
            "binary_fpr",
            "false_negatives",
            "false_positives",
            "latency_ms_per_1000",
        ]
    ]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(
        f"Done in {time.perf_counter() - t_start:.1f}s. Start the API with:  uvicorn backend.main:app"
    )
    return metrics


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
