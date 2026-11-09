"""Loads the trained model and turns flows into predictions + explanations."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .preprocessing import NORMAL_LABEL, Schema, validate_and_prepare

MODEL_FILE = "ids_model.joblib"
METRICS_FILE = "metrics.json"


class ModelNotAvailable(RuntimeError):
    pass


def _fmt_value(v: Any) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4g}"
    return str(v)


class IDSModel:
    def __init__(self, artifact: dict[str, Any], metrics: dict[str, Any] | None = None):
        self.preprocessor = artifact["preprocessor"]
        self.classifier = artifact["classifier"]
        self.schema = Schema.from_dict(artifact["schema"])
        self.classes = [str(c) for c in artifact["classes"]]
        self.feature_names: list[str] = list(artifact["feature_names"])
        self.default_threshold = float(artifact.get("default_threshold", 0.5))
        self.metadata: dict[str, Any] = artifact.get("metadata", {})
        self.metrics = metrics
        self.normal_idx = self.classes.index(NORMAL_LABEL)
        self.attack_idx = np.array(
            [i for i in range(len(self.classes)) if i != self.normal_idx]
        )
        self._explainer = None

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, model_dir: str | Path) -> "IDSModel":
        model_dir = Path(model_dir)
        path = model_dir / MODEL_FILE
        if not path.exists():
            raise ModelNotAvailable(
                f"No trained model found at {path}. Train one first: python -m training.train"
            )
        try:
            # NOTE: joblib/pickle files can run code when loaded. Only load models you trained yourself.
            artifact = joblib.load(path)
        except Exception as exc:  # corrupted file, incompatible library versions...
            raise ModelNotAvailable(f"Could not load {path}: {exc}") from exc
        metrics = None
        mpath = model_dir / METRICS_FILE
        if mpath.exists():
            try:
                metrics = json.loads(mpath.read_text())
            except json.JSONDecodeError:
                metrics = None
        return cls(artifact, metrics)

    @property
    def name(self) -> str:
        return str(self.metadata.get("model_name", type(self.classifier).__name__))

    def _get_explainer(self):
        if self._explainer is None:
            import shap

            self._explainer = shap.TreeExplainer(self.classifier)
        return self._explainer

    def warm_up(self) -> None:
        try:
            self._get_explainer()
        except Exception:
            self._explainer = None

    def _shap_for_class(self, X: np.ndarray, class_idx: np.ndarray) -> np.ndarray:
        sv = self._get_explainer().shap_values(X, check_additivity=False)
        arr = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)
        if arr.ndim == 2:
            arr = arr[..., None]
        n_feat, n_cls = X.shape[1], len(self.classes)
        if (
            arr.shape[1] != n_feat and arr.shape[0] == n_cls
        ):  # (classes, rows, features)
            arr = np.transpose(arr, (1, 2, 0))
        return arr[np.arange(len(X)), :, class_idx]

    def _score(self, prepared: pd.DataFrame, threshold: float) -> dict[str, np.ndarray]:
        X = np.asarray(self.preprocessor.transform(prepared), dtype=float)
        proba = self.classifier.predict_proba(X)
        p_attack = np.clip(1.0 - proba[:, self.normal_idx], 0.0, 1.0)
        best_pos = np.argmax(proba[:, self.attack_idx], axis=1)
        best_attack = self.attack_idx[best_pos]
        return {
            "X": X,
            "proba": proba,
            "p_attack": p_attack,
            "best_attack": best_attack,
            "is_attack": p_attack >= threshold,
        }

    def _contributions(self, X_row_values: pd.Series, shap_row: np.ndarray, top_k: int):
        order = np.argsort(-np.abs(shap_row))[:top_k]
        feats = []
        for j in order:
            name = self.feature_names[j]
            val = X_row_values[name]
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)
            feats.append(
                {
                    "feature": name,
                    "value": val,
                    "contribution": float(shap_row[j]),
                    "direction": "toward" if shap_row[j] >= 0 else "away",
                }
            )
        return feats

    def _explanation_text(
        self,
        explained: str,
        prediction: str,
        p_attack: float,
        threshold: float,
        feats: list[dict[str, Any]],
    ) -> str:
        if prediction == "Attack":
            head = (
                f"Flagged as an attack ({explained}) with attack probability {p_attack:.1%} "
                f"(threshold {threshold:.0%})."
            )
        else:
            head = (
                f"Judged normal: attack probability is only {p_attack:.1%} "
                f"(threshold {threshold:.0%})."
            )
        toward = [f for f in feats if f["direction"] == "toward"][:3]
        away = [f for f in feats if f["direction"] == "away"][:2]
        parts = [head]
        if toward:
            parts.append(
                f"Main factors supporting '{explained}': "
                + "; ".join(
                    f"{f['feature']} = {_fmt_value(f['value'])} ({f['contribution']:+.3f})"
                    for f in toward
                )
                + "."
            )
        if away:
            parts.append(
                f"Factors pointing the other way: "
                + "; ".join(
                    f"{f['feature']} = {_fmt_value(f['value'])} ({f['contribution']:+.3f})"
                    for f in away
                )
                + "."
            )
        parts.append(
            "Contributions are SHAP values: larger magnitude = stronger influence on this class."
        )
        return " ".join(parts)

    def _result_row(self, i: int, s: dict[str, np.ndarray]) -> dict[str, Any]:
        is_attack = bool(s["is_attack"][i])
        p = float(s["p_attack"][i])
        best = int(s["best_attack"][i])
        return {
            "index": int(i),
            "prediction": "Attack" if is_attack else "Normal",
            "attack_type": self.classes[best] if is_attack else None,
            "most_likely_attack": self.classes[best],
            "confidence": p if is_attack else 1.0 - p,
            "attack_probability": p,
            "type_probability": float(s["proba"][i, best]),
            "explained_class": None,
            "explanation": None,
            "top_features": None,
        }

    def _explain_rows(
        self,
        rows: list[int],
        s,
        prepared: pd.DataFrame,
        results: list[dict],
        threshold: float,
        top_k: int,
        target_class: str | None = None,
    ) -> None:
        if not rows:
            return
        rows_arr = np.asarray(rows)
        if target_class is not None:
            class_idx = np.full(len(rows_arr), self.classes.index(target_class))
        else:
            class_idx = np.array(
                [
                    s["best_attack"][r] if s["is_attack"][r] else self.normal_idx
                    for r in rows_arr
                ]
            )
        shap_vals = self._shap_for_class(s["X"][rows_arr], class_idx)
        for r, ci, sv in zip(rows_arr, class_idx, shap_vals):
            feats = self._contributions(prepared.iloc[r], sv, top_k)
            res = results[r]
            res["explained_class"] = self.classes[int(ci)]
            res["top_features"] = feats
            res["explanation"] = self._explanation_text(
                self.classes[int(ci)],
                res["prediction"],
                res["attack_probability"],
                threshold,
                feats,
            )

    def predict(
        self, df: pd.DataFrame, threshold: float | None = None, explain_limit: int = 0
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        thr = self.default_threshold if threshold is None else float(threshold)
        prepared, warnings = validate_and_prepare(df, self.schema)
        s = self._score(prepared, thr)
        results = [self._result_row(i, s) for i in range(len(prepared))]

        if explain_limit > 0 and s["is_attack"].any():
            flagged = np.flatnonzero(s["is_attack"])
            top = flagged[np.argsort(-s["p_attack"][flagged])][:explain_limit]
            try:
                self._explain_rows(
                    [int(r) for r in top], s, prepared, results, thr, top_k=6
                )
            except (
                Exception
            ) as exc:  # never fail a prediction because an explanation failed
                warnings.append(
                    f"Explanations unavailable for this request ({type(exc).__name__})."
                )

        n_attacks = int(s["is_attack"].sum())
        dist: dict[str, int] = {NORMAL_LABEL: len(results) - n_attacks}
        for r in results:
            if r["prediction"] == "Attack":
                dist[r["attack_type"]] = dist.get(r["attack_type"], 0) + 1
        return {
            "model": self.name,
            "threshold": thr,
            "n_flows": len(results),
            "n_attacks": n_attacks,
            "n_normal": len(results) - n_attacks,
            "attack_distribution": dist,
            "warnings": warnings,
            "inference_ms": (time.perf_counter() - t0) * 1000.0,
            "results": results,
        }

    def explain(
        self,
        flow: pd.DataFrame,
        top_k: int = 8,
        target_class: str | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        thr = self.default_threshold if threshold is None else float(threshold)
        if target_class is not None and target_class not in self.classes:
            from .preprocessing import InputValidationError

            raise InputValidationError(
                f"Unknown class '{target_class}'.", {"valid_classes": self.classes}
            )
        prepared, warnings = validate_and_prepare(flow, self.schema)
        s = self._score(prepared, thr)
        results = [self._result_row(0, s)]
        self._explain_rows([0], s, prepared, results, thr, top_k, target_class)
        return {
            "model": self.name,
            "threshold": thr,
            "result": results[0],
            "warnings": warnings,
        }

    def info(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "classes": self.classes,
            "default_threshold": self.default_threshold,
            "features": {
                "numeric": self.schema.numeric_features,
                "categorical": self.schema.categorical_features,
                "categories": self.schema.categories,
            },
            "metadata": self.metadata,
            "metrics": self.metrics,
        }
