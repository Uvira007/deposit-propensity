"""
Compute metrics for all models and identify the best model based on ROC-AUC
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

def evaluate_models(results: dict[str, tuple[Any, np.ndarray, np.ndarray]],
                    y_test: np.ndarray | pd.Series,
                    metrics: list[str] | None = None) -> pd.DataFrame:
    """
    Compute metrics for each model. results: model_name -> (model, y_pred, y_pred_proba)
    Returns a dataframe with model_name as index and metrics as columns.
    """
    metrics = metrics or ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = []
    for name, (_model, y_pred, y_pred_proba) in results.items():
        row = dict[str, Any](model=name)
        if "accuracy" in metrics:
            row["accuracy"] = accuracy_score(y_test, y_pred)
        if "precision" in metrics:
            row["precision"] = precision_score(y_test, y_pred, zero_division=0)
        if "recall" in metrics:
            row["recall"] = recall_score(y_test, y_pred, zero_division=0)
        if "f1" in metrics:
            row["f1"] = f1_score(y_test, y_pred, zero_division=0)
        if "roc_auc" in metrics:
            try:
                row["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
            except ValueError:
                row["roc_auc"] = np.nan  # e.g. if only one class present in y_test
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")

def get_best_model(metrics_df: pd.DataFrame, metric: str = "roc_auc") -> str:
    """Identify best model based on specified metric (default: roc_auc)"""
    if metric not in metrics_df.columns:
        raise ValueError(f"Metric {metric} not found in metrics_df columns")
    return metrics_df[metric].idxmax()
