"""
SHAP Analysis for the best model to improve interpretability and 
feature importance score
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

def run_shap_analysis(
    model: Any,
    X: pd.DataFrame,
    feature_names: list[str],
    plots_dir: Path,
    model_name: str,
    max_display: int = 15,
    sample_size: int | None = 1000,
):
    """
    Run TreeExplainer on the model, save summary and bar plots
    X is the data to explain (e.g. test set or other sample)
    """
    plots_dir = Path(plots_dir)
    # Sample data for SHAP if sample_size is specified
    if sample_size is not None and len(X) > sample_size:
        X = X.sample(sample_size, random_state=42)
    X_arr = np.asarray(X)
    explainer = shap.TreeExplainer(model, X_arr, feature_names=feature_names)
    shap_values = explainer.shap_values(X_arr)
    # Binary classification: Tree explainers may return list [neg, pos]; use positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    feature_importance = np.abs(shap_values).mean(axis=0)
    # Save summary plot
    shap.summary_plot(shap_values, X, feature_names=feature_names, 
                      show=False, max_display=max_display)
    shap_plot = plt.gcf()
    shap_plot.savefig(plots_dir / f"{model_name}_shap_summary.png", bbox_inches='tight')
    plt.close()

    # Bar plot (mean |SHAP|)
    shap.summary_plot(shap_values, X, feature_names=feature_names, 
                      show=False, plot_type="bar", max_display=max_display)
    plt.tight_layout()
    plt.savefig(plots_dir / f"{model_name}_shap_importance.png", bbox_inches='tight')
    plt.close()