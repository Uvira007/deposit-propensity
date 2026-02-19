"""
Resampling for imbalanced data: SMOTE and class weight computation
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE


def apply_smote(X: pd.DataFrame, y: pd.Series,
                k_neighbors: int = 5,
                random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to training data. Returns resampled X and y."""
    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    return pd.DataFrame(X_res, columns = X.columns), pd.Series(y_res, name=y.name)

def compute_class_weights(y: pd.Series):
    """
    Compute scale_pos_weight for gradient boosting (e.g. XGBoost):
    count(negative) / count(positive). use 1.0 if no positive samples
    """
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    if n_pos == 0:
        return 1.0
    return float(n_neg/n_pos)
