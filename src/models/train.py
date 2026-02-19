"""
Train multiple Tree based classifiers with optional class weighting
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

def train_all_models(X_train: pd.DataFrame,
                     y_train: pd.Series,
                     X_test: pd.DataFrame,
                     scale_pos_weight: float,
                     use_class_weight: bool,
                     random_state: int = 42,
                     model_params: dict[str, dict[str, Any]] | None = None,
                     ):
    """
    Train Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost.
    Returns dict: model_name -> (fitted_model, y_pred, y_pred_proba)
    """
    params = model_params or {}
    out = {}

    # Decision Tree
    kw = dict[str, Any](random_state=random_state)
    if use_class_weight:
        kw["class_weight"] = "balanced"
    dt = DecisionTreeClassifier(**(params.get("decision_tree", {}) | kw))
    dt.fit(X_train, y_train)
    out["decision_tree"] = (dt, dt.predict(X_test), dt.predict_proba(X_test)[:, 1])

    # Random Forest
    rf = RandomForestClassifier(**(params.get("random_forest", {}) | kw))
    rf.fit(X_train, y_train)
    out["random_forest"] = (rf, rf.predict(X_test), rf.predict_proba(X_test)[:, 1])

    # XGBoost
    xgb_kw = dict[str, str | Any](random_state=random_state, eval_metric="logloss")
    if use_class_weight:
        xgb_kw["scale_pos_weight"] = scale_pos_weight
    xgb_model = xgb.XGBClassifier(**(params.get("xgboost", {}) | xgb_kw))
    xgb_model.fit(X_train, y_train)
    out["xgboost"] = (xgb_model, xgb_model.predict(X_test), 
                      xgb_model.predict_proba(X_test)[:, 1])
    
    # LightGBM
    lgb_kw = dict[str, Any](random_state=random_state, verbose = -1)
    if use_class_weight:
        lgb_kw["scale_pos_weight"] = scale_pos_weight
    lgb_model = lgb.LGBMClassifier(**(params.get("lightgbm", {}) | lgb_kw))
    lgb_model.fit(X_train, y_train)
    out["lightgbm"] = (lgb_model, lgb_model.predict(X_test), 
                      lgb_model.predict_proba(X_test)[:, 1])
    
    # CatBoost
    cb_kw = dict[str, Any](random_state=random_state, verbose = 0)
    if use_class_weight:
        cb_kw["scale_pos_weight"] = scale_pos_weight
    cb_model = cb.CatBoostClassifier(**(params.get("catboost", {}) | cb_kw))
    cb_model.fit(X_train, y_train)
    out["catboost"] = (cb_model, cb_model.predict(X_test), 
                      cb_model.predict_proba(X_test)[:, 1])
    
    return out
