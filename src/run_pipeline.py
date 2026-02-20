"""
Main Pipeline: Load Data, Preprocess, Resample, Train, Evaluate
Run from the project root: python -m src.run_pipeline
"""
import json
import pickle
from pathlib import Path

from sklearn.metrics import roc_curve

from src.config import load_config, get_paths
from src.data.load_data import load_bank_marketing
from src.preprocessing.preprocess import preprocess_data
from src.sampling.resample import apply_smote, compute_class_weights
from src.models.train import train_all_models
from src.evaluation.evaluate import evaluate_models, get_best_model
from src.interpretability.shap_analysis import run_shap_analysis

def main(config_path: Path | None = None):
    # Load config and paths
    config = load_config(config_path)
    paths = get_paths(config)
    seed = config["project"]["seed"]
    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    samp_cfg = config["sampling"]
    model_cfg = config["models"]
    eval_cfg = config["evaluation"]
    shap_cfg = config["shap"]

    # Output dirs
    for key in ("models_dir", "plots_dir", "metrics_dir", "dashboard_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)


    # Load data
    df = load_bank_marketing(paths["data_dir"],
                             filename = data_cfg["filename"],
                             download_if_missing= True,
                             )

    # Preprocess data and split training and test set
    X_train, X_test, y_train, y_test, transformer, feature_names = preprocess_data(
        df,
        target_column = data_cfg["target_column"],
        positive_class = data_cfg["positive_class"],
        categorical_columns = prep_cfg["categorical_columns"],
        drop_columns = prep_cfg["drop_columns"],
        test_size = data_cfg["test_size"],
        random_state = data_cfg["random_state"]
    )

    # class weight from original (imbalanced) train set for use in model
    scale_pos_weight = compute_class_weights(y_train)

    # Resample training data using SMOTE
    if samp_cfg["use_smote"]:
        X_train_resampled, y_train_resampled = apply_smote(X_train, y_train, 
                                                           k_neighbors=samp_cfg["smote_k_neighbors"],
                                                           random_state=samp_cfg["random_state"])
    else:
        X_train_resampled, y_train_resampled = X_train, y_train

    class_weights = compute_class_weights(y_train_resampled)

    model_params = {
        k: v
        for k, v in model_cfg.items()
        if k != "random_state" and isinstance(v, dict) 
    }
    
    # Train models
    models = train_all_models(X_train_resampled, 
                              y_train_resampled, 
                              X_test,
                              scale_pos_weight=scale_pos_weight,
                              use_class_weight=samp_cfg["use_class_weight"],
                              random_state=model_cfg["random_state"],
                              model_params=model_params,
                              )

    # Evaluate models
    evaluation_results = evaluate_models(models, 
                                         y_test,
                                         metrics=eval_cfg["metrics"],
                                         )
    
    #Pick the best model
    best_name = get_best_model(evaluation_results, metric = "roc_auc")
    best_model, _, _ = models[best_name]

    #Save metrics
    metrics_path = paths["metrics_dir"] / "model_comparison.json"
    evaluation_results.round(4).to_json(metrics_path, orient="index", indent=2)
    evaluation_results.round(4).to_csv(paths["metrics_dir"] / "model_comparison.csv")

    # Save best model and preprocessor
    with open(paths["models_dir"] / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(paths["models_dir"] / "preprocessor.pkl", "wb") as f:
        pickle.dump(transformer, f)
    with open(paths["models_dir"] / "feature_names.json", "wb") as f:
        pickle.dump(feature_names, f)
    with open(paths["models_dir"] / "best_model_name.txt", "w") as f:
        f.write(best_name)

    # Save dashboard artifacts: ROC curves and predictions for all models
    roc_curves = {}
    evaluation_data = {"y_test": y_test.astype(int).tolist(), "models": {}}
    for name, (model, y_pred, y_pred_proba) in models.items():
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_curves[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        evaluation_data["models"][name] = {
            "y_pred": y_pred.astype(int).tolist(),
            "y_pred_proba": y_pred_proba.tolist(),
        }
    with open(paths["dashboard_dir"] / "roc_curves.json", "w") as f:
        json.dump(roc_curves, f, indent=2)
    with open(paths["dashboard_dir"] / "evaluation_data.json", "w") as f:
              json.dump(evaluation_data, f, indent=2)


    # SHAP on best model
    run_shap_analysis(
        model=best_model,
        X=X_test,
        feature_names=feature_names,
        plots_dir=paths["plots_dir"],
        model_name=best_name,
        max_display=shap_cfg["max_display_features"],
        sample_size=shap_cfg["sample_size"]
    )


if __name__ == "__main__":
    main()