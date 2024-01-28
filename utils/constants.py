REGRESSION = "regression"
CLASSIFICATION = "classification"

ALLOWED_REGRESSION_METRICS = ["rmse", "mse", "mae"]
ALLOWED_CLASSIFICATION_METRICS = {
    0: [  # not_skewed
        "accuracy",
        "precision",
        "recall",
        "f1",
    ],  # f1 or accuracy are the best?
    1: [  # skewed
        "precision",
        "recall",
        "f1",
        "auc",
    ],  # auc is the best?
}

METRIC_INTERPRETATION_MAP = {
    "accuracy": "Higher is better. (between 0 and 1). model is correct <<value_percent>> times",
    "precision": "Higher is better. (between 0 and 1). model is correct <<value_percent>> times. Macro average precision is calculated for multi-class classification. There are others for multi-class classification: micro, weighted, samples",
    "recall": "Higher is better. (between 0 and 1). model identified <<value_percent>> of positive samples correctly. Macro average recall is calculated for multi-class classification. There are others for multi-class classification: micro, weighted, samples",
    "f1": "Higher is better. (between 0 and 1). model is correct <<value_percent>> times. Macro average f1 is calculated for multi-class classification. There are others for multi-class classification: micro, weighted, samples",
    "auc": "Higher is better",
    "mse": "Lower is better",
    "rmse": "Lower is better",
    "mae": "Lower is better",
}

DATA_FOLDER = "input"
MODEL_FOLDER = "models"

MNIST = "mnist"
