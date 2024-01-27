import logging
from typing import Any, Dict, List, Optional

import numpy as np
from schema.evaluation import validate
from sklearn import metrics
from utils.classification_evaluations import (
    accuracy,
    accuracy_v2,
    false_negative,
    false_positive,
    true_positive,
)
from utils.constants import (
    ALLOWED_CLASSIFICATION_METRICS,
    ALLOWED_REGRESSION_METRICS,
    CLASSIFICATION,
    METRIC_INTERPRETATION_MAP,
    REGRESSION,
)


class Evaluation:
    """
    Class for evaluation
    """

    def __init__(
        self,
        problem_type: str,
        skewed: bool = True,
        metric: Optional[str] = None,
    ):
        data = validate(
            {
                "problem_type": problem_type,
                "skewed": skewed,
                "metric": metric,
            }
        )

        self.metric = data.metric
        self.skewed = data.skewed
        self.problem_type = data.problem_type

    def evaluate(self, y_true: List[float], y_pred: List[float]) -> Dict[str, Any]:
        metrics_to_calculate = []
        if self.metric:
            metrics_to_calculate = [self.metric]

        if not metrics_to_calculate:
            if self.problem_type == CLASSIFICATION:
                metrics_to_calculate = ALLOWED_CLASSIFICATION_METRICS[self.skewed]

            if self.problem_type == REGRESSION:
                metrics_to_calculate = ALLOWED_REGRESSION_METRICS

        logging.info(f"Metrics to calculate : {metrics_to_calculate}")

        calculated_metrics = {}
        for metric in metrics_to_calculate:
            calculated_metrics[metric] = getattr(self, f"_calculate_{metric}")(
                y_true, y_pred
            )
            interpretation = METRIC_INTERPRETATION_MAP[metric]
            interpretation = interpretation.replace(
                "<<value_percent>>",
                str(calculated_metrics[metric] * 100),  # type: ignore
            )
            calculated_metrics["interpretation"] = interpretation

        return calculated_metrics  # type: ignore

    def _calculate_accuracy(self, y_true: List[float], y_pred: List[float]) -> float:  # type: ignore
        # Use one of the following:
        # 1. accuracy(y_true, y_pred)
        # 2. accuracy_v2(y_true, y_pred)
        # 3. metrics.accuracy_score(y_true, y_pred)
        # All three should give the same result

        assert (
            accuracy(y_true, y_pred)
            == accuracy_v2(y_true, y_pred)
            == metrics.accuracy_score(y_true, y_pred)  # type: ignore
        )  # type: ignore

        return metrics.accuracy_score(y_true, y_pred)  # type: ignore

    def _calculate_precision(self, y_true: List[float], y_pred: List[float]) -> float:
        """
        Function to calculate precision
        :param y_true: list of true values
        :param y_pred: list of predicted values
        :return: precision score
        """
        classes = np.unique(y_true)  # type: ignore
        precision = 0.0
        for class_ in classes:
            # all classes except current are 0
            temp_true = [1 if p == class_ else 0 for p in y_true]
            temp_pred = [1 if p == class_ else 0 for p in y_pred]

            tp = true_positive(temp_true, temp_pred)
            fp = false_positive(temp_true, temp_pred)

            temp_precision = tp / (tp + fp)

            precision += temp_precision

        precision /= len(classes)
        return precision

    def _calculate_recall(self, y_true: List[float], y_pred: List[float]) -> float:
        """
        Function to calculate recall
        :param y_true: list of true values
        :param y_pred: list of predicted values
        :return: recall score
        """
        classes = np.unique(y_true)  # type: ignore
        recall = 0.0
        for class_ in classes:
            # all classes except current are 0
            temp_true = [1 if p == class_ else 0 for p in y_true]
            temp_pred = [1 if p == class_ else 0 for p in y_pred]

            tp = true_positive(temp_true, temp_pred)
            fn = false_negative(temp_true, temp_pred)

            temp_recall = tp / (tp + fn)

            recall += temp_recall

        recall /= len(classes)
        return recall

    def _calculate_f1(self, y_true: List[float], y_pred: List[float]) -> float:
        # Use one of the following:
        # 1. f1(self._calculate_precision(y_true, y_pred), self._calculate_recall(y_true, y_pred))
        # 2. metrics.f1_score(y_true, y_pred)
        # Both should give the same result

        return metrics.f1_score(y_true, y_pred, average="macro")  # type: ignore

    def _calculate_auc(self, y_true: List[float], y_pred: List[float]) -> float:
        """
        Function to calculate auc score
        :param y_true: list of true values
        :param y_pred: list of predicted values
        :return: auc score
        """
        return metrics.roc_auc_score(y_true, y_pred)  # type: ignore

    def _calculate_mse(self, y_true: List[float], y_pred: List[float]) -> float:
        """
        Function to calculate mse
        :param y_true: list of true values
        :param y_pred: list of predicted values
        :return: mse score
        """
        return metrics.mean_squared_error(y_true, y_pred)  # type: ignore

    def _calculate_rmse(self, y_true: List[float], y_pred: List[float]) -> float:
        """
        Function to calculate rmse
        :param y_true: list of true values
        :param y_pred: list of predicted values
        :return: rmse score
        """
        return np.sqrt(metrics.mean_squared_error(y_true, y_pred))  # type: ignore

    def _calculate_mae(self, y_true: List[float], y_pred: List[float]) -> float:
        """
        Function to calculate mae
        :param y_true: list of true values
        :param y_pred: list of predicted values
        :return: mae score
        """
        return metrics.mean_absolute_error(y_true, y_pred)  # type: ignore

    def run_example(self):
        y_true = []
        y_pred = []
        if self.problem_type == REGRESSION:
            y_true = [1, 2, 3, 1, 2, 3, 1, 2, 3]
            y_pred = [2, 1, 3, 1, 2, 3, 3, 1, 2]

        if self.problem_type == CLASSIFICATION:
            if not self.skewed:
                y_true = [0, 1, 1, 1, 0, 0, 0, 1]
                y_pred = [0, 1, 0, 1, 0, 1, 0, 0]
            else:
                y_true = [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
                y_pred = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]

        logging.info(f"y_true : {y_true}")
        logging.info(f"y_pred : {y_pred}")

        metrics = self.evaluate(y_true, y_pred)  # type: ignore
        logging.info(f"Metrics : {metrics}")
