from typing import List

import numpy as np


def accuracy(y_true: List[float], y_pred: List[float]) -> float:
    """
    Function to calculate accuracy
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: accuracy score
    """
    # initialize a simple counter for correct predictions
    correct_counter = 0
    # loop over all elements of y_true
    # and y_pred "together"
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            # if prediction is equal to truth, increase the counter
            correct_counter += 1

    # return accuracy
    # which is correct predictions over the number of samples
    return correct_counter / len(y_true)


def accuracy_v2(y_true: List[float], y_pred: List[float]) -> float:  # type: ignore
    """
    Function to calculate accuracy using tp/tn/fp/fn
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: accuracy score
    """
    classes = np.unique(y_true)  # type: ignore
    accuracy_score = 0.0
    for class_ in classes:
        # all classes except current are 0
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        tp = true_positive(temp_true, temp_pred)
        fp = false_positive(temp_true, temp_pred)
        fn = false_negative(temp_true, temp_pred)
        tn = true_negative(temp_true, temp_pred)
        temp_accuracy_score = (tp + tn) / (tp + tn + fp + fn)
        accuracy_score += temp_accuracy_score

    accuracy_score /= len(classes)
    return accuracy_score


def true_positive(y_true: List[int], y_pred: List[int]) -> float:
    """
    Function to calculate True Positives
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of true positives
    """
    # initialize
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
    return tp


def true_negative(y_true: List[int], y_pred: List[int]) -> float:
    """
    Function to calculate True Negatives
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of true negatives
    """
    # initialize
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
    return tn


def false_positive(y_true: List[int], y_pred: List[int]) -> float:
    """
    Function to calculate False Positives
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of false positives
    """
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
    return fp


def false_negative(y_true: List[int], y_pred: List[int]) -> float:
    """
    Function to calculate False Negatives
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of false negatives
    """
    # initialize
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1
    return fn


def f1(precision: float, recall: float) -> float:
    """
    Function to calculate f1 score
    :param precision: precision score
    :param recall: recall score
    :return: f1 score
    """
    f1 = 2 * precision * recall / (precision + recall)
    return f1
