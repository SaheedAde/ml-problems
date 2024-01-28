import logging

from src.evaluation import Evaluation
from src.folds import Fold
from utils.constants import CLASSIFICATION, REGRESSION

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fold(problem_type=REGRESSION).run_example()
    Fold(problem_type=CLASSIFICATION).run_example()
    Fold(problem_type=CLASSIFICATION, dataset_balanced=True).run_example()

    Evaluation(problem_type=REGRESSION).run_example()
    Evaluation(problem_type=CLASSIFICATION).run_example()
    Evaluation(problem_type=CLASSIFICATION, skewed=False).run_example()
