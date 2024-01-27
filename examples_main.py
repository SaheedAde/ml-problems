import logging

from src.evaluation import Evaluation
from src.folds import Fold

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fold(problem_type="regression").run_example()
    Fold(problem_type="classification").run_example()
    Fold(problem_type="classification", dataset_balanced=True).run_example()

    Evaluation(problem_type="regression").run_example()
    Evaluation(problem_type="classification").run_example()
    Evaluation(problem_type="classification", skewed=False).run_example()
