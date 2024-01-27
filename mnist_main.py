import logging

from src.folds import Fold

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # In the exploration notebook, the dataset is not balanced so we use stratified kfold.
    # However on choosing the evaluation metrics, the data is not skewed so we use accuracy/f1
    # Our goal is a classification problem
    Fold(problem_type="regression").run_example()
