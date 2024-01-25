import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn import datasets, model_selection
from utils.constants import CLASSIFICATION, REGRESSION


class Fold:
    """
    Class for fold
    """

    def __init__(self, problem_type: str, dataset_balanced: Optional[bool] = None):
        if problem_type not in [CLASSIFICATION, REGRESSION]:
            raise ValueError("Problem type not supported")

        if problem_type == CLASSIFICATION:
            if dataset_balanced not in [True, False]:
                raise ValueError("Dataset balanced not supported")

        if problem_type == REGRESSION and dataset_balanced is not None:
            logging.warning(
                "Regression doesn't need dataset balanced parameter, clearing it"
            )
            dataset_balanced = None

        self.problem_type = problem_type
        self.dataset_balanced = dataset_balanced

        self.create_fold = self._do_nothing
        if problem_type == "regression":
            self.create_fold = self._stratified_kfold_regression

    def _do_nothing(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def _stratified_kfold_regression(self, data: pd.DataFrame) -> pd.DataFrame:
        # we create a new column called kfold and fill it with -1
        data["kfold"] = -1

        # the next step is to randomize the rows of the data
        data = data.sample(frac=1).reset_index(drop=True)  # type: ignore

        # calculate the number of bins by Sturge's rule
        # I take the floor of the value, you can also
        # just round it
        num_bins = int(np.floor(1 + np.log2(len(data))))

        # bin targets
        data.loc[:, "bins"] = pd.cut(data["target"], bins=num_bins, labels=False)  # type: ignore

        # initiate the kfold class from model_selection module
        kf = model_selection.StratifiedKFold(n_splits=5)

        # fill the new kfold column
        # note that, instead of targets, we use bins!
        bins_col = data.bins.values  # type: ignore
        for f, (_, fold_idx) in enumerate(kf.split(X=data, y=bins_col)):  # type: ignore
            data.loc[fold_idx, "kfold"] = f

        # drop the bins column
        data = data.drop("bins", axis=1)  # type: ignore
        # return dataframe with folds
        return data

    def run_example(self):
        if self.problem_type == "regression":
            # we create a sample dataset with 15000 samples
            # and 100 features and 1 target
            X, y = datasets.make_regression(  # type: ignore
                n_samples=15000, n_features=100, n_targets=1
            )
            # create a dataframe out of our numpy arrays
            df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
            df.loc[:, "target"] = y

            logging.info(f"Dataset before fold creation is : {df.head()}")

            # create folds
            df = self.create_fold(df)

            logging.info("Dataset after fold creation")
            logging.info(f"Head : {df.head()}")
            logging.info(f"Tail : {df.tail()}")
            return
