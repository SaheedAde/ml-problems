import logging

import numpy as np
import pandas as pd
from schema.fold import validate
from sklearn import datasets, model_selection
from utils.constants import REGRESSION


class Fold:
    """
    Class for fold
    """

    def __init__(
        self,
        problem_type: str,
        dataset_balanced: bool = False,
        target_col_name: str = "target",
        num_of_folds: int = 5,
    ):
        data = validate(
            {"problem_type": problem_type, "dataset_balanced": dataset_balanced}
        )

        self.num_of_folds = num_of_folds
        self.problem_type = data.problem_type
        self.target_col_name = target_col_name
        self.dataset_balanced = data.dataset_balanced

        self.create_fold = self._do_nothing
        if problem_type == REGRESSION:
            self.create_fold = self._stratified_kfold_regression
            return

        self.create_fold = self._classification_fold

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
        data.loc[:, "bins"] = pd.cut(  # type: ignore
            data[self.target_col_name], bins=num_bins, labels=False
        )

        # initiate the kfold class from model_selection module
        kf = model_selection.StratifiedKFold(n_splits=self.num_of_folds)

        # fill the new kfold column
        # note that, instead of targets, we use bins!
        bins_col = data.bins.values  # type: ignore
        for f, (_, fold_idx) in enumerate(kf.split(X=data, y=bins_col)):  # type: ignore
            data.loc[fold_idx, "kfold"] = f

        # drop the bins column
        data = data.drop("bins", axis=1)  # type: ignore
        # return dataframe with folds
        return data

    def _classification_fold(self, data: pd.DataFrame) -> pd.DataFrame:
        # we create a new column called kfold and fill it with -1
        data["kfold"] = -1

        # the next step is to randomize the rows of the data
        data = data.sample(frac=1).reset_index(drop=True)  # type: ignore

        if self.dataset_balanced:
            # initiate the kfold class from model_selection module
            kf = model_selection.KFold(n_splits=self.num_of_folds)

            # fill the new kfold column
            for fold, (trn_, val_) in enumerate(kf.split(X=data)):  # type: ignore
                data.loc[val_, "kfold"] = fold  # type: ignore

            return data

        # fetch targets
        y = data[self.target_col_name].values  # type: ignore

        # initiate the kfold class from model_selection module
        kf = model_selection.StratifiedKFold(n_splits=self.num_of_folds)

        # fill the new kfold column
        for f, (t_, v_) in enumerate(kf.split(X=data, y=y)):  # type: ignore
            data.loc[v_, "kfold"] = f

        return data

    def run_example(self):
        if self.problem_type == REGRESSION:
            # we create a sample dataset with 15000 samples
            # and 100 features and 1 target
            X, y = datasets.make_regression(  # type: ignore
                n_samples=15000, n_features=100, n_targets=1
            )
            # create a dataframe out of our numpy arrays
            df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
            df.loc[:, self.target_col_name] = y

            logging.info(f"Dataset before REGRESSION fold creation is : {df.head()}")

            # create folds
            df = self.create_fold(df)

            logging.info("Dataset after REGRESSION fold creation")
            logging.info(f"Head : {df.head()}")
            logging.info(f"Tail : {df.tail()}")
            return

        if self.dataset_balanced:
            # we create a sample dataset with 15000 samples
            # and 100 features and 1 target
            text = "BALANCED"
            X, y = datasets.make_classification(  # type: ignore
                n_samples=15000, n_features=100, n_classes=2
            )
        else:
            # we create a sample dataset with 15000 samples
            # and 100 features and 1 target
            text = "UNBALANCED"
            X, y = datasets.make_classification(  # type: ignore
                n_samples=15000, n_features=100, n_classes=2, weights=[0.95, 0.05]
            )

        # create a dataframe out of our numpy arrays
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df.loc[:, self.target_col_name] = y

        logging.info(
            f"Dataset before {text} CLASSIFICATION fold creation is : {df.head()}"
        )

        # create folds
        df = self.create_fold(df)

        logging.info(f"Dataset after {text} CLASSIFICATION fold creation")
        logging.info(f"Head : {df.head()}")
        logging.info(f"Tail : {df.tail()}")
