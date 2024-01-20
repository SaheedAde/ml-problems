import numpy as np
import pandas as pd
from sklearn import model_selection


class Fold:
    """
    Class for fold
    """

    def __init__(self, problem_type: str, dataset_balanced: bool):
        if problem_type not in ["classification", "regression"]:
            raise ValueError("Problem type not supported")
        if dataset_balanced not in [True, False]:
            raise ValueError("Dataset balanced not supported")

        self.create_fold = _default_kfold
        self.problem_type = problem_type
        self.dataset_balanced = dataset_balanced

        if problem_type == "regression":
            self.create_fold = _stratified_kfold_regression

    def _stratified_kfold_regression(self, data: pd.DataFrame) -> pd.DataFrame:
        # we create a new column called kfold and fill it with -1
        data["kfold"] = -1

        # the next step is to randomize the rows of the data
        data = data.sample(frac=1).reset_index(drop=True)

        # calculate the number of bins by Sturge's rule
        # I take the floor of the value, you can also
        # just round it
        num_bins = int(np.floor(1 + np.log2(len(data))))

        # bin targets
        data.loc[:, "bins"] = pd.cut(data["target"], bins=num_bins, labels=False)

        # initiate the kfold class from model_selection module
        kf = model_selection.StratifiedKFold(n_splits=5)

        # fill the new kfold column
        # note that, instead of targets, we use bins!
        for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
            data.loc[v_, "kfold"] = f

        # drop the bins column
        data = data.drop("bins", axis=1)
        # return dataframe with folds
        return data
