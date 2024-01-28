import argparse
import json
import logging
import os

import joblib  # type: ignore
import pandas as pd
from src.evaluation import Evaluation
from src.folds import Fold
from src.model_dispatcher import models
from utils.constants import CLASSIFICATION, DATA_FOLDER, MNIST, MODEL_FOLDER

DATA_FILENAME = f"{MNIST}_data.csv"
RAW_DATA_PATH = os.path.join(DATA_FOLDER, DATA_FILENAME)
FOLD_DATA_PATH = os.path.join(DATA_FOLDER, f"folds_{DATA_FILENAME}")
MODEL_PATH = os.path.join(MODEL_FOLDER, MNIST, "MODEL_NAME")

NUMBER_OF_FOLDS = 5


def create_folds_from_raw_data() -> None:
    # Check if file exists
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"File {DATA_FILENAME} not found in {DATA_FOLDER}. Please run the exploration notebook first."
        )

    if os.path.exists(FOLD_DATA_PATH):
        logging.info("Folds already created")
        return

    logging.info("Creating folds")
    df = pd.read_csv(RAW_DATA_PATH)  # type: ignore
    fold_df = Fold(
        problem_type=CLASSIFICATION,
        dataset_balanced=False,
        target_col_name="class",
        num_of_folds=NUMBER_OF_FOLDS,
    ).create_fold(df)
    fold_df.to_csv(FOLD_DATA_PATH, index=False)
    logging.info("Folds created")


def run(fold: int, model: str):
    # read the training data with folds
    df = pd.read_csv(FOLD_DATA_PATH)  # type: ignore

    # training data is where kfold is not equal to provided fold
    # also, note that we reset the index
    df_train = df[df.kfold != fold].reset_index(drop=True)  # type: ignore

    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)  # type: ignore

    # drop the class column from dataframe and convert it to
    # a numpy array by using .values.
    # target is class column in the dataframe
    x_train = df_train.drop("class", axis=1).values  # type: ignore
    y_train = df_train["class"].values  # type: ignore

    # similarly, for validation, we have
    x_valid = df_valid.drop("class", axis=1).values  # type: ignore
    y_valid = df_valid["class"].values  # type: ignore

    # fetch the model from model_dispatcher
    clf = models.get(model)
    if not clf:
        raise ValueError(f"Model {model} not found in model_dispatcher")

    # fit the model on training data
    clf.fit(x_train, y_train)  # type: ignore

    # create predictions for validation samples
    preds = clf.predict(x_valid)  # type: ignore

    # calculate & print accuracy
    metrics = Evaluation(problem_type=CLASSIFICATION, skewed=False).evaluate(
        y_valid,  # type: ignore
        preds,  # type: ignore
    )
    logging.info(f"Fold={fold}, Metrics={metrics}")
    accuracy = metrics["accuracy"]
    logging.info(f">>>>>>>>>>>>Fold={fold}, Accuracy={accuracy}")

    # create folder if it does not exist
    model_path = MODEL_PATH.replace("MODEL_NAME", model)
    os.makedirs(model_path, exist_ok=True)

    # save the model
    joblib.dump(  # type: ignore
        clf,
        os.path.join(model_path, f"fold_{fold}.bin"),
    )

    # get or create evaluation json
    evaluation_json_path = os.path.join(model_path, "evaluation.json")
    if os.path.exists(evaluation_json_path):
        with open(evaluation_json_path, "r+") as f:
            evaluation_json = json.load(f)  # type: ignore
    else:
        evaluation_json = {}

    evaluation_json[f"fold_{fold}"] = metrics
    with open(evaluation_json_path, "w") as f:
        json.dump(evaluation_json, f, indent=4)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # In the exploration notebook, the dataset is not balanced so we use stratified kfold.
    # However on choosing the evaluation metrics, the data is not skewed so we use accuracy/f1
    # Our goal is a classification problem

    create_folds_from_raw_data()

    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add the different arguments you need and their type
    parser.add_argument("--model", type=str)
    # read the arguments from the command line
    args = parser.parse_args()

    for fold in range(NUMBER_OF_FOLDS):
        run(fold=fold, model=args.model)
