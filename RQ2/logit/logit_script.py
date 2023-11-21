import os
import chardet
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    average_precision_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
)

from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.dummy import DummyClassifier
from copy import copy
import numpy as np
import joblib
import wandb
import logging


logging.basicConfig(level=logging.INFO)

random_state = 7
no_of_active_features = 15
total_features = 13

scorers_for_gridcv = {
    "accuracy_score": make_scorer(accuracy_score),
    "precision_score": make_scorer(precision_score),
    "recall_score": make_scorer(recall_score),
    "fbeta_score": make_scorer(fbeta_score, beta=0.5),
    "balanced_accuracy_score": make_scorer(balanced_accuracy_score),
    "average_precision_score": make_scorer(average_precision_score),
}


class CustomScore:
    def __init__(
        self, accuracy, precision, recall, balanced_accuracy, fbeta, avg_precision
    ) -> None:
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.balanced_accuracy = balanced_accuracy
        self.fbeta = fbeta
        self.avg_precision = avg_precision


def get_all_scores(y_real, y_pred, y_scores) -> CustomScore:
    accuracy = accuracy_score(y_real, y_pred)
    precision = precision_score(y_real, y_pred)
    recall = recall_score(y_real, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_real, y_pred)
    fbeta = fbeta_score(y_real, y_pred, beta=0.5)
    avg_precision = average_precision_score(y_real, y_scores)

    return CustomScore(
        accuracy, precision, recall, balanced_accuracy, fbeta, avg_precision
    )


def convert_scores_to_dict(custom_scores: CustomScore) -> dict:
    metrics = {}
    metrics["accuracy"] = round(custom_scores.accuracy, 2)
    metrics["precision"] = round(custom_scores.precision, 2)
    metrics["recall"] = round(custom_scores.recall, 2)
    metrics["balanced_accuracy"] = round(custom_scores.balanced_accuracy, 2)
    metrics["fbeta"] = round(custom_scores.fbeta, 2)
    metrics["avg_precision"] = round(custom_scores.avg_precision, 2)
    return metrics


def imbalanced_performance_summary(scores: CustomScore, model_name: str):
    metrics = convert_scores_to_dict(scores)
    table = make_table(metrics, model_name)
    metrics_chart = wandb.visualize("wandb/metrics/v1", table)
    return metrics_chart


def make_table(metrics, model_name):
    columns = ["metric_name", "metric_value", "model_name"]
    table_content = [[name, value, model_name] for name, value in metrics.items()]

    table = wandb.Table(columns=columns, data=table_content)

    return table


def get_encoding(file_path):
    with open(file_path, "rb") as f:
        data = f.read(10000)
    return chardet.detect(data).get("encoding")


def read_files(file_path: str):
    if file_path == None:
        print("Enter filepath")
        return

    considered_features = ['max_breadth', 'density', 'virality', 'depth', 'structural_heterogeneity', 'characteristic_distance', 'size_of_scc', 'layer_ratio']
    
    x_train_file = "./rq2_x_train_df.csv"

    train_file = os.path.join(file_path + os.path.sep + x_train_file)

    train_df = pd.read_csv(
        train_file, index_col=0, sep=",", encoding=get_encoding(train_file)
    )

    X_train_val = train_df.iloc[:, 3:total_features]
    y_train_val = train_df.iloc[:, total_features]
    
    X_train_val = X_train_val[considered_features]

    return X_train_val, y_train_val


def nested_loop(X_train_val: pd.DataFrame, y_train_val: pd.Series):
    outer_cv = RepeatedKFold(n_splits=5, n_repeats=1)

    logit_avg_prec = 0

    trained_logit_model = None

    loop_index = 0

    for train_index, val_index in outer_cv.split(X_train_val.to_numpy()):
        
        logging.info('round: ' + str(loop_index))

        loop_index += 1

        X_train, X_val = (
            X_train_val.iloc[train_index, :],
            X_train_val.iloc[val_index, :],
        )

        y_train, Y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

        normalized_df = copy(X_train)
        cd_first_quantile = np.quantile(normalized_df["size"], 0.25)
        cd_third_quantile = np.quantile(normalized_df["size"], 0.75)
        normalized_df["depth"] = np.log(normalized_df["depth"])
        normalized_df["size"] = np.log(normalized_df["size"])
        normalized_df["max_breadth"] = np.log(normalized_df["max_breadth"])
        normalized_df["strongly_cc"] = np.log(normalized_df["strongly_cc"])
        normalized_df["characteristic_distance"] = np.log(
            normalized_df["characteristic_distance"]
            + cd_first_quantile**2 / cd_third_quantile
        )

        scaler = StandardScaler().set_output(transform="pandas")
        scaled_X_train = scaler.fit_transform(normalized_df)

        smotetomek = smotetomek_as_cleaner()
        scaled_resampled_X_train, scaled_resampled_y_train = smotetomek.fit_resample(
            scaled_X_train, y_train
        )

        logging.info('initiating inner cv for round ' + str(loop_index))
        inner_cv = RepeatedKFold(n_splits=5, n_repeats=3)

        logit_result = fit_multiple_models(
            scaled_resampled_X_train, scaled_resampled_y_train, inner_cv
        )

        logit_model = logit_result.best_estimator_

        run = wandb.init(
            project="thesis", group="logistic", job_type="epoch_" + str(loop_index)
        )

        logit_Y_pred = logit_model.best_estimator_.predict(X_val)
        logit_Y_pred_proba = logit_model.best_estimator_.predict_proba(X_val)
        logit_custom_score = get_all_scores(
            Y_val, logit_Y_pred, logit_Y_pred_proba[:, 1]
        )
        scores_dict = convert_scores_to_dict(logit_custom_score)
        run.log(scores_dict)

        joblib.dump(logit_result, "logit.joblib")
        artifact = run.Artifact(
            "logit_model_epoch_"+str(loop_index),
            type="model",
            description="GridSearch output for logistic classifier",
            metadata={"parameters": logit_result.best_params_, "metrics": scores_dict},
        )

        artifact.add_file('clf.joblib') # or 'clf.pkl'

        run.log_artifact(artifact)

        run.finish()

        logging.info('logged to wandb')

        if logit_avg_prec < logit_custom_score.avg_precision:
            logit_avg_prec = logit_custom_score.avg_precision
            trained_logit_model = logit_model

    normalized_df = copy(X_train_val)
    cd_first_quantile = np.quantile(normalized_df["size"], 0.25)
    cd_third_quantile = np.quantile(normalized_df["size"], 0.75)
    normalized_df["depth"] = np.log(normalized_df["depth"])
    normalized_df["size"] = np.log(normalized_df["size"])
    normalized_df["max_breadth"] = np.log(normalized_df["max_breadth"])
    normalized_df["strongly_cc"] = np.log(normalized_df["strongly_cc"])
    normalized_df["characteristic_distance"] = np.log(
        normalized_df["characteristic_distance"]
        + cd_first_quantile**2 / cd_third_quantile
    )

    scaler = StandardScaler().set_output(transform="pandas")
    scaled_X_train_val = scaler.fit_transform(normalized_df)
    smotetomek = get_data_pipeline_with_smotetomek()
    (
        scaled_resampled_X_train_val,
        scaled_resampled_y_train_val,
    ) = smotetomek.fit_resample(scaled_X_train_val, y_train_val)

    logistic = LogisticRegression(
        **trained_logit_model.best_params_, random_state=random_state
    )
    refit_logit = logistic.fit(
        scaled_resampled_X_train_val, scaled_resampled_y_train_val
    )

    return refit_logit


def get_data_pipeline_with_smotetomek():
    smotetomek = SMOTETomek(
        smote=SMOTE(sampling_strategy="all"),
        tomek=TomekLinks(
            sampling_strategy="majority",
        ),
        random_state=random_state,
    )

    return smotetomek


def smotetomek_as_cleaner():
    smotetomek_as_cleaner = SMOTETomek(
        tomek=TomekLinks(),
        random_state=random_state,
    )

    return smotetomek_as_cleaner


def model_fitting_loop_with_grid_search(
    model, x_train_df, y_train_df, inner_cv, grid_param, model_name
):
    clf = GridSearchCV(
        estimator=model,
        cv=inner_cv,
        refit="average_precision_score",
        param_grid=grid_param,
        scoring=scorers_for_gridcv,
        verbose=3,
        n_jobs=-1,
    )

    result = clf.fit(x_train_df, y_train_df)
    return result


def fit_dummy_classifier(x_train_df, y_train_df, constant):
    dummy_clf = DummyClassifier(strategy="constant", constant=constant)
    dummy_clf.fit(x_train_df, y_train_df)
    return dummy_clf


def fit_multiple_models(x_train_df, y_train_df, inner_cv):
    # Logistic Regression
    logit_grid = {
        "penalty": ["l1", "l2", "elasticnet", None],
        "dual": [True, False],
        "C": [_ for _ in range(1, 10, 1)],
        "fit_intercept": [True, False],
        "solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
        "n_jobs": [-1],
    }

    logit_model = LogisticRegression()
    logit_result = model_fitting_loop_with_grid_search(
        logit_model, x_train_df, y_train_df, inner_cv, logit_grid, "Logistic Regression"
    )

    # log the following param for every model
    # best_estimator_
    # best_score_
    # best_params_
    # scorer_

    return logit_result


def flip_true_false(y):
    copy_y = copy(y)
    flipped_y = [0 if item == 1 else 1 for item in copy_y]
    return flipped_y


def work():
    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"

    # csv_path = '/root'
    csv_path = "."

    logging.info('reading files')
    X_train_val, y_train_val = read_files(csv_path)

    logging.info('started the loop')
    # call the nested loop to get all the trained models
    print(X_train_val.shape, y_train_val.shape)
    nested_loop(X_train_val, y_train_val)


if __name__ == "__main__":
    work()
