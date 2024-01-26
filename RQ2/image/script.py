# read df input from pickled files
# use smote upgrade

import os
import chardet
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    precision_recall_curve,
    average_precision_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)
from time import time
import xgboost as xgb
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold, GridSearchCV, train_test_split
from sklearn.dummy import DummyClassifier
from copy import copy
import numpy as np
import wandb
import joblib
import lightgbm as lgb
from typing import Dict, Any, List
import pickle
import logging
from flytekit import task, workflow, dynamic
import smote_variants as sv
import warnings
warnings.filterwarnings("ignore")

wandb_project = "test01"
random_state = 7
no_of_active_features = 15
optimization_metric = "average_precision_score"

scorers_for_gridcv = {
    "accuracy_score": make_scorer(accuracy_score),
    "precision_score": make_scorer(precision_score),
    "recall_score": make_scorer(recall_score),
    "fbeta_score": make_scorer(fbeta_score, beta=0.5),
    "balanced_accuracy_score": make_scorer(balanced_accuracy_score),
    "average_precision_score": make_scorer(average_precision_score),
    "roc_auc": make_scorer(roc_auc_score),
}


class CustomScore:
    def __init__(self, accuracy, precision, recall, balanced_accuracy, fbeta, avg_precision, roc_auc) -> None:
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.balanced_accuracy = balanced_accuracy
        self.fbeta = fbeta
        self.avg_precision = avg_precision
        self.roc_auc = roc_auc

    def get_default_metric(self) -> float:
        return self.avg_precision

class OutputClass:            
    def __init__(self, logit, dt, rf, xgb, lgb) -> None:
        self.logit = logit
        self.dt = dt
        self.rf = rf
        self.xgb = xgb
        self.lgb = lgb

class CLFOutput:
    def __init__(self, gridsearch_dict: dict, score: float) -> None:
        self.gridsearch_dict = gridsearch_dict
        self.score = score 

def get_all_scores(y_real, y_pred, y_scores) -> CustomScore:
    accuracy = accuracy_score(y_real, y_pred)
    precision = precision_score(y_real, y_pred)
    recall = recall_score(y_real, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_real, y_pred)
    fbeta = fbeta_score(y_real, y_pred, beta=0.5)
    avg_precision = average_precision_score(y_real, y_scores)
    roc_auc = roc_auc_score(y_real, y_scores)

    return CustomScore(accuracy, precision, recall, balanced_accuracy, fbeta, avg_precision, roc_auc)


def convert_scores_to_dict(custom_scores: CustomScore):
    metrics = {}

    metrics["accuracy"] = round(custom_scores.accuracy, 2)
    metrics["precision"] = round(custom_scores.precision, 2)
    metrics["recall"] = round(custom_scores.recall, 2)
    metrics["balanced_accuracy"] = round(custom_scores.balanced_accuracy, 2)
    metrics["fbeta"] = round(custom_scores.fbeta, 2)
    metrics["avg_precision"] = round(custom_scores.avg_precision, 2)
    metrics["roc_auc"] = round(custom_scores.roc_auc, 2)

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
        logging.error("READING_FILES: %s", "file path must be provided.")
        return

    false_feature_file = "rq2_vosoughi_False_features.csv"
    true_feature_file = "rq2_vosoughi_True_features.csv"
    true_file = os.path.join(file_path + os.path.sep + true_feature_file)
    false_file = os.path.join(file_path + os.path.sep + false_feature_file)

    vosoughi_false_df = pd.read_csv(false_file, index_col=0, sep=",", encoding=get_encoding(false_file))
    vosoughi_true_df = pd.read_csv(true_file, index_col=0, sep=",", encoding=get_encoding(true_file))
    vosoughi_df = pd.concat([vosoughi_false_df, vosoughi_true_df])
    return vosoughi_df


def read_pickled_input_files(file_path: str):
    if file_path == None:
        logging.error("READING_FILES: %s", "file path must be provided.")
        return

    X_train_val = None
    y_train_val = None
    X_test = None
    y_test = None

    # wandb.init(project=wandb_project)

    with open(file_path + "/x_train_val.pickle", "rb") as file:
        X_train_val = pickle.load(file)
        # joblib.dump(X_train_val, "x_train_val.joblib")
        # x_train_val_artifact = wandb.Artifact("x_train_val.joblib", type="dataset")
        # x_train_val_artifact.add_file("x_train_val.joblib")
        # wandb.log_artifact(x_train_val_artifact)

    with open(file_path + "/y_train_val.pickle", "rb") as file:
        y_train_val = pickle.load(file)
        # joblib.dump(y_train_val, "y_train_val.joblib")
        # y_train_val_artifact = wandb.Artifact("y_train_val.joblib", type="dataset")
        # y_train_val_artifact.add_file("y_train_val.joblib")
        # wandb.log_artifact(y_train_val_artifact)

    with open(file_path + "/x_test.pickle", "rb") as file:
        X_test = pickle.load(file)
        # joblib.dump(X_test, "x_test.joblib")
        # x_test_artifact = wandb.Artifact("x_test.joblib", type="dataset")
        # x_test_artifact.add_file("x_test.joblib")
        # wandb.log_artifact(x_test_artifact)

    with open(file_path + "/y_test.pickle", "rb") as file:
        y_test = pickle.load(file)
        # joblib.dump(y_test, "y_test.joblib")
        # y_test_artifact = wandb.Artifact("y_test.joblib", type="dataset")
        # y_test_artifact.add_file("y_test.joblib")
        # wandb.log_artifact(y_test_artifact)

    # wandb.finish()
    
    return X_train_val, X_test, y_train_val, y_test


def filter_and_split_df(df: pd.DataFrame):
    test_size = 0.15

    filtered_df = df[(df.characteristic_distance != -99999) & (df.depth != 0)]
    filtered_df = filtered_df.drop(columns=["avg_cluster_coef"], axis=1)
    filtered_df = filtered_df.drop(["weakly_cc"], axis=1)
    filtered_df = filtered_df.drop(["strongly_cc"], axis=1)
    filtered_df = filtered_df.drop(["size"], axis=1)
    total_features = no_of_active_features - 4

    X = filtered_df.iloc[:, 3:total_features]
    Y = filtered_df.iloc[:, total_features]
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    return X_train_val, X_test, y_train_val, y_test


@workflow
def nested_loop() -> list[OutputClass]:

    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"

    # FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.DEBUG)

    try:
        # wandb.init(project=wandb_project)
        # wandb.alert(title="Started", text="Your run has started. Mark the time.")
        # wandb.finish()

        start = time()

        logging.info("WORK: %s", "initiating processing, reading files")
        csv_path = "."
        X_train_val, X_test, y_train_val, y_test = read_pickled_input_files(csv_path)

        # call the nested loop to get all the trained models
        logging.info("WORK: %s", f"shapes of input: {X_train_val.shape}, {X_test.shape}, {y_train_val.shape}, {y_test.shape}")

        logging.info("WORK: %s", "entering nested loop")

        outer_cv = RepeatedKFold(n_splits=2, n_repeats=1)

        loop_index = 0

        logging.info("NESTED_LOOP: %s", "loop starts")

        loop_outputs = []

        for train_index, val_index in outer_cv.split(X_train_val.to_numpy()):

            loop_index += 1
            epoch_str = "epoch_" + str(loop_index)

            logging.info("NESTED_LOOP: %s", f"inside loop epoch {loop_index}")

            X_train, X_val = (X_train_val.iloc[train_index, :], X_train_val.iloc[val_index, :])

            y_train, Y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

            logging.info("NESTED_LOOP: %s", "feature scaling")
            normalized_df = copy(X_train)
            cd_first_quantile = np.quantile(normalized_df["characteristic_distance"], 0.25)
            cd_third_quantile = np.quantile(normalized_df["characteristic_distance"], 0.75)
            normalized_df["depth"] = np.log(normalized_df["depth"])
            normalized_df["max_breadth"] = np.log(normalized_df["max_breadth"])
            # normalized_df["size"] = np.log(normalized_df["size"])
            # normalized_df["strongly_cc"] = np.log(normalized_df["strongly_cc"])
            normalized_df["characteristic_distance"] = np.log(normalized_df["characteristic_distance"] + cd_first_quantile**2 / cd_third_quantile)

            scaler = StandardScaler().set_output(transform="pandas")
            scaled_X_train = scaler.fit_transform(normalized_df)
            scaled_resampled_X_train, scaled_resampled_y_train = oversample_data(scaled_X_train.to_numpy(), y_train.to_numpy())

            inner_cv = RepeatedKFold(n_splits=5, n_repeats=3)

            logging.info("NESTED_LOOP: %s", f"entering model fitting for {epoch_str}")

            
            dt_output = fit_dt_model(
                x_train_df=scaled_resampled_X_train, y_train_df=scaled_resampled_y_train, X_val=X_val, Y_val=Y_val, inner_cv=inner_cv, epoch_str=epoch_str
            )
            logit_output = fit_logistic_model(
                x_train_df=scaled_resampled_X_train, y_train_df=scaled_resampled_y_train, X_val=X_val, Y_val=Y_val, inner_cv=inner_cv, epoch_str=epoch_str
            )
            
            # rf_output = fit_rf_model(
            #     x_train_df=scaled_resampled_X_train, y_train_df=scaled_resampled_y_train, X_val=X_val, Y_val=Y_val, inner_cv=inner_cv, epoch_str=epoch_str
            # )
            rf_output = None
            xgb_output = fit_xgb_model(
                x_train_df=scaled_resampled_X_train, y_train_df=scaled_resampled_y_train, X_val=X_val, Y_val=Y_val, inner_cv=inner_cv, epoch_str=epoch_str
            )
            lgb_output = fit_lgb_model(
                x_train_df=scaled_resampled_X_train, y_train_df=scaled_resampled_y_train, X_val=X_val, Y_val=Y_val, inner_cv=inner_cv, epoch_str=epoch_str
            )
            loop_outputs.append(OutputClass(logit_output, dt_output, rf_output, xgb_output, lgb_output))
            logging.info("NESTED_LOOP: %s", f"model fitting for {epoch_str} completed.")
            

        logging.info("NESTED_LOOP: %s", f"loop_outputs has {len(loop_outputs)} items.")
        # refitt = refitting_models(loop_outputs=loop_outputs, X_train_val=X_train_val, y_train_val=y_train_val)

        end = time()
        time_taken = str(end - start)
        logging.info("NESTED_LOOP: %s", f"{epoch_str} re-fitting logging compelte, end of epoch. it took {time_taken} seconds")

        # wandb.init(project=wandb_project)
        # wandb.alert(title="Complete", text="Your run is complete. Check the board.")
        # wandb.finish()

    except Exception as error:
        logging.info("MAIN: %s", "ERROR: some error happened, could not finish.")
        logging.info("MAIN: %s", error)

        # wandb.init(project=wandb_project)
        # wandb.alert(title="Error", text="Your run was interrupted by some exception.")
        # wandb.finish()
    return loop_outputs

@workflow
def main_wf():
    loop_outputs = nested_loop()
    refitt = refitting_models(loop_outputs=loop_outputs)
    loop_outputs >> refitt


@task(container_image="istiyaksiddiquee/flyte-for-kube:test17")
def refitting_models(
    loop_outputs: list[OutputClass]
) -> None:

    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"

    csv_path = "."
    X_train_val, _, y_train_val, _ = read_pickled_input_files(csv_path)

    loop_counter = 0

    logit_epoch_id = -1
    dt_epoch_id = -1
    rf_epoch_id = -1
    xgb_epoch_id = -1
    lgb_epoch_id = -1

    logit_avg_prec = 0
    dt_avg_prec = 0
    rf_avg_prec = 0
    xgb_avg_prec = 0
    lgb_avg_prec = 0

    trained_dt_model = None
    trained_rf_model = None
    trained_logit_model = None
    trained_xgb_model = None
    trained_lgb_model = None

    for loop_item in loop_outputs:

        logit_result = loop_item.logit.gridsearch_dict
        dt_result = loop_item.dt.gridsearch_dict
        # rf_result = loop_item.rf.gridsearch_dict
        xgb_result = loop_item.xgb.gridsearch_dict
        lgb_result = loop_item.lgb.gridsearch_dict
    
        if logit_result != None and logit_avg_prec < loop_item.logit.score:
            logit_avg_prec = loop_item.logit.score
            trained_logit_model = logit_result
            logit_epoch_id = loop_counter

        if dt_result != None and dt_avg_prec < loop_item.dt.score:
            dt_avg_prec = loop_item.dt.score
            trained_dt_model = dt_result
            dt_epoch_id = loop_counter

        # if rf_result != None and rf_avg_prec < loop_item.rf.score:
        #     rf_avg_prec = loop_item.rf.score
        #     trained_rf_model = rf_result
            # rf_epoch_id = loop_counter

        if xgb_result != None and xgb_avg_prec < loop_item.xgb.score:
            xgb_avg_prec = loop_item.xgb.score
            trained_xgb_model = xgb_result
            xgb_epoch_id = loop_counter

        if lgb_result != None and lgb_avg_prec < loop_item.lgb.score:
            lgb_avg_prec = loop_item.lgb.score
            trained_lgb_model = lgb_result
            lgb_epoch_id = loop_counter
        
        loop_counter += 1

    logging.info("REFITTING_MODELS: %s", f"models retrieved, re-fitting starts")
    normalized_df = copy(X_train_val)
    cd_first_quantile = np.quantile(normalized_df["characteristic_distance"], 0.25)
    cd_third_quantile = np.quantile(normalized_df["characteristic_distance"], 0.75)
    normalized_df["depth"] = np.log(normalized_df["depth"])
    normalized_df["max_breadth"] = np.log(normalized_df["max_breadth"])
    # normalized_df["size"] = np.log(normalized_df["size"])
    # normalized_df["strongly_cc"] = np.log(normalized_df["strongly_cc"])
    normalized_df["characteristic_distance"] = np.log(normalized_df["characteristic_distance"] + cd_first_quantile**2 / cd_third_quantile)

    scaler = StandardScaler().set_output(transform="pandas")
    scaled_X_train_val = scaler.fit_transform(normalized_df)
    scaled_resampled_X_train_val, scaled_resampled_y_train_val = oversample_data(scaled_X_train_val.to_numpy(), y_train_val.to_numpy())

    # dummy_false = fit_dummy_classifier(scaled_resampled_X_train_val, scaled_resampled_y_train_val, 0)

    if trained_logit_model != None:
        # store logit model
        logistic = LogisticRegression(**trained_logit_model)
        refit_logit = logistic.fit(scaled_resampled_X_train_val, scaled_resampled_y_train_val)

        wandb.init(project=wandb_project, group="logit", job_type="final")
        joblib.dump(refit_logit, "logit.joblib")
        logit_artifact = wandb.Artifact(
            "Logistic-Model",
            type="model",
            description="selected Logistic model",
            metadata={
                "parameters": trained_logit_model,
                "epoch": logit_epoch_id,
            },
        )

        logit_artifact.add_file("logit.joblib")
        wandb.log_artifact(logit_artifact)
        wandb.finish()

    if trained_dt_model != None:
        # store dt model
        dt = DecisionTreeClassifier(**trained_dt_model)
        refit_dt = dt.fit(scaled_resampled_X_train_val, scaled_resampled_y_train_val)

        wandb.init(project=wandb_project, group="dt", job_type="final")
        joblib.dump(refit_dt, "dt.joblib")
        dt_artifact = wandb.Artifact(
            "DT-Model",
            type="model",
            description="selected DT model",
            metadata={
                "parameters": trained_dt_model,
                "epoch": dt_epoch_id,
            },
        )

        dt_artifact.add_file("dt.joblib")
        wandb.log_artifact(dt_artifact)
        wandb.finish()

    if trained_rf_model != None:
        # store rf model
        rf = RandomForestClassifier(**trained_rf_model)
        refit_rf = rf.fit(scaled_resampled_X_train_val, scaled_resampled_y_train_val)

        wandb.init(project=wandb_project, group="rf", job_type="final")
        joblib.dump(refit_rf, "rf.joblib")
        rf_artifact = wandb.Artifact(
            "RF-Model",
            type="model",
            description="selected RF model",
            metadata={
                "parameters": trained_rf_model,
                "epoch": rf_epoch_id,
            },
        )

        rf_artifact.add_file("rf.joblib")
        wandb.log_artifact(rf_artifact)
        wandb.finish()

    if trained_xgb_model != None:
        # store xgb model
        xgboost = xgb.XGBClassifier(objective="binary:hinge", nthread=4, seed=random_state)
        xgboost = xgboost.set_params(**trained_xgb_model)
        refit_xgb = xgboost.fit(scaled_resampled_X_train_val, scaled_resampled_y_train_val)

        wandb.init(project=wandb_project, group="xgb", job_type="final")
        joblib.dump(refit_xgb, "xgb.joblib")
        xgb_artifact = wandb.Artifact(
            "XGB-Model",
            type="model",
            description="selected XGB model",
            metadata={
                "parameters": trained_xgb_model,
                "epoch": xgb_epoch_id,
            },
        )

        xgb_artifact.add_file("xgb.joblib")
        wandb.log_artifact(xgb_artifact)
        wandb.finish()

    if trained_lgb_model != None:
        # store lgb model
        lgb_model = lgb.LGBMClassifier(objective="binary", random_state=42)
        lgb_model = lgb_model.set_params(**trained_lgb_model)
        refit_lgb = lgb_model.fit(scaled_resampled_X_train_val, scaled_resampled_y_train_val)

        wandb.init(project=wandb_project, group="lgb", job_type="final")
        joblib.dump(refit_lgb, "lgb.joblib")
        lgb_artifact = wandb.Artifact(
            "LGB-Model",
            type="model",
            description="selected LGB model",
            metadata={
                "parameters": trained_lgb_model,
                "epoch": lgb_epoch_id,
            },
        )

        lgb_artifact.add_file("lgb.joblib")
        wandb.log_artifact(lgb_artifact)
        wandb.finish()

    return

def oversample_data(X: pd.Series, y: pd.Series):
    
    oversampler = sv.polynom_fit_SMOTE_poly()
    X_samp, y_samp = oversampler.sample(X, y)
    X_samp, y_samp = pd.DataFrame(X_samp), pd.Series(y_samp)

    # smotetomek = SMOTETomek(
    #     smote=SMOTE(sampling_strategy="all"),
    #     tomek=TomekLinks(
    #         sampling_strategy="majority",
    #     ),
    #     random_state=random_state,
    # )
    # X_samp, y_samp = smotetomek.fit_resample(X, y)
    return X_samp, y_samp


def fit_dummy_classifier(x_train_df, y_train_df, constant):
    dummy_clf = DummyClassifier(strategy="constant", constant=constant)
    dummy_clf.fit(x_train_df, y_train_df)
    return dummy_clf


@task(container_image="istiyaksiddiquee/flyte-for-kube:test17")
def fit_logistic_model(
    x_train_df: pd.Series, y_train_df: pd.Series, X_val: pd.Series, Y_val: pd.Series, inner_cv: RepeatedKFold, epoch_str: str
) -> CLFOutput:
    # Logistic Regression

    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"

    logging.info("FIT_LOGIT_MODEL: %s", f"logit run scheduled for {epoch_str}.")

    logit_result = None
    logit_best_grid_param = None
    logit_score = None

    try:
        logit_grid = {
            "penalty": ["l2"],
        }
        # logit_grid = {
        #     "penalty": ["l1", "l2", "elasticnet"],
        #     "dual": [True, False],
        #     "C": [_ for _ in range(1, 10, 1)],
        #     "fit_intercept": [True, False],
        #     "max_iter": [500],
        #     "solver": ["lbfgs", "newton-cg", "newton-cholesky", "sag", "saga"],
        #     "n_jobs": [-1],
        # }
        logit_model = LogisticRegression()

        clf = GridSearchCV(
            estimator=logit_model,
            cv=inner_cv,
            refit=optimization_metric,
            param_grid=logit_grid,
            scoring=scorers_for_gridcv,
            verbose=0,
            n_jobs=-1,
        )

        logit_result = clf.fit(x_train_df, y_train_df)

    except Exception as error:
        logging.error("Could not fit Logistic model")
        logging.error("An exception occurred:", error)

    if logit_result != None:
        logit_model = logit_result.best_estimator_
        logit_best_grid_param = logit_model.get_params()
        wandb.init(project=wandb_project, group="logit", job_type=epoch_str)

        logit_Y_pred = logit_model.predict(X_val)
        logit_Y_pred_proba = logit_model.predict_proba(X_val)
        logit_custom_score = get_all_scores(Y_val, logit_Y_pred, logit_Y_pred_proba[:, 1])
        wandb.log(convert_scores_to_dict(logit_custom_score))

        logit_cv_result_df = pd.DataFrame(logit_result.cv_results_)
        # logit_cv_result_table = wandb.Table(dataframe=logit_cv_result_df)
        logit_cv_result_artifact = wandb.Artifact("logit_cv_result_artifact_" + epoch_str, type="cv_result")
        # logit_cv_result_artifact.add(logit_cv_result_table, "logit_cv_result_table_" + epoch_str)
        logit_cv_file_name = f"./logit_cv_result_{epoch_str}.csv"
        logit_cv_result_df.to_csv(logit_cv_file_name)
        logit_cv_result_artifact.add_file(logit_cv_file_name)
        wandb.log_artifact(logit_cv_result_artifact)

        wandb.finish()
        logit_score = logit_custom_score.get_default_metric()

    logging.info("FIT_LOGIT_MODEL: %s", "logit run completed.")
    clf_output = CLFOutput(logit_best_grid_param, logit_score)

    return clf_output


@task(container_image="istiyaksiddiquee/flyte-for-kube:test17")
def fit_dt_model(x_train_df: pd.Series, y_train_df: pd.Series, X_val: pd.Series, Y_val: pd.Series, inner_cv: RepeatedKFold, epoch_str: str) -> CLFOutput:
    # Decision Tree

    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"

    logging.info("FIT_DT_MODEL: %s", f"dt run scheduled for {epoch_str}.")

    dt_result = None
    dt_score = None
    dt_best_grid_param = None
    try:

        # dt_grid = {
        #     "criterion": ["gini", "entropy", "log_loss"],
        #     "splitter": ["best", "random"],
        #     "max_depth": [_ for _ in range(1, 10, 1)],
        #     "min_samples_split": [_ for _ in range(1, 10, 1)],
        #     "min_samples_leaf": [_ for _ in range(1, 10, 1)],
        # }
        dt_grid = {"criterion": ["gini"]}
        dt_clf = DecisionTreeClassifier(random_state=random_state)

        clf = GridSearchCV(
            estimator=dt_clf,
            cv=inner_cv,
            refit=optimization_metric,
            param_grid=dt_grid,
            scoring=scorers_for_gridcv,
            verbose=0,
            n_jobs=-1,
        )

        dt_result = clf.fit(x_train_df, y_train_df)

    except Exception as error:
        logging.error("Could not fit Decision Tree model")
        logging.error("An exception occurred:", error)

    if dt_result != None:
        dt_model = dt_result.best_estimator_
        dt_best_grid_param = dt_model.get_params()
        wandb.init(project=wandb_project, group="dt", job_type=epoch_str)
        dt_Y_pred = dt_model.predict(X_val)
        dt_Y_pred_proba = dt_model.predict_proba(X_val)
        dt_custom_score = get_all_scores(Y_val, dt_Y_pred, dt_Y_pred_proba[:, 1])
        wandb.log(convert_scores_to_dict(dt_custom_score))

        dt_cv_result_df = pd.DataFrame(dt_result.cv_results_)
        # dt_cv_result_table = wandb.Table(dataframe=dt_cv_result_df)
        dt_cv_result_artifact = wandb.Artifact("dt_cv_result_artifact_" + epoch_str, type="cv_result")
        # dt_cv_result_artifact.add(dt_cv_result_table, "dt_cv_result_table_" + epoch_str)
        dt_cv_file_name = f"./dt_cv_result_{epoch_str}.csv"
        dt_cv_result_df.to_csv(dt_cv_file_name)
        dt_cv_result_artifact.add_file(dt_cv_file_name)
        wandb.log_artifact(dt_cv_result_artifact)

        wandb.finish()
        dt_score = dt_custom_score.get_default_metric()

    logging.info("FIT_DT_MODEL: %s", "dt run completed.")
    
    clf_output = CLFOutput(dt_best_grid_param, dt_score)

    return clf_output



@task(container_image="istiyaksiddiquee/flyte-for-kube:test17")
def fit_svc_model(x_train_df: pd.Series, y_train_df: pd.Series, X_val: pd.Series, Y_val: pd.Series, inner_cv: RepeatedKFold, epoch_str: str) -> CLFOutput:
    # SVC

    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"

    logging.info("FIT_SVC_MODEL: %s", f"svc run scheduled for {epoch_str}.")

    svc_result = None
    svc_score = None
    svc_best_grid_param = None

    try:
        # SVC
        svc_grid = {"C": [0.1]}
        # svc_grid = {
        #     "C": [0.1],
        #     "kernel": ["linear", "poly", "rbf", "sigmoid"],
        #     "degree": [_ for _ in range(1, 5, 1)],
        #     "gamma": ["scale", "auto"],
        #     "decision_function_shape": ["ovo", "ovr"],
        #     "shrinking": [True, False],
        #     "coef0": [0.0, 0.1, 0.01, 0.5, 1],
        # }
        svc_model = SVC()

        clf = GridSearchCV(
            estimator=svc_model,
            cv=inner_cv,
            refit=optimization_metric,
            param_grid=svc_grid,
            scoring=scorers_for_gridcv,
            verbose=0,
            n_jobs=-1,
        )

        svc_result = clf.fit(x_train_df, y_train_df)

    except Exception as error:
        logging.error("Could not fit SVC model")
        logging.error("An exception occurred:", error)

    if svc_result != None:
        svc_model = svc_result.best_estimator_
        svc_best_grid_param = svc_model.get_params()

        wandb.init(project=wandb_project, group="svc", job_type=epoch_str)
        svc_Y_pred = svc_model.predict(X_val)
        svc_Y_pred_proba = svc_model.predict_proba(X_val)
        svc_custom_score = get_all_scores(Y_val, svc_Y_pred, svc_Y_pred_proba[:, 1])
        wandb.log(convert_scores_to_dict(svc_custom_score))

        svc_cv_result_df = pd.DataFrame(svc_result.cv_results_)
        # svc_cv_result_table = wandb.Table(dataframe=svc_cv_result_df)
        svc_cv_result_artifact = wandb.Artifact("svc_cv_result_artifact_" + epoch_str, type="cv_result")
        # svc_cv_result_artifact.add(svc_cv_result_table, "svc_cv_result_table_" + epoch_str)
        svc_cv_file_name = f"./svc_cv_result_{epoch_str}.csv"
        svc_cv_result_df.to_csv(svc_cv_file_name)
        svc_cv_result_artifact.add_file(svc_cv_file_name)
        wandb.log_artifact(svc_cv_result_artifact)

        wandb.finish()

        svc_score = svc_custom_score.get_default_metric()

    logging.info("FIT_SVC_MODEL: %s", "svc run completed.")

    clf_output = CLFOutput(svc_best_grid_param, svc_score)

    return clf_output

@task(container_image="istiyaksiddiquee/flyte-for-kube:test17")
def fit_rf_model(x_train_df: pd.Series, y_train_df: pd.Series, X_val: pd.Series, Y_val: pd.Series, inner_cv: RepeatedKFold, epoch_str: str) -> CLFOutput:
    # Random Forest

    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"

    logging.info("FIT_RF_MODEL: %s", f"rf run scheduled for {epoch_str}.")

    rf_result = None
    rf_score = None
    rf_best_grid_param = None

    try:
        rf_grid = {"criterion": ["gini"]}

        # rf_grid = {
        #     "criterion": ["gini", "entropy", "log_loss"],
        #     "max_depth": [_ for _ in range(1, 10, 1)],
        #     "max_features": ["sqrt", "log2", None],
        #     "min_samples_leaf": [_ for _ in range(1, 10, 1)],
        # }
        rf_model = RandomForestClassifier()

        clf = GridSearchCV(
            estimator=rf_model,
            cv=inner_cv,
            refit=optimization_metric,
            param_grid=rf_grid,
            scoring=scorers_for_gridcv,
            verbose=0,
            n_jobs=-1,
        )

        rf_result = clf.fit(x_train_df, y_train_df)

    except Exception as error:
        logging.error("Could not fit Random Forest model")
        logging.error("An exception occurred:", error)

    if rf_result != None:
        rf_model = rf_result.best_estimator_
        rf_best_grid_param = rf_model.get_params()
        wandb.init(project=wandb_project, group="rf", job_type=epoch_str)
        rf_Y_pred = rf_model.predict(X_val)
        rf_Y_pred_proba = rf_model.predict_proba(X_val)
        rf_custom_score = get_all_scores(Y_val, rf_Y_pred, rf_Y_pred_proba[:, 1])
        wandb.log(convert_scores_to_dict(rf_custom_score))

        rf_cv_result_df = pd.DataFrame(rf_result.cv_results_)
        # rf_cv_result_table = wandb.Table(dataframe=rf_cv_result_df)
        rf_cv_result_artifact = wandb.Artifact("rf_cv_result_artifact_" + epoch_str, type="cv_result")
        # rf_cv_result_artifact.add(rf_cv_result_table, "rf_cv_result_table_" + epoch_str)
        rf_cv_file_name = f"./rf_cv_result_{epoch_str}.csv"
        rf_cv_result_df.to_csv(rf_cv_file_name)
        rf_cv_result_artifact.add_file(rf_cv_file_name)
        wandb.log_artifact(rf_cv_result_artifact)

        wandb.finish()

        rf_score = rf_custom_score.get_default_metric()

    logging.info("FIT_RF_MODEL: %s", "rf run completed.")

    clf_output = CLFOutput(rf_best_grid_param, rf_score)

    return clf_output


@task(container_image="istiyaksiddiquee/flyte-for-kube:test17")
def fit_xgb_model(x_train_df: pd.Series, y_train_df: pd.Series, X_val: pd.Series, Y_val: pd.Series, inner_cv: RepeatedKFold, epoch_str: str) -> CLFOutput:
    # XGB

    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"

    logging.info("FIT_XGB_MODEL: %s", f"xgb run scheduled for {epoch_str}.")

    xgb_result = None
    xgb_score = None
    xgb_best_grid_param = None

    try:
        # XGB
        xgb_grid = {
            "learning_rate": [0.1],
        }
        # xgb_grid = {
        #     # "n_estimators": range(60, 220, 40),
        #     "learning_rate": [0.1, 0.01, 0.05],
        #     "booster": ["gbtree", "gblinear", "dart"],
        # }

        xgb_model = xgb.XGBClassifier(objective="binary:hinge", nthread=4, seed=random_state)

        clf = GridSearchCV(
            estimator=xgb_model,
            cv=inner_cv,
            refit=optimization_metric,
            param_grid=xgb_grid,
            scoring=scorers_for_gridcv,
            verbose=0,
            n_jobs=-1,
        )

        xgb_result = clf.fit(x_train_df, y_train_df)

    except Exception as error:
        logging.error("Could not fit XGB model")
        logging.error("An exception occurred:", error)

    if xgb_result != None:
        try:
            xgb_model = xgb_result.best_estimator_
            xgb_best_grid_param = xgb_model.get_params()
            wandb.init(project=wandb_project, group="xgb", job_type=epoch_str)
            xgb_Y_pred = xgb_model.predict(X_val)
            xgb_Y_pred_proba = xgb_model.predict_proba(X_val)
            xgb_custom_score = get_all_scores(Y_val, xgb_Y_pred, xgb_Y_pred_proba[:, 1])
            wandb.log(convert_scores_to_dict(xgb_custom_score))

            xgb_cv_result_df = pd.DataFrame(xgb_result.cv_results_)
            # xgb_cv_result_table = wandb.Table(dataframe=xgb_cv_result_df)
            xgb_cv_result_artifact = wandb.Artifact("xgb_cv_result_artifact_" + epoch_str, type="cv_result")
            # xgb_cv_result_artifact.add(xgb_cv_result_table, "xgb_cv_result_table_" + epoch_str)
            xgb_cv_file_name = f"./xgb_cv_result_{epoch_str}.csv"
            xgb_cv_result_df.to_csv(xgb_cv_file_name)
            xgb_cv_result_artifact.add_file(xgb_cv_file_name)
            wandb.log_artifact(xgb_cv_result_artifact)

            wandb.finish()

            xgb_score = xgb_custom_score.get_default_metric()
        except Exception as e:
            logging.error("FIT_XGB_MODEL: %s", "Exception happened inside xgb result processor.")
            logging.error("FIT_XGB_MODEL: %s", e)

    logging.info("FIT_XGB_MODEL: %s", "xgb run completed.")
    clf_output = CLFOutput(xgb_best_grid_param, xgb_score)

    return clf_output


@task(container_image="istiyaksiddiquee/flyte-for-kube:test17")
def fit_lgb_model(x_train_df: pd.Series, y_train_df: pd.Series, X_val: pd.Series, Y_val: pd.Series, inner_cv: RepeatedKFold, epoch_str: str) -> CLFOutput:
    # LGB

    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
    
    logging.info("FIT_LGB_MODEL: %s", f"lgb run scheduled for {epoch_str}.")

    lgb_result = None
    lgb_score = None
    lgb_best_grid_param = None
    
    try:
        # LGB
        lgb_grid = {"num_leaves": [31]}

        # lgb_grid = {
        #     "learning_rate": [0.001, 0.005, 0.01],
        #     "n_estimators": [8, 16, 24],
        #     "num_leaves": [6, 8, 12],  # large num_leaves helps improve accuracy but might lead to over-fitting
        #     "boosting_type": ["gbdt", "dart"],  # for better accuracy -> try dart
        #     "subsample": [0.7, 0.75],
        #     "reg_alpha": [1, 1.2],
        #     "reg_lambda": [1, 1.2, 1.4],
        # }

        lgb_model = lgb.LGBMClassifier(objective="binary", random_state=42)

        clf = GridSearchCV(
            estimator=lgb_model,
            cv=inner_cv,
            refit=optimization_metric,
            param_grid=lgb_grid,
            scoring=scorers_for_gridcv,
            verbose=0,
            n_jobs=-1,
        )

        lgb_result = clf.fit(x_train_df, y_train_df)

    except Exception as error:
        logging.error("Could not fit XGB model")
        logging.error("An exception occurred:", error)

    if lgb_result != None:
        lgb_model = lgb_result.best_estimator_
        lgb_best_grid_param = lgb_model.get_params()
        wandb.init(project=wandb_project, group="lgb", job_type=epoch_str)

        lgb_Y_pred = lgb_model.predict(X_val)
        lgb_Y_pred_proba = lgb_model.predict_proba(X_val)
        lgb_custom_score = get_all_scores(Y_val, lgb_Y_pred, lgb_Y_pred_proba[:, 1])
        wandb.log(convert_scores_to_dict(lgb_custom_score))

        lgb_cv_result_df = pd.DataFrame(lgb_result.cv_results_)
        # lgb_cv_result_table = wandb.Table(dataframe=lgb_cv_result_df)
        lgb_cv_result_artifact = wandb.Artifact("lgb_cv_result_artifact_" + epoch_str, type="cv_result")
        # lgb_cv_result_artifact.add(lgb_cv_result_table, "lgb_cv_result_table_" + epoch_str)
        lgb_cv_file_name = f"./lgb_cv_result_{epoch_str}.csv"
        lgb_cv_result_df.to_csv(lgb_cv_file_name)
        lgb_cv_result_artifact.add_file(lgb_cv_file_name)
        wandb.log_artifact(lgb_cv_result_artifact)

        wandb.finish()
        lgb_score = lgb_custom_score.get_default_metric()

    logging.info("FIT_LGB_MODEL: %s", "lgb run completed.")
    clf_output = CLFOutput(lgb_best_grid_param, lgb_score)

    return clf_output


# def flip_true_false(y):
#     copy_y = copy(y)
#     flipped_y = [0 if item == 1 else 1 for item in copy_y]
#     return flipped_y


if __name__ == "__main__":
    nested_loop()
