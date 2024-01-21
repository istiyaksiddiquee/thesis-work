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

wandb_project = "test01"
random_state = 7
total_cv = 5
no_of_active_features = 15
optimization_metric = "average_precision_score"
data_folder = "segment" # alternatives: segment, shuttle, one_yeast, three_yeast

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


@task(container_image="istiyaksiddiquee/flyte-for-kube:test10")
def fit_logistic_model(x_train_val_df: pd.Series, y_train_val_df: pd.Series) -> None:
    # Logistic Regression

    logging.info("FIT_LOGIT_MODEL: %s", f"logit run scheduled.")

    logit_result = None

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
            cv=total_cv,
            refit=optimization_metric,
            param_grid=logit_grid,
            scoring=scorers_for_gridcv,
            verbose=0,
            n_jobs=-1,
        )

        logit_result = clf.fit(x_train_val_df, y_train_val_df)

    except Exception as error:
        logging.error("Could not fit Logistic model")
        logging.error("An exception occurred:", error)

    if logit_result != None:
        wandb.init(project=wandb_project, group="logit", job_type="final")
        logit_model = logit_result.best_estimator_
        joblib.dump(logit_model, "logit.joblib")
        logit_artifact = wandb.Artifact(
            "Logistic-Model",
            type="model",
            description="selected Logistic model",
            metadata={"parameters": logit_result.best_params_},
        )

        logit_artifact.add_file("logit.joblib")
        wandb.log_artifact(logit_artifact)
        wandb.finish()

    logging.info("FIT_LOGIT_MODEL: %s", "logit run completed.")
    return


@task(container_image="istiyaksiddiquee/flyte-for-kube:test10")
def fit_dt_model(x_train_val_df: pd.Series, y_train_val_df: pd.Series) -> None:
    # Decision Tree

    logging.info("FIT_DT_MODEL: %s", f"dt run scheduled.")

    dt_result = None

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
            cv=total_cv,
            refit=optimization_metric,
            param_grid=dt_grid,
            scoring=scorers_for_gridcv,
            verbose=0,
            n_jobs=-1,
        )

        dt_result = clf.fit(x_train_val_df, y_train_val_df)

    except Exception as error:
        logging.error("Could not fit Decision Tree model")
        logging.error("An exception occurred:", error)

    if dt_result != None:
        wandb.init(project=wandb_project, group="dt", job_type="final")
        dt_model = dt_result.best_estimator_
        joblib.dump(dt_model, "dt.joblib")
        dt_artifact = wandb.Artifact(
            "Decision Tree Model",
            type="model",
            description="selected DT model",
            metadata={"parameters": dt_result.best_params_},
        )

        dt_artifact.add_file("dt.joblib")
        wandb.log_artifact(dt_artifact)
        wandb.finish()

    logging.info("FIT_DT_MODEL: %s", "dt run completed.")
    return

@task(container_image="istiyaksiddiquee/flyte-for-kube:test10")
def fit_rf_model(x_train_val_df: pd.Series, y_train_val_df: pd.Series) -> None:
    # Random Forest

    logging.info("FIT_RF_MODEL: %s", f"rf run scheduled.")

    rf_result = None
    
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
            cv=total_cv,
            refit=optimization_metric,
            param_grid=rf_grid,
            scoring=scorers_for_gridcv,
            verbose=0,
            n_jobs=-1,
        )

        rf_result = clf.fit(x_train_val_df, y_train_val_df)

    except Exception as error:
        logging.error("Could not fit Random Forest model")
        logging.error("An exception occurred:", error)

    if rf_result != None:
        
        wandb.init(project=wandb_project, group="rf", job_type="final")
        
        rf_model = rf_result.best_estimator_
        joblib.dump(rf_model, "rf.joblib")
        rf_artifact = wandb.Artifact(
            "Random Forest Model",
            type="model",
            description="selected RF model",
            metadata={"parameters": rf_result.best_params_},
        )

        rf_artifact.add_file("rf.joblib")
        wandb.log_artifact(rf_artifact)
        wandb.finish()

    logging.info("FIT_RF_MODEL: %s", "rf run completed.")
    return


@task(container_image="istiyaksiddiquee/flyte-for-kube:test10")
def fit_xgb_model(x_train_val_df: pd.Series, y_train_val_df: pd.Series) -> None:
    # XGB

    logging.info("FIT_XGB_MODEL: %s", f"xgb run scheduled.")

    xgb_result = None
    
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
            cv=total_cv,
            refit=optimization_metric,
            param_grid=xgb_grid,
            scoring=scorers_for_gridcv,
            verbose=0,
            n_jobs=-1,
        )

        xgb_result = clf.fit(x_train_val_df, y_train_val_df)

    except Exception as error:
        logging.error("Could not fit XGB model")
        logging.error("An exception occurred:", error)

    if xgb_result != None:
        wandb.init(project=wandb_project, group="xgb", job_type="final")
        
        xgb_model = xgb_result.best_estimator_
        joblib.dump(xgb_model, "xgb.joblib")
        xgb_artifact = wandb.Artifact(
            "XGB Model",
            type="model",
            description="selected XGB model",
            metadata={"parameters": xgb_result.best_params_},
        )

        xgb_artifact.add_file("xgb.joblib")
        wandb.log_artifact(xgb_artifact)
        wandb.finish()

    logging.info("FIT_XGB_MODEL: %s", "xgb run completed.")
    return


@task(container_image="istiyaksiddiquee/flyte-for-kube:test10")
def fit_lgb_model(x_train_val_df: pd.Series, y_train_val_df: pd.Series) -> None:
    # LGB

    logging.info("FIT_LGB_MODEL: %s", f"lgb run scheduled.")
    lgb_result = None
    lgb_score = None
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
            cv=total_cv,
            refit=optimization_metric,
            param_grid=lgb_grid,
            scoring=scorers_for_gridcv,
            verbose=0,
            n_jobs=-1,
        )

        lgb_result = clf.fit(x_train_val_df, y_train_val_df)

    except Exception as error:
        logging.error("Could not fit LGB model")
        logging.error("An exception occurred:", error)

    if lgb_result != None:
        
        wandb.init(project=wandb_project, group="lgb", job_type="final")
            
        lgb_model = lgb_result.best_estimator_
        joblib.dump(lgb_model, "lgb.joblib")
        lgb_artifact = wandb.Artifact(
            "LGB Model",
            type="model",
            description="selected LGB model",
            metadata={"parameters": lgb_result.best_params_},
        )

        lgb_artifact.add_file("lgb.joblib")
        wandb.log_artifact(lgb_artifact)
        wandb.finish()

    logging.info("FIT_LGB_MODEL: %s", "lgb run completed.")
    return


@workflow()
def additional_workflow():
    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"

    # FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.DEBUG)

    try:
        # wandb.init(project=wandb_project)
        # wandb.alert(title="Started", text="Your run has started. Mark the time.")
        # wandb.finish()

        logging.info("ADDITIONAL_WORKFLOW: %s", "initiating processing, reading files")
        csv_path = os.path.join('.', data_folder)
        X_train_val, X_test, y_train_val, y_test = read_pickled_input_files(csv_path)

        # call the nested loop to get all the trained models
        logging.info("ADDITIONAL_WORKFLOW: %s", f"shapes of input: {X_train_val.shape}, {X_test.shape}, {y_train_val.shape}, {y_test.shape}")

        logging.info("NESTED_LOOP: %s", "feature scaling")
        normalized_df = copy(X_train_val)
        cd_first_quantile = np.quantile(normalized_df["characteristic_distance"], 0.25)
        cd_third_quantile = np.quantile(normalized_df["characteristic_distance"], 0.75)
        normalized_df["depth"] = np.log(normalized_df["depth"])
        normalized_df["max_breadth"] = np.log(normalized_df["max_breadth"])
        # normalized_df["size"] = np.log(normalized_df["size"])
        # normalized_df["strongly_cc"] = np.log(normalized_df["strongly_cc"])
        normalized_df["characteristic_distance"] = np.log(normalized_df["characteristic_distance"] + cd_first_quantile**2 / cd_third_quantile)

        scaler = StandardScaler().set_output(transform="pandas")
        scaled_X_train = scaler.fit_transform(normalized_df)
        scaled_resampled_X_train, scaled_resampled_y_train = oversample_data(scaled_X_train.to_numpy(), y_train_val.to_numpy())

        logging.info("ADDITIONAL_WORKFLOW: %s", "entering model fitting task")
        fit_logistic_model(x_train_val_df=scaled_resampled_X_train, y_train_val_df=scaled_resampled_y_train)
        fit_dt_model(x_train_val_df=scaled_resampled_X_train, y_train_val_df=scaled_resampled_y_train)
        fit_rf_model(x_train_val_df=scaled_resampled_X_train, y_train_val_df=scaled_resampled_y_train)
        fit_xgb_model(x_train_val_df=scaled_resampled_X_train, y_train_val_df=scaled_resampled_y_train)
        fit_lgb_model(x_train_val_df=scaled_resampled_X_train, y_train_val_df=scaled_resampled_y_train)
        
        logging.info("ADDITIONAL_WORKFLOW: %s", "model fitting complete.")
        
    except Exception as error:
        logging.info("ADDITIONAL_WORKFLOW: %s", "ERROR: some error happened, could not finish.")
        logging.info("ADDITIONAL_WORKFLOW: %s", error)

    return
