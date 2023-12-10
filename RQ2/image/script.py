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
from typing import Dict, Any
import pickle
import logging


random_state = 7
no_of_active_features = 15

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
        print("Enter filepath")
        return

    false_feature_file = "rq2_vosoughi_False_features.csv"
    true_feature_file = "rq2_vosoughi_True_features.csv"
    true_file = os.path.join(file_path + os.path.sep + true_feature_file)
    false_file = os.path.join(file_path + os.path.sep + false_feature_file)

    vosoughi_false_df = pd.read_csv(false_file, index_col=0, sep=",", encoding=get_encoding(false_file))
    vosoughi_true_df = pd.read_csv(true_file, index_col=0, sep=",", encoding=get_encoding(true_file))
    vosoughi_df = pd.concat([vosoughi_false_df, vosoughi_true_df])
    return vosoughi_df


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


def nested_loop(X_train_val: pd.DataFrame, y_train_val: pd.Series) -> None:
    start = time()

    outer_cv = RepeatedKFold(n_splits=2, n_repeats=1)

    logit_epoch_id = -1
    dt_epoch_id = -1
    rf_epoch_id = -1
    svc_epoch_id = -1
    xgb_epoch_id = -1
    lgb_epoch_id = -1

    logit_avg_prec = 0
    dt_avg_prec = 0
    rf_avg_prec = 0
    svc_avg_prec = 0
    xgb_avg_prec = 0
    lgb_avg_prec = 0

    trained_dt_model = None
    trained_rf_model = None
    trained_svc_model = None
    trained_logit_model = None
    trained_xgb_model = None
    trained_lgb_model = None

    loop_index = 0

    logging.info("NESTED_LOOP: %s", "loop starts")

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
        normalized_df["characteristic_distance"] = np.log(
            normalized_df["characteristic_distance"] + cd_first_quantile**2 / cd_third_quantile
        )

        scaler = StandardScaler().set_output(transform="pandas")
        scaled_X_train = scaler.fit_transform(normalized_df)

        smotetomek = get_data_pipeline_with_smotetomek()
        scaled_resampled_X_train, scaled_resampled_y_train = smotetomek.fit_resample(scaled_X_train, y_train)

        inner_cv = RepeatedKFold(n_splits=5, n_repeats=3)

        logging.info("NESTED_LOOP: %s", "entering model fitting")
        dt_result, svc_result, rf_result, logit_result, xgb_result, lgb_result = fit_multiple_models(
            x_train_df=scaled_resampled_X_train, y_train_df=scaled_resampled_y_train, inner_cv=inner_cv
        )

        if logit_result != None:
            logit_model = logit_result.best_estimator_
            wandb.init(project="thesis", group="logit", job_type=epoch_str)

            logit_Y_pred = logit_model.predict(X_val)
            logit_Y_pred_proba = logit_model.predict_proba(X_val)
            logit_custom_score = get_all_scores(Y_val, logit_Y_pred, logit_Y_pred_proba[:, 1])
            wandb.log(convert_scores_to_dict(logit_custom_score))

            logit_cv_result_df = pd.DataFrame(logit_result.cv_results_)
            logit_cv_result_table = wandb.Table(dataframe=logit_cv_result_df)
            logit_cv_result_artifact = wandb.Artifact("logit_cv_result_artifact_" + epoch_str, type="cv_result")
            logit_cv_result_artifact.add(logit_cv_result_table, "logit_cv_result_table_" + epoch_str)
            logit_cv_file_name = f"./logit_cv_result_{epoch_str}.csv"
            logit_cv_result_df.to_csv(logit_cv_file_name)
            logit_cv_result_artifact.add_file(logit_cv_file_name)
            wandb.log_artifact(logit_cv_result_artifact)

            wandb.finish()

            if logit_avg_prec < logit_custom_score.avg_precision:
                logit_avg_prec = logit_custom_score.avg_precision
                trained_logit_model = logit_model
                logit_epoch_id = loop_index

        if dt_result != None:
            dt_model = dt_result.best_estimator_

            wandb.init(project="thesis", group="dt", job_type="epoch_" + str(loop_index))
            dt_Y_pred = dt_model.predict(X_val)
            dt_Y_pred_proba = dt_model.predict_proba(X_val)
            dt_custom_score = get_all_scores(Y_val, dt_Y_pred, dt_Y_pred_proba[:, 1])
            wandb.log(convert_scores_to_dict(dt_custom_score))

            dt_cv_result_df = pd.DataFrame(dt_result.cv_results_)
            dt_cv_result_table = wandb.Table(dataframe=dt_cv_result_df)
            dt_cv_result_artifact = wandb.Artifact("dt_cv_result_artifact_" + epoch_str, type="cv_result")
            dt_cv_result_artifact.add(dt_cv_result_table, "dt_cv_result_table_" + epoch_str)
            dt_cv_file_name = f"./dt_cv_result_{epoch_str}.csv"
            dt_cv_result_df.to_csv(dt_cv_file_name)
            dt_cv_result_artifact.add_file(dt_cv_file_name)
            wandb.log_artifact(dt_cv_result_artifact)

            wandb.finish()

            if dt_avg_prec < dt_custom_score.avg_precision:
                dt_avg_prec = dt_custom_score.avg_precision
                trained_dt_model = dt_model
                dt_epoch_id = loop_index

        if svc_result != None:
            svc_model = svc_result.best_estimator_

            wandb.init(project="thesis", group="svc", job_type="epoch_" + str(loop_index))
            svc_Y_pred = svc_model.predict(X_val)
            svc_Y_pred_proba = svc_model.predict_proba(X_val)
            svc_custom_score = get_all_scores(Y_val, svc_Y_pred, svc_Y_pred_proba[:, 1])
            wandb.log(convert_scores_to_dict(svc_custom_score))

            svc_cv_result_df = pd.DataFrame(svc_result.cv_results_)
            svc_cv_result_table = wandb.Table(dataframe=svc_cv_result_df)
            svc_cv_result_artifact = wandb.Artifact("svc_cv_result_artifact_" + epoch_str, type="cv_result")
            svc_cv_result_artifact.add(svc_cv_result_table, "svc_cv_result_table_" + epoch_str)
            svc_cv_file_name = f"./svc_cv_result_{epoch_str}.csv"
            svc_cv_result_df.to_csv(svc_cv_file_name)
            svc_cv_result_artifact.add_file(svc_cv_file_name)
            wandb.log_artifact(svc_cv_result_artifact)

            wandb.finish()

            if svc_avg_prec < svc_custom_score.avg_precision:
                svc_avg_prec = svc_custom_score.avg_precision
                trained_svc_model = svc_model
                svc_epoch_id = loop_index

        if rf_result != None:
            rf_model = rf_result.best_estimator_

            wandb.init(project="thesis", group="rf", job_type="epoch_" + str(loop_index))
            rf_Y_pred = rf_model.predict(X_val)
            rf_Y_pred_proba = rf_model.predict_proba(X_val)
            rf_custom_score = get_all_scores(Y_val, rf_Y_pred, rf_Y_pred_proba[:, 1])
            wandb.log(convert_scores_to_dict(rf_custom_score))

            rf_cv_result_df = pd.DataFrame(rf_result.cv_results_)
            rf_cv_result_table = wandb.Table(dataframe=rf_cv_result_df)
            rf_cv_result_artifact = wandb.Artifact("rf_cv_result_artifact_" + epoch_str, type="cv_result")
            rf_cv_result_artifact.add(rf_cv_result_table, "rf_cv_result_table_" + epoch_str)
            rf_cv_file_name = f"./rf_cv_result_{epoch_str}.csv"
            rf_cv_result_df.to_csv(rf_cv_file_name)
            rf_cv_result_artifact.add_file(rf_cv_file_name)
            wandb.log_artifact(rf_cv_result_artifact)

            wandb.finish()

            if rf_avg_prec < rf_custom_score.avg_precision:
                rf_avg_prec = rf_custom_score.avg_precision
                trained_rf_model = rf_model
                rf_epoch_id = loop_index

        if xgb_result != None:
            xgb_model = xgb_result.best_estimator_

            wandb.init(project="thesis", group="xgb", job_type="epoch_" + str(loop_index))
            xgb_Y_pred = xgb_model.predict(X_val)
            xgb_Y_pred_proba = xgb_model.predict_proba(X_val)
            xgb_custom_score = get_all_scores(Y_val, xgb_Y_pred, xgb_Y_pred_proba[:, 1])
            wandb.log(convert_scores_to_dict(xgb_custom_score))

            xgb_cv_result_df = pd.DataFrame(xgb_result.cv_results_)
            xgb_cv_result_table = wandb.Table(dataframe=xgb_cv_result_df)
            xgb_cv_result_artifact = wandb.Artifact("xgb_cv_result_artifact_" + epoch_str, type="cv_result")
            xgb_cv_result_artifact.add(xgb_cv_result_table, "xgb_cv_result_table_" + epoch_str)
            xgb_cv_file_name = f"./xgb_cv_result_{epoch_str}.csv"
            xgb_cv_result_df.to_csv(xgb_cv_file_name)
            xgb_cv_result_artifact.add_file(xgb_cv_file_name)
            wandb.log_artifact(xgb_cv_result_artifact)

            wandb.finish()

            if xgb_avg_prec < xgb_custom_score.avg_precision:
                xgb_avg_prec = xgb_custom_score.avg_precision
                trained_xgb_model = xgb_model
                xgb_epoch_id = loop_index

        if lgb_result != None:
            lgb_model = lgb_result.best_estimator_
            wandb.init(project="thesis", group="lgb", job_type=epoch_str)

            lgb_Y_pred = lgb_model.predict(X_val)
            lgb_Y_pred_proba = lgb_model.predict_proba(X_val)
            lgb_custom_score = get_all_scores(Y_val, lgb_Y_pred, lgb_Y_pred_proba[:, 1])
            wandb.log(convert_scores_to_dict(lgb_custom_score))

            lgb_cv_result_df = pd.DataFrame(lgb_result.cv_results_)
            lgb_cv_result_table = wandb.Table(dataframe=lgb_cv_result_df)
            lgb_cv_result_artifact = wandb.Artifact("lgb_cv_result_artifact_" + epoch_str, type="cv_result")
            lgb_cv_result_artifact.add(lgb_cv_result_table, "lgb_cv_result_table_" + epoch_str)
            lgb_cv_file_name = f"./lgb_cv_result_{epoch_str}.csv"
            lgb_cv_result_df.to_csv(lgb_cv_file_name)
            lgb_cv_result_artifact.add_file(lgb_cv_file_name)
            wandb.log_artifact(lgb_cv_result_artifact)

            wandb.finish()

            if lgb_avg_prec < lgb_custom_score.avg_precision:
                lgb_avg_prec = lgb_custom_score.avg_precision
                trained_lgb_model = lgb_model
                lgb_epoch_id = loop_index

    normalized_df = copy(X_train_val)
    cd_first_quantile = np.quantile(normalized_df["size"], 0.25)
    cd_third_quantile = np.quantile(normalized_df["size"], 0.75)
    normalized_df["depth"] = np.log(normalized_df["depth"])
    normalized_df["max_breadth"] = np.log(normalized_df["max_breadth"])
    # normalized_df["size"] = np.log(normalized_df["size"])
    # normalized_df["strongly_cc"] = np.log(normalized_df["strongly_cc"])
    normalized_df["characteristic_distance"] = np.log(
        normalized_df["characteristic_distance"] + cd_first_quantile**2 / cd_third_quantile
    )

    scaler = StandardScaler().set_output(transform="pandas")
    scaled_X_train_val = scaler.fit_transform(normalized_df)
    smotetomek = get_data_pipeline_with_smotetomek()
    (
        scaled_resampled_X_train_val,
        scaled_resampled_y_train_val,
    ) = smotetomek.fit_resample(scaled_X_train_val, y_train_val)

    # dummy_false = fit_dummy_classifier(scaled_resampled_X_train_val, scaled_resampled_y_train_val, 0)

    if trained_logit_model != None:
        # store logit model
        logistic = LogisticRegression(**trained_logit_model.get_params())
        refit_logit = logistic.fit(scaled_resampled_X_train_val, scaled_resampled_y_train_val)

        wandb.init(project="thesis", group="logit", job_type="final")
        joblib.dump(refit_logit, "logit.joblib")
        logit_artifact = wandb.Artifact(
            "Logistic-Model",
            type="model",
            description="selected Logistic model",
            metadata={
                "parameters": trained_logit_model.get_params(),
                "epoch": logit_epoch_id,
            },
        )

        logit_artifact.add_file("logit.joblib")
        wandb.log_artifact(logit_artifact)
        wandb.finish()

    if trained_dt_model != None:
        # store dt model
        dt = DecisionTreeClassifier(**trained_dt_model.get_params())
        refit_dt = dt.fit(scaled_resampled_X_train_val, scaled_resampled_y_train_val)

        wandb.init(project="thesis", group="dt", job_type="final")
        joblib.dump(refit_dt, "dt.joblib")
        dt_artifact = wandb.Artifact(
            "DT-Model",
            type="model",
            description="selected DT model",
            metadata={
                "parameters": trained_dt_model.get_params(),
                "epoch": dt_epoch_id,
            },
        )

        dt_artifact.add_file("dt.joblib")
        wandb.log_artifact(dt_artifact)
        wandb.finish()

    if trained_rf_model != None:
        # store rf model
        rf = RandomForestClassifier(**trained_rf_model.get_params())
        refit_rf = rf.fit(scaled_resampled_X_train_val, scaled_resampled_y_train_val)

        wandb.init(project="thesis", group="rf", job_type="final")
        joblib.dump(refit_rf, "rf.joblib")
        rf_artifact = wandb.Artifact(
            "RF-Model",
            type="model",
            description="selected RF model",
            metadata={
                "parameters": trained_rf_model.get_params(),
                "epoch": rf_epoch_id,
            },
        )

        rf_artifact.add_file("rf.joblib")
        wandb.log_artifact(rf_artifact)
        wandb.finish()

    if trained_svc_model != None:
        # store svc model
        svm = LogisticRegression(**trained_svc_model.get_params())
        refit_svm = svm.fit(scaled_resampled_X_train_val, scaled_resampled_y_train_val)

        wandb.init(project="thesis", group="svc", job_type="final")
        joblib.dump(refit_svm, "svc.joblib")
        svc_artifact = wandb.Artifact(
            "SVC-Model",
            type="model",
            description="selected SVC model",
            metadata={
                "parameters": trained_svc_model.get_params(),
                "epoch": svc_epoch_id,
            },
        )

        svc_artifact.add_file("svc.joblib")
        wandb.log_artifact(svc_artifact)
        wandb.finish()

    if trained_xgb_model != None:
        # store xgb model
        xgboost = xgb.XGBClassifier(objective="binary:hinge", nthread=4, seed=random_state)
        xgboost = xgboost.set_params(**trained_xgb_model.get_xgb_params())
        refit_xgb = xgboost.fit(scaled_resampled_X_train_val, scaled_resampled_y_train_val)

        wandb.init(project="thesis", group="xgb", job_type="final")
        joblib.dump(refit_xgb, "xgb.joblib")
        xgb_artifact = wandb.Artifact(
            "XGB-Model",
            type="model",
            description="selected XGB model",
            metadata={
                "parameters": trained_xgb_model.get_params(),
                "epoch": xgb_epoch_id,
            },
        )

        xgb_artifact.add_file("xgb.joblib")
        wandb.log_artifact(xgb_artifact)
        wandb.finish()

    if trained_lgb_model != None:
        # store lgb model
        lgb_model = lgb.LGBMClassifier(objective="binary", random_state=42)
        lgb_model = lgb_model.set_params(**trained_lgb_model.get_params())
        refit_lgb = lgb_model.fit(scaled_resampled_X_train_val, scaled_resampled_y_train_val)

        wandb.init(project="thesis", group="lgb", job_type="final")
        joblib.dump(refit_lgb, "lgb.joblib")
        lgb_artifact = wandb.Artifact(
            "LGB-Model",
            type="model",
            description="selected LGB model",
            metadata={
                "parameters": trained_lgb_model.get_params(),
                "epoch": lgb_epoch_id,
            },
        )

        lgb_artifact.add_file("lgb.joblib")
        wandb.log_artifact(lgb_artifact)
        wandb.finish()

    # joblib.dump(refit_dt, "decision_tree")
    # joblib.dump(refit_rf, "random_forest")
    # joblib.dump(refit_logit, "logistic")
    # joblib.dump(refit_xgb, "xgboost")

    end = time()
    print("took : " + str(end - start))
    return None


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


def generic_grid_search(
    model: Any,
    x_train_df: pd.Series,
    y_train_df: pd.Series,
    inner_cv: RepeatedKFold,
    grid_param: Dict,
    model_name: str,
):
    result = None
    # try:
    #     clf = GridSearchCV(
    #         estimator=model,
    #         cv=inner_cv,
    #         refit="average_precision_score",
    #         param_grid=grid_param,
    #         scoring=scorers_for_gridcv,
    #         # verbose=1,
    #         n_jobs=-1,
    #     )

    #     result = clf.fit(x_train_df, y_train_df)
    # except:
    #     print(f"Could not fit {model_name} model")

    logging.info("Generic_Grid_Search: %s", f"fitting {model_name}")
    
    clf = GridSearchCV(
        estimator=model,
        cv=inner_cv,
        refit="average_precision_score",
        param_grid=grid_param,
        scoring=scorers_for_gridcv,
        # verbose=1,
        n_jobs=-1,
    )

    result = clf.fit(x_train_df, y_train_df)

    return result


def fit_dummy_classifier(x_train_df, y_train_df, constant):
    dummy_clf = DummyClassifier(strategy="constant", constant=constant)
    dummy_clf.fit(x_train_df, y_train_df)
    return dummy_clf


def fit_multiple_models(x_train_df: pd.Series, y_train_df: pd.Series, inner_cv: RepeatedKFold):
    # Logistic Regression
    # logit_grid = {
    #     "penalty": ["l1", "l2", "elasticnet", None],
    #     "dual": [True, False],
    #     "C": [_ for _ in range(1, 10, 1)],
    #     "fit_intercept": [True, False],
    #     "solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
    #     "n_jobs": [-1],
    # }

    logit_grid = {
        "penalty": ["l2"],
    }

    dt_grid = {"criterion": ["gini"]}

    svc_grid = {"C": [0.1]}

    rf_grid = {"criterion": ["gini"]}

    xgb_grid = {
        "colsample_bytree": [0.7],
    }

    logit_model = LogisticRegression()
    logit_result = generic_grid_search(
        logit_model, x_train_df, y_train_df, inner_cv, logit_grid, "Logistic Regression"
    )

    # # Decision Tree
    # dt_grid = {
    #     "criterion": ["gini", "entropy", "log_loss"],
    #     "splitter": ["best", "random"],
    #     "max_depth": [4, 5, 6, 7],
    #     "min_samples_split": [2, 5, 10],
    #     "min_samples_leaf": [4, 5, 6, 7],
    # }

    dt_clf = DecisionTreeClassifier(random_state=random_state)
    dt_result = generic_grid_search(dt_clf, x_train_df, y_train_df, inner_cv, dt_grid, "Decision Tree")

    # # SVC
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
    svc_result = generic_grid_search(
        svc_model, x_train_df, y_train_df, inner_cv, svc_grid, "Support Vector Machine"
    )

    # # Random Forest
    # rf_grid = {
    #     "criterion": ["gini", "entropy", "log_loss"],
    #     "max_depth": [_ for _ in range(1, 10, 1)],
    #     "max_features": ["sqrt", "log2", None],
    #     "min_samples_split": [_ for _ in range(1, 10, 1)],
    #     "min_samples_leaf": [_ for _ in range(1, 10, 1)],
    # }

    rf_model = RandomForestClassifier()
    rf_result = generic_grid_search(
        rf_model, x_train_df, y_train_df, inner_cv, rf_grid, "Random Forest"
    )

    # # XGBoost
    # xgb_grid = {
    #     "max_depth": range(2, 10, 1),
    #     "n_estimators": range(60, 220, 40),
    #     "learning_rate": [0.1, 0.01, 0.05],
    #     "booster": ["gbtree", "gblinear", "dart"],
    #     "max_depth": [6],
    #     "min_child_weight": [11],
    #     "subsample": [0.8],
    #     "colsample_bytree": [0.7],
    # }

    xgb_model = xgb.XGBClassifier(objective="binary:hinge", nthread=4, seed=random_state)
    xgb_result = generic_grid_search(xgb_model, x_train_df, y_train_df, inner_cv, xgb_grid, "XGBoost")

    lgb_grid = {"num_leaves": [31]}
    lgb_model = lgb.LGBMClassifier(objective="binary", random_state=42)
    lgb_result = generic_grid_search(lgb_model, x_train_df, y_train_df, inner_cv, lgb_grid, "LightGBM")

    logging.info("Fit_Multiple_Models: %s", "one run complete, returning to nested loop")
    
    return (dt_result, svc_result, rf_result, logit_result, xgb_result, lgb_result)


# def flip_true_false(y):
#     copy_y = copy(y)
#     flipped_y = [0 if item == 1 else 1 for item in copy_y]
#     return flipped_y


def work():
    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"

    # FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.DEBUG)

    # csv_path = '/root'
    csv_path = "."

    logging.info("WORK: %s", "initiating processing, reading files")
    df = read_files(csv_path)

    logging.info("WORK: %s", "filtering and splitting data")
    X_train_val, X_test, y_train_val, y_test = filter_and_split_df(df)

    logging.info("WORK: %s", "storing splits")
    with open("./x_train_val.pickle", "wb") as file:
        pickle.dump(X_train_val, file)

    with open("./y_train_val.pickle", "wb") as file:
        pickle.dump(y_train_val, file)

    with open("./x_test.pickle", "wb") as file:
        pickle.dump(X_test, file)

    with open("./y_test.pickle", "wb") as file:
        pickle.dump(y_test, file)

    # call the nested loop to get all the trained models
    print(X_train_val.shape, X_test.shape, y_train_val.shape, y_test.shape)

    logging.info("WORK: %s", "entering nested loop")
    nested_loop(X_train_val=X_train_val, y_train_val=y_train_val)


if __name__ == "__main__":
    work()
