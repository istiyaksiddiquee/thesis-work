import os
import argparse
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
)
import xgboost as xgb
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from imblearn.combine import SMOTETomek
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold, GridSearchCV, train_test_split
from sklearn.dummy import DummyClassifier
import mlflow
from copy import copy
import numpy as np
import pickle
from flytekit import task, workflow, dynamic, ImageSpec, Resources
from flytekit.remote import FlyteRemote
from flytekit.configuration import Config, PlatformConfig


random_state = 7
no_of_active_features = 15

custom_image = ImageSpec(
    python_version="3.10",
    packages=["scikit-learn", "pandas", "numpy", "chardet", "xgboost", "imbalanced-learn"],
    base_image="python:3.10-slim-buster"
)


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


scorers_for_gridcv = {
    "accuracy_score": make_scorer(accuracy_score),
    "precision_score": make_scorer(precision_score),
    "recall_score": make_scorer(recall_score),
    "fbeta_score": make_scorer(fbeta_score, beta=0.5),
    "balanced_accuracy_score": make_scorer(balanced_accuracy_score),
    "average_precision_score": make_scorer(average_precision_score),
}


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

    vosoughi_false_df = pd.read_csv(
        false_file, index_col=0, sep=",", encoding=get_encoding(false_file)
    )
    vosoughi_true_df = pd.read_csv(
        true_file, index_col=0, sep=",", encoding=get_encoding(true_file)
    )
    vosoughi_df = pd.concat([vosoughi_false_df, vosoughi_true_df])
    return vosoughi_df


def calculate_tpr_fpr(y_real, y_pred):
    """
    Calculates the True Positive Rate (tpr) and the True Negative Rate (fpr) based on real and predicted observations

    Args:
        y_real: The list or series with the real classes
        y_pred: The list or series with the predicted classes

    Returns:
        tpr: The True Positive Rate of the classifier
        fpr: The False Positive Rate of the classifier
    """

    # Calculates the confusion matrix and recover each element
    cm = confusion_matrix(y_real, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    # Calculates tpr and fpr
    tpr = TP / (TP + FN)  # sensitivity - true positive rate
    fpr = 1 - TN / (TN + FP)  # 1-specificity - false positive rate

    prec = TP/(TP+FP)
    rec = TP/(TP+FN)

    return tpr, fpr, prec, rec


def get_all_roc_coordinates(y_real, y_proba):
    """
    Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a threshold for the predicion of the class.

    Args:
        y_real: The list or series with the real classes.
        y_proba: The array with the probabilities for each class, obtained by using the `.predict_proba()` method.

    Returns:
        tpr_list: The list of TPRs representing each threshold.
        fpr_list: The list of FPRs representing each threshold.
    """
    tpr_list = [0]
    fpr_list = [0]
    for i in range(len(y_proba)):
        threshold = y_proba[i]
        y_pred = y_proba >= threshold
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list


def filter_and_split_df(df: pd.DataFrame):
    test_size = 0.20

    filtered_df = df[(df.characteristic_distance != -99999) & (df.depth != 0)]
    filtered_df = filtered_df.drop(columns=["avg_cluster_coef"], axis=1)
    filtered_df = filtered_df.drop(["weakly_cc"], axis=1)
    total_features = no_of_active_features - 2

    X = filtered_df.iloc[:, 3:total_features]
    Y = filtered_df.iloc[:, total_features]
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    return X_train_val, X_test, y_train_val, y_test

@task(container_image="istiyaksiddiquee/flyte-base-image:1.0.0")
def nested_loop(X_train_val: pd.DataFrame, y_train_val: pd.Series):

    outer_cv = RepeatedKFold(n_splits=5, n_repeats=1)

    dt_avg_prec = 0
    rf_avg_prec = 0
    svc_avg_prec = 0
    logit_avg_prec = 0
    xgb_avg_prec = 0

    trained_dt_model = None
    trained_rf_model = None
    trained_svc_model = None
    trained_logit_model = None
    trained_xgb_model = None

    for train_index, val_index in outer_cv.split(X_train_val.to_numpy()):
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

        smotetomek = get_data_pipeline_with_smotetomek()  
        scaled_resampled_X_train, scaled_resampled_y_train = smotetomek.fit_resample(scaled_X_train, y_train)

        inner_cv = RepeatedKFold(n_splits=5, n_repeats=3)
        dt_model, svc_model, rf_model, logit_model, xgb_model = fit_multiple_models(
            scaled_resampled_X_train, scaled_resampled_y_train, inner_cv
        )

        logit_Y_pred = logit_model.best_estimator_.predict(X_val)
        logit_Y_pred_proba = logit_model.best_estimator_.predict_proba(X_val)
        logit_custom_score = get_all_scores(Y_val, logit_Y_pred, logit_Y_pred_proba[:, 1])

        # dt_Y_pred = dt_model.predict(X_val)
        # dt_Y_pred_proba = dt_model.predict_proba(X_val)
        # dt_custom_score = get_all_scores(Y_val, dt_Y_pred, dt_Y_pred_proba)

        # svc_Y_pred = svc_model.predict(X_val)
        # svc_Y_pred_proba = svc_model.predict_proba(X_val)
        # svc_custom_score = get_all_scores(Y_val, svc_Y_pred, svc_Y_pred_proba)

        # rf_Y_pred = rf_model.predict(X_val)
        # rf_Y_pred_proba = rf_model.predict_proba(X_val)
        # rf_custom_score = get_all_scores(Y_val, rf_Y_pred, rf_Y_pred_proba)

        # xgb_Y_pred = xgb_model.predict(X_val)
        # xgb_Y_pred_proba = xgb_model.predict_proba(X_val)
        # xgb_custom_score = get_all_scores(Y_val, xgb_Y_pred, xgb_Y_pred_proba)

        # if dt_avg_prec < dt_custom_score.avg_precision:
        #     dt_avg_prec = dt_custom_score.avg_precision
        #     trained_dt_model = dt_model

        # if rf_avg_prec < rf_custom_score.avg_precision:
        #     rf_avg_prec = rf_custom_score.avg_precision
        #     trained_rf_model = rf_model

        # if svc_avg_prec < svc_custom_score.avg_precision:
        #     svc_avg_prec = svc_custom_score.avg_precision
        #     trained_svc_model = svc_model

        # if xgb_avg_prec < xgb_custom_score.avg_precision:
        #     xgb_avg_prec = xgb_custom_score.avg_precision
        #     trained_xgb_model = xgb_model

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
    scaled_resampled_X_train_val, scaled_resampled_y_train_val = smotetomek.fit_resample(scaled_X_train_val, y_train_val)

    # dt = DecisionTreeClassifier(
    #     **trained_dt_model.best_params_, random_state=random_state
    # )
    # refit_dt = dt.fit(scaled_X_train_val, y_train_val)

    # svm = SVC(**trained_svc_model.best_params_, random_state=random_state)
    # refit_svm = svm.fit(scaled_X_train_val, y_train_val)

    # rf = RandomForestClassifier(
    #     **trained_rf_model.best_params_, random_state=random_state
    # )
    # refit_rf = rf.fit(scaled_X_train_val, y_train_val)

    # xgboost = xgb.XGBClassifier(objective="binary:hinge", nthread=4, seed=random_state)
    # xgboost = xgboost.set_params(**trained_xgb_model)
    # refit_xgb = xgboost.fit(scaled_X_train_val, y_train_val)

    logistic = LogisticRegression(
        **trained_logit_model.best_params_, random_state=random_state
    )
    refit_logit = logistic.fit(scaled_resampled_X_train_val, scaled_resampled_y_train_val)

    refit_dt = None
    refit_svm = None
    refit_rf = None
    refit_xgb = None

    return refit_dt, refit_svm, refit_rf, refit_logit, refit_xgb


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


@dynamic
def fit_multiple_models(x_train_df, y_train_df, inner_cv):
    # Logistic Regression
    logit_grid = {
        # "penalty": ["l1", "l2", "elasticnet", None],
        # "dual": [True, False],
        # "C": [_ for _ in range(1, 10, 1)],
        "fit_intercept": [True, False],
        # "solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
        # "n_jobs": [-1],
    }

    logit_model = LogisticRegression()
    logit_result = model_fitting_loop_with_grid_search(
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

    # dt_clf = DecisionTreeClassifier(random_state=random_state)
    # dt_result = model_fitting_loop_with_grid_search(dt_clf, x_train_df, y_train_df, inner_cv, dt_grid, 'Decision Tree')

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
    # svc_model = SVC()
    # svc_result = model_fitting_loop_with_grid_search(
    #     svc_model, x_train_df, y_train_df, inner_cv, svc_grid, 'Support Vector Machine'
    # )

    # # Random Forest
    # rf_grid = {
    #     "criterion": ["gini", "entropy", "log_loss"],
    #     "max_depth": [_ for _ in range(1, 10, 1)],
    #     "max_features": ["sqrt", "log2", None],
    #     "min_samples_split": [_ for _ in range(1, 10, 1)],
    #     "min_samples_leaf": [_ for _ in range(1, 10, 1)],
    # }

    # rf_model = RandomForestClassifier()
    # rf_result = model_fitting_loop_with_grid_search(rf_model, x_train_df, y_train_df, inner_cv, rf_grid, 'Random Forest')

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

    # xgb_model = xgb.XGBClassifier(
    #     objective="binary:hinge", nthread=4, seed=random_state
    # )
    # xgb_result = model_fitting_loop_with_grid_search(
    #     xgb_model, x_train_df, y_train_df, inner_cv, xgb_grid, 'XGBoost'
    # )

    # log the following param for every model
    # best_estimator_
    # best_score_
    # best_params_
    # scorer_

    return (None,
        None,
        None,
        logit_result,
        None)

    # return (
    #     dt_result.best_estimator_,
    #     svc_result.best_estimator_,
    #     rf_result.best_estimator_,
    #     logit_result.best_estimator_,
    #     xgb_result.best_estimator_,
    # )


def flip_true_false(y):
    copy_y = copy(y)
    flipped_y = [0 if item == 1 else 1 for item in copy_y]
    return flipped_y

@task(container_image="ghcr.io/flyteorg/flytekit:py3.10-1.9.1", limits=Resources(mem="3000Mi", cpu="1", ephemeral_storage="3000Mi"))
def dummy_task():
    sum = 0
    for i in range(1000):
        sum += i

    return sum

@workflow
def work():

    # df = read_files(".")
    # X_train_val, X_test, y_train_val, y_test = filter_and_split_df(df)

    # print(X_train_val.shape, X_test.shape, y_train_val.shape, y_test.shape)
    # refit_dt, refit_svm, refit_rf, refit_logit, refit_xgb = nested_loop(
    #     X_train_val, y_train_val
    # )

    print('here')
    summation = dummy_task()
    return summation

    # with open('./logit.pickle', 'wb') as file:
    #     pickle.dump(refit_logit, file)
    
    # with open('./logit.pickle', 'rb') as file:
    #     refit_logit = pickle.load(file)

    # normalized_df = copy(X_test)
    # cd_first_quantile = np.quantile(normalized_df["size"], 0.25)
    # cd_third_quantile = np.quantile(normalized_df["size"], 0.75)
    # normalized_df["depth"] = np.log(normalized_df["depth"])
    # normalized_df["size"] = np.log(normalized_df["size"])
    # normalized_df["max_breadth"] = np.log(normalized_df["max_breadth"])
    # normalized_df["strongly_cc"] = np.log(normalized_df["strongly_cc"])
    # normalized_df["characteristic_distance"] = np.log(
    #     normalized_df["characteristic_distance"]
    #     + cd_first_quantile**2 / cd_third_quantile
    # )

    # scaler = StandardScaler().set_output(transform="pandas")
    # scaled_X_test = scaler.fit_transform(normalized_df)
    

    # logit_Y_pred = refit_logit.predict(scaled_X_test)
    # logit_Y_pred_proba = refit_logit.predict_proba(scaled_X_test)
    # logit_custom_score = get_all_scores(y_test, logit_Y_pred, logit_Y_pred_proba[:, 1])
    # logit_precision, logit_recall, logit_thresholds = precision_recall_curve(
    #     y_test, logit_Y_pred_proba[:, 1]
    # )
    # flipped_logit_custom_score = get_all_scores(
    #     y_test, logit_Y_pred, logit_Y_pred_proba[:, 0]
    # )
    # (
    #     flipped_logit_precision,
    #     flipped_logit_recall,
    #     flipped_logit_thresholds,
    # ) = precision_recall_curve(flip_true_false(y_test), logit_Y_pred_proba[:, 0])

    # print(
    #     logit_custom_score.accuracy,
    #     logit_custom_score.precision,
    #     logit_custom_score.recall,
    #     logit_custom_score.avg_precision,
    # )
    # print(
    #     flipped_logit_custom_score.accuracy,
    #     flipped_logit_custom_score.precision,
    #     flipped_logit_custom_score.recall,
    #     flipped_logit_custom_score.avg_precision,
    # )

    # print(confusion_matrix(y_test, logit_Y_pred))

if __name__ == "__main__":
    # get data
    # filter data, split data, prepare separate df
    # get into nested loop
    # print(os.path.sep)

    # Create the parser and add arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument(dest="run-id", help="run id for tracking in mlflow")

    # # Parse and print the results
    # args = parser.parse_args()

    # mlflow.set_tracking_uri("sqlite:///mlflow.db")
    # mlflow.set_experiment("rq2-round-" + args.run - id)
    

    # FlyteRemote object is the main entrypoint to API
    remote = FlyteRemote(
        config=Config(platform=PlatformConfig(endpoint='172.19.81.227:30080', insecure=True, insecure_skip_verify=False)),
        default_project="fakeray",
        default_domain="development",
    )

    # Execute
    execution = remote.execute(
        work, inputs={}, execution_name="workflow-execution-6", version="0.0.1", wait=True, image_config="ghcr.io/flyteorg/flytekit:py3.10-1.9.1"
    )

    # Or use execution_name_prefix to avoid repeated execution names
    # execution = remote.execute(
    #     flyte_workflow, inputs={"mean": 1}, execution_name_prefix="flyte", wait=True
    # )
    
# DecisionTreeClassifier
# SVC
# RandomForestClassifier
# LogisticRegression
# XGBClassifier
# VotingClassifier
