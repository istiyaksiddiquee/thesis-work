import os
import wandb
import joblib
import pickle
import logging
import chardet
import numpy as np
import pandas as pd
from copy import copy
import xgboost as xgb
from time import time
import lightgbm as lgb
import smote_variants as sv
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    average_precision_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC
from flytekit import task, workflow
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold, GridSearchCV, train_test_split, KFold

random_state = 7
wandb_project = "RQ2RUN5"
no_of_active_features = 15
optimization_metric = "balanced_accuracy_score"

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

@workflow
def nested_loop() -> list[CLFOutput]:

    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    # FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.DEBUG)

    try:

        logging.info("NESTED_LOOP: %s", "initiating processing, reading files")
        csv_path = "."
        X_train_val, X_test, y_train_val, y_test = read_pickled_input_files(csv_path)

        # call the nested loop to get all the trained models
        logging.info("NESTED_LOOP: %s", f"shapes of input: {X_train_val.shape}, {X_test.shape}, {y_train_val.shape}, {y_test.shape}")

        logging.info("NESTED_LOOP: %s", "entering nested loop")

        # outer_cv = RepeatedKFold(n_splits=2, n_repeats=1)
        outer_cv = KFold(n_splits=5)

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
            normalized_df["size"] = np.log(normalized_df["size"])
            normalized_df["strongly_cc"] = np.log(normalized_df["strongly_cc"])
            normalized_df["characteristic_distance"] = np.log(normalized_df["characteristic_distance"] + cd_first_quantile**2 / cd_third_quantile)

            scaler = StandardScaler().set_output(transform="pandas")
            scaled_X_train = scaler.fit_transform(normalized_df)
            scaled_resampled_X_train, scaled_resampled_y_train = oversample_data(scaled_X_train.to_numpy(), y_train.to_numpy())
            # scaled_resampled_X_train, scaled_resampled_y_train = pd.DataFrame(scaled_X_train.to_numpy()), pd.Series(y_train.to_numpy())

            inner_cv = RepeatedKFold(n_splits=5, n_repeats=3)

            logging.info("NESTED_LOOP: %s", f"entering model fitting for {epoch_str}")

            logit_output = fit_logistic_model(
                x_train_df=scaled_resampled_X_train, y_train_df=scaled_resampled_y_train, X_val=X_val, Y_val=Y_val, inner_cv=inner_cv, epoch_str=epoch_str
            )
            dt_output = fit_dt_model(
                x_train_df=scaled_resampled_X_train, y_train_df=scaled_resampled_y_train, X_val=X_val, Y_val=Y_val, inner_cv=inner_cv, epoch_str=epoch_str
            )
            rf_output = fit_rf_model(
                x_train_df=scaled_resampled_X_train, y_train_df=scaled_resampled_y_train, X_val=X_val, Y_val=Y_val, inner_cv=inner_cv, epoch_str=epoch_str
            )
            xgb_output = fit_xgb_model(
                x_train_df=scaled_resampled_X_train, y_train_df=scaled_resampled_y_train, X_val=X_val, Y_val=Y_val, inner_cv=inner_cv, epoch_str=epoch_str
            )
            lgb_output = fit_lgb_model(
                x_train_df=scaled_resampled_X_train, y_train_df=scaled_resampled_y_train, X_val=X_val, Y_val=Y_val, inner_cv=inner_cv, epoch_str=epoch_str
            )
            loop_outputs.append(logit_output)
            loop_outputs.append(dt_output)
            loop_outputs.append(rf_output)
            loop_outputs.append(xgb_output)
            loop_outputs.append(lgb_output)
            
        logging.info("NESTED_LOOP: %s", f"model fitting for {epoch_str} completed.")
        
    except Exception as error:
        logging.error("NESTED_LOOP: %s", "ERROR: some error happened, could not finish.")
        logging.error("NESTED_LOOP: %s", error)

        wandb.init(project=wandb_project)
        wandb.alert(title="Error", text="Your run was interrupted by some exception.")
        wandb.finish()
    return loop_outputs

@workflow
def main_wf():
    
    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    wandb.init(project=wandb_project)
    wandb.alert(title="Started", text="Your run has started. Mark the time.")
    wandb.finish()

    loop_outputs = nested_loop()
    refitt = refitting_models(loop_outputs=loop_outputs)
    loop_outputs >> refitt

    logging.info("MAIN_WF: %s", f"workflow finished.")
    return


@task(container_image="istiyaksiddiquee/flyte-for-kube:RQ2RUN5")
def refitting_models(
    loop_outputs: list[CLFOutput]
) -> None:

    logging.info("REFITTING_MODELS: %s", "Model Refitting starts.")
    
    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    csv_path = "/root/workflows"
    X_train_val, _, y_train_val, _ = read_pickled_input_files(csv_path)

    loop_counter = 0

    logit_epoch_id = -1
    rf_epoch_id = -1
    dt_epoch_id = -1
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

    loop_counter = 0
    total_iteration = int (len(loop_outputs) / 5)
    
    for i in range(total_iteration):
        
        index = 5 * i + 0
        logit_output = loop_outputs[index]

        index = 5 * i + 1
        dt_output = loop_outputs[index]
        
        index = 5 * i + 2
        rf_output = loop_outputs[index]
        
        index = 5 * i + 3
        xgb_output = loop_outputs[index]
        
        index = 5 * i + 4
        lgb_output = loop_outputs[index]
        
        if logit_output != None and logit_output.score != None:
            if logit_avg_prec < logit_output.score:
                logit_avg_prec = logit_output.score
                trained_logit_model = logit_output.gridsearch_dict
                logit_epoch_id = loop_counter

        if dt_output != None and dt_output.score != None: 
            if dt_avg_prec < dt_output.score:
                dt_avg_prec = dt_output.score
                trained_dt_model = dt_output.gridsearch_dict
                dt_epoch_id = loop_counter

        if rf_output != None and rf_output.score != None:
            if rf_avg_prec < rf_output.score:
                rf_avg_prec = rf_output.score
                trained_rf_model = rf_output.gridsearch_dict
                rf_epoch_id = loop_counter

        if xgb_output != None and xgb_output.score != None:
            if xgb_avg_prec < xgb_output.score:
                xgb_avg_prec = xgb_output.score
                trained_xgb_model = xgb_output.gridsearch_dict
                xgb_epoch_id = loop_counter

        if lgb_output != None and lgb_output.score != None: 
            if lgb_avg_prec < lgb_output.score:
                lgb_avg_prec = lgb_output.score
                trained_lgb_model = lgb_output.gridsearch_dict
                lgb_epoch_id = loop_counter
        
        loop_counter += 1

    logging.info("REFITTING_MODELS: %s", f"models retrieved, re-fitting starts")
    normalized_df = copy(X_train_val)
    cd_first_quantile = np.quantile(normalized_df["characteristic_distance"], 0.25)
    cd_third_quantile = np.quantile(normalized_df["characteristic_distance"], 0.75)
    normalized_df["depth"] = np.log(normalized_df["depth"])
    normalized_df["max_breadth"] = np.log(normalized_df["max_breadth"])
    normalized_df["size"] = np.log(normalized_df["size"])
    normalized_df["strongly_cc"] = np.log(normalized_df["strongly_cc"])
    normalized_df["characteristic_distance"] = np.log(normalized_df["characteristic_distance"] + cd_first_quantile**2 / cd_third_quantile)

    scaler = StandardScaler().set_output(transform="pandas")
    scaled_X_train_val = scaler.fit_transform(normalized_df)
    scaled_resampled_X_train_val, scaled_resampled_y_train_val = oversample_data(scaled_X_train_val.to_numpy(), y_train_val.to_numpy())
    # scaled_resampled_X_train_val, scaled_resampled_y_train_val = pd.DataFrame(scaled_X_train_val.to_numpy()), pd.Series(y_train_val.to_numpy())

    # dummy_false = fit_dummy_classifier(scaled_resampled_X_train_val, scaled_resampled_y_train_val, 0)

    logging.info("REFITTING_MODELS: %s", "data ready, initiating processing")
    
    if trained_logit_model != None:
        # store logit model
        logging.info("REFITTING_MODELS: %s", "processing logit model.")
        logistic = LogisticRegression(**trained_logit_model)
        refit_logit = logistic.fit(scaled_resampled_X_train_val.values, scaled_resampled_y_train_val.values)

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
        logging.info("REFITTING_MODELS: %s", "processing dt model.")
        dt = DecisionTreeClassifier(**trained_dt_model)
        refit_dt = dt.fit(scaled_resampled_X_train_val.values, scaled_resampled_y_train_val.values)

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
        logging.info("REFITTING_MODELS: %s", "processing rf model.")
        rf = RandomForestClassifier(**trained_rf_model)
        refit_rf = rf.fit(scaled_resampled_X_train_val.values, scaled_resampled_y_train_val.values)

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
        logging.info("REFITTING_MODELS: %s", "processing xgb model.")
        xgboost = xgb.XGBClassifier(objective="binary:hinge", nthread=4, seed=random_state)
        xgboost = xgboost.set_params(**trained_xgb_model)
        refit_xgb = xgboost.fit(scaled_resampled_X_train_val.values, scaled_resampled_y_train_val.values)

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
        logging.info("REFITTING_MODELS: %s", "processing lgb model.")
        lgb_model = lgb.LGBMClassifier(objective="binary", random_state=42)
        lgb_model = lgb_model.set_params(**trained_lgb_model)
        refit_lgb = lgb_model.fit(scaled_resampled_X_train_val.values, scaled_resampled_y_train_val.values)

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

    logging.info("REFITTING_MODELS: %s", "process complete, returning to base.")
    
    wandb.init(project=wandb_project)
    wandb.alert(title="Complete", text="Your run is complete. Check the board.")
    wandb.finish()
    return

def oversample_data(X: pd.Series, y: pd.Series):
    
    oversampler = sv.polynom_fit_SMOTE_poly()
    X_samp, y_samp = oversampler.sample(X, y)
    X_samp, y_samp = pd.DataFrame(X_samp), pd.Series(y_samp)

    return X_samp, y_samp


def fit_dummy_classifier(x_train_df, y_train_df, constant):
    dummy_clf = DummyClassifier(strategy="constant", constant=constant)
    dummy_clf.fit(x_train_df, y_train_df)
    return dummy_clf


@task(container_image="istiyaksiddiquee/flyte-for-kube:RQ2RUN5")
def fit_logistic_model(
    x_train_df: pd.Series, y_train_df: pd.Series, X_val: pd.Series, Y_val: pd.Series, inner_cv: RepeatedKFold, epoch_str: str
) -> CLFOutput:
    # Logistic Regression

    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    logging.info("FIT_LOGIT_MODEL: %s", f"logit run scheduled for {epoch_str}.")
    logging.info("FIT_LOGIT_MODEL: %s", f"{x_train_df.shape}, {y_train_df.shape}, {X_val.shape}, {Y_val.shape}.")

    logit_result = None
    logit_best_grid_param = None
    logit_score = None

    try:
        # logit_grid = {
        #     "penalty": ["l2"],
        # }
        logit_grid = {
            "penalty": ["l1", "l2", "elasticnet"],
            "dual": [True, False],
            "C": [_ for _ in range(1, 10, 1)],
            "fit_intercept": [True, False],
            "max_iter": [500],
            "solver": ["lbfgs", "newton-cg", "newton-cholesky", "sag", "saga"],
            "n_jobs": [-1],
        }
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

        logit_result = clf.fit(x_train_df.values, y_train_df.values)

    except Exception as error:
        logging.error("FIT_LOGIT_MODEL: %s", "Could not fit Logistic model.")
        logging.error("FIT_LOGIT_MODEL: %s", f"An exception occurred: {error}")

    if logit_result != None:
        logit_model = logit_result.best_estimator_
        logit_best_grid_param = logit_model.get_params()
        wandb.init(project=wandb_project, group="logit", job_type=epoch_str)

        logit_Y_pred = logit_model.predict(X_val.values)
        logit_Y_pred_proba = logit_model.predict_proba(X_val.values)
        logit_custom_score = get_all_scores(Y_val.values, logit_Y_pred, logit_Y_pred_proba[:, 1])
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


@task(container_image="istiyaksiddiquee/flyte-for-kube:RQ2RUN5")
def fit_dt_model(x_train_df: pd.Series, y_train_df: pd.Series, X_val: pd.Series, Y_val: pd.Series, inner_cv: RepeatedKFold, epoch_str: str) -> CLFOutput:
    # Decision Tree

    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    logging.info("FIT_DT_MODEL: %s", f"dt run scheduled for {epoch_str}.")

    dt_result = None
    dt_score = None
    dt_best_grid_param = None
    try:

        # dt_grid = {"criterion": ["gini"]}
        dt_grid = {
            "criterion": ["gini", "entropy", "log_loss"],
            "splitter": ["best", "random"],
            "max_depth": [_ for _ in range(1, 5, 1)],
            "min_samples_split": [_ for _ in range(1, 5, 1)],
            "min_samples_leaf": [_ for _ in range(1, 10, 1)],
        }
        
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

        dt_result = clf.fit(x_train_df.values, y_train_df.values)

    except Exception as error:
        logging.error("FIT_DT_MODEL: %s", "Could not fit Decision Tree model")
        logging.error("FIT_DT_MODEL: %s", f"An exception occurred: {error}")

    if dt_result != None:
        dt_model = dt_result.best_estimator_
        dt_best_grid_param = dt_model.get_params()
        wandb.init(project=wandb_project, group="dt", job_type=epoch_str)
        dt_Y_pred = dt_model.predict(X_val.values)
        dt_Y_pred_proba = dt_model.predict_proba(X_val.values)
        dt_custom_score = get_all_scores(Y_val.values, dt_Y_pred, dt_Y_pred_proba[:, 1])
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

@task(container_image="istiyaksiddiquee/flyte-for-kube:RQ2RUN5")
def fit_rf_model(x_train_df: pd.Series, y_train_df: pd.Series, X_val: pd.Series, Y_val: pd.Series, inner_cv: RepeatedKFold, epoch_str: str) -> CLFOutput:
    # Random Forest

    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    logging.info("FIT_RF_MODEL: %s", f"rf run scheduled for {epoch_str}.")

    rf_result = None
    rf_score = None
    rf_best_grid_param = None

    try:
        # rf_grid = {"criterion": ["gini"]}

        rf_grid = {
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [_ for _ in range(1, 5, 1)],
            "min_samples_leaf": [_ for _ in range(1, 10, 1)],
        }
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

        rf_result = clf.fit(x_train_df.values, y_train_df.values)

    except Exception as error:
        logging.error("FIT_RF_MODEL: %s", f"Could not fit Random Forest model.")
        logging.error("FIT_RF_MODEL: %s", f"An exception occurred: {error}")

    if rf_result != None:
        rf_model = rf_result.best_estimator_
        rf_best_grid_param = rf_model.get_params()
        wandb.init(project=wandb_project, group="rf", job_type=epoch_str)
        rf_Y_pred = rf_model.predict(X_val.values)
        rf_Y_pred_proba = rf_model.predict_proba(X_val.values)
        rf_custom_score = get_all_scores(Y_val.values, rf_Y_pred, rf_Y_pred_proba[:, 1])
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


@task(container_image="istiyaksiddiquee/flyte-for-kube:RQ2RUN5")
def fit_xgb_model(x_train_df: pd.Series, y_train_df: pd.Series, X_val: pd.Series, Y_val: pd.Series, inner_cv: RepeatedKFold, epoch_str: str) -> CLFOutput:
    # XGB

    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    logging.info("FIT_XGB_MODEL: %s", f"xgb run scheduled for {epoch_str}.")

    xgb_result = None
    xgb_score = None
    xgb_best_grid_param = None

    try:
        # XGB
        # xgb_grid = {
        #     "learning_rate": [0.1],
        # }
        xgb_grid = {
            # "n_estimators": range(60, 220, 40),
            "learning_rate": [0.1, 0.01, 0.05],
            "booster": ["gbtree", "gblinear", "dart"],
        }

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

        xgb_result = clf.fit(x_train_df.values, y_train_df.values)

    except Exception as error:
        logging.error("FIT_XGB_MODEL: %s", "Could not fit XGB model")
        logging.error("FIT_XGB_MODEL: %s", f"An exception occurred: {error}")

    if xgb_result != None:
        try:
            xgb_model = xgb_result.best_estimator_
            xgb_best_grid_param = xgb_model.get_params()
            
            wandb.init(project=wandb_project, group="xgb", job_type=epoch_str)
            
            xgb_Y_pred = xgb_model.predict(X_val.values)
            xgb_Y_pred_proba = xgb_model.predict_proba(X_val.values)
            xgb_custom_score = get_all_scores(Y_val.values, xgb_Y_pred, xgb_Y_pred_proba[:, 1])
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


@task(container_image="istiyaksiddiquee/flyte-for-kube:RQ2RUN5")
def fit_lgb_model(x_train_df: pd.Series, y_train_df: pd.Series, X_val: pd.Series, Y_val: pd.Series, inner_cv: RepeatedKFold, epoch_str: str) -> CLFOutput:
    # LGB

    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    
    logging.info("FIT_LGB_MODEL: %s", f"lgb run scheduled for {epoch_str}.")

    lgb_result = None
    lgb_score = None
    lgb_best_grid_param = None
    
    try:
        # LGB
        # lgb_grid = {"num_leaves": [31]}

        lgb_grid = {
            "learning_rate": [0.001, 0.005, 0.01],
            "n_estimators": [8, 16, 24],
            "num_leaves": [6, 8, 12],  # large num_leaves helps improve accuracy but might lead to over-fitting
            "boosting_type": ["gbdt", "dart"],  # for better accuracy -> try dart
            "subsample": [0.7, 0.75],
            "reg_alpha": [1, 1.2],
            "reg_lambda": [1, 1.2, 1.4],
        }

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

        lgb_result = clf.fit(x_train_df.values, y_train_df.values)

    except Exception as error:
        logging.error("FIT_LGB_MODEL: %s", f"Could not fit XGB model")
        logging.error("FIT_LGB_MODEL: %s", f"An exception occurred: {error}")

    if lgb_result != None:
        
        lgb_model = lgb_result.best_estimator_
        lgb_best_grid_param = lgb_model.get_params()
        wandb.init(project=wandb_project, group="lgb", job_type=epoch_str)

        lgb_Y_pred = lgb_model.predict(X_val.values)
        lgb_Y_pred_proba = lgb_model.predict_proba(X_val.values)
        lgb_custom_score = get_all_scores(Y_val.values, lgb_Y_pred, lgb_Y_pred_proba[:, 1])
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

if __name__ == "__main__":
    main_wf()
