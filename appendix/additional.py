import os
import pandas as pd
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

import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier
from copy import copy
import numpy as np
import wandb
import joblib
import lightgbm as lgb
import pickle
import logging
from flytekit import task, workflow
import smote_variants as sv

total_cv = 5
random_state = 7
no_of_active_features = 15
wandb_project = "RQ2RUN5"
optimization_metric = "balanced_accuracy"
# data_folder = "segment"
# data_folder = "shuttle"
# data_folder = "one_yeast"
# data_folder = "three_yeast"
data_folder = "thesis"


scorers_for_gridcv = {
    "accuracy_score": make_scorer(accuracy_score),
    "precision_score": make_scorer(precision_score, average=None),
    "recall_score": make_scorer(recall_score, average=None),
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
    precision = precision_score(y_real, y_pred, average=None)
    recall = recall_score(y_real, y_pred, average=None)
    balanced_accuracy = balanced_accuracy_score(y_real, y_pred)
    fbeta = fbeta_score(y_real, y_pred, beta=0.5)
    avg_precision = average_precision_score(y_real, y_scores)
    roc_auc = roc_auc_score(y_real, y_scores)

    return CustomScore(accuracy, precision, recall, balanced_accuracy, fbeta, avg_precision, roc_auc)


def convert_scores_to_dict(custom_scores: CustomScore):
    metrics = {}

    metrics["accuracy"] = round(custom_scores.accuracy, 2)
    metrics["precision_0"] = round(custom_scores.precision[0], 2)
    metrics["precision_1"] = round(custom_scores.precision[1], 2)
    metrics["recall_0"] = round(custom_scores.recall[0], 2)
    metrics["recall_1"] = round(custom_scores.recall[1], 2)
    metrics["balanced_accuracy"] = round(custom_scores.balanced_accuracy, 2)
    metrics["fbeta"] = round(custom_scores.fbeta, 2)
    metrics["avg_precision"] = round(custom_scores.avg_precision, 2)
    metrics["roc_auc"] = round(custom_scores.roc_auc, 2)

    return metrics


def oversample_data(X: pd.Series, y: pd.Series):

    oversampler = sv.polynom_fit_SMOTE_poly()
    X_samp, y_samp = oversampler.sample(X, y)
    X_samp, y_samp = pd.DataFrame(X_samp), pd.Series(y_samp)

    return X_samp, y_samp


def read_pickled_input_files(file_path: str):
    if file_path == None:
        logging.error("READING_FILES: %s", "file path must be provided.")
        return

    X_train_val = None
    y_train_val = None
    X_test = None
    y_test = None

    with open(os.path.join(file_path, "x_train_val_full.pickle"), "rb") as file:
        X_train_val = pickle.load(file)

    with open(os.path.join(file_path, "y_train_val_full.pickle"), "rb") as file:
        y_train_val = pickle.load(file)

    with open(os.path.join(file_path, "x_test_full.pickle"), "rb") as file:
        X_test = pickle.load(file)

    with open(os.path.join(file_path, "y_test_full.pickle"), "rb") as file:
        y_test = pickle.load(file)

    return X_train_val, X_test, y_train_val, y_test


def process_gridcv_results(cv_results, group, total_cv=5):
    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    
    
    for i in range(0, total_cv):
        acc_score = round(cv_results['mean_test_accuracy_score'][i], 2)
        prec_0 = round(cv_results['mean_test_precision_0'][i], 2)
        prec_1 = round(cv_results['mean_test_precision_1'][i], 2)
        rec_0 = round(cv_results['mean_test_recall_0'][i], 2)
        rec_1 = round(cv_results['mean_test_recall_1'][i], 2)
        fbeta = round(cv_results['mean_test_fbeta_score'][i], 2)
        ba_score = round(cv_results['mean_test_balanced_accuracy_score'][i], 2)
        avg_prec = round(cv_results['mean_test_average_precision_score'][i], 2)
        roc_auc = round(cv_results['mean_test_roc_auc'][i], 2)
        
        wandb.init(project='test01', group=group, job_type="epoch_"+ str(i+1))
        wandb.log({"accuracy": acc_score, "precision_0": prec_0, "precision_1": prec_1, "recall_0": rec_0, "recall_1": rec_1, "fbeta": fbeta, "balanced_accuracy": ba_score, "average_precision": avg_prec, "roc_auc": roc_auc})
        
    return

@task(container_image="istiyaksiddiquee/flyte-for-thesis:RQ2RUN5")
def fit_logistic_model(x_train_val_df: pd.Series, y_train_val_df: pd.Series) -> None:
    # Logistic Regression

    logging.info("FIT_LOGIT_MODEL: %s", f"logit run scheduled.")

    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    
    logit_result = None

    y_train_val_df.replace(to_replace='positive', value=1, inplace=True)
    y_train_val_df.replace(to_replace='negative', value=0, inplace=True)

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
            cv=total_cv,
            refit=optimization_metric,
            param_grid=logit_grid,
            scoring=optimization_metric,
            verbose=0,
            n_jobs=-1,
        )

        logit_result = clf.fit(x_train_val_df, y_train_val_df)

    except Exception as error:
        logging.error("Could not fit Logistic model")
        logging.error("An exception occurred:", error)

    if logit_result != None:
        process_gridcv_results(logit_result.cv_results_, "logit")
        
        wandb.init(project=wandb_project, group="logit", job_type="final")
        
        logit_model = logit_result.best_estimator_
        joblib.dump(logit_model, "logit.joblib")
        logit_artifact = wandb.Artifact(
            "Logistic-Model",
            type="model",
            description="selected Logistic model",
            metadata={
                "best_parameters": logit_result.best_params_,
                "best_score": logit_result.best_score_
            },
        )

        logit_artifact.add_file("logit.joblib")
        wandb.log_artifact(logit_artifact)
        wandb.finish()

    logging.info("FIT_LOGIT_MODEL: %s", "logit run completed.")
    return


@task(container_image="istiyaksiddiquee/flyte-for-thesis:RQ2RUN5")
def fit_dt_model(x_train_val_df: pd.Series, y_train_val_df: pd.Series) -> None:
    # Decision Tree

    logging.info("FIT_DT_MODEL: %s", f"dt run scheduled.")

    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    
    dt_result = None

    y_train_val_df.replace(to_replace='positive', value=1, inplace=True)
    y_train_val_df.replace(to_replace='negative', value=0, inplace=True)
    
    try:
        dt_grid = {
            "criterion": ["gini", "entropy", "log_loss"],
            "splitter": ["best", "random"],
            "max_depth": [_ for _ in range(1, 5, 1)],
            "min_samples_split": [_ for _ in range(2, 5, 1)],
            "min_samples_leaf": [_ for _ in range(2, 10, 1)],
        }
        # dt_grid = {"criterion": ["gini"]}
        dt_clf = DecisionTreeClassifier(random_state=random_state)

        clf = GridSearchCV(
            estimator=dt_clf,
            cv=total_cv,
            refit=optimization_metric,
            param_grid=dt_grid,
            scoring=optimization_metric,
            verbose=0,
            n_jobs=-1,
        )

        dt_result = clf.fit(x_train_val_df, y_train_val_df)

    except Exception as error:
        logging.error("Could not fit Decision Tree model")
        logging.error("An exception occurred:", error)

    if dt_result != None:
        process_gridcv_results(dt_result.cv_results_, "logit")
        
        wandb.init(project=wandb_project, group="dt", job_type="final")
        dt_model = dt_result.best_estimator_
        joblib.dump(dt_model, "dt.joblib")
        dt_artifact = wandb.Artifact(
            "Decision-Tree-Model",
            type="model",
            description="selected DT model",
            metadata={
                "best_parameters": dt_result.best_params_,
                "best_score": dt_result.best_score_
            },
        )

        dt_artifact.add_file("dt.joblib")
        wandb.log_artifact(dt_artifact)
        wandb.finish()

    logging.info("FIT_DT_MODEL: %s", "dt run completed.")
    return


@task(container_image="istiyaksiddiquee/flyte-for-thesis:RQ2RUN5")
def fit_rf_model(x_train_val_df: pd.Series, y_train_val_df: pd.Series) -> None:
    # Random Forest

    logging.info("FIT_RF_MODEL: %s", f"rf run scheduled.")

    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    
    rf_result = None
    y_train_val_df.replace(to_replace='positive', value=1, inplace=True)
    y_train_val_df.replace(to_replace='negative', value=0, inplace=True)

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
            cv=total_cv,
            refit=optimization_metric,
            param_grid=rf_grid,
            scoring=optimization_metric,
            verbose=0,
            n_jobs=-1,
        )

        rf_result = clf.fit(x_train_val_df, y_train_val_df)

    except Exception as error:
        logging.error("Could not fit Random Forest model")
        logging.error("An exception occurred:", error)

    if rf_result != None:
        process_gridcv_results(rf_result.cv_results_, "logit")
        
        wandb.init(project=wandb_project, group="rf", job_type="final")

        rf_model = rf_result.best_estimator_
        joblib.dump(rf_model, "rf.joblib")
        rf_artifact = wandb.Artifact(
            "Random-Forest-Model",
            type="model",
            description="selected RF model",
            metadata={
                "best_parameters": rf_result.best_params_,
                "best_score": rf_result.best_score_
            },
        )

        rf_artifact.add_file("rf.joblib")
        wandb.log_artifact(rf_artifact)
        wandb.finish()

    logging.info("FIT_RF_MODEL: %s", "rf run completed.")
    return


@task(container_image="istiyaksiddiquee/flyte-for-thesis:RQ2RUN5")
def fit_xgb_model(x_train_val_df: pd.Series, y_train_val_df: pd.Series) -> None:
    # XGB

    logging.info("FIT_XGB_MODEL: %s", f"xgb run scheduled.")

    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    
    xgb_result = None
    y_train_val_df.replace(to_replace='positive', value=1, inplace=True)
    y_train_val_df.replace(to_replace='negative', value=0, inplace=True)

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
            cv=total_cv,
            refit=optimization_metric,
            param_grid=xgb_grid,
            scoring=optimization_metric,
            verbose=0,
            n_jobs=-1,
        )

        xgb_result = clf.fit(x_train_val_df, y_train_val_df)

    except Exception as error:
        logging.error("Could not fit XGB model")
        logging.error("An exception occurred:", error)

    if xgb_result != None:
        process_gridcv_results(xgb_result.cv_results_, "logit")
        
        wandb.init(project=wandb_project, group="xgb", job_type="final")

        xgb_model = xgb_result.best_estimator_
        joblib.dump(xgb_model, "xgb.joblib")
        xgb_artifact = wandb.Artifact(
            "XGB-Model",
            type="model",
            description="selected XGB model",
            metadata={
                "best_parameters": xgb_result.best_params_,
                "best_score": xgb_result.best_score_
            },
        )

        xgb_artifact.add_file("xgb.joblib")
        wandb.log_artifact(xgb_artifact)
        wandb.finish()

    logging.info("FIT_XGB_MODEL: %s", "xgb run completed.")
    return


@task(container_image="istiyaksiddiquee/flyte-for-thesis:RQ2RUN5")
def fit_lgb_model(x_train_val_df: pd.Series, y_train_val_df: pd.Series) -> None:
    # LGB

    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    
    logging.info("FIT_LGB_MODEL: %s", f"lgb run scheduled.")

    y_train_val_df.replace(to_replace='positive', value=1, inplace=True)
    y_train_val_df.replace(to_replace='negative', value=0, inplace=True)
    
    lgb_result = None
    
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
            cv=total_cv,
            refit=optimization_metric,
            param_grid=lgb_grid,
            scoring=optimization_metric,
            verbose=0,
            n_jobs=-1,
        )

        lgb_result = clf.fit(x_train_val_df, y_train_val_df)

    except Exception as error:
        logging.error("Could not fit LGB model")
        logging.error("An exception occurred:", error)

    if lgb_result != None:
        process_gridcv_results(lgb_result.cv_results_, "logit")
        
        wandb.init(project=wandb_project, group="lgb", job_type="final")

        lgb_model = lgb_result.best_estimator_
        joblib.dump(lgb_model, "lgb.joblib")
        lgb_artifact = wandb.Artifact(
            "LGB-Model",
            type="model",
            description="selected LGB model",
            metadata={
                "best_parameters": lgb_result.best_params_,
                "best_score": lgb_result.best_score_
            },
        )

        lgb_artifact.add_file("lgb.joblib")
        wandb.log_artifact(lgb_artifact)
        wandb.finish()

    logging.info("FIT_LGB_MODEL: %s", "lgb run completed.")
    return


@task(container_image="istiyaksiddiquee/flyte-for-thesis:RQ2RUN5")
def fit_dummy_classifier(x: pd.Series, y: pd.Series):

    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    y.replace(to_replace='positive', value=1, inplace=True)
    y.replace(to_replace='negative', value=0, inplace=True)

    dummy_stratified = DummyClassifier(strategy="stratified")
    dummy_frequent = DummyClassifier(strategy="most_frequent")

    stratified_dummy_cls = dummy_stratified.fit(x, y)
    most_freq_dummy_cls = dummy_frequent.fit(x, y)

    wandb.init(project=wandb_project, group="dummy", job_type="final")
    joblib.dump(stratified_dummy_cls, "stratified_dummy_cls.joblib")
    joblib.dump(most_freq_dummy_cls, "most_freq_dummy_cls.joblib")

    str_dum_artifact = wandb.Artifact("Stratified-Dummy-Cls", type="model", description="trained stratified dummy model")

    most_freq_dum_artifact = wandb.Artifact("Most-Freq-Dummy-Cls", type="model", description="trained most freq dummy model")

    str_dum_artifact.add_file("stratified_dummy_cls.joblib")
    most_freq_dum_artifact.add_file("most_freq_dummy_cls.joblib")

    wandb.log_artifact(str_dum_artifact)
    wandb.log_artifact(most_freq_dum_artifact)
    wandb.finish()

    return


@workflow()
def additional_workflow():
    
    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.DEBUG)

    try:
        # wandb.init(project=wandb_project)
        # wandb.alert(title="Started", text="Your run has started. Mark the time.")
        # wandb.finish()

        logging.info("ADDITIONAL_WORKFLOW: %s", "initiating processing, reading files")
        csv_path = os.path.join(".", data_folder)
        X_train_val, X_test, y_train_val, y_test = read_pickled_input_files(csv_path)
        print(y_train_val)
        
        # call the nested loop to get all the trained models
        logging.info("ADDITIONAL_WORKFLOW: %s", f"shapes of input: {X_train_val.shape}, {X_test.shape}, {y_train_val.shape}, {y_test.shape}")

        logging.info("NESTED_LOOP: %s", "feature scaling")
        normalized_df = copy(X_train_val)
        cd_first_quantile = np.quantile(normalized_df["characteristic_distance"], 0.25)
        cd_third_quantile = np.quantile(normalized_df["characteristic_distance"], 0.75)
        normalized_df["depth"] = np.log(normalized_df["depth"])
        normalized_df["max_breadth"] = np.log(normalized_df["max_breadth"])
        normalized_df["characteristic_distance"] = np.log(normalized_df["characteristic_distance"] + cd_first_quantile**2 / cd_third_quantile)
        normalized_df["size"] = np.log(normalized_df["size"])
        normalized_df["strongly_cc"] = np.log(normalized_df["strongly_cc"])

        scaler = StandardScaler().set_output(transform="pandas")
        scaled_X_train = scaler.fit_transform(normalized_df)
        scaled_resampled_X_train, scaled_resampled_y_train = oversample_data(scaled_X_train.to_numpy(), y_train_val.to_numpy())

        logging.info("ADDITIONAL_WORKFLOW: %s", "entering model fitting task")
        fit_dt_model(x_train_val_df=scaled_resampled_X_train, y_train_val_df=scaled_resampled_y_train)
        fit_logistic_model(x_train_val_df=scaled_resampled_X_train, y_train_val_df=scaled_resampled_y_train)
        fit_xgb_model(x_train_val_df=scaled_resampled_X_train, y_train_val_df=scaled_resampled_y_train)
        fit_rf_model(x_train_val_df=scaled_resampled_X_train, y_train_val_df=scaled_resampled_y_train)
        fit_lgb_model(x_train_val_df=scaled_resampled_X_train, y_train_val_df=scaled_resampled_y_train)
        fit_dummy_classifier(x=scaled_resampled_X_train, y=scaled_resampled_y_train)
        
        logging.info("ADDITIONAL_WORKFLOW: %s", "model fitting complete.")


    except Exception as error:
        logging.info("ADDITIONAL_WORKFLOW: %s", "ERROR: some error happened, could not finish.")
        logging.info("ADDITIONAL_WORKFLOW: %s", error)

        wandb.init(project=wandb_project)
        wandb.alert(title="Error", text="Something happened. Check the logs.")
        wandb.finish()

    return

@task(container_image="istiyaksiddiquee/flyte-for-thesis:RQ2RUN5")
def finishing_alert() -> None:

    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    wandb.init(project=wandb_project)
    wandb.alert(title="Finished", text="The run has finished. Check the results.")
    wandb.finish()
    
    return 


@workflow
def main_wf():
    
    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    # wandb.init(project=wandb_project)
    # wandb.alert(title="Started", text="Your run has started. Mark the time.")
    # wandb.finish()

    loop_outputs = additional_workflow()
    refitt = finishing_alert()
    loop_outputs >> refitt

    logging.info("MAIN_WF: %s", f"workflow finished.")
    return

if __name__ == "__main__":
    main_wf()
