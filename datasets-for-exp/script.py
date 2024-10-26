import os
import wandb
import joblib
import pickle
import logging
import numpy as np
import pandas as pd
from copy import copy
import xgboost as xgb
import lightgbm as lgb
import smote_variants as sv
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    average_precision_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import StratifiedKFold, HalvingGridSearchCV

# RQ2Final1, RQ3Final1, RQ3Final2 
# RQ2Final2, RQ3Final3, RQ3Final4, RQ4Final1

# experiment related settings

# direction: check the following before running an experiment 
# 1. change wandb_project and folder_path according to the experiment name
# 2. check notebook for normalization column and selected column
# 3. check codebase related settings

ds = 1
trial = False
quant = True
feature_selection = 0
data_imputation = 0
wandb_project = "RQ2Comb4"
folder_path = "RQ2Final1"

removed_columns = ['size', 'max_breadth']
normalization_columns = ['size', 'max_breadth', 'characteristic_distance']

if feature_selection == 1:
    normalization_columns = list(set(normalization_columns)-set(removed_columns))

# codebase related settings
random_state = 7
optimization_metric = "average_precision_score"
default_metric = "val_average_precision"

class CLFOutput:
    def __init__(self, gridsearch_dict: dict, score: float) -> None:
        self.gridsearch_dict = gridsearch_dict
        self.score = score 

def precision_0(y, y_pred):
    returned_arr = precision_score(y_true=y, y_pred=y_pred, average=None)
    return returned_arr[0]

def precision_1(y, y_pred):
    returned_arr = precision_score(y_true=y, y_pred=y_pred, average=None)
    return returned_arr[1]

def recall_0(y, y_pred):
    returned_arr = recall_score(y_true=y, y_pred=y_pred, average=None)
    return returned_arr[0]

def recall_1(y, y_pred):
    returned_arr = recall_score(y_true=y, y_pred=y_pred, average=None)
    return returned_arr[1]

def get_all_scores(y_real, y_pred, y_scores) -> dict:
    
    accuracy = accuracy_score(y_real, y_pred)
    precision0 = precision_0(y_real, y_pred)
    precision1 = precision_1(y_real, y_pred)
    recall0 = recall_0(y_real, y_pred)
    recall1 = recall_1(y_real, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_real, y_pred)
    fbeta = fbeta_score(y_real, y_pred, beta=0.5)
    avg_precision = average_precision_score(y_real, y_scores)
    roc_auc = roc_auc_score(y_real, y_scores)

    return {
        "val_accuracy": accuracy, 
        "val_precision_0": precision0, 
        "val_precision_1": precision1, 
        "val_recall_0": recall0, 
        "val_recall_1": recall1, 
        "val_fbeta": fbeta, 
        "val_balanced_accuracy": balanced_accuracy,
        "val_average_precision": avg_precision,
        "val_roc_auc": roc_auc
    }    


def read_pickled_input_files(file_path: str):
    
    if file_path == None:
        logging.error("READING_FILES: %s", "file path must be provided.")
        return

    X_train_val = None
    y_train_val = None

    X_train_val_file_name = file_path + "_x_train_val.pickle"
    y_train_val_file_name = file_path + "_y_train_val.pickle"
        
    with open(os.path.join(file_path, X_train_val_file_name), "rb") as file:
        X_train_val = pickle.load(file)
    
    with open(os.path.join(file_path, y_train_val_file_name) , "rb") as file:
        y_train_val = pickle.load(file)
    
    return X_train_val, y_train_val

# @workflow
def nested_loop() -> list[CLFOutput]:

    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    # FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format="%(asctime)s LINE: %(lineno)d - %(message)s", level=logging.DEBUG)

    try:

        logging.info("NESTED_LOOP: %s", "initiating processing, reading files")
        # csv_path = "."
        X_train_val, y_train_val = read_pickled_input_files(folder_path)
        
        # call the nested loop to get all the trained models
        logging.info("NESTED_LOOP: %s", f"shapes of input: {X_train_val.shape}, {y_train_val.shape}")

        logging.info("NESTED_LOOP: %s", "entering nested loop")

        outer_cv = StratifiedKFold(n_splits=5)

        loop_index = 0

        logging.info("NESTED_LOOP: %s", "loop starts")

        loop_outputs = []

        for train_index, val_index in outer_cv.split(X_train_val.to_numpy(), y_train_val.to_numpy()):

            loop_index += 1
            epoch_str = "epoch_" + str(loop_index)

            logging.info("NESTED_LOOP: %s", f"inside loop epoch {loop_index}")

            X_train, X_val = (X_train_val.iloc[train_index, :], X_train_val.iloc[val_index, :])

            y_train, Y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

            if feature_selection == 1:
                X_train = X_train[X_train.columns[~X_train.columns.isin(removed_columns)]]
                X_val = X_val[X_val.columns[~X_val.columns.isin(removed_columns)]]
                
            logging.info("NESTED_LOOP: %s", "feature scaling")
            normalized_df = copy(X_train)
            
            log_normalizer = FunctionTransformer(func=np.log)
            scaler = StandardScaler()
            
            preprocessor = ColumnTransformer([
                ('log_norm', log_normalizer, normalization_columns),  # Columns to log-normalize
                ('scale', scaler, list(X_train.columns))  # Columns to scale
            ])
            
            if quant == True:
                cd_first_quantile = np.quantile(normalized_df["characteristic_distance"], 0.25)
                cd_third_quantile = np.quantile(normalized_df["characteristic_distance"], 0.75)
                normalized_df["characteristic_distance"] = normalized_df["characteristic_distance"] + cd_first_quantile**2 / cd_third_quantile

                val_cd_first_quantile = np.quantile(X_val["characteristic_distance"], 0.25)
                val_cd_third_quantile = np.quantile(X_val["characteristic_distance"], 0.75)
                X_val["characteristic_distance"] = X_val["characteristic_distance"] + val_cd_first_quantile**2 / val_cd_third_quantile
                
            scaled_X_train = preprocessor.fit_transform(normalized_df)
            scaled_X_val = preprocessor.transform(X_val)
            
            # only for x_train
            scaled_resampled_X_train, scaled_resampled_y_train = pd.DataFrame(scaled_X_train), pd.Series(y_train)
            
            if data_imputation == 1:
                scaled_resampled_X_train, scaled_resampled_y_train = oversample_data(scaled_X_train, y_train)

            inner_cv = StratifiedKFold(n_splits=5)

            logging.info("NESTED_LOOP: %s", f"entering model fitting for {epoch_str}")
            
            # inside the grid search we are passing the scaled, transformed, resampled x_train. so there is a slight data leakage in the grid search process
            # but we cannot avoid it, since even if we shift this transformation down to the grid search and combine it with a proprocessor there, 
            # the same thing will happen as grid search process will not run the transformation on top of the training set leaving the hold out test set 
            
            logit_output = fit_logistic_model(
                x_train_df=scaled_resampled_X_train, y_train_df=scaled_resampled_y_train, X_val=scaled_X_val, Y_val=Y_val, inner_cv=inner_cv, epoch_str=epoch_str
            )
            dt_output = fit_dt_model(
                x_train_df=scaled_resampled_X_train, y_train_df=scaled_resampled_y_train, X_val=scaled_X_val, Y_val=Y_val, inner_cv=inner_cv, epoch_str=epoch_str
            )            
            rf_output = fit_rf_model(
                x_train_df=scaled_resampled_X_train, y_train_df=scaled_resampled_y_train, X_val=scaled_X_val, Y_val=Y_val, inner_cv=inner_cv, epoch_str=epoch_str
            )            
            xgb_output = fit_xgb_model(
                x_train_df=scaled_resampled_X_train, y_train_df=scaled_resampled_y_train, X_val=scaled_X_val, Y_val=Y_val, inner_cv=inner_cv, epoch_str=epoch_str
            )
            lgb_output = fit_lgb_model(
                x_train_df=scaled_resampled_X_train, y_train_df=scaled_resampled_y_train, X_val=scaled_X_val, Y_val=Y_val, inner_cv=inner_cv, epoch_str=epoch_str
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

        if not trial:
            wandb.init(project=wandb_project)
            wandb.alert(title="Error", text="Your run was interrupted by some exception.")
            wandb.finish()
    return loop_outputs

# @workflow
def main_wf():
    
    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    if not trial:
        wandb.init(project=wandb_project)
        wandb.alert(title="Started", text="Your run has started. Mark the time.")
        wandb.finish()

    loop_outputs = nested_loop()
    refitting_models(loop_outputs=loop_outputs)
    # loop_outputs >> refitt

    logging.info("MAIN_WF: %s", f"workflow finished.")
    return

# @task(container_image="istiyaksiddiquee/thesis-round-two:"+wandb_project)
def refitting_models(
    loop_outputs: list[CLFOutput]
) -> None:

    logging.info("REFITTING_MODELS: %s", "Model Refitting starts.")
    
    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    
    try:
            
        X_train_val, y_train_val = read_pickled_input_files(folder_path)
        
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
        
        if feature_selection == 1:
            X_train_val = X_train_val[X_train_val.columns[~X_train_val.columns.isin(removed_columns)]]
            
        normalized_df = copy(X_train_val)
        
        log_normalizer = FunctionTransformer(func=np.log)
        scaler = StandardScaler()
        
        preprocessor = ColumnTransformer([
            ('log_norm', log_normalizer, normalization_columns),  # Columns to log-normalize
            ('scale', scaler, list(X_train_val.columns))  # Columns to scale
        ])
        
        if quant == True:
            cd_first_quantile = np.quantile(normalized_df["characteristic_distance"], 0.25)
            cd_third_quantile = np.quantile(normalized_df["characteristic_distance"], 0.75)
            normalized_df["characteristic_distance"] = normalized_df["characteristic_distance"] + cd_first_quantile**2 / cd_third_quantile
            
        scaled_X_train_val = preprocessor.fit_transform(normalized_df)
        
        # only for x_train
        scaled_resampled_X_train_val, scaled_resampled_y_train_val = pd.DataFrame(scaled_X_train_val), pd.Series(y_train_val)
        
        if data_imputation == 1:
            scaled_resampled_X_train_val, scaled_resampled_y_train_val = oversample_data(scaled_X_train_val, y_train_val)
        
        logging.info("REFITTING_MODELS: %s", "data ready, initiating processing")
        
        stratified_dummy_cls = fit_dummy_classifier(scaled_resampled_X_train_val, scaled_resampled_y_train_val, "stratified")
        most_freq_dummy_cls = fit_dummy_classifier(scaled_resampled_X_train_val, scaled_resampled_y_train_val, "most_frequent")
        
        # train fit_default_rf_classifier with resampled scaled x y
        default_rf = fit_default_rf_classifier(scaled_resampled_X_train_val, scaled_resampled_y_train_val)
        
        if not trial:
            wandb.init(project=wandb_project, group="dummy", job_type="final")
            joblib.dump(stratified_dummy_cls, "stratified_dummy_cls.joblib")
            joblib.dump(most_freq_dummy_cls, "most_freq_dummy_cls.joblib")
            joblib.dump(default_rf, "default_rf.joblib")
            
            str_dum_artifact = wandb.Artifact(
                "Stratified-Dummy-Cls",
                type="model",
                description="trained stratified dummy model"
            )

            most_freq_dum_artifact = wandb.Artifact(
                "Most-Freq-Dummy-Cls",
                type="model",
                description="trained most freq dummy model"
            )
            
            default_rf_artifact = wandb.Artifact(
                "Default-RF-Cls",
                type="model",
                description="trained default rf model"
            )
            
            str_dum_artifact.add_file("stratified_dummy_cls.joblib")
            most_freq_dum_artifact.add_file("most_freq_dummy_cls.joblib")
            default_rf_artifact.add_file("default_rf.joblib")
            
            wandb.log_artifact(str_dum_artifact)
            wandb.log_artifact(most_freq_dum_artifact)
            wandb.log_artifact(default_rf_artifact)
            wandb.finish()

        print('----------------------------------------------------------------------------')
        print('Final Fit Starts')
        print('----------------------------------------------------------------------------')

        if trained_logit_model != None:
            # store logit model
            logging.info("REFITTING_MODELS: %s", "processing logit model.")
            logistic = LogisticRegression(**trained_logit_model)
            refit_logit = logistic.fit(scaled_resampled_X_train_val, scaled_resampled_y_train_val)
            
            if not trial:
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
            refit_dt = dt.fit(scaled_resampled_X_train_val, scaled_resampled_y_train_val)

            if not trial:
                
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
            refit_rf = rf.fit(scaled_resampled_X_train_val, scaled_resampled_y_train_val)

            if not trial:
                
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
            refit_xgb = xgboost.fit(scaled_resampled_X_train_val, scaled_resampled_y_train_val)

            if not trial:
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
            refit_lgb = lgb_model.fit(scaled_resampled_X_train_val, scaled_resampled_y_train_val)
            
            if not trial:

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


        print('----------------------------------------------------------------------------')
        print('Final Fit Ends')
        print('----------------------------------------------------------------------------')
                
        logging.info("REFITTING_MODELS: %s", "process complete, returning to base.")
        
        if not trial:        
            wandb.init(project=wandb_project)
            wandb.alert(title="Complete", text="Your run is complete. Check the board.")
            wandb.finish()
        
    except Exception as error:
        logging.error("NESTED_LOOP: %s", "ERROR: some error happened, could not finish.")
        logging.error("NESTED_LOOP: %s", error)

        if not trial:
            wandb.init(project=wandb_project)
            wandb.alert(title="Error", text="Your run was interrupted by some exception inside model fitting.")
            wandb.finish()

    return

def oversample_data(X: pd.Series, y: pd.Series):
    
    oversampler = sv.polynom_fit_SMOTE_poly()
    X_samp, y_samp = oversampler.sample(X, y)
    X_samp, y_samp = pd.DataFrame(X_samp), pd.Series(y_samp)

    return X_samp, y_samp

def fit_dummy_classifier(x: pd.Series, y: pd.Series, strategy: str):
    
    dummy_clf = DummyClassifier(strategy=strategy)
    dummy_clf.fit(x, y)
    return dummy_clf

def fit_default_rf_classifier(x: pd.Series, y: pd.Series):
    
    rf_model = RandomForestClassifier()
    rf_model.fit(x, y)
    return rf_model

# @task(container_image="istiyaksiddiquee/thesis-round-two:"+wandb_project)
def fit_logistic_model(
    x_train_df: pd.Series, y_train_df: pd.Series, X_val: pd.Series, Y_val: pd.Series, inner_cv: StratifiedKFold, epoch_str: str
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
        
        logit_grid = None
        
        if ds == 1:
            logit_grid = {
                "penalty": ["l1", "l2", "elasticnet"],
                "dual": [True, False],
                "C": [_ for _ in range(1, 11, 1)],
                "fit_intercept": [True, False],
                "max_iter": [500],
                "solver": ["lbfgs", "newton-cg", "sag", "saga"],
                "n_jobs": [-1],
            }
        else:
            logit_grid = {
                "penalty": ["l1", "l2", "elasticnet"],
                "dual": [True, False],
                "C": [_ for _ in range(1, 11, 1)],
                "fit_intercept": [True, False],
                "max_iter": [500],
                "solver": ["lbfgs", "newton-cg", "sag", "saga"],
                "n_jobs": [-1],
            }
        
        if trial == True:
            logit_grid = {"penalty": ["l2"]}
        
        logit_model = LogisticRegression()

        clf = HalvingGridSearchCV(
            estimator=logit_model,
            factor=4,
            cv=inner_cv,
            refit=True,
            param_grid=logit_grid,
            scoring="average_precision",
            verbose=3,
            n_jobs=-1,
        )

        logit_result = clf.fit(x_train_df, y_train_df)

    except Exception as error:
        logging.error("FIT_LOGIT_MODEL: %s", "Could not fit Logistic model.")
        logging.error("FIT_LOGIT_MODEL: %s", f"An exception occurred: {error}")

    if logit_result != None:
        
        logit_model = logit_result.best_estimator_
        logit_best_grid_param = logit_model.get_params()

        logit_Y_pred = logit_model.predict(X_val)
        logit_Y_pred_proba = logit_model.predict_proba(X_val)
        
        logit_custom_score = get_all_scores(Y_val, logit_Y_pred, logit_Y_pred_proba[:, 1])
        logit_score = logit_custom_score[default_metric]
        
        if not trial:
            wandb.init(project=wandb_project, group="logit", job_type=epoch_str)
            
            wandb.log(logit_custom_score)
            
            wandb.log(
                {
                    "best_parameters": logit_result.best_params_,
                    "best_score": logit_result.best_score_
                }
            )
            
            logit_cv_result_artifact = wandb.Artifact(
                "logit_cv_result_artifact_" + epoch_str, 
                type="cv_result"
            )
            
            logit_cv_file_name = f"./logit_cv_result_{epoch_str}.csv"
            logit_cv_result_df = pd.DataFrame(logit_result.cv_results_)
            logit_cv_result_df.to_csv(logit_cv_file_name)
            logit_cv_result_artifact.add_file(logit_cv_file_name)
            wandb.log_artifact(logit_cv_result_artifact)

            # store entire grid search clf object
            joblib.dump(logit_result, f"logit_gscv_{epoch_str}.joblib")
            logit_gscv = wandb.Artifact(
                "Logistic-GSCV",
                type="gscv",
                description="trained grid search object",
                metadata={
                    "best_parameters": logit_result.best_params_,
                    "best_score": logit_result.best_score_,
                    "epoch": epoch_str,
                },
            )

            logit_gscv.add_file(f"logit_gscv_{epoch_str}.joblib")
            wandb.log_artifact(logit_gscv)
            
            wandb.finish()
        

    logging.info("FIT_LOGIT_MODEL: %s", "logit run completed.")
    clf_output = CLFOutput(logit_best_grid_param, logit_score)

    return clf_output

# @task(container_image="istiyaksiddiquee/thesis-round-two:"+wandb_project)
def fit_dt_model(x_train_df: pd.Series, y_train_df: pd.Series, X_val: pd.Series, Y_val: pd.Series, inner_cv: StratifiedKFold, epoch_str: str) -> CLFOutput:
    # Decision Tree

    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    logging.info("FIT_DT_MODEL: %s", f"dt run scheduled for {epoch_str}.")

    dt_result = None
    dt_score = None
    dt_best_grid_param = None
    try:

        dt_grid = None
        if ds == 1:
            dt_grid = {
                "max_depth": [_ for _ in np.arange(1, 40 + 10, 10)],
                # "max_features": ['sqrt', 'log2', None],
                "min_samples_split": [_ for _ in np.arange(1, 40 + 5, 5)],
                "min_samples_leaf": [_ for _ in np.arange(1, 30 + 5, 5)],
                "min_impurity_decrease": [_ for _ in np.arange(0.005, 0.1+0.02, 0.02)]
            }
        else:
            dt_grid = {
                "max_depth": [_ for _ in np.arange(1, 20 + 5, 5)],
                # "max_features": ['sqrt', 'log2', None],
                "min_samples_leaf": [_ for _ in np.arange(1, 15 + 3, 3)],
                "min_samples_split": [_ for _ in np.arange(1, 10 + 1, 1)],
                "min_impurity_decrease": [_ for _ in np.arange(0.005, 0.1+0.02, 0.02)]
            }
        
        if trial == True:
            dt_grid = {"criterion": ["gini"]}
        
        dt_clf = DecisionTreeClassifier(random_state=random_state)

        clf = HalvingGridSearchCV(
            estimator=dt_clf,
            factor=4,
            cv=inner_cv,
            refit=True,
            param_grid=dt_grid,
            scoring="average_precision",
            verbose=3,
            n_jobs=-1,
        )

        dt_result = clf.fit(x_train_df, y_train_df)

    except Exception as error:
        logging.error("FIT_DT_MODEL: %s", "Could not fit Decision Tree model")
        logging.error("FIT_DT_MODEL: %s", f"An exception occurred: {error}")

    if dt_result != None:
        dt_model = dt_result.best_estimator_
        dt_best_grid_param = dt_model.get_params()
        
        dt_Y_pred = dt_model.predict(X_val)
        dt_Y_pred_proba = dt_model.predict_proba(X_val)
        
        dt_custom_score = get_all_scores(Y_val, dt_Y_pred, dt_Y_pred_proba[:, 1])
        dt_score = dt_custom_score[default_metric]
        
        if not trial:
            wandb.init(project=wandb_project, group="dt", job_type=epoch_str)
            
            wandb.log(dt_custom_score)
            
            wandb.log(
                {
                    "best_parameters": dt_result.best_params_,
                    "best_score": dt_result.best_score_
                }
            )

            dt_cv_result_df = pd.DataFrame(dt_result.cv_results_)
            dt_cv_result_artifact = wandb.Artifact(
                "dt_cv_result_artifact_" + epoch_str, 
                type="cv_result"
            )
            
            dt_cv_file_name = f"./dt_cv_result_{epoch_str}.csv"
            dt_cv_result_df = pd.DataFrame(dt_result.cv_results_)
            dt_cv_result_df.to_csv(dt_cv_file_name)
            dt_cv_result_artifact.add_file(dt_cv_file_name)
            wandb.log_artifact(dt_cv_result_artifact)

            # store entire grid search clf object
            joblib.dump(dt_result, f"dt_gscv_{epoch_str}.joblib")
            dt_gscv = wandb.Artifact(
                "DT-GSCV",
                type="gscv",
                description="trained grid search object",
                metadata={
                    "best_parameters": dt_result.best_params_,
                    "best_score": dt_result.best_score_,
                    "epoch": epoch_str,
                },
            )

            dt_gscv.add_file(f"dt_gscv_{epoch_str}.joblib")
            wandb.log_artifact(dt_gscv)
            
            wandb.finish()
        

    logging.info("FIT_DT_MODEL: %s", "dt run completed.")
    
    clf_output = CLFOutput(dt_best_grid_param, dt_score)

    return clf_output

# @task(container_image="istiyaksiddiquee/thesis-round-two:"+wandb_project)
def fit_rf_model(x_train_df: pd.Series, y_train_df: pd.Series, X_val: pd.Series, Y_val: pd.Series, inner_cv: StratifiedKFold, epoch_str: str) -> CLFOutput:
    # Random Forest

    os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
    os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    logging.info("FIT_RF_MODEL: %s", f"rf run scheduled for {epoch_str}.")

    rf_result = None
    rf_score = None
    rf_best_grid_param = None

    try:
        
        rf_grid = None 
        if ds == 1:
            rf_grid = {
                    "max_depth": [_ for _ in np.arange(1, 50 + 15, 15)],
                    "n_estimators": [_ for _ in np.arange(50, 500 + 100, 100)],
                    # "max_features": ['sqrt', 'log2', None], 
                    "min_samples_leaf": [_ for _ in np.arange(1, 30 + 10, 10)],
                    "min_samples_split": [_ for _ in np.arange(2, 30 + 5, 5)],
                    "criterion": ["entropy"],
                    "min_impurity_decrease": [_ for _ in np.arange(0.005, 0.1+0.04, 0.04)]
            }
        else:
            rf_grid = {
                    "max_depth": [_ for _ in np.arange(1, 20 + 5, 5)],
                    "n_estimators": [_ for _ in np.arange(50, 300 + 100, 100)],
                    # "max_features": ['sqrt', 'log2', None], 
                    "min_samples_leaf": [_ for _ in np.arange(1, 20 + 5, 5)],
                    "min_samples_split": [_ for _ in np.arange(2, 20 + 5, 5)],
                    # "criterion": ["entropy"],
                    "min_impurity_decrease": [_ for _ in np.arange(0.005, 0.1+0.02, 0.02)]
            }
        
        if trial == True:
            rf_grid = {"criterion": ["gini"]}
            
        rf_model = RandomForestClassifier()

        clf = HalvingGridSearchCV(
            estimator=rf_model,
            factor=4,
            cv=inner_cv,
            refit=True,
            param_grid=rf_grid,
            scoring="average_precision",
            verbose=3,
            n_jobs=-1,
        )

        rf_result = clf.fit(x_train_df, y_train_df)

    except Exception as error:
        logging.error("FIT_RF_MODEL: %s", f"Could not fit Random Forest model.")
        logging.error("FIT_RF_MODEL: %s", f"An exception occurred: {error}")

    if rf_result != None:
        rf_model = rf_result.best_estimator_
        rf_best_grid_param = rf_model.get_params()
        
        rf_Y_pred = rf_model.predict(X_val)
        rf_Y_pred_proba = rf_model.predict_proba(X_val)
        
        rf_custom_score = get_all_scores(Y_val, rf_Y_pred, rf_Y_pred_proba[:, 1])
        rf_score = rf_custom_score[default_metric]
        
        if not trial:
            wandb.init(project=wandb_project, group="rf", job_type=epoch_str)
            
            wandb.log(rf_custom_score)
            
            wandb.log(
                {
                    "best_parameters": rf_result.best_params_,
                    "best_score": rf_result.best_score_
                }
            )

            rf_cv_result_df = pd.DataFrame(rf_result.cv_results_)
            
            rf_cv_result_artifact = wandb.Artifact(
                "rf_cv_result_artifact_" + epoch_str, 
                type="cv_result"
            )
            
            rf_cv_file_name = f"./rf_cv_result_{epoch_str}.csv"
            rf_cv_result_df = pd.DataFrame(rf_result.cv_results_)
            rf_cv_result_df.to_csv(rf_cv_file_name)
            rf_cv_result_artifact.add_file(rf_cv_file_name)
            wandb.log_artifact(rf_cv_result_artifact)

            # store entire grid search clf object
            joblib.dump(rf_result, f"rf_gscv_{epoch_str}.joblib")
            rf_gscv = wandb.Artifact(
                "RF-GSCV",
                type="gscv",
                description="trained grid search object",
                metadata={
                    "best_parameters": rf_result.best_params_,
                    "best_score": rf_result.best_score_,
                    "epoch": epoch_str,
                },
            )

            rf_gscv.add_file(f"rf_gscv_{epoch_str}.joblib")
            wandb.log_artifact(rf_gscv)
            
            wandb.finish()

    logging.info("FIT_RF_MODEL: %s", "rf run completed.")

    clf_output = CLFOutput(rf_best_grid_param, rf_score)

    return clf_output

# @task(container_image="istiyaksiddiquee/thesis-round-two:"+wandb_project)
def fit_xgb_model(x_train_df: pd.Series, y_train_df: pd.Series, X_val: pd.Series, Y_val: pd.Series, inner_cv: StratifiedKFold, epoch_str: str) -> CLFOutput:
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
        
        xgb_grid = None
        if ds == 1:
            xgb_grid = {   
                    # "gamma": [_ for _ in np.arange(0.1, 1+0.1, 0.1)],
                    # "subsample": [_ for _ in np.arange(0.2, 1 + 0.2, 0.2)],
                    "max_depth": [_ for _ in np.arange(1, 30 + 10, 10)],
                    # "reg_alpha": [_ for _ in np.arange(1, 10 + 2, 2)],
                    "reg_lambda": [_ for _ in np.arange(1, 10 + 2, 2)],
                    "n_estimators": [_ for _ in np.arange(100, 250+50, 50)],
                    "learning_rate": [_ for _ in np.arange(0.1, 1 + 0.2, 0.2)],
                    # "colsample_bytree": [_ for _ in np.arange(0.2, 0.9+0.1, 0.1)],
                    "min_child_weight": [_ for _ in np.arange(1, 7+2, 2)],
            }
        else:
            xgb_grid = {   
                # "gamma": [_ for _ in np.arange(0.1, 1+0.1, 0.1)],
                # "subsample": [_ for _ in np.arange(0.2, 1 + 0.2, 0.2)],
                "max_depth": [_ for _ in np.arange(1, 10 + 2, 2)],
                # "reg_alpha": [_ for _ in np.arange(1, 10 + 2, 2)],
                "reg_lambda": [_ for _ in np.arange(1, 10 + 2, 2)],
                "n_estimators": [_ for _ in np.arange(100, 300+100, 100)],
                "learning_rate": [_ for _ in np.arange(0.1, 1 + 0.2, 0.2)],
                "min_child_weight": [_ for _ in np.arange(1, 7+2, 2)],
            }


        if trial == True:
            xgb_grid = {"learning_rate": [0.1]}
            
        xgb_model = xgb.XGBClassifier(objective="binary:hinge", nthread=4, seed=random_state)

        clf = HalvingGridSearchCV(
            estimator=xgb_model,
            factor=4,
            cv=inner_cv,
            refit=True,
            param_grid=xgb_grid,
            scoring="average_precision",
            verbose=3,
            n_jobs=-1,
        )

        xgb_result = clf.fit(x_train_df, y_train_df)

    except Exception as error:
        logging.error("FIT_XGB_MODEL: %s", "Could not fit XGB model")
        logging.error("FIT_XGB_MODEL: %s", f"An exception occurred: {error}")

    if xgb_result != None:
        xgb_model = xgb_result.best_estimator_
        xgb_best_grid_param = xgb_model.get_params()
        
        xgb_Y_pred = xgb_model.predict(X_val)
        xgb_Y_pred_proba = xgb_model.predict_proba(X_val)
        
        xgb_custom_score = get_all_scores(Y_val, xgb_Y_pred, xgb_Y_pred_proba[:, 1])
        xgb_score = xgb_custom_score[default_metric]
        
        if not trial:
            try:
                wandb.init(project=wandb_project, group="xgb", job_type=epoch_str)
                
                wandb.log(xgb_custom_score)
                
                wandb.log(
                    {
                        "best_parameters": xgb_result.best_params_,
                        "best_score": xgb_result.best_score_
                    }
                )
                
                xgb_cv_result_artifact = wandb.Artifact(
                    "xgb_cv_result_artifact_" + epoch_str, 
                    type="cv_result"
                )
                
                xgb_cv_file_name = f"./xgb_cv_result_{epoch_str}.csv"
                xgb_cv_result_df = pd.DataFrame(xgb_result.cv_results_)
                xgb_cv_result_df.to_csv(xgb_cv_file_name)
                xgb_cv_result_artifact.add_file(xgb_cv_file_name)
                wandb.log_artifact(xgb_cv_result_artifact)
                
                # store entire grid search clf object
                joblib.dump(xgb_result, f"xgb_gscv_{epoch_str}.joblib")
                xgb_gscv = wandb.Artifact(
                    "XGB-GSCV",
                    type="gscv",
                    description="trained grid search object",
                    metadata={
                        "best_parameters": xgb_result.best_params_,
                        "best_score": xgb_result.best_score_,
                        "epoch": epoch_str,
                    },
                )

                xgb_gscv.add_file(f"xgb_gscv_{epoch_str}.joblib")
                wandb.log_artifact(xgb_gscv)
                
                wandb.finish()
                
            except Exception as e:
                logging.error("FIT_XGB_MODEL: %s", "Exception happened inside xgb result processor.")
                logging.error("FIT_XGB_MODEL: %s", e)

    logging.info("FIT_XGB_MODEL: %s", "xgb run completed.")
    clf_output = CLFOutput(xgb_best_grid_param, xgb_score)

    return clf_output

# @task(container_image="istiyaksiddiquee/thesis-round-two:"+wandb_project)
def fit_lgb_model(x_train_df: pd.Series, y_train_df: pd.Series, X_val: pd.Series, Y_val: pd.Series, inner_cv: StratifiedKFold, epoch_str: str) -> CLFOutput:
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
        
        lgb_grid = None
        if ds == 1:
            lgb_grid = {
                    "max_depth": [_ for _ in np.arange(1, 10+3, 3)],
                    # "lambda_l1": [_ for _ in np.arange(1, 10+2, 2)],
                    "lambda_l2": [_ for _ in np.arange(1, 10+3, 3)],
                    "num_leaves": [_ for _ in np.arange(50, 80+15, 15)],
                    "learning_rate": [_ for _ in np.arange(0.1, 1+0.3, 0.3)],
                    "min_data_in_leaf": [_ for _ in np.arange(50, 250+50, 50)],
                    "min_gain_to_split": [_ for _ in np.arange(0.1, 1+0.3, 0.3)],
            }
        else:
            lgb_grid = {
                "max_depth": [_ for _ in np.arange(1, 10+3, 3)],
                # "lambda_l1": [_ for _ in np.arange(1, 10+2, 2)],
                "lambda_l2": [_ for _ in np.arange(1, 10+3, 3)],
                "num_leaves": [_ for _ in np.arange(50, 80+15, 15)],
                "learning_rate": [_ for _ in np.arange(0.1, 1+0.3, 0.3)],
                "min_data_in_leaf": [_ for _ in np.arange(10, 30+15, 15)],
                "min_gain_to_split": [_ for _ in np.arange(0.1, 1+0.2, 0.2)],
            }
        lgb_model = lgb.LGBMClassifier(objective="binary", random_state=42)

        if trial == True:
            lgb_grid = {"num_leaves": [31]}
        
        clf = HalvingGridSearchCV(
            estimator=lgb_model,
            factor=4,
            cv=inner_cv,
            refit=True,
            param_grid=lgb_grid,
            scoring="average_precision",
            verbose=3,
            n_jobs=-1,
        )

        lgb_result = clf.fit(x_train_df, y_train_df)

    except Exception as error:
        logging.error("FIT_LGB_MODEL: %s", f"Could not fit XGB model")
        logging.error("FIT_LGB_MODEL: %s", f"An exception occurred: {error}")

    if lgb_result != None:
        
        lgb_model = lgb_result.best_estimator_
        lgb_best_grid_param = lgb_model.get_params()

        lgb_Y_pred = lgb_model.predict(X_val)
        lgb_Y_pred_proba = lgb_model.predict_proba(X_val)
        
        lgb_custom_score = get_all_scores(Y_val, lgb_Y_pred, lgb_Y_pred_proba[:, 1])
        lgb_score = lgb_custom_score[default_metric]
            
        if not trial:
            wandb.init(project=wandb_project, group="lgb", job_type=epoch_str)
            
            wandb.log(lgb_custom_score)
            
            wandb.log(
                {
                    "best_parameters": lgb_result.best_params_,
                    "best_score": lgb_result.best_score_
                }
            )

            lgb_cv_result_artifact = wandb.Artifact(
                "lgb_cv_result_artifact_" + epoch_str, 
                type="cv_result"
            )
            
            lgb_cv_file_name = f"./lgb_cv_result_{epoch_str}.csv"
            lgb_cv_result_df = pd.DataFrame(lgb_result.cv_results_)
            lgb_cv_result_df.to_csv(lgb_cv_file_name)
            lgb_cv_result_artifact.add_file(lgb_cv_file_name)
            wandb.log_artifact(lgb_cv_result_artifact)

            # store entire grid search clf object
            joblib.dump(lgb_result, f"lgb_gscv_{epoch_str}.joblib")
            lgb_gscv = wandb.Artifact(
                "LGB-GSCV",
                type="gscv",
                description="trained grid search object",
                metadata={
                    "best_parameters": lgb_result.best_params_,
                    "best_score": lgb_result.best_score_,
                    "epoch": epoch_str,
                },
            )

            lgb_gscv.add_file(f"lgb_gscv_{epoch_str}.joblib")
            wandb.log_artifact(lgb_gscv)

            wandb.finish()

    logging.info("FIT_LGB_MODEL: %s", "lgb run completed.")
    clf_output = CLFOutput(lgb_best_grid_param, lgb_score)

    return clf_output

if __name__ == "__main__":
    main_wf()