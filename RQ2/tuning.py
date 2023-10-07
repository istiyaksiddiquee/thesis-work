import os

import pandas as pd
import sklearn.datasets
import sklearn.metrics
import xgboost as xgb
from ray import tune
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split


def load_dataset(data_dir='./'):
    seed = 7
    test_size = 0.25
    vosoughi_false_df = pd.read_csv('./rq2_vosoughi_False_features.csv', index_col=0, sep=',', encoding='utf-8')
    vosoughi_true_df = pd.read_csv('./rq2_vosoughi_True_features.csv', index_col=0, sep=',', encoding='utf-8')
    vosoughi_df = pd.concat([vosoughi_false_df, vosoughi_true_df])
    vosoughi_df.characteristic_distance.replace(to_replace=-99999, value=float("NaN"), inplace=True)
    vosoughi_df.structural_heterogeneity.replace(to_replace=0.0, value=float("NaN"), inplace=True)
    X = vosoughi_df.iloc[:,3:15]
    Y = vosoughi_df.iloc[:,15]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test    

def train_breast_cancer(config: dict):
     
     # Split into train and test set
     train_x, test_x, train_y, test_y = load_dataset()
     # Build input matrices for XGBoost
     train_set = xgb.DMatrix(train_x, label=train_y)
     test_set = xgb.DMatrix(test_x, label=test_y)
     # Train the classifier, using the Tune callback
     xgb.train(
         config,
         train_set,
         evals=[(test_set, "eval")],
         verbose_eval=False,
         callbacks=[TuneReportCheckpointCallback(filename="model.xgb")],
     )


def get_best_model_checkpoint(results):
    best_bst = xgb.Booster()
    best_result = results.get_best_result()
    
    with best_result.checkpoint.as_directory() as best_checkpoint_dir:
        best_bst.load_model(os.path.join(best_checkpoint_dir, "model.xgb"))
    accuracy = 1.0 - best_result.metrics["eval-error"]
    print(f"Best model parameters: {best_result.config}")
    print(f"Best model total accuracy: {accuracy:.4f}")
    return best_bst


def tune_xgboost():
    search_space = {
        # You can mix constants with search space objects.
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error", "auc"],
        "max_depth": tune.randint(1, 10),
        "min_child_weight": tune.choice([1, 2, 3]),
        "subsample": tune.uniform(0.5, 1.0),
        "eta": tune.loguniform(1e-4, 1e-1),
        "gamma": tune.randint(1, 10),
        "alpha": tune.randint(1, 5),
        "lambda": tune.randint(1, 5),
        "grow_policy": ["depthwise", "lossguide"], 
        "tree_method": ["auto", "exact", "approx", "hist"],
        "sampling_method": ["uniform", "gradient_based"]
    }
    # This will enable aggressive early stopping of bad trials.
    scheduler = ASHAScheduler(
        max_t=10, grace_period=1, reduction_factor=2  # 10 training iterations
    )
    
    tuner = tune.Tuner(
        train_breast_cancer,
        tune_config=tune.TuneConfig(
            metric="eval-logloss",
            mode="min",
            scheduler=scheduler,
            num_samples=10,
        ),
        param_space=search_space,
    )
    results = tuner.fit()

    return results


if __name__ == "__main__":
    
    results = tune_xgboost()

    # Load the best model checkpoint.
    best_bst = get_best_model_checkpoint(results)

    # You could now do further predictions with
    # best_bst.predict(...)
