from sklearn import svm
from sklearn import datasets
from joblib import dump, load
import os
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
os.environ["WANDB_PROJECT"] = "test-1"

# run = wandb.init(project="test-1", job_type="demo_run_2")

wandb.init(project="thesis")
wandb.alert(title="High Loss", text="Loss is increasing rapidly")

# with wandb.init() as run:
#     run.log({"a": 1, "b": 2})


# def imbalanced_performance_summary(model= None, X = None, ):

#     metrics = {name: utils.round_2(metric) for name, metric in metrics.items()}
#     calculate.make_table()


# clf = svm.SVC()
X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=7
)

# clf.fit(X_train, y_train)


# dump(clf, "clf.joblib")


# # Create a new artifact with metadata
# params = {"max_depth": 5, "n_estimators": 100}  # replace with your model's parameters
# metrics = {
#     "accuracy": 0.95,
#     "precision": 0.96,
#     "recall": 0.94,
# }  # replace with your model's metrics
# artifact = wandb.Artifact(
#     "recommender_model",
#     type="model",
#     description="Random forest classifier for book recommendations",
#     metadata={"parameters": params, "metrics": metrics},
# )


# # # # Add the model file to the artifact
# artifact.add_file('clf.joblib') # or 'clf.pkl'


# # # # Save the artifact
# run.log_artifact(artifact)

# model_at = run.use_artifact("recommender_model:latest")
# model_dir = model_at.download()
# print("model: ", os.path.join(model_dir, "clf.joblib"))
# model = load(os.path.join(model_dir, "clf.joblib"))

# y_pred = model.predict(X_test)

# print(accuracy_score(y_test, y_pred))

# # # Finish the run
# run.finish()
