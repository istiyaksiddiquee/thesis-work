from sklearn import svm
from sklearn import datasets
from joblib import dump, load
import os
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
os.environ["WANDB_ENTITY"] = "istiyaksiddiquee"
os.environ["WANDB_PROJECT"] = "thesis"

# run = wandb.init(project="test-1", job_type="demo_run_2")

wandb.init(project="RQ2RUN1")

model_at = wandb.use_artifact("Logistic-Model:latest")
model_dir = model_at.download()
# print("model: ", os.path.join(model_dir, "clf.joblib"))
model = load(os.path.join(model_dir, "logit.joblib"))

with open("./x_test.pickle", "rb") as file:
    X_test = pickle.load(file)

with open("./y_test.pickle", "rb") as file:
    y_test = pickle.load(file)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))

# # # Finish the run
# run.finish()
