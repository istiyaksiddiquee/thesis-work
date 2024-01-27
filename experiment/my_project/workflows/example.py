"""A simple Flyte example."""

import typing
import numpy as np
from flytekit import task, workflow
from sklearn import datasets
from sklearn.linear_model import LogisticRegression


@workflow
def wf() -> None:

  print("task starts")
  iris = datasets.load_iris()
  X = iris.data[:, :2]
  Y = iris.target
  fitting_task(X=X, Y=Y)
  print("task returned")
  return

@task(container_image="istiyaksiddiquee/dummy-test-for-flyte:test01")
def fitting_task(
   X: np.ndarray,
   Y: np.ndarray
) -> None:
  print("inside task")
  logreg = LogisticRegression(C=1e5)
  logreg.fit(X, Y)
  return

if __name__ == "__main__":
    wf()
