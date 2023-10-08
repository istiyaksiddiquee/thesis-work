import os
import wandb
from wandb.sklearn import calculate

os.environ["WANDB_API_KEY"] = "b21f4406f3966154b12e98de3bef934216952a54"
os.environ["WANDB_ENTITY"]="istiyaksiddiquee"
os.environ["WANDB_PROJECT"]="test-1"

wandb.init(project="test-1")

with wandb.init() as run:
    run.log({"a": 1, "b": 2})


def imbalanced_performance_summary(model= None, X = None, ):
    
    metrics = {name: utils.round_2(metric) for name, metric in metrics.items()}
    calculate.make_table()