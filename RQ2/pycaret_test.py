# # create a custom function
# c = compare_models(include=['lr','ridge','br','et','rf','ada',ExponentialRegressor()],sort='Confidence')


import pycaret
from pycaret.classification import *
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import chardet
from sklearn.model_selection import train_test_split


def get_encoding(file_path):

    with open(file_path, 'rb') as f:
        data = f.read(10000)
    return chardet.detect(data).get("encoding")


def b_accuracy(y, y_pred):
    return balanced_accuracy_score(y, y_pred)
# add it to PyCaret


false_file_path = './rq2_vosoughi_False_features.csv'
true_file_path = './rq2_vosoughi_True_features.csv'
feature_names = [
    'depth',
    'size',
    'max_breadth',
    'virality',
    'strongly_cc',
    'weakly_cc',
    'size_of_scc',
    'avg_cluster_coef',
    'density',
    'layer_ratio',
    'structural_heterogeneity',
    'characteristic_distance',
    'veracity'
]

false_df = pd.read_csv(false_file_path, encoding=get_encoding(false_file_path))
true_df = pd.read_csv(true_file_path, encoding=get_encoding(true_file_path))
df = pd.concat([false_df, true_df])
df = df[df.depth != 0]
print(df.shape)

X_train, X_test, y_train, y_test = train_test_split(df, test_size=0.33, random_state=42)

classf = setup(data=df[feature_names], target='veracity',
               train_size=0.8, n_jobs=6, normalize=True, session_id=3934)
add_metric(id='ba', name="Balanced Accuracy", score_func=b_accuracy)
compare_models(sort='Balanced Accuracy')
