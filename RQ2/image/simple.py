
import logging
import pickle
import pandas as pd
import numpy as np
import smote_variants as sv


def read_pickled_input_files(file_path: str):
    if file_path == None:
        logging.error("READING_FILES: %s", "file path must be provided.")
        return

    X_train_val = None
    y_train_val = None
    X_test = None
    y_test = None

    with open("./x_train_val.pickle", "rb") as file:
        X_train_val = pickle.load(file)

    with open("./y_train_val.pickle", "rb") as file:
        y_train_val = pickle.load(file)

    with open("./x_test.pickle", "rb") as file:
        X_test = pickle.load(file)

    with open("./y_test.pickle", "rb") as file:
        y_test = pickle.load(file)

    return X_train_val, X_test, y_train_val, y_test


def oversample_data(X: pd.Series, y: pd.Series):
    oversampler = sv.polynom_fit_SMOTE_poly()
    X_samp, y_samp = oversampler.sample(X, y)

    # smotetomek = SMOTETomek(
    #     smote=SMOTE(sampling_strategy="all"),
    #     tomek=TomekLinks(
    #         sampling_strategy="majority",
    #     ),
    #     random_state=random_state,
    # )
    # X_samp, y_samp = smotetomek.fit_resample(X, y)
    X_samp, y_samp = pd.DataFrame(X_samp), pd.Series(y_samp)
    return X_samp, y_samp


X_train_val, X_test, y_train_val, y_test = read_pickled_input_files(".")
resampled_x, resampled_y = oversample_data(X_train_val.to_numpy(), y_train_val.to_numpy())
print(resampled_x.shape, resampled_y.shape)