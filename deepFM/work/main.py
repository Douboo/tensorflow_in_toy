
import os
import sys

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold

sys.path.append("../..")
print(sys.path)
import config
from metrics import gini_norm
from DataReader import FeatureDictionary, DataParser
from deepFM import DeepFM

import shutil
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from utils.common import timeit, choose_gpu


gini_scorer = make_scorer(gini_norm, greater_is_better=True, needs_proba=True)


def load_data():

    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTest = pd.read_csv(config.TEST_FILE)

    def preprocess(df):
        cols = [c for c in df.columns if c not in ["id", "target"]]
        df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)
        df["ps_car_13_x_ps_reg_03"] = df["ps_car_13"] * df["ps_reg_03"]
        return df

    dfTrain = preprocess(dfTrain)
    dfTest = preprocess(dfTest)

    cols = [c for c in dfTrain.columns if c not in ["id", "target"]]
    cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    X_train = dfTrain[cols].values
    y_train = dfTrain["target"].values
    X_test = dfTest[cols].values
    ids_test = dfTest["id"].values
    cat_features_indices = [i for i,c in enumerate(cols) if c in config.CATEGORICAL_COLS]

    return dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices


def run_base_model_dfm(dfTrain, dfTest, folds, prefix, dfm_params):
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                           numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS)
    data_parser = DataParser(feat_dict=fd)
    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)

    dfm_params["feature_size"] = fd.feat_dim
    dfm_params["field_size"] = len(Xi_train[0])


    _get = lambda x, l: [x[i] for i in l]
    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)

        Xi_train_ = np.array(Xi_train_, dtype='int32')
        Xv_train_ = np.array(Xv_train_, dtype='float32')
        y_train_ = np.array(y_train_, dtype=np.int8)
        Xi_valid_ = np.array(Xi_valid_, dtype='int32')
        Xv_valid_ = np.array(Xv_valid_, dtype='float32')
        y_valid_ = np.array(y_valid_, dtype=np.int8)


        dfm = DeepFM(**dfm_params).build_model()
        dfm.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=[keras.metrics.AUC(name='auc')])
        print(dfm.summary())

        checkpoint_dir = "../checkpoints/{}_cpt_" + str(i)
        log_dir = "../logs/{}_train_logs_" + str(i)

        checkpoint_dir = checkpoint_dir.format(prefix)
        log_dir = log_dir.format(prefix)

        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        shutil.rmtree(log_dir, ignore_errors=True)
        os.makedirs(log_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "weights.hdf5")
        callbacks = [
            ModelCheckpoint(checkpoint_path,
                            monitor="val_loss",
                            save_best_only=True),
            EarlyStopping(patience=5, monitor="val_loss"),
            TensorBoard(log_dir=log_dir)
        ]

        dfm.fit((Xi_train_, Xv_train_), y_train_,
                epochs=50,
                # epochs=1,
                batch_size=64,
                validation_data=((Xi_valid_, Xv_valid_), y_valid_),
                verbose=2, callbacks=callbacks
                )

@timeit
def main():
    choose_gpu()

    # load data
    dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = load_data()

    # folds
    folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS,
                                 # n_splits=2,
                                 shuffle=True,
                                 random_state=config.RANDOM_SEED).split(X_train, y_train))


    # ------------------ DeepFM Model ------------------
    # params
    dfm_params = {
        "use_fm": True,
        "use_deep": True,
        "embedding_size": 8,
        "deep_layers": [32, 32]
    }
    run_base_model_dfm(dfTrain, dfTest, folds, "deepFM_bn", dfm_params)

    # ------------------ FM Model ------------------
    fm_params = dfm_params.copy()
    fm_params["use_deep"] = False
    run_base_model_dfm(dfTrain, dfTest, folds, "fm", fm_params)


    # ------------------ DNN Model ------------------
    dnn_params = dfm_params.copy()
    dnn_params["use_fm"] = False
    run_base_model_dfm(dfTrain, dfTest, folds, "dnn", dnn_params)

if __name__ == "__main__":
    main()



