#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from padal.core.algo.clfs.stack.classfiers import  XgboostClassfier, LogistRegressionClassfier, RandomForestClassfiers
from padal.core.algo.clfs.stack.paramselect import BayesianOptimise
from sklearn.externals import joblib
import json
from multiprocessing import Pool
from sklearn.preprocessing import MinMaxScaler
import os
import csv

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(["label", "uid"], axis=1).values
    y = df["label"].values
    uid = df["uid"].values
    return X, y, uid

def get_discrete_data():
    train_path = "../data/train_discrete.csv"
    test_path = "../data/test_discrete.csv"
    train_X, train_y, train_uid = load_data(train_path)
    test_X, test_y, test_uid = load_data(test_path)
    return train_X, test_X, train_y, test_y, train_uid, test_uid



def get_rank_data():
    train_path = "../data/train_rank.csv"
    test_path = "../data/test_rank.csv"
    train_X, train_y, train_uid = load_data(train_path)
    test_X, test_y, test_uid = load_data(test_path)
    return train_X, test_X, train_y, test_y, train_uid, test_uid


def get_num_data():
    train_path = "../data/train_nd.csv"
    test_path = "../data/test_nd.csv"
    train_X, train_y, train_uid = load_data(train_path)

    test_X, test_y, test_uid = load_data(test_path)
    return train_X, test_X, train_y, test_y, train_uid, test_uid


def get_raw_data():
    train_path = "../data/train_raw.csv"
    test_path = "../data/test_raw.csv"
    train_X, train_y, train_uid = load_data(train_path)
    test_X, test_y, test_uid = load_data(test_path)
    return train_X, test_X, train_y, test_y, train_uid, test_uid


def get_train_data():
    raw_data = get_raw_data()
    num_data = get_num_data()
    discrete_data = get_discrete_data()
    rank_data = get_rank_data()
    datas = {"raw": raw_data,
             "num": num_data,
             "discrete": discrete_data,
             "rank": rank_data
             }
    return datas


def get_classfiers():
    clfs = {
        "lr": LogistRegressionClassfier(),
        "xgb": XgboostClassfier(),
        "rf": RandomForestClassfiers()
    }
    return clfs


def _train(data, clf, clf_name, data_name, seed=1):
    train_X, test_X, train_y, test_y, train_uid, test_uid = data
    clf.fit(train_X, train_y)
    best_params = clf.best_params_
    # best_clf = clf.best_estimator_
    # name = data_name + "_" + clf_name + "_" + str(seed)
    # train_pred = cross_valid_train(train_X, train_y, clf)
    # train_score = roc_auc_score(train_y, train_pred)
    # print("train auc score is:", train_score)
    test_pred = clf.predict_proba(test_X)[:, 1]
    test_score = roc_auc_score(test_y, test_pred)
    print("best param is:", )
    print("test auc score is:", test_score)
    best_params.update({"pid": data_name, "score": test_score})
    save_params(best_params, clf_name)
    # joblib.dump(clf, './model/{0}.pkl'.format(name))
    # save_pred(train_uid, train_pred, name, "train")
    # save_pred(test_uid, test_pred, name, "test")


def save_pred(uid, pred, name, kind):
    test_result = pd.DataFrame(columns=["uid", "score"])
    test_result.uid = uid
    test_result.score = pred
    test_result.to_csv('./pred/{0}/{1}.csv'.format(kind, name), index=None)


def save_params(param, clf_name):
    if not os.path.exists("param"):
        os.mkdir("param")
    csv_path = "./param/" + clf_name + ".csv"
    df = pd.DataFrame([param])
    append_df_to_csv(df, csv_path)


def append_df_to_csv(df, csv_path, sep=","):
    import os
    if not os.path.isfile(csv_path):
        df.to_csv(csv_path, mode='a', index=False, sep=sep)
    elif len(df.columns) != len(pd.read_csv(csv_path, nrows=1, sep=sep).columns):
        raise Exception("Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(len(pd.read_csv(csv_path, nrows=1, sep=sep).columns)) + " columns.")
    elif not (df.columns == pd.read_csv(csv_path, nrows=1, sep=sep).columns).all():
        raise Exception("Columns and column order of dataframe and csv file do not match!!")
    else:
        df.to_csv(csv_path, mode='a', index=False, sep=sep, header=False)


def cross_valid_train(X, y, clf):
    from sklearn.cross_validation import KFold
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score
    scores = []
    # t1 = time.time()
    cv = KFold(len(y), n_folds=10)
    train_prob = np.zeros((X.shape[0],))
    for i, (train_index, test_index) in enumerate(cv):
        train_X = X[train_index]
        train_y = y[train_index]
        test_X = X[test_index]
        test_y = y[test_index]
        new_clf = clone(clf)
        new_clf.fit(train_X, train_y)
        # test_prediction_cv = self.get_base_predict(new_model, test_X_cv, test_index)
        test_prob = new_clf.predict_proba(test_X)[:, 1]
        train_prob[test_index] = test_prob

        # 获得验证集上的分数
        score = roc_auc_score(test_y, test_prob)
        scores.append(score)
    train_prob = train_prob.reshape((X.shape[0], 1))
    return train_prob


def get_data_by_clf(data, clf_name):
    train_X, test_X, train_y, test_y, train_uid, test_uid = data
    if clf_name == "lr":
        train_X = MinMaxScaler.fit_transform(train_X)
        test_X = MinMaxScaler.fit_transform(test_X)

    return train_X, test_X, train_y, test_y, train_uid, test_uid


def fine_tunning(clf_name=None, data_name=None):
    # pool = Pool(5)
    datas = get_train_data().items()
    pools = []
    for data_name_, data in datas:
        for clf_name_, clf in get_classfiers().items():
            if clf_name and clf_name != clf_name_:
                continue
            if data_name and data_name_ != data_name:
                continue
            # data = get_data_by_clf(data, clf_name)
            _train(data, clf, clf_name, data_name, seed=1)

    # for data_name, data, clf_name, clf in pools:
    #     if clf_name == "xgb":
    #         _train(data, clf, clf_name, data_name)
    #     pool.apply_async(_train, args=(data, clf, clf_name, data_name))
    # pool.close()
    # pool.join()

if __name__ == "__main__":
    fine_tunning(clf_name="xgb")







