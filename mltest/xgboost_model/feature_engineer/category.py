#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from padal.core.algo.clfs.stack.feature_select import KmeansClusterFeature, GradientBoostingEmbedding, RandomForestEmbedding


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(["label", "uid"], axis=1).values
    y = df["label"].values
    uid = df["uid"].values
    return X, y, uid


def get_gbdt_feature():
    train_path = "../data/train_raw.csv"
    test_path = "../data/test_raw.csv"
    train_X, train_y, train_uid = load_data(train_path)
    test_X, test_y, test_uid = load_data(test_path)
    kf = GradientBoostingEmbedding(n_estimators=100, max_depth=3, learning_rate=0.02)
    kf.fit(train_X, train_y)
    train_X = kf.transform(train_X)
    test_X = kf.transform(test_X)
    feature_names = ["gbdt_" + str(i) for i in range(train_X.shape[1])]
    # df_train = pd.DataFrame(np.hstack([train_X, train_y.reshape(-1, 1), train_uid.reshape(-1, 1)]),
    #                         columns=np.hstack([feature_names, ["label"], ["uid"]]))
    # df_test = pd.DataFrame(np.hstack([test_X, test_y.reshape(-1, 1), test_uid.reshape(-1, 1)]),
    #                        columns=np.hstack([feature_names, ["label"], ["uid"]]))

    df_train = pd.DataFrame(train_X, columns=feature_names)
    df_train["label"] = train_y
    df_train["uid"] = train_uid

    df_test = pd.DataFrame(test_X, columns=feature_names)
    df_test["label"] = test_y
    df_test["uid"] = test_uid
    # df_train = pd.DataFrame(n

    df_train.to_csv("../data/train_gbdt.csv")
    df_test.to_csv("../data/test_gbdt.csv")


def get_kmeans_feature(n_clusers=10):
    train_path = "../data/train_raw.csv"
    test_path = "../data/test_raw.csv"
    train_X, train_y, train_uid = load_data(train_path)
    test_X, test_y, test_uid = load_data(test_path)
    kf = KmeansClusterFeature(n_clusters=n_clusers, merge_origin=False)
    train_X = kf.fit_transform(train_X)
    test_X = kf.fit_transform(test_X)
    print("train x:", train_X.shape)
    print(train_y.reshape(-1, 1).shape)
    feature_names = ["kmeans_"+str(i)for i in range(train_X.shape[1])]
    df_train = pd.DataFrame(train_X, columns=feature_names)
    df_train["label"] = train_y
    df_train["uid"] = train_uid

    df_test = pd.DataFrame(test_X, columns=feature_names)
    df_test["label"] = test_y
    df_test["uid"] = test_uid
    # df_train = pd.DataFrame(np.hstack([train_X, train_y.reshape(-1,1), train_uid.reshape(-1,1)]),
    #                         columns=np.hstack([feature_names, ["label"], ["uid"]]))
    # df_test = pd.DataFrame(np.hstack([test_X, test_y.reshape(-1,1), test_uid.reshape(-1,1)]),
    #                         columns=np.hstack([feature_names, ["label"], ["uid"]]))

    df_train.to_csv("../data/train_kmeans.csv")
    df_test.to_csv("../data/test_kmeans.csv")


def get_rf_features():
    train_path = "../data/train_raw.csv"
    test_path = "../data/test_raw.csv"
    train_X, train_y, train_uid = load_data(train_path)
    test_X, test_y, test_uid = load_data(test_path)
    kf = RandomForestEmbedding(n_estimators=50, max_depth=8, n_jobs=3, sparse_output=False)
    kf.fit(train_X, train_y)
    train_X = kf.transform(train_X)
    test_X = kf.transform(test_X)
    feature_names = ["rf_" + str(i) for i in range(train_X.shape[1])]
    ss = np.hstack([train_X, train_y.reshape(-1,1), train_uid.reshape(-1,1)])
    print("ss:", ss.shape)
    df_train = pd.DataFrame(train_X,columns=feature_names)
    df_train["label"] = train_y
    df_train["uid"] = train_uid
    print(df_train.shape)

    df_test = pd.DataFrame(test_X, columns=feature_names)
    df_test["label"] = test_y
    df_test["uid"] = test_uid
    print(df_test.shape)

    df_train.to_csv("../data/train_rf.csv")
    df_test.to_csv("../data/test_rf.csv")


if __name__ == "__main__":
    get_kmeans_feature()
    # get_gbdt_feature()
    # get_rf_features()






