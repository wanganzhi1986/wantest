#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from padal.core.algo.clfs.stack.feature_enigneer import DiscreteBinsFeature
from scipy import stats


def get_discrete_feature():
    df_train = pd.read_csv("../data/train_raw.csv")
    df_test = pd.read_csv("../data/test_raw.csv")
    print("feaure names is:", list(df_train.columns))
    feature_names = df_train.columns.drop(["label", "uid"])
    train_X = df_train[feature_names].values
    train_y = df_train["label"].values
    test_X = df_test[feature_names].values
    test_y = df_test["label"].values

    # 训练集转换
    db_train = DiscreteBinsFeature(in_feature_names=feature_names, merge_origin=False,n_bins=20)
    train_X_ = db_train.fit_transform(train_X)
    discrete_train_X = np.hstack([train_X_, df_train[["uid", "label"]].values])
    discrete_train_names = np.hstack([db_train.out_feature_names, ["uid", "label"]])
    df_train_discrete = pd.DataFrame(discrete_train_X, columns=discrete_train_names)

    # 测试集转换
    db_test = DiscreteBinsFeature(in_feature_names=feature_names, merge_origin=False, n_bins=20)
    test_X_ = db_test.fit_transform(test_X)
    discrete_test_X = np.hstack([test_X_, df_test[["uid", "label"]].values])
    discrete_test_names = np.hstack([db_test.out_feature_names, ["uid", "label"]])
    df_test_discrete = pd.DataFrame(discrete_test_X, columns=discrete_test_names)
    df_train_discrete.to_csv('../data/train_discrete.csv', index=None)
    df_test_discrete.to_csv('../data/test_discrete.csv',index=None)


def get_num_discrete_feature():
    train_x = pd.read_csv('../data/train_discrete.csv')
    test_x = pd.read_csv('../data/test_discrete.csv')

    cols = ["label", "uid"]
    for i in range(20):
        cols.append("n"+str(i+1))
        train_x["n"+str(i+1)] = (train_x == (i+1)).sum(axis=1)
        test_x["n" + str(i+1)] = (train_x == (i + 1)).sum(axis=1)

    train_x[cols].to_csv('../data/train_nd.csv', index=None)
    test_x[cols].to_csv('../data/test_nd.csv', index=None)


def get_rank_features():
    # feature_type = pd.read_csv('../data/features_type.csv')
    # numeric_feature = list(feature_type[feature_type.type == 'numeric'].feature)

    # print("numeric feature is:", numeric_feature)

    # rank特征的命名：在原始特征前加'r',如'x1'的rank特征为'rx1'

    # 三份数据集分别排序，使用的时候需要归一化。
    # 更合理的做法是merge到一起排序，这个我们也试过，效果差不多，因为数据分布相对比较一致。

    test = pd.read_csv('../data/test_raw.csv')
    test_rank = test[["label", "uid"]]
    for feature in test.columns:
        if feature == "uid" or feature == "label":
            continue
        test_rank.loc['r' + feature] = test[feature].rank(method='max')
    test_rank.to_csv('../data/test_rank.csv', index=None)

    train = pd.read_csv('../data/train_raw.csv')
    train_rank = train[["label", "uid"]]
    for feature in train.columns:
        if feature == "uid" or feature == "label":
            continue
        train_rank.loc['r' + feature] = train[feature].rank(method='max')
    train_rank.to_csv('../data/train_rank.csv', index=None)


def main():
    # get_discrete_feature()
    get_num_discrete_feature()
    # get_rank_features()


if __name__ == "__main__":
    main()








