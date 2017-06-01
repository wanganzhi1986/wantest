#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations, permutations


class FeaturePairs(object):

    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path


def get_feature_pairs(df, kind="permutation"):
    columns = list(df.columns)
    values = df.values
    col_pairs = []
    val_pairs = []
    if kind == "combination":
        col_pairs = list(combinations(columns, r=2))
        val_pairs = list(combinations(values, r=2))

    if kind == "permutation":
        col_pairs = list(permutations(columns, r=2))
        val_pairs = list(permutations(values, r=2))

    return col_pairs, val_pairs


def get_pairs_plus_data(eps=0.01, num=None):
    train_path = "../data/origin/train_raw.csv"
    test_path = "../data/origin/test_raw.csv"
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_feature = df_train.drop(["label", "uid"], axis=1).values
    label = df_train["label"].values

    cols_pairs, val_pairs = get_feature_pairs(df_feature)
    plus_corr, df_plus = get_pairs_plus_corr(cols_pairs, val_pairs, label)


def get_pairs_correlation(col_pairs, val_pairs):
    # col_pairs, val_pairs = get_feature_pairs(df)
    corr_pairs = map(lambda a, b: get_corr(a, b), val_pairs)
    return dict(zip(col_pairs, corr_pairs))


def get_corr(a, b):
    coef, _ = stats.pearsonr(a, b)
    return coef


def get_pairs_pvalues(col_pairs, val_pairs):
    # col_pairs, val_pairs = get_feature_pairs(df)
    corr_pairs = map(lambda a, b: get_pvalue(a, b), val_pairs)
    return dict(zip(col_pairs, corr_pairs))


def get_pvalue(a, b):
    _, pvalue = stats.pearsonr(a, b)
    return pvalue


def get_pairs_labels_corr(cols, values, label):
    corrs = map(lambda value: get_corr(value, label), values)
    return dict(zip(cols, corrs))


def get_pairs_labels_pvalue(cols, values, label):
    pvlaues = map(lambda value: get_pvalue(value, label), values)
    return dict(zip(cols, pvlaues))


def get_pairs_plus_corr(col_pairs, val_pairs, label):
    plus_values = map(lambda a, b: a + b, val_pairs)
    cols = map(lambda x: ":".join(list(x)), col_pairs)
    label_corr = get_pairs_labels_corr(cols, plus_values, label)
    df_plus = pd.DataFrame(plus_values, columns=cols)
    df_plus["label"] = label
    return label_corr, df_plus


def get_pairs_sub_corr(col_pairs, val_pairs, label):
    sub_values = map(lambda a, b: a + b, val_pairs)
    cols = map(lambda x: ":".join(list(x)), col_pairs)
    label_corr = get_pairs_labels_corr(cols, sub_values, label)
    df_sub = pd.DataFrame(sub_values, columns=cols)
    # plus = dict(zip(cols, plus_values))
    return label_corr, df_sub


def get_pairs_mul_corr(col_pairs, val_pairs, label):
    mul_values = map(lambda a, b: a * b, val_pairs)
    cols = map(lambda x: ":".join(list(x)), col_pairs)
    label_corr = get_pairs_labels_corr(cols, mul_values, label)
    df_sub = pd.DataFrame(mul_values, columns=cols)
    df_sub["label"] = label
    # plus = dict(zip(cols, plus_values))
    return label_corr, df_sub


def get_pairs_divide_corr(col_pairs, val_pairs, label):
    divide_values = map(lambda a, b: a + b, val_pairs)
    cols = map(lambda x: ":".join(list(x)), col_pairs)
    label_corr = get_pairs_labels_corr(cols, divide_values, label)
    df_divide = pd.DataFrame(divide_values, columns=cols)
    # plus = dict(zip(cols, plus_values))
    return label_corr, df_divide


def find_corr_pairs_sub(train_x, train_y, eps=0.01, num=None):
    feature_size = len(train_x[0])
    feature_corr_list = []
    for i in range(feature_size):
        if i % 50 == 0:
            print i
        for j in range(feature_size):
            if i < j:
                corr = stats.pearsonr(train_x[:, i] - train_x[:, j], train_y)
                if abs(corr[0]) < eps:
                    continue
                feature_corr = (i, j, abs(corr[0]))
                feature_corr_list.append(feature_corr)
    sort_feature_list = sorted(feature_corr_list, key=lambda x: x[2], reverse=True)
    if num and num < len(sort_feature_list):
        return sort_feature_list[:num]
    return feature_corr_list


def find_corr_pairs_plus(train_x, train_y, eps=0.01, num=None):
    feature_size = len(train_x[0])
    feature_corr_list = []
    for i in range(feature_size):
        if i % 50 == 0:
            print i
        for j in range(feature_size):
            if i < j:
                corr = stats.pearsonr(train_x[:,i] + train_x[:,j], train_y)
                if abs(corr[0]) < eps:
                    continue
                feature_corr = (i, j, corr[0])
                feature_corr_list.append(feature_corr)
    sort_feature_list = sorted(feature_corr_list, key=lambda x: x[2], reverse=True)
    if num and num < len(sort_feature_list):
        return sort_feature_list[:num]
    return feature_corr_list


def find_corr_pairs_divide(train_x, train_y, eps=0.01, num=None):
    feature_size = len(train_x[0])
    feature_corr_list = []
    for i in range(feature_size):
        if i % 50 == 0:
            print i
        for j in range(feature_size):
            if i != j:
                try:
                    res = train_x[:, i] / train_x[:, j]
                    corr = stats.pearsonr(res, train_y)
                    if abs(corr[0]) < eps:
                        continue
                    feature_corr = (i, j, abs(corr[0]))
                    feature_corr_list.append(feature_corr)
                except ValueError:
                    print 'divide 0'
    sort_feature_list = sorted(feature_corr_list, key=lambda x: x[2], reverse=True)
    if num and num < len(sort_feature_list):
        return sort_feature_list[:num]
    return feature_corr_list


def find_corr_pairs_sub_mul(train_x, train_y, sorted_corr_sub, eps=0.01, num=None):
    feature_size = len(train_x[0])
    feature_corr_list = []
    for i in range(len(sorted_corr_sub)):
        ind_i = sorted_corr_sub[i][0]
        ind_j = sorted_corr_sub[i][1]
        if i % 100 == 0:
            print i
        for j in range(feature_size):
            if j != ind_i and j != ind_j:
                res = (train_x[:, ind_i] - train_x[:, ind_j]) * train_x[:, j]
                corr = stats.pearsonr(res, train_y)
                if abs(corr[0]) < eps:
                    continue
                feature_corr = (ind_i, ind_j, j, corr[0])
                feature_corr_list.append(feature_corr)
    sort_feature_list = sorted(feature_corr_list, key=lambda x: x[2], reverse=True)
    if num and num < len(sort_feature_list):
        return sort_feature_list[:num]
    return feature_corr_list


def get_feature_pairs_plus_data():
    train_path = "../data/extend/M1/train_raw.csv"
    test_path = "../data/extend/M1/test_raw.csv"
    train_X, train_y, train_uid, feature_names = load_data(train_path)
    test_X, test_y, test_uid, feature_names = load_data(test_path)
    trains = []
    tests = []
    feature_corr_list = find_corr_pairs_plus(train_X, train_y)
    corr_names = []
    for k,  (i, j, corr) in enumerate(feature_corr_list):
        corr_names.append(feature_names[i] + ":" + feature_names[j])
        trains.append(train_X[:, i] + train_X[:, j])
        tests.append(test_X[:, i] + test_X[:, j])

    df_train = pd.DataFrame(np.column_stack(trains+[train_y]), columns=corr_names+["label"])
    df_train["uid"] = train_uid
    df_test = pd.DataFrame(np.column_stack(tests + [test_y]), columns=corr_names + ["label"])
    df_test["uid"] = test_uid
    df_train.to_csv("../data/extend/M1/train_plus.csv")
    df_test.to_csv("../data/extend/M1/test_plus.csv")


def get_feature_pairs_sub_data():
    train_path = "../data/extend/M1/train_raw.csv"
    test_path = "../data/extend/M1/test_raw.csv"
    train_X, train_y, train_uid, feature_names = load_data(train_path)
    test_X, test_y, test_uid, feature_names = load_data(test_path)
    trains = []
    tests = []
    feature_corr_list = find_corr_pairs_sub(train_X, train_y)
    corr_names = []
    for k, (i, j, corr) in enumerate(feature_corr_list):
        corr_names.append(feature_names[i] + ":" + feature_names[j])
        trains.append(train_X[:, i] - train_X[:, j])
        tests.append(test_X[:, i] - test_X[:, j])

    df_train = pd.DataFrame(np.column_stack(trains + [train_y]), columns=corr_names + ["label"])
    df_train["uid"] = train_uid
    df_test = pd.DataFrame(np.column_stack(tests + [test_y]), columns=corr_names + ["label"])
    df_test["uid"] = test_uid
    df_train.to_csv("../data/extend/M1/train_sub.csv")
    df_test.to_csv("../data/extend/M1/train_sub.csv")


def get_feature_pairs_sub_mul_data():
    train_path = "../data/extend/M1/train_raw.csv"
    test_path = "../data/extend/M1/test_raw.csv"
    train_X, train_y, train_uid, feature_names = load_data(train_path)
    test_X, test_y, test_uid, feature_names = load_data(test_path)
    trains = []
    tests = []
    feature_corr_list = find_corr_pairs_sub_mul(train_X, train_y)
    corr_names = []
    for k, (i, j, corr) in enumerate(feature_corr_list):
        corr_names.append(feature_names[i] + ":" + feature_names[j])
        trains.append(train_X[:, i] - train_X[:, j])
        tests.append(test_X[:, i] - test_X[:, j])

    df_train = pd.DataFrame(np.column_stack(trains + [train_y]), columns=corr_names + ["label"])
    df_test = pd.DataFrame(np.column_stack(tests + [test_y]), columns=corr_names + ["label"])
    df_train.to_csv("../data/extend/M1/train_sub_mul.csv")
    df_test.to_csv("../data/extend/M1/test_sub_mul.csv")


def get_feature_divide_data():
    pass


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(["label", "uid"], axis=1).values
    feature_names = list(df.drop(["label", "uid"], axis=1).columns)
    y = df["label"].values
    uid = df["uid"].values
    return X, y, uid, feature_names

if __name__ == "__main__":
    get_feature_pairs_plus_data()
    get_feature_pairs_sub_data()

