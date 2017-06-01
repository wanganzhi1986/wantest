#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os


train_path = "../data/train.csv"
test_path = "../data/test.csv"

def add_uid():
    train_path = "../data/train.txt"
    test_path = "../data/test.txt"
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    print ("feature names is:", df_train.columns)

    df_train["uid"] = np.array(["kn"+str(i+1) for i in range(df_train.shape[0])])
    df_test["uid"] = np.array(["kn"+str(i+1) for i in range(df_test.shape[0])])

    df_train.to_csv("../data/train.csv", index=None)
    df_test.to_csv("../data/test.csv", index=None)
    # os.remove("../data/train.txt")
    # os.remove("../data/test.txt")


def fill_nan():
    train_path = "../data/train.csv"
    test_path = "../data/test.csv"
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_train.fillna(df_train.median())
    df_test.fillna(df_test.median())
    df_train = prepocess(df_train)
    df_test = prepocess(df_test)
    print("feature names is:", df_train.columns)
    df_train.to_csv("../data/train_raw.csv", index=None)
    df_test.to_csv("../data/test_raw.csv", index=None)


def prepocess(df):
    df = df.fillna(df.median())

    drop_columns = []
    drop_columns.extend(df.columns[(df.isnull().sum() / float(df.shape[0])) > 0.9])
    drop_columns.extend(df.columns[np.std(df) < 1e-8].tolist())
    drop_columns = list(set(drop_columns))
    print ("drop feauture is:", drop_columns)
    df = df.drop(drop_columns, axis=1)
    return df


if __name__ == "__main__":
    add_uid()
    fill_nan()


