#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
import os
from sklearn.cross_validation import train_test_split
from metadata import MetaData
import hashlib


# 数据管理器
class DataManager(object):
    def __init__(self,
                 dataset_name=None,
                 X=None,
                 y=None,
                 data_path=None,
                 metadata=None,
                 feature_names=None,
                 label_name="label",

                 ):
        self.feature_names = feature_names
        self.label_name = label_name
        self.metadata = metadata
        # self.train_path = train_path
        # self.test_path = test_path
        # self.train_X = train_X,
        # self.test_X = test_X
        # self.train_y = train_y
        # self.test_y = test_y
        # self.test_size = test_size
        # self.all_data = all_data
        self.data_path = data_path
        self.X = X
        self.y = y
        self.dataset_name = dataset_name

        self.data = {}
        self.feature_info = {}

        if self.X is None and self.data_path is None:
            raise ValueError("no available data is use")

        if self.dataset_name is None:
            self.dataset_name = self.make_dataset_name()

    def parse_data(self, X, y=None):
        if type(X) == pd.DataFrame:
            cols = sorted(list(X.columns))
            if self.label_name in cols:
                if X.shape[1] <= 2:
                    raise ValueError("feature data not exist")
                cols.remove(self.label_name)
                X_ = X[cols].values
                y_ = X[self.label_name].values
                feature_names = list(X[X.columns.drop(self.label_name)].columns)

            else:
                if y is None:
                    raise ValueError("label value is not exist")
                else:
                    X_ = X[cols].values
                    if type(y) == pd.DataFrame:
                        y_ = y.values
                    elif type(y) == pd.Series:
                        y_ = y

                    elif type(y) == np.ndarray and len(y.shape) == 1:
                        y_ = y

                    else:
                        raise ValueError("label value type is invalid")
                    feature_names = list(X.columns)

        elif type(X) == np.ndarray:
            if y is None:
                raise ValueError("label value is  unknown")
            if len(X.shape) == 1:
                X_ = X.reshape(-1,1)
            else:
                X_ = X
            if type(y) == pd.DataFrame:
                y_ = y.values
            elif type(y) == pd.Series:
                y_ = y
            elif type(y) == np.ndarray and len(y.shape) == 1:
                y_ = y
            else:
                raise ValueError("label value type is invalid")

            if self.feature_names is None:
                feature_names = ["feature"+str(i) for i in range(X_.shape[1])]
            else:
                feature_names = self.feature_names
        else:
            raise ValueError("data type is invalid")
        return X_, y_, feature_names

    def fit(self, X=None, y=None):
        self.make_data()
        self.make_metadata()

    def make_dataset_name(self):
        if self.data_path:
            basepath = os.path.basename(self.data_path)
            dataset_name, _ = os.path.splitext(basepath)
        else:
            m = hashlib.md5()
            m.update(self.X)
            dataset_name = m.hexdigest()
        return dataset_name

    # 生成特征数据
    def make_metadata(self):
        if self.metadata:
            self.feature_info = self.metadata.features_info

        if not self.data:
            self.make_data()
        feature_names = self.feature_names
        train_X = self.data.get("train_X")
        test_X = self.data.get("test_X")
        if self.all_data:
            X = np.vstack([train_X, test_X])
        else:
            X = train_X

        df = pd.DataFrame(X, columns=feature_names)
        metadata = MetaData(label_name=self.label_name)
        metadata.fit(df)
        self.metadata = metadata
        self.feature_info = metadata.features_info

    # 生成模型数据
    def make_data(self):
        if self.train_X is None:
            train_data = self.load_data(self.train_path)
            train_X, train_y, train_feature_names = self.parse_data(train_data)
        else:
            train_X, train_y, train_feature_names = self.parse_data(self.train_X, self.train_y)

        if self.test_X is None:
            test_feature_names = train_feature_names
            if self.test_path:
                test_data = self.load_data(self.test_path)
                test_X, test_y, test_feature_names = self.parse_data(test_data)
            else:
                train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=self.test_size )

        else:
            test_X, test_y, test_feature_names = self.parse_data(self.test_X, self.test_y)

        if sorted(train_feature_names) == sorted(test_feature_names):

            self.data["train_X"] = train_X
            self.data["train_y"] = train_y
            self.data["test_X"] = test_X
            self.data["test_y"] = test_y
            self.feature_names = sorted(train_feature_names)
        else:
            raise ValueError("test data feature name not conform train data feature name")

    def load_data(self, path, format="csv"):
        if path:
            basename = os.path.basename(os.path.expanduser(path))
            file_name, prefix = basename.split(".")
            try:
                if prefix == ".npy":
                    X = np.load(path)
                else:
                    X = pd.read_csv(path)
                return X
            except Exception as e:
                logging.log(e)




