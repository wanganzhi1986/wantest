#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import imputation
from sklearn.preprocessing import MinMaxScaler
from scipy import sparse
from scipy import stats
from collections import namedtuple

TrainData = namedtuple("TrainData",["train_features", "train_label", "feature_names", "label_name"])

TestData = namedtuple("TestData", ["test_features", "test_label", "feature_names", "label_name"])


class PrepocessBase(object):

    def __init__(self,
                 train_path,
                 test_path,
                 exclude_columns=None,
                 sc=None,
                 split_value=0.3,
                 include_columns=None,
                 data_format="csv",
                 label_column="label",
                 replace_strategy="median"):

        if not label_column:
            raise ValueError("label columns not exists")

        if not train_path:
            raise ValueError("train data not exists")
        if include_columns and len(set(include_columns)) < len(include_columns):
            raise ValueError("include colums has same column")
        if exclude_columns and len(set(exclude_columns)) < len(exclude_columns):
            raise ValueError("exclude columns has same column")

        if exclude_columns and label_column in exclude_columns:
            raise  ValueError("label columns must not delete")

        if include_columns and label_column not in include_columns:
            include_columns.append(label_column)

        self.train_path = train_path
        self.test_path = test_path
        self.data_format = data_format
        self.include_columns = include_columns
        self.exclude_columns = exclude_columns
        self.label_column = label_column
        self.split_value = split_value
        self.replace_strategy = replace_strategy
        self.sc = sc

        self.feature_names = []
        self.train_data = None
        self.test_data = None

        if self.include_columns and self.label_column not in self.include_columns:
            self.include_columns.append(self.label_column)


    def fit(self):
        df_train_X, df_train_y, df_test_X, df_test_y = self.read_data()
        df_train_X = self.replace_nans(df_train_X)
        df_test_X = self.replace_nans(df_test_X)

        self.train_data = TrainData(df_train_X.values, df_train_y[self.label_column].values, list(df_train_X.columns), self.label_column)
        self.test_data = TestData(df_test_X.values, df_test_y[self.label_column].values, list(df_test_X.columns), self.label_column)
        return self

    # 读取数据
    def read_data(self):
        if self.test_path:
            df_train_X, df_train_y = self.read_raw_data(self.train_path)
            df_test_X, df_test_y = self.read_raw_data(self.test_path)


        else:
            df_train_X, df_train_y = self.read_raw_data(self.train_path)
            cols = df_train_X.columns
            train_X = df_train_X.values
            train_y = df_train_y.values
            train_X, test_X, train_y, test_y = train_test_split(train_X, train_y)
            df_train_X = pd.DataFrame(train_X, columns=cols)
            df_test_X = pd.DataFrame(test_X, columns=cols)
            df_train_y = pd.DataFrame(train_y, columns=[self.label_column])
            df_test_y = pd.DataFrame(test_y, columns=[self.label_column])

        return df_train_X, df_train_y, df_test_X, df_test_y


    def read_raw_data(self, path):
        if not self.sc:
            if self.data_format == "csv":
                df_train = pd.read_csv(path)

            elif self.data_format == "json":
                df_train = pd.read_json(path)

            elif self.data_format == "xlsx":
                df_train = pd.read_excel(path)

            else:
                raise ValueError("data format is invalid")
        else:
            pass

        cols = df_train.columns.tolist()
        include_columns = self.include_columns
        exclude_columns = self.exclude_columns
        if include_columns:
            include_columns = list(set(include_columns).intersection(set(cols)))

        if exclude_columns:
            exclude_columns = list(set(exclude_columns).intersection(set(cols)))

        if include_columns:
            df_train = df_train[include_columns]

        if exclude_columns:
            df_train = df_train.drop(exclude_columns, axis=1)

        # 将无穷的数值换成空值
        df_train = df_train.replace([np.inf, -np.inf], np.NaN)
        df_train_label = df_train[self.label_column]
        df_train_features = df_train.drop([self.label_column], axis=1)
        # self.feature_names = df_train_features.columns.tolist()
        # X = df_train_features.values
        # y = df_train_label.values
        return df_train_features, df_train_label


    def transform(self):
        train_X, test_X, train_y, test_y = self.read_data()
        train_X, test_X = self.replace_nan(train_X, test_X)
        return train_X, test_X, train_y, test_y

    def dummy_from_categorical(data):

        #make dummy variables from categoricals
        categorical = data.select_dtypes(include=[object])

        for col in categorical.columns:

            if  categorical[col].nunique() > 2:
                dummy_features = pd.get_dummies(categorical[col])
                dummy_features.columns = ['is_' + '_'.join([col] + c.split()).lower() for c in dummy_features.columns]

                data.drop(col,axis=1,inplace=True)
                data = data.merge(dummy_features,left_index=True,right_index=True)

        return data

    def replace_nans(self, df_features):
        miss_rate = 0.9
        # 将无穷的值替换成空值

        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        drop_columns = []
        # 删去缺失值过多的列
        drop_columns.extend(df_features.columns[(df_features.isnull().sum() / float(df_features.shape[0])) > miss_rate])
        drop_columns = list(set(drop_columns))
        print ('drop columns is:', drop_columns)
        allowed_strategies = ["mean", "median", "most_frequent"]
        if self.replace_strategy not in allowed_strategies:
            raise ValueError("replace strategy not exist")

        # 按照指定的策略进行填充
        if self.replace_strategy == "median":
            df_features = df_features.fillna(df_features.median())

        if self.replace_strategy == "mean":
            df_features = df_features.fillna(df_features.mean())

        if self.replace_strategy == "most_frequent":
            cols = df_features.columns.tolist()
            mode = stats.mode(df_features.values)
            freq_value = dict(zip(cols, list(mode[0][0])))
            df_features = df_features.fillna(freq_value)

        if self.replace_strategy == "zero":
            df_features = df_features.fillna(0)

        return df_features

    # 保存预处理后的结果
    def save(self, folder):
        pass


# 对数据建立预处理
class PreprocessBulider(object):

    def __init__(self, metadata,
                 feature_names=None,
                 label_name=None,
                 include_columns=None,
                 exclude_columns=None,
                 var_threshold=0.001,
                 null_threshold=0.85,
                 corr_threshold=0.0001

                 ):

        if label_name is None:
            raise ValueError("label column not given")

        if feature_names is None:
            raise ValueError(" feature columns name is not given")

        if exclude_columns and label_name in exclude_columns:
            raise ValueError("label name: %s must not in exclude columns"%label_name)

        if label_name not in feature_names:
            self.feature_names = feature_names + [label_name]
        self.feature_names = feature_names
        self.metadata = metadata

        self.label_name = label_name

        self.include_columns = include_columns
        self.exclude_columns = exclude_columns

        # 设定删除某个特征的阈值
        self.null_threshold = null_threshold
        self.var_threshold = var_threshold
        self.corr_threshold = corr_threshold

        self.pipelines = []
        self.drop_columns = []
        self.discrete_columns = []

    def run(self, X):
        if self.include_columns:
            if self.exclude_columns:
                columns = self.include_columns - self.exclude_columns
            else:
                columns = self.include_columns
        else:
            if self.exclude_columns:
                columns = self.feature_names - self.exclude_columns

            else:
                columns = self.feature_names + [self.label_name]

        if type(X) == pd.DataFrame:
            df = X

        elif type(X) == pd.Series:
            df = pd.DataFrame(X, columns=self.feature_names)

        else:
            raise ValueError("input X value %s is not illegal" % str(type(X)))
        df = df[columns]
        self._build_preprocess(df)

    # 建立预处理分析器
    def _build_preprocess(self, df):

        mappers = [self.handle_null_feature, self.handle_category_feature,
                   self.handle_onehot_feature, self.handle_scale_feature]
        for col_name in list(df.columns):
            col_value = df[col_name]
            metadata = self.metadata.get(col_name)
            if self.hanlde_invaild_feature(metadata=metadata):
                self.drop_columns.append(col_name)
                continue

            if self.handle_continue_feature(metadata):
                self.discrete_columns.append(col_name)

            for _mapper in mappers:
                self.pipelines.append((col_name, _mapper(metadata)))


    # 处理含有缺失值的特征
    def handle_null_feature(self, metadata):
        from sklearn.preprocessing import FunctionTransformer
        from sklearn.preprocessing import Imputer
        stats = metadata.get("stats")
        attr_type = metadata.get("attr")
        null_ratio = stats.get("null_ratio")

        if null_ratio < 0.3:
            if attr_type == "discrete_numerical":
                return FunctionTransformer(lambda x: -1)

        if 0.3 <= null_ratio < 0.85:
            if attr_type == "discrete_numerical":
                return FunctionTransformer(lambda x: -1)

            if attr_type == "continue_numerical":
                return Imputer(strategy="median")
        return None


    # 处理类别特征
    def handle_category_feature(self, metadata):
        from sklearn.preprocessing import LabelBinarizer
        from sklearn.preprocessing import OneHotEncoder
        attr_type = metadata.get("attr")
        if attr_type and attr_type == "categorical":
            return LabelBinarizer
        return None

    # 处理连续特征
    def handle_continue_feature(self, metadata):
        attr_type = metadata.get("attr")
        if attr_type and attr_type == "continue_numerical":
            return True
        return False

    def handle_onehot_feature(self, metadata):
        from sklearn.preprocessing import OneHotEncoder
        stats = metadata.get('stats')
        unique_value = stats.get("unique_value")
        if unique_value <= 5:
            return OneHotEncoder
        return None


    # 处理长尾特征
    def handle_longtail_feature(self, metadata):
        pass

    # 处理特征的归一化
    def handle_scale_feature(self, metadata):
        from sklearn.preprocessing import MinMaxScaler
        stats = metadata.get("stats")
        mean_value = stats.get("mean_value")
        var_value = stats.get("var_value")
        if abs(mean_value) > 5:
            return MinMaxScaler
        return None

    # 处理无效的特征
    def hanlde_invaild_feature(self, metadata):
        stats = metadata.get("stats")
        null_ratio = stats.get("null_ratio")
        var_value = stats.get("var_value")
        corr_value = stats.get("corr_value")
        # 缺失值超过指定的阈值，删除此列
        if null_ratio > self.null_threshold:
            return True

        # 方差值小于指定的阈值，删除此列
        if abs(var_value) < self.var_threshold:
            return True

        # 与目标变量的相关系数小于指定的阈值，删除此列
        if abs(corr_value) < self.corr_threshold:
            return True

    # 处理用户的自定义的特征
    def handle_user_define_feaure(self, metadata):
        pass












