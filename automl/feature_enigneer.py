#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from collections import Counter
from bisect import bisect_left
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from datamanager import DataManager


class FeatureData(TransformerMixin):

    def __init__(self,
                 dataset_name=None,
                 feature_stragety=None,
                 label_name=None):
        # 指定选择的特征策略
        self.feature_stragety = feature_stragety
        self.label_name = label_name
        self.dataset_name = dataset_name

        self.mapper = {"sort": self.get_sort_feature, "kmeans": self.get_kmeans_feature,
                       "discrete": self.get_discrete_feature, "count": self.get_count_feauture,
                       "gbdt": self.get_gbdt_feature, "weight": self.get_weight_feature
                       }

        self.datamanagers = {}
        self.transforms = {}

    def fit(self, X, y):

        for stragety, func in self.mapper.items():
            if self.feature_stragety and stragety not in self.feature_stragety:
                continue
            self.datamanagers[stragety] = self.get_data_manager(X, y, stragety)
            self.transforms[stragety] = self.get_data_transform(func)

    # 获得转换后的数据
    def transform(self, X):
        feature_data = {}
        for name, manager in self.datamanagers.items():
            feature_data[name] = manager.train_X

        return feature_data

    # 得到数据对象
    def get_data_manager(self, X, y, func, feature_stragety):

        dataset_name = self.dataset_name + "_" + feature_stragety
        feature_names = [feature_stragety+"_"+str(i) for i in range(X.shape[0])]
        if self.label_name is None:
            self.label_name = "label"
        X_ = func(X, y)
        datamanager = DataManager(X=X_, y=y,
                                  dataset_name=dataset_name,
                                  feature_names=feature_names,
                                  label_name=self.label_name
                                  )
        return datamanager

    # 获得数据的转换器
    def get_data_transform(self, func):
        transformer = FunctionTransformer(func=func)
        return transformer


    # 获得排序特征
    def get_sort_feature(self, X, y=None):
        pass

    # 获得计数特征
    def get_count_feauture(self, X, y=None):
        pass


    # 获得离散特征
    def get_discrete_feature(self, X, y=None):
        pass


    # 获得kmeans分类特征
    def get_kmeans_feature(self, X, y=None):
        pass


    # 获得gbdt的分类特征
    def get_gbdt_feature(self, X, y):
        pass

    # 获得加权特征
    def get_weight_feature(self, X, y):
        pass








# 获得排序特征
class SortedFeature(TransformerMixin):

    def __init__(self, in_feature_names, numeric_features):
        self.in_feature_names = in_feature_names
        self.numeric_features = numeric_features

        self.out_feature_names = []

    def fit(self, X, y):
        pass

    def transform(self, X):
        pass


# 连续变量离散化
class DiscreteBinsFeature(TransformerMixin):
    def __init__(self,
                 in_feature_names=None,
                 n_bins=32,
                 thresholds_groups=None,
                 merge_origin=True,
                 bin_vector=False,
                 fill_nan=True
                 ):
        """
        :param in_feature_names: 输入的特征的个数
        :param n_bins: 离散的区间的个数
        :param thresholds_groups: 区间分割点， 如果提供的话，则忽略n_bins
        """

        self.in_feature_names = in_feature_names

        self._threshold_groups = thresholds_groups
        if self._threshold_groups is None:
            self._threshold_groups = []
        self._n_bins = n_bins
        self.merge_origin = merge_origin
        self.df_index = None
        self.df_merge = None

        self.bins_index = []
        self.out_feature_names = []

    def fit(self, X, y=None):
        if self.in_feature_names is not None:
            if X.shape[1] != len(self.in_feature_names):
                raise ValueError("feature names nums not match feature names num")
        else:
            self.in_feature_names = ["discret_feature"+str(i) for i in len(X.shape[1])]

        self.out_feature_names = [name + '_bin' for name in self.in_feature_names]

        if self._threshold_groups:
            return
        self._fit(X, y)
        return self

    def _fit(self, X, y):
        df = pd.DataFrame(X, columns=self.in_feature_names)
        print("in feature names is:", len(self.in_feature_names))
        for col in self.in_feature_names:
            col_values = df[col].values
            thresholds = self.get_bins_threshold(col_values, self._n_bins)
            self._threshold_groups.append(thresholds)
            bin_index = self.get_bins_index(col_values, thresholds)
            self.bins_index.append(np.array(bin_index))

    # 获得这个值得索引
    def transform(self, X):
        if self.bins_index:
            # print np.hstack(self.bins_index).shape
            # print np.column_stack(self.bins_index).shape
            X_ = np.column_stack(self.bins_index)

            if X_.shape[0] == X.shape[0] and X_.shape[1] == X.shape[1]:
                if self.merge_origin:
                    merge_ = np.hstack([X, X_])
                    self.df_merge = pd.concat((pd.DataFrame(X, columns=self.in_feature_names),
                                               pd.DataFrame(X_, columns=self.out_feature_names)
                                               ), axis=1
                                              )
                    return merge_
                self.df_index = pd.DataFrame(X_, columns=self.out_feature_names)
                return X_
            else:
                raise ValueError("X_ shape and X shape not match")
        return X


    # 获得区间的索引
    def get_bins_index(self, col_values, thresholds):
        if thresholds:
            return [bisect_left(thresholds, value) for value in col_values]
        else:
            return col_values


    # 获得划分区间的离散点
    def get_bins_threshold(self, col_values, n_bins):
        histogram = sorted(Counter(col_values).items())
        # 划分的数据的个数要大于区间的个数，如果区间个数大于点的个数，则至少有一个区间将会是空区间
        if len(histogram) <= n_bins:
            self._threshold_groups.append(None)
            return

        thresholds = []
        n_bins = self._n_bins
        # 要划分的值，将这些值分到区间中去
        n_items = float(len(col_values))
        ih = 0
        ih_lim = len(histogram)
        while n_bins > 1:
            # 均分的区间的大小
            expected_bin_size = n_items / n_bins
            count = 0
            while count < expected_bin_size:
                count += histogram[ih][1]
                ih += 1
            # 添加到上个区间的最后一个值
            last_addition = histogram[ih - 1][1]
            # print('last addition is:', last_addition)
            # 如果添加了某个值使得这个区间内的个数超过了区间的范围，那么则去衡量添加这个数后超过的大小与
            # 不添加这个数后再需要添加多少个数才能得到范围进行衡量对比，如果超过的程度大于不足的程度。那么
            # 这个数就添加到下一个区间内否则添加到上一个区间
            if count - expected_bin_size > expected_bin_size - (count - last_addition) \
                    and count != last_addition:
                ih -= 1
                count -= last_addition
            # print("ih is:", ih)
            # 如果得到了范围的边界，那么就跳出
            if ih == ih_lim:
                break
            # 上一个区间的最后一个值和下一个区间的第一个值得中间值添加到划分点的集合中

            # print("hist 0:", histogram[ih - 1][0])
            # print("hist 1:", histogram[ih][0])
            # print("threshold is:", (histogram[ih - 1][0] + histogram[ih][0]) / 2 )
            thresholds.append((histogram[ih - 1][0] + histogram[ih][0]) / 2)
            n_items -= count
            n_bins -= 1
        return thresholds



if __name__ == "__main__":
    import sklearn_pandas

    # 测试划分区间的功能：
    X = np.array([2, 1, 4, 3, 4, 1, 2, 3, 4, 4, 4, 2, 2, 1, 3, 2])
    print("count result is:", sorted(Counter(X).items()))

    bin_disctete = DiscreteBinsFeature(n_bins=3)
    thresolds = bin_disctete.get_bins_threshold(col_values=X, n_bins=3)
    result = bin_disctete.get_bins_index(col_values=X, thresholds=thresolds)
    # bin_disctete.fit(X, y=None)
    # result = bin_disctete.transform(X)
    print("threshold result is:", thresolds)
    print("bin result is:", result)

    import tarfile
    filepath = "/Users/wangwei/workplace/credit/train.7z"
    tar = tarfile.open(filepath)
    tar.extractall()
    tar.close()




