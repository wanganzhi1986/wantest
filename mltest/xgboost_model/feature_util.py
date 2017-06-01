#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


# 得到排序特征
def get_sorted_feature(X, y):
    return np.argsort(X, axis=0) + 1


# 得到计数特征
def get_count_feature(X, y):
    from sklearn.preprocessing import Binarizer



# 得到离散特征
def get_disctete_feature(X, y):

    pass


def get_null_feature(X, y):
    pass


# 得到kmeans的分类特征
def get_kmeans_feature(X, y):
    pass


# 得到gbdt的分类特征
def get_gbdt_feature(X, y):
    pass