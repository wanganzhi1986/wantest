#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import entropy


class MetaData(object):

    def __init__(self, label_name, skipna=None):
        self.skipna = skipna
        self.label_name = label_name

        self.features_info = {}
        self.feature_curving = {}

    def fit(self, X):
        if type(X) == pd.DataFrame:
            self._parse_df(X)

        else:
            columns = ["feature" + str(i) for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=columns)
            self._parse_df(X)

    # 解析数据，获得每列的信息
    def _parse_df(self, df):
        for col_name in list(df.columns):
            col_value = df[col_name]
            label_value = df[self.label_name]
            self.features_info[col_name] = self.get_feature_info(col_value, label_value)

    # 获得每个特征的信息
    def get_feature_info(self, col_value, label_value):
        feature_info = {}
        attr_type = self.get_attr_type_info(col_value)
        stats_value = self.get_data_stats_info(col_value)
        corr_value = self.get_corr_value_info(col_value, label_value)
        feature_info["stats"] = stats_value
        feature_info["corr"] = corr_value
        feature_info["attr"] = attr_type
        return feature_info

    # 获得一列特征的缺失值信息
    def get_data_null_ratio(self, col_value):
        null_ratio = 0
        if col_value:
            null_count = sum([1 if np.isnan(v) else 0 for v in col_value])
            null_ratio = null_count/len(col_value)
        return null_ratio

    # 获得数据类型信息
    def get_attr_type_info(self, col_value):
        dtype = col_value.dtype
        if dtype in (np.object,):
            attribute_type = 'categorical'
        elif dtype in (np.int, np.int32, np.int64, np.float, np.float32,
                       np.float64, int, float):
            # 判断数据是否是连续数据
            if len(np.unique(col_value)) >= 10:
                attribute_type = "continue_numerical"

            else:
                attribute_type = "discrete_numerical"
        else:
            raise ValueError("no data type is supported")

        return attribute_type

    def get_data_stats_info(self, col_value):
        if type(col_value) == pd.Series:
            df = pd.DataFrame(col_value)

        elif type(col_value) == pd.DataFrame:
            df = col_value

        elif type(col_value) == list:
            df = pd.DataFrame(np.array(col_value))

        else:
            raise ValueError('data type not supported')

        null_ratio = 1 - float(df.count()/df.shape[0])

        mean_value = float(df.mean())
        var_value = float(df.var())
        unique_count = len(np.unique(df.values))

        return {
            "null_ratio": round(null_ratio, 4),
            "mean_value": round(mean_value, 4),
            "var_value": round(var_value, 4),
            "unique_count": unique_count
        }


    # 得到与目标的相关性
    def get_corr_value_info(self, col_value, label_value):

        if type(col_value) == pd.DataFrame:
            col_value = col_value.values
        elif type(col_value) == pd.Series:
            col_value = col_value

        elif type(col_value) == list:
            col_value = np.array(col_value)
        pearson_value, _ = pearsonr(col_value, label_value)
        entropy_value = entropy(col_value, base=2)
        spearm_value = spearmanr(col_value, label_value).correlation

        return {
            "pearson": round(pearson_value, 4),
            "entropy": round(entropy_value, 4),
            "spearm": round(float(spearm_value), 4)
        }

    # 得到绘制特征分布的曲线
    def get_feature_curving(self):
        pass


if __name__ == "__main__":
    columns = ["MP0050048", "MP0041080","MP0045048", "MP0042110", "MS0050001", "mp_months", "MP0050003",
             "MP0110001", "MP0110008", "MP0110012","MP0110010", "mp_district_41", "MP0050047","MP0045110",
             "MP0050028", "MP0044026", "MP0050021", "MP0041053", "MP0045115", "MP0041060", "MP0050018",
             "MS0050011", "mp_district_13", "MP0042067", "MP0110029", "label"]
    import pandas as pd
    import pprint
    train_path = "/Users/wangwei/workplace/data_train_mp.csv"
    df = pd.read_csv(train_path)
    df = df[columns]
    meatadata = MetaData(label_name="label")
    meatadata.fit(df)
    print("feature info is:")
    pprint.pprint(meatadata.features_info)











