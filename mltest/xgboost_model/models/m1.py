#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from util import TrainModel, get_all_train_data, get_all_df
from itertools import combinations
from util import merge_data

clf_names = ["lr", "xgb"]

lr_best_param = {"C": 0.01}
xgb_best_param = {'reg_alpha': 8.159752615976961,
                  'colsample_bytree': 0.8673698837685357,
                  'learning_rate': 0.23349173391703357,
                  'min_child_weight': 28,
                  'n_estimators': 235,
                  'reg_lambda': 481.84913690161386,
                  'max_depth': 4,
                  'gamma': 0.10233577351305655,
                  "subsample": 0.8560391704853437
                  }

params = {
    "lr": lr_best_param,
    "xgb": xgb_best_param
}


def train(train_best_param=True):
    stage = "M1"
    train_datas, test_datas = get_all_train_data(stage)
    for data_name, train_data in train_datas.items():
        for clf_name in clf_names:
            train_X, train_y, train_uid = train_data
            test_X, test_y, test_uid = test_datas.get(data_name)
            if train_best_param:
                best_param = params.get(clf_name)
            else:
                best_param = None
            tm = TrainModel(clf_name=clf_name,
                            train_X=train_X,
                            train_y=train_y,
                            test_X=test_X,
                            test_y=test_y,
                            dataset_name=data_name,
                            best_param=best_param,
                            train_uid=train_uid,
                            test_uid=test_uid,
                            stage="M1"
                            )
            tm.train(train_X, train_y)
            tm.predict(test_X, test_y)
            train_time = tm.interval
            print("train time is:", train_time)


if __name__ == "__main__":
    import os
    df1 = pd.DataFrame([{"id":1,"a":1, "b":2}, {"id":2,"a":4, "b":2}, {"id":3,"a":3, "b":5}])
    df2 = pd.DataFrame([{"id":1,"c": 1, "d": 2}, {"id":2,"c": 4, "d": 2}, {"id":3,"c": 3, "d": 5}])
    df3 = pd.DataFrame([{"id":1,"e": 1, "f": 2}, {"id":2,"e": 4, "f": 2}, {"id":3,"e": 3, "f": 5}])
    df4 = pd.DataFrame([{"id":1,"g":1, "h":2}, {"id":2,"g":4, "h":2}, {"id":3,"g":3, "h":5}])

    # df = {"df1":df1, "df2":df2, "df3":df3,"df4":df4}
    # r = merge_data(df, r=3, on="id")
    # print r
    for r in os.listdir("../data/exclude/M3"):
        print r




