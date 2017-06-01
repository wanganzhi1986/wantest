#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from util import TrainModel, get_all_train_data, get_all_df
from util import merge_data
import os
import logging

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


def train(train_best_param=True, seed=1):
    stage = "M2"
    print("模型第二阶段训练:")
    train_datas, test_datas = get_all_train_data(stage)
    if not train_datas:
        print("没有数据，重新合并数据")
        make_merge2_data()
        train_datas, test_datas = get_all_train_data(stage)
    print("开始训练了:")
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
                            stage=stage,
                            seed=seed
                            )
            tm.train()
            tm.predict()
            train_time = tm.interval
            test_score = tm.test_score

            print("模型:{0},数据:{1}, 分数:{2}".format(clf_name, data_name, test_score))
            print("train time is:", train_time)


def make_merge2_data():
    m1_path = "../data/extend/" + "M2"
    if not os.path.exists(m1_path):
        os.mkdir(m1_path)
    train_m1_data, test_m1_data = get_all_df(stage="M1")
    merge_data(train_m1_data, path=m1_path, kind="train",  r=2, on="uid")
    merge_data(test_m1_data, path=m1_path, kind="test", r=2, on="uid")


if __name__ == "__main__":
    train()












