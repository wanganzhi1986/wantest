#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV
import sklearn.cross_validation
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator
import os
import xgboost as xgb
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from padal.core.algo.clfs.stack.util import get_time_interval
from datetime import datetime
from itertools import combinations
import logging


class TrainModel(object):
    def __init__(self,
                 clf_name,
                 stage,
                 train_X=None,
                 test_X=None,
                 train_y=None,
                 test_y=None,
                 train_uid=None,
                 test_uid=None,
                 dataset_name=None,
                 best_param=None,
                 train_param_num=2,
                 seed=1

                 ):
        self.clf_name = clf_name
        self.best_param = best_param
        self.dataset_name = dataset_name
        self.train_param_num = train_param_num
        self.train_uid = train_uid
        self.test_uid = test_uid
        self.seed = seed
        self.stage = stage
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y

        self.name = self.dataset_name + "_" + self.clf_name + "_" + str(self.seed)
        self.ensemble = self.check_train(self.name)
        self.train_pred = None
        self.test_pred = None
        self.test_score = None
        self.interval = None

    def train(self):
        if self.train_X is None and self.train_y is None:
            raise ValueError("train data must be given")
        start = datetime.now()
        if self.best_param is None:
            self.best_param = self.get_best_param()
        # 如果没有模型，那么再进行训练
        if self.ensemble is None:
            self.ensemble = self.get_classfier().get(self.clf_name).set_params(**self.best_param)
            self.ensemble.fit(self.train_X, self.train_y)
        self.train_pred = self.cross_valid_train(self.train_X, self.train_y, self.ensemble)
        self.test_pred = self.ensemble.predict_proba(self.test_X)[:, 1]
        self.save_model(self.ensemble, self.name)
        # path = "../result/" + self.dataset_name
        self.save_pred(uid=self.train_uid, pred=self.train_pred, label=self.train_y, name=name, kind="train", stage="M2")
        end = datetime.now()
        self.interval = get_time_interval(start, end)

    def predict(self):
        name = self.dataset_name + "_" + self.clf_name + "_" + str(self.seed)
        if self.test_X is None:
            raise ValueError("test data must be given")
        if self.ensemble is None:
            self.ensemble = self.load_model(name)
        self.test_pred = self.ensemble.predict_proba(self.test_X)[:, 1]
        self.save_pred(uid=self.test_uid, pred=self.test_pred, label=self.test_y, name=name, kind="test", stage="M2")

    # 获得模型的训练评分
    def score(self, metric="auc", kind="test"):
        score = 0
        if kind == "test":
            if self.test_y is None:
                raise ValueError("test label must be given")
            if self.test_pred is None:
                self.predict()
            if self.test_pred is None:
                raise ValueError( "predict fail")
            score = roc_auc_score(self.test_y, self.test_pred)

        if kind == "train":
            if self.train_y is None:
                raise ValueError("test label must be given")
            if self.test_pred is None:
                self.train()
            if self.train_pred is None:
                raise ValueError( "predict fail")
            score = roc_auc_score(self.train_y, self.train_pred)
        return score

    # 检查模型是否已经训练，如果已经训练直接返回，否则进行训练
    def check_train(self, name):
        mdir = "../result/model"
        if os.path.exists(mdir) and os.listdir(mdir):
            for fp in os.listdir(mdir):
                base_name = os.path.basename(fp)
                name_, _ = os.path.splitext(base_name)
                if name == name_:
                    return joblib.load(os.path.join(mdir, base_name))
        return None

    # 获得已经保存的超参数
    def get_best_param(self):
        best_params = self.load_params(self.clf_name)
        max_score = 0
        for best_param in best_params:
            clf = self.get_classfier().get(self.clf_name).set_params(**best_param)
            clf.fit(self.train_X, self.train_y)
            pred = clf.predict_proba(self.test_X)[:, 1]
            score = roc_auc_score(self.test_y, pred)
            if score > max_score:
                self.best_param = best_param
        return self.best_param

    def load_params(self, clf_name):
        param_path = "../param/" + clf_name + ".csv"
        df = pd.read_csv(param_path)
        if self.dataset_name:
            df = df[df["pid"] == self.dataset_name]
        df_param = df.sort_values(by="score").head(self.train_param_num) \
            .drop(["pid", "score"], axis=1)
        return df_param.to_dict(orient="records")

    # 保存模型
    def save_model(self, model, name):
        if not os.path.exists("../result/model"):
            os.mkdir("../result/model")
        model_name = "{0}.pkl".format(name)
        model_path = os.path.join("../result/model", model_name)
        joblib.dump(model, model_path)

    # 加载模型
    def load_model(self, name):
        try:
            model = joblib.load(os.path.join("../result/models", name))
            return model
        except Exception as e:
            return None


    def save_pred(self, uid, pred, label, name, kind, stage):
        if not os.path.exists("../result/pred/" + kind + "/" + stage):
            os.mkdir("../result/pred/" + kind + "/" + stage)
        test_result = pd.DataFrame(columns=["uid", "score", "label"])
        test_result.uid = uid
        test_result.score = pred
        test_result.label = label
        test_result.to_csv('../result/pred/{0}/{1}/{2}.csv'.format(kind, stage, name), index=None)

    def cross_valid_train(self, X, y, clf):
        from sklearn.cross_validation import KFold
        from sklearn.base import clone
        from sklearn.metrics import roc_auc_score
        scores = []
        # t1 = time.time()
        cv = KFold(len(y), n_folds=10)
        train_prob = np.zeros((X.shape[0],))
        for i, (train_index, test_index) in enumerate(cv):
            train_X = X[train_index]
            train_y = y[train_index]
            test_X = X[test_index]
            test_y = y[test_index]
            new_clf = clone(clf)
            new_clf.fit(train_X, train_y)
            # test_prediction_cv = self.get_base_predict(new_model, test_X_cv, test_index)
            test_prob = new_clf.predict_proba(test_X)[:, 1]
            train_prob[test_index] = test_prob

            # 获得验证集上的分数
            score = roc_auc_score(test_y, test_prob)
            scores.append(score)
        train_prob = train_prob.reshape((X.shape[0], 1))
        return train_prob

    def get_classfier(self):
        clfs = {
            "lr": LogisticRegression(),
            "xgb": xgb.XGBClassifier()
        }
        return clfs


def parse_data(path):
    df = pd.read_csv(path)
    X = df.drop(["label", "uid"], axis=1).values
    y = df["label"].values
    uid = df["uid"].values
    return X, y, uid


# 获得融合阶段的所有的训练数据
def get_all_train_data(stage):
    data_path = "../data/extend/" + stage
    trains = {}
    tests = {}
    if not os.path.exists(data_path):
        print("路径不存在")
        os.mkdir(data_path)
        return trains, tests
    fps = os.listdir(data_path)
    if not fps:
        print("文件不存在")
        return trains, tests
    for fp in os.listdir(data_path):
        base_name = os.path.basename(fp)
        fp_name, _ = os.path.splitext(base_name)
        kind = fp_name.split("_")[0]
        name = fp_name.split("_")[1]
        if kind == "train":
            trains[name] = parse_data(os.path.join(data_path, base_name))
        if kind == "test":
            tests[name] = parse_data(os.path.join(data_path, base_name))
    return trains, tests


def get_all_df(stage):
    data_path = "../data/extend/" + stage
    trains = {}
    tests = {}
    print('获得目录下的所有csv文件')
    for fp in os.listdir(data_path):
        base_name = os.path.basename(fp)
        fp_name, _ = os.path.splitext(base_name)
        kind = fp_name.split("_")[0]
        name = fp_name.split("_")[1]
        if kind == "train":
            trains[name] = pd.read_csv(os.path.join(data_path, base_name))
        if kind == "test":
            tests[name] = pd.read_csv(os.path.join(data_path, base_name))
    return trains, tests


# 合并多个csv文件:data格式:{"name1": df}
def merge_data(data, path, kind,  r=2, on="uid", dp="label"):
    names = data.keys()
    dfs = data.values()
    dfs_ = [df.drop(dp, axis=1) for df in dfs]
    print("开始合并文件")
    merge_names = list(combinations(names, r))
    # merge_dfs = [list(combination)for combination in list(combinations(dfs, r))]
    merge_dfs = map(lambda combination: reduce(lambda l, r: pd.merge(left=l, right=r, how="left", on=on), list(combination)),
                    list(combinations(dfs_, r)))
    print("开始保存文件")
    for i, (names, df) in enumerate(zip(merge_names, merge_dfs)):
        file_name = kind + "_" + ":".join(list(names))
        if path[-1] != "/":
            path += "/"
        if not os.path.exists(path):
            os.mkdir(path)
        df[dp] = data.get(names[0])[dp]
        df.to_csv(path + file_name + ".csv", index=None)
    print("文件保存完成")



