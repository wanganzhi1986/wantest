#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time
import datetime
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score, log_loss
import copy
import json
import os
import codecs
from sklearn import cross_validation
import util
from collections import namedtuple


class MultiClassfier(BaseEstimator):

    def __init__(self,
                 estimators,
                 stack_by_prob=True,
                 folds=10,
                 metric="auc",
                 n_classes=2,
                 save_base_dump=False,
                 oob_flag=True
                 ):

        self.classfiers = estimators
        self.folds = folds
        self.metric = metric
        self.n_classes = n_classes
        self.save_base_dump = save_base_dump
        self.oob_flag = oob_flag

        self.blend_train = None
        self.blend_test = None
        self.models = None
        self.train_infos = None
        self.train_probs = None
        self.train_preds = None
        self.train_cv_models = None

    def _fit(self, X, y):
        train_preds = []
        train_probs = []
        train_infos = dict()
        train_models = dict()
        train_cv_models = dict()
        for clf_name, clf in self.classfiers.items():
            train_pred, train_prob, train_info, train_cv_model, train_model = self.make_train_predict(X, y, clf)
            train_preds.append(train_pred)
            train_probs.append(train_prob)
            train_infos[clf_name] = train_info
            train_models[clf_name] = train_model
            train_models[clf_name] = train_cv_model

        self.train_probs = np.array(train_probs).T
        self.train_preds = np.array(train_preds).T
        self.train_infos = train_infos
        self.train_models = train_models
        self.train_cv_models = train_cv_models

    def make_train_predict(self, X, y, clf):
        eval_metric = self.get_eval_metric()

        scores = []

        t1 = time.time()
        cv = cross_validation.KFold(len(y), n_folds=self.folds)
        train_prob = np.zeros(X.shape[0], self.n_classes-1)
        train_pred = np.zeros(X.shape[0], self.n_classes-1)
        cv_models = []
        models = []
        train_info = {}
        for i, (train_index, test_index) in enumerate(cv):
            train_X = X[train_index]
            train_y = y[train_index]
            test_X = X[test_index]
            test_y = y[test_index]
            new_clf = clone(clf)

            model_id = util.get_model_id(clf)
            dump_file = util.get_cache_file(model_id, test_index, )
            new_clf.fit(train_X, train_y)

            # test_prediction_cv = self.get_base_predict(new_model, test_X_cv, test_index)
            test_prob = new_clf.predit_prob(test_X)[:, 1]
            train_prob[test_index] = test_prob
            train_pred[test_index] = new_clf.predict(test_X)

            # 获得验证集上的分数
            score = eval_metric(test_y, test_prob)
            print("score is:", score)
            scores.append(score)
            cv_models[i] = copy.deepcopy(new_clf)

        clf.fit(X,y)
        models.append(clf)
        t2 = time.time()
        train_info["cv_time"] = t2 - t1
        train_info["cv_score_mean"] = np.mean(scores)
        train_info["cv_score_std"] = np.std(scores)
        return train_pred, train_prob, train_info, cv_models, models


     # 获得预测的评价方法
    def get_eval_metric(self):
        if self.metric.lower() == "auc":
            eval_metric = roc_auc_score
        elif self.metric.lower() == "logloss":
            eval_metric = log_loss
        else:
            raise ValueError("Got a unrecognized metric name: %s" % self.metric)
        return eval_metric

    def fit(self, X, y):
        self._fit(X, y)

    # 获得测试集的预测值
    def predict(self, X):
        test_preds = []
        test_probs = []
        for clf_name, clf in self.classfiers:
            if self.oob_flag:
                models = self.train_cv_models.get(clf_name)
                test_prob = np.zeros(X.shape[0], self.folds)
                test_pred = []

                for j in range(self.folds):
                    test_prob[:, j] = models[j].predict_proba(X)[:, 1]
                    test_pred.append(models[j].predict(X))
                test_prob = test_prob.mean(1)
                test_pred = np.array([np.argmax(np.bincount(x))for x in test_pred])
                test_probs.append(test_prob)
                test_preds.append(test_pred)
            else:
                model = self.train_models.get(clf_name)
                test_preds.append(model.predict(X))

        return np.array(test_preds).T

    # 获得测试集的预测概率
    def predict_prob(self, X):
        test_probs = []
        for clf_name, clf in self.classfiers:
            if self.oob_flag:
                models = self.train_cv_models.get(clf_name)
                test_prob = np.zeros(X.shape[0], self.folds)
                for j in range(self.folds):
                    test_prob[:, j] = models[j].predict_proba(X)[:, 1]
                test_prob = test_prob.mean(1)
                test_probs.append(test_prob)
            else:
                model = self.train_models.get(clf_name)
                test_probs.append(model.predict_prob(X)[:, 1])

        return np.array(test_probs).T

    # 计算测试集上的评分
    def evaluate(self, y_true, y_preds):

        for clf_name, clf in self.classfiers:
            y_pred = y_preds.get(clf_name)
            if y_pred:
                score = roc_auc_score(y_true, y_true)
            else:
                raise ValueError("predict values not exist")
            self.train_infos[clf_name]["test_score"] = score



    # # 获得测试集的训练信息
    # def get_test_info(self, X):
    #     for clf_name, clf in self.classfiers.items():
    #














    def predict_prob(self, X):
        pass
