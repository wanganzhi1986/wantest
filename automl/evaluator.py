#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import backend
import pipeline
import sklearn.cross_validation
from sklearn.cross_validation import KFold
from sklearn.base import clone, BaseEstimator
import copy
import util
from backend import Backend
import time


class AbstractEvaluator(BaseEstimator):

    def __init__(self, dataset_name,
                 classfier,
                 clf_name,
                 config,
                 output_dir=None,
                 metric="auc",
                 n_folds=10,
                 verbose=True,
                 eval_time_limit=None
                 ):
        self.config = config
        self.classfier = classfier
        self.output_dir = output_dir
        self.metric = metric
        self.n_folds = n_folds
        self.verbose = verbose
        self.eval_time_limit = eval_time_limit
        self.clf_name = clf_name

        self.backend = backend.Backend(self.output_dir)
        self.model = None
        self.start_time = time.time()
        self.duration = None

        self.dataset_name = dataset_name
        self.model = None
        self.cv_models = None
        self.train_pred = None
        self.train_prob = None
        self.test_prob = None
        self.test_pred = None
        self.score = None
        self.instance_ids = []


    def fit(self, X, y):
        raise NotImplementedError

    # 对结果进行预测
    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def finish_up(self, loss=None, train_prob=None, test_prob=None,  file_output=True):
        self.duration = time.time() - self.start_time

        if file_output:
            self.save_file(loss, train_prob, test_prob)
        else:
            loss_, additional_run_info_ = None, None

        if loss_ is not None:
            return self.duration, loss_, self.seed, additional_run_info_

        return self.duration, loss, self.seed, additional_run_info


    # 保存结果
    def save_file(self, model=None, train_prob=None, test_prob=None):
        if model:
            model_identifier = util.get_model_identifier(self.dataset_name, self.clf_name, model)
            self.backend.save_model(model, model_identifier)

        model_id = util.get_model_id(model)
        if train_prob:
            train_prob_identifier = self.dataset_name + "_" + self.clf_name
            self.backend.save_predictions_as_npy(train_prob, "train", train_prob_identifier)

        if test_prob:
            test_prob_identifier = self.dataset_name + "_" + self.clf_name + "_" + model_id
            self.backend.save_predictions_as_npy(test_prob, "test", test_prob_identifier)


class CrossValidationEvaluator(AbstractEvaluator):

    def __init__(self,
                 dataset_name,
                 classfier,
                 output_dir,
                 config=None,
                 metric="auc",
                 n_folds=10,
                 verbose=True,
                 eval_time_limit=None,
                 file_output=True
                 ):

        super(CrossValidationEvaluator, self).__init__(
                                                       classfier=classfier,
                                                       config=config,
                                                       output_dir=output_dir,
                                                       metric=metric,
                                                       verbose=verbose,
                                                       n_folds=n_folds,
                                                       eval_time_limit=eval_time_limit
                                                       )

        self.dataset_name = dataset_name
        self.model = None
        self.cv_models = None
        self.train_pred = None
        self.train_prob = None
        self.test_prob = None
        self.test_pred = None
        self.score = None
        self.instance_ids = []
        self.file_output = file_output

    def fit(self, X, y):
        if self.config is None:
            self.model = self.classfier()
        else:
            if type(self.config) == dict:
                self.model = clone(self.classfier).set_params(**self.config)
            else:
                raise ValueError("config type: %s is invalid"%str(type(self.config)))
        self._fit(X, y)
        if self.file_output:
            self.save_file(self.model, self.train_prob, self.test_prob)

    def _fit(self, X, y, eval_time_limit=None):
        eval_metric = util.get_eval_metric(self.metric)
        cv = KFold(len(y), n_folds=self.n_folds)
        train_prob = np.zeros((X.shape[0],))
        train_pred = np.zeros((X.shape[0],))
        cv_models = [None for i in range(self.n_folds)]
        scores = []
        start = time.time()
        for i, (train_index, test_index) in enumerate(cv):
            train_X = X[train_index]
            train_y = y[train_index]
            test_X = X[test_index]
            test_y = y[test_index]
            new_clf = clone(self.classfier)
            new_clf.fit(train_X, train_y)

            # test_prediction_cv = self.get_base_predict(new_model, test_X_cv, test_index)
            test_prob = new_clf.predict_proba(test_X)[:, 1]
            train_prob[test_index] = test_prob
            train_pred[test_index] = new_clf.predict(test_X)

            # 获得验证集上的分数
            score = eval_metric(test_y, test_prob)
            print(self.classfier.__name__ + "___score is:", score)
            scores.append(score)
            cv_models[i] = copy.deepcopy(new_clf)
            self.instance_ids.append(self.dataset_name+'_'+str(test_index))
            end = time.time()
            # 如果超过了时间的界限，则终止程序
            if eval_time_limit:
                if end - start > eval_time_limit:
                    break

        self.train_prob = train_prob.reshape((X.shape[0], 1))
        self.train_pred = train_pred.reshape((X.shape[0], 1))
        self.model.fit(X, y)
        self.score = np.mean(scores)

    def predict(self, X):
        test_pred = self.model.predict(X)
        self.test_pred = test_pred
        return self.test_pred

    def predict_proba(self, X):
        test_prob = self.model.predict_proba(X)[:, 1]
        self.test_prob = test_prob
        if self.file_output:
            self.save_file(test_prob)
        return test_prob








