#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from bayes_opt import BayesianOptimization
from bayes_opt.helpers import UtilityFunction, unique_rows, PrintLog, acq_max
import smac
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import pandas as pd
from sklearn.cross_validation import cross_val_score
import json
import collections
import time
from datetime import  datetime
import util
import os, errno
from sklearn.base import BaseEstimator, clone
import logging


# 初始化的点
class InitalDesign(object):

    def __init__(self, init_points, estimator, model,
                 target_func,
                 pbounds=None):
        self.model = model
        self.target_func = target_func

        if pbounds is None:
            raise ValueError("params bounds must be given")

        if init_points:
            if type(init_points) is int:

                points = []
                for pbound in pbounds.values():
                    x = np.random.uniform(pbound[0], pbound[1], size=init_points)
                    # points.append(x)
                    if pbound[2] == "Integer":
                        points.append(x.astype(int))
                    else:
                        points.append(x)

                df = pd.DataFrame(np.array(points), index=pbounds.keys()).T
                self.init_points = df.to_dict(orient="records")

            elif type(init_points) is dict:
                self.init_points = init_points

            else:
                raise ValueError("init points type not supported")
        else:
            raise ValueError("init search points must given")

    def run(self, X, y):
        param_x = []
        param_y = []
        for init_point in self.init_points:
            param_x.append(init_point.values())
            param_y.append(self.target_func(init_point))
        return np.array(param_x), np.array(param_y)


RunKey = collections.namedtuple(
    'RunKey', ['config_id'])

RunValue = collections.namedtuple(
    'RunValue', ['cost', 'time', 'status', 'score', 'additional_info'])


class BayesianOptimise(BaseEstimator):

    def __init__(self,
                 estimator,
                 save_dir=None,
                 acq_func="ei",
                 model=None,
                 target_func=None,
                 init_points=5,
                 model_params=None,
                 pbounds=None,
                 verbose=True,
                 n_iters=20,
                 kappa=2.576,
                 xi=0.0,
                 refit=True,
                 reuse=True

                 ):

        self.acq_func = acq_func
        self.target_func = target_func
        self.estimator = estimator
        self.init_points = init_points
        self.pbounds = pbounds
        self.verbose = verbose
        self.n_iters = n_iters
        self.kappa = kappa
        self.xi = xi
        self.save_dir = save_dir
        self.refit = refit
        self.reuse = reuse
        self.save_dir = save_dir
        if self.save_dir is not None:
            self.create_save_dir()

        self.model_params = model_params

        self.output_file = None
        self.max_value = None
        self.best_params_ = None
        self.train_infos = []
        self.best_score_ = None
        self.best_estimator_ = None
        self.plog = PrintLog(list(pbounds.keys()))
        if model is None:
            if model_params is None:
                self.model = GaussianProcessRegressor(
                kernel=Matern(),
                n_restarts_optimizer=25,
                    )
            else:
                if type(model_params) is dict:
                    self.model = GaussianProcessRegressor(**model_params)
                else:
                    raise ValueError("model params type must be dict type")

        if self.acq_func not in ["ucb", "ei", "poi"]:
            raise ValueError("acquistion_function %s not been implemented" %self.acq_func)

    def fit(self, X, y):
        if self.target_func is None:
            self.target_func = lambda x: cross_val_score(clone(self.estimator).set_params(**x), X, y, 'roc_auc', cv=10).mean()
        self.run(X, y)

        if self.refit:
            best_estimator = clone(self.estimator).set_params(**self.best_params_)
            if y is not None:
                best_estimator.fit(X, y)
            else:
                best_estimator.fit(X)
            self.best_estimator_ = best_estimator
        return self

    def run(self, X, y):
        param_names = self.pbounds.keys()
        bounds = np.array([(pbound[0], pbound[1]) for pbound in self.pbounds.values()])
        data_type = [pbound[2] for pbound in self.pbounds.values()]
        # 获得运行的初始值,首先读取已经训练的结果,如果不存在,则进行初始随机化点
        try:
            init_param_X, init_param_y = self.load_interation()
        except Exception as e:
            init_design = InitalDesign(self.init_points, self.estimator, self.model, self.target_func,  self.pbounds)
            init_param_X, init_param_y = init_design.run(X, y)
        self.util = UtilityFunction(kind=self.acq_func, kappa=self.kappa, xi=self.xi)
        self.best_score_ = init_param_y.max()
        self.best_params_ = dict(zip(self.pbounds.keys(), init_param_X[init_param_y.argmax()]))
        # print("init value is:")
        # print(init_param_y)

        x_max = self.choose_next(init_param_X, init_param_y)

        # 打印训练结果
        if self.verbose:
            self.plog.print_header(initialization=False)

        param_X = init_param_X
        param_y = init_param_y
        for i in range(self.n_iters):
            # 如果失败,读取保存的文件继续执行
            start = datetime.now()
            train_info = dict()
            try:
                # 判断所获得最大值是否已经存在,如果存在,随机获取一个新的值
                pwarning = False
                if np.any((param_X - x_max).sum(axis=1) == 0):
                    # x_max = []
                    # for bound in self.pbounds.values():
                        # v = np.random.uniform(bound[0], bound[1])
                        # if bound[2] == "Integer":
                        #     x_max.append(int(v))
                        # else:
                        #     x_max.append(v)

                    x_max = np.array([np.random.uniform(bound[0], bound[1]) for bound in self.pbounds.values()])
                    # pwarning = True

                param = []
                for i in range(len(data_type)):
                    if data_type[i] == "Integer":
                        param.append(int(x_max[i]))
                    else:
                        param.append(float(x_max[i]))
                new_est_param = dict(zip(param_names, param))
                # 进行参数性能的评价
                new_param_y = self.target_func(new_est_param)
                # print("new param y is:")
                # print(new_param_y)
                if new_param_y > self.best_score_:
                    self.best_params_ = new_est_param
                    self.best_score_ = new_param_y
                param_X = np.vstack((param_X, x_max.reshape((1, -1))))
                param_y = np.append(param_y, new_param_y)
                x_max = self.choose_next(param_X, param_y)
                if self.verbose:
                    self.plog.print_step(param_X[-1], param_y[-1], warning=pwarning)
                i += 1
                end = datetime.now()
                interval = util.get_time_interval(start, end)
                train_info["iters"] = i
                train_info["status"] = "success"
                train_info["time"] = interval
                train_info["score"] = param_y[-1]
                train_info.update(dict(zip(self.pbounds.keys(), param_X[-1])))
                self.train_infos.append(train_info)
            except Exception as e:
                logging.exception(e)
                print("run exception is:", str(e))
                end = datetime.now()
                i += 1
                train_info["iters"] = i
                train_info["status"] = "fail"
                train_info["time"] = util.get_time_interval(start, end)
                train_info["score"] = param_y[-1]
                train_info.update(dict(zip(self.pbounds.keys(), param_X[-1])))
                self.train_infos.append(train_info)
                self.save_iterations(self.train_infos)
                continue
        print("best params is:", self.best_params_)
        self.save_iterations(self.train_infos)
        if self.verbose:
            self.plog.print_summary()


    # 选择下一个运行的参数配置,
    def choose_next(self, X, y):
        """choose next value from given model
        Parameters
        ----------
        X : array of shape [n_run_samples, n_features]
            has run hyperparamter config nums
        y : array of shape [n_run_samples]
            The predicted target values from given target algothrim.
        """
        # if self.model_params:
        #     self.model.set_params(**self.model_params)
        ur = unique_rows(X)
        self.model.fit(X[ur], y[ur])
        y_max = self.best_score_
        bounds = np.array([(pbound[0], pbound[1])for pbound in self.pbounds.values()])
        x_max = acq_max(ac=self.util.utility,
                        gp=self.model,
                        y_max=y_max,
                        bounds=bounds)
        return x_max

    #绘制参数运行的曲线
    def plot_curve(self):
        pass

    # 保存每次迭代的信息
    def save_iterations(self, train_infos):
        try:
            df = pd.DataFrame(train_infos)
            df.to_csv(self.output_file, encoding="utf-8")
        except Exception as e:
            raise

    # 创建保存的文件夹
    def create_save_dir(self):

        try:
            os.makedirs(self.save_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        self.output_file = os.path.join(self.save_dir, 'results.csv')

    # 加载训练的历史信息
    def load_interation(self):

        path = os.path.join(self.save_dir, "result.csv")
        df = pd.read_csv(path)
        param_X = df[df["status"] == "success"][self.pbounds.keys()].values
        param_y = df["score"].values
        return param_X, param_y

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)


class SMBO(object):

    def __init__(self):
        pass


if __name__ == "__main__":
    features = ['MP0110012','MP0110008','MP0110010','MP0110029','MP0041080','MP0042110','MP0045110',
    'MP0044026' ,'MP0041053','MP0045115' ,'MP0042067','MP0050048','MP0050021','MP0050047',
    'MP0050028','MP0050018','MP0041060','MP0110011','MP0110002','MP0110003',
    'MP0110005' ,'MP0110006', 'MP0110007', 'MP0110009', 'MP0110022', 'MP0110023', 'MP0110024',
    'MP0110025', 'MP0110026', 'MP0110027', 'MP0110030', 'MP0110032', 'MP0110033', 'MP0110036',
    'MP0110037' ,'MP0110045']

    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import MinMaxScaler, Normalizer
    import xgboost as xgb
    train_path = "/Users/wangwei/workplace/xgboost/test.txt"
    test_path = "/Users/wangwei/workplace/xgboost/test.txt"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df = pd.concat((df_train, df_test))
    y = df["label"].values
    # X = df[features].fillna(df.median()).values
    X = df[df.columns.drop("label")].fillna(df.median()).values
    # train_y = df["label"].values
    # train_X = df_train[df.columns.drop("label")].fillna(0.0).values
    print("train X shape is:", X.shape)
    train_X, test_X, train_y, test_y = train_test_split(X, y)

    # train_X = MinMaxScaler().fit_transform(train_X)
    # test_X = MinMaxScaler().fit_transform(test_X)
    # train_X = Normalizer().fit_transform(train_X)
    # test_X = Normalizer().fit_transform(test_X)
    # lr_params = {"C": (0.01, 1)}
    xgb_params ={
        "n_estimators": (10, 1000, "Integer"),
        "learning_rate": (0.001, 1, "Float"),
        "max_depth": (2, 10, "Integer"),
    }
    # baysopt = GridSearchCV(LogisticRegression(), param_grid={"C":np.arange(0.001, 2, 0.001)})
    start = datetime.now()
    baysopt = BayesianOptimise(
        estimator=xgb.XGBClassifier(),
        save_dir="/Users/wangwei/workplace/bays_train",
        init_points=5,
        acq_func="ei",
        pbounds=xgb_params
    )
    # clf = xgb.XGBClassifier(n_estimators=400, max_depth=7, learning_rate=0.02)
    #
    baysopt.fit(train_X, train_y)
    # baysopt.fit(X, y)
    # clf.fit(train_X, train_y)
    # df_test = pd.read_csv(test_path)
    # test_y = df_test["label"].values
    # test_X = df_test[df_test.columns.drop("label")].fillna(0.0).values
    # print("test shape is:", test_X.shape)
    pred_y = baysopt.predict_proba(test_X)[:, 1]
    score = roc_auc_score(test_y, pred_y)
    end = datetime.now()
    print("score is:", score)
    interval = util.get_time_interval(start, end)
    print("optimise time is: %s"%interval)














