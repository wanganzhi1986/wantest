#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from padal.core.algo.clfs.stack.classfiers import  XgboostClassfier, LogistRegressionClassfier, RandomForestClassfiers
from padal.core.algo.clfs.stack.paramselect import BayesianOptimise
from multiprocessing import Pool
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
import os
from datetime import datetime
import time
import copy
from sklearn.base import clone


def load_data(path):
    df = pd.read_csv(path)
    df["uid"] = np.array(["kn_" + str(i) for i in range(df.shape[0])])
    X = df.drop(["label", "uid"], axis=1).values
    y = df["label"].values
    uid = df["uid"].values
    return X, y, uid


def get_corr_plus_data():
    train_path = "../data/train_plus.csv"
    test_path = "../data/test_plus.csv"
    train_X, train_y, train_uid = load_data(train_path)
    test_X, test_y, test_uid = load_data(test_path)
    return train_X, test_X, train_y, test_y, train_uid, test_uid


def get_origin_data():
    train_path = "../data/train.csv"
    test_path = "../data/test.csv"
    train_X, train_y, train_uid = load_data(train_path)
    test_X, test_y, test_uid = load_data(test_path)
    return train_X, test_X, train_y, test_y, train_uid, test_uid


def get_discrete_data():
    train_path = "../data/train_discrete.csv"
    test_path = "../data/test_discrete.csv"
    train_X, train_y, train_uid = load_data(train_path)
    test_X, test_y, test_uid = load_data(test_path)
    return train_X, test_X, train_y, test_y, train_uid, test_uid



def get_rank_data():
    train_path = "../data/train_rank.csv"
    test_path = "../data/test_rank.csv"
    train_X, train_y, train_uid = load_data(train_path)
    test_X, test_y, test_uid = load_data(test_path)
    return train_X, test_X, train_y, test_y, train_uid, test_uid


def get_num_data():
    train_path = "../data/train_nd.csv"
    test_path = "../data/test_nd.csv"
    train_X, train_y, train_uid = load_data(train_path)

    test_X, test_y, test_uid = load_data(test_path)
    return train_X, test_X, train_y, test_y, train_uid, test_uid


def get_raw_data():
    train_path = "../data/train_raw.csv"
    test_path = "../data/test_raw.csv"
    train_X, train_y, train_uid = load_data(train_path)
    test_X, test_y, test_uid = load_data(test_path)
    # train_X = MinMaxScaler().fit_transform(train_X)
    # test_X = MinMaxScaler().fit_transform(test_X)
    return train_X, test_X, train_y, test_y, train_uid, test_uid


def get_merge_two_data():
    datas = get_train_data().items()
    merge_data = {}
    for data_name1, data1 in datas:
        train_X1, test_X1, train_y1, test_y1, train_uid, test_uid = data1
        for data_name2, data2 in datas:
            train_X2, test_X2, train_y2, test_y2, train_uid, test_uid = data2
            if data_name1 == data_name2:
                continue
            data_name = data_name1 + ":" + data_name2
            train_X = np.hstack([train_X1, train_X2])
            test_X = np.hstack([test_X1, test_X2])
            train_y = train_y1
            test_y = test_y1
            merge_data[data_name] = (train_X, test_X, train_y, test_y, train_uid, test_uid)
    return merge_data


def get_merge_all_data():
    train_Xs = []
    test_Xs = []
    train_y_ = None
    test_y_ = None,
    train_uid_ = None,
    test_uid_ = None
    for data_name, data in get_train_data().items():
        # if data_name == "num":
        #     continue
        train_X, test_X, train_y, test_y, train_uid, test_uid = data
        train_y_ = train_y
        test_y_ = test_y
        train_uid_ = train_uid
        test_uid_ = test_uid
        train_Xs.append(train_X)
        test_Xs.append(test_X)
    train_X = np.hstack(train_Xs)
    test_X = np.hstack(test_Xs)
    return train_X, test_X, train_y_, test_y_, train_uid_, test_uid_


def _train(train_X, train_y, test_X, test_y, clf):
    clf.fit(train_X, train_y)
    best_params = clf.best_params
    pred_y = clf.predict(test_X)


def get_lr_model(train_X, train_y, test_X, test_y, best_param=None):
    from sklearn.linear_model import LogisticRegression
    # train_X = MinMaxScaler().fit_transform(train_X)
    # test_X = MinMaxScaler().fit_transform(test_X)
    print train_X.shape
    print test_X.shape
    best_param = {"C": 0.04}
    if best_param:
        clf = LogisticRegression().set_params(**best_param)
        clf.fit(train_X, train_y)
    else:
        param_grid = {"C": np.arange(0.01, 1, 0.01)}
        clf = GridSearchCV(LogisticRegression(), param_grid=param_grid)
        clf.fit(train_X, train_y)
        # clf = LogistRegressionClassfier(C=0.02)
        # clf.fit(train_X, train_y)
        best_param = clf.best_params_
        print("best param is:", best_param)

    # train_pred = clf.predict_proba(train_X)[:, 1]
    train_pred = cross_valid_train(train_X, train_y, clf)
    test_pred = clf.predict_proba(test_X)[:, 1]
    auc_score = roc_auc_score(test_y, test_pred)
    print ("auc score is:", auc_score)
    return auc_score, best_param, clf, train_pred, test_pred


def get_tuning_xgboost_model(train_X, train_y, test_X, test_y):
    clf = XgboostClassfier()
    clf.fit(train_X, train_y)
    best_param = clf.best_params_
    pred_y = clf.predict_proba(test_X)[:, 1]
    auc_score = roc_auc_score(test_y, pred_y)
    print("best param is:", best_param)
    print("auc score is:", auc_score)



def get_xgboost_model(train_X, train_y, test_X, test_y, best_param=None):
    max_score = 0
    # for max_depth in range(1, 10):
    best_param = {'reg_alpha': 8.159752615976961,
                  'colsample_bytree': 0.8673698837685357,
                  'learning_rate': 0.23349173391703357,
                  'min_child_weight': 28,
                  'n_estimators': 235,
                  'reg_lambda': 481.84913690161386,
                  'max_depth': 4,
                  'gamma': 0.10233577351305655,
                  "subsample": 0.8560391704853437
                  }

    # best_param = {'reg_alpha': 363,
    #               'colsample_bytree':0.6632,
    #               'learning_rate':0.0128,
    #               'min_child_weight': 3,
    #               'n_estimators': 453,
    #               'reg_lambda': 463,
    #               'max_depth': 7,
    #               'gamma': 0.3244,
    #               "subsample": 0.036
    #               }
    # clf = xgb.XGBClassifier(max_depth=8,
    #                             n_estimators=800,
    #                             learning_rate=0.02,
    #                             gamma=0.1,
    #                             min_child_weight=20,
    #                             colsample_bytree=0.3,
    #                             reg_alpha=100,
    #                             reg_lambda=550,
    #                             silent=True,
    #                             # scale_pos_weight=1500.0/13458.0,
    #                             subsample=0.7,
    #                             )
    clf = xgb.XGBClassifier().set_params(**best_param)
    clf.fit(train_X, train_y)
    # train_pred = clf.predict_proba(train_X)[:, 1]

    train_pred = cross_valid_train(train_X, train_y, clf)
    test_pred = clf.predict_proba(test_X)[:, 1]
    auc_score = roc_auc_score(test_y, test_pred)
    # if auc_score > max_score:
    #     max_score = auc_score
    #     print("depth is:", max_depth)
    #     print("max score is:", max_score)
    return auc_score, best_param, clf, train_pred, test_pred


def save_model(model, name):
    if not os.path.exists("model"):
        os.mkdir("model")
    joblib.dump(model, './model/{0}.pkl'.format(name))


def save_pred(uid, pred, label, name, kind):
    if not os.path.exists("pred/"+kind):
        os.mkdir("pred/"+kind)
    # print("uid", uid.shape)
    # print("pred:", pred.shape)
    # print('label:', label.shape)
    test_result = pd.DataFrame(columns=["uid", "score", "label"])
    test_result.uid = uid
    test_result.score = pred
    test_result.label = label
    test_result.to_csv('./pred/{0}/{1}.csv'.format(kind, name), index=None)


def cross_valid_train(X, y, clf):
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


def get_train_data():
    raw_data = get_raw_data()
    num_data = get_num_data()
    discrete_data = get_discrete_data()
    rank_data = get_rank_data()
    datas = {"raw": raw_data,
             "num": num_data,
             "discrete": discrete_data,
             "rank": rank_data
             }
    return datas


def get_train_model():
    models = {
        "lr": get_lr_model,
        "xgb": get_xgboost_model
    }
    return models


def train_origin(seed=1, clf_name=None, data_name=None, save_file=True):
    d = {}
    for data_name_, data in get_train_data().items():
        print("data name is:", data_name_)
        train_X, test_X, train_y, test_y, train_uid, test_uid = data
        # auc_score = get_xgboost_model(train_X, train_y, test_X, test_y)
        for model_name, clf in get_train_model().items():
            if clf_name and model_name != clf_name:
                continue
            if data_name and data_name_ != data_name:
                continue
            name = data_name_ + "_" + model_name + "_" + str(seed)
            auc_score, best_param, model, train_pred, test_pred = clf(train_X, train_y, test_X, test_y)
            print("data:%{0}_{1} score is:".format(data_name_, model_name), auc_score)
            print("param is:", best_param)
            if save_file:
                save_model(model, name)
                save_pred(uid=train_uid, pred=train_pred, label=train_y, name=name, kind="train")
                save_pred(uid=test_uid, pred=test_pred, label=test_y,  name=name, kind="test")
        # d[data_name] = auc_score
    print ("各数据集的评分:", d)
    return d


def train_merge_two(seed=1, clf_name=None, save_file=True):
    result = {}
    for data_name, data in get_merge_two_data().items():
        train_X, test_X, train_y, test_y, train_uid, test_uid = data
        # auc_score = get_xgboost_model(train_X, train_y, test_X, test_y)
        for model_name, clf in get_train_model().items():
            if clf_name and model_name != clf_name:
                continue
            name = data_name + "_" + model_name + "_" + str(seed)
            auc_score, best_param, model, train_pred, test_pred = clf(train_X, train_y, test_X, test_y)
            print("data:%{0}_{1} score is:".format(data_name, model_name), auc_score)
            print("param is:", best_param)
            if save_file:
                save_model(model, name)
                save_pred(uid=train_uid, pred=train_pred, label=train_y, name=name, kind="train")
                save_pred(uid=test_uid, pred=test_pred, label=test_y, name=name, kind="test")
    print ("各数据集的评分:", result)
    return result


def train():
    result = {}
    result["origin"] = train_origin()
    result["merge2"] = train_merge_two()
    print("final result is:", result)
    return result

if __name__ == "__main__":
    # train()
    # train_origin(seed=1, clf_name="lr", data_name="raw")
    # train_merge_two(seed=1, clf_name="lr")
    # train_X, test_X, train_y, test_y, train_uid, test_uid = get_origin_data()
    # auc_score, best_param, clf, train_pred, test_pred=get_xgboost_model(train_X, train_y, test_X, test_y)
    # print("origin auc score is:", auc_score)

    # train_X, test_X, train_y, test_y, train_uid, test_uid = get_num_data()
    # auc_score, best_param, clf, train_pred, test_pred = get_xgboost_model(train_X, train_y, test_X, test_y)
    # print("raw auc score is:", auc_score)

    train_X, test_X, train_y, test_y, train_uid, test_uid = get_corr_plus_data()
    print("train x shape is:", train_X.shape)
    # auc_score, best_param, clf, train_pred, test_pred = get_xgboost_model(train_X, train_y, test_X, test_y)
    get_tuning_xgboost_model(train_X, train_y, test_X, test_y)
    # print("raw auc score is:", auc_score)

















