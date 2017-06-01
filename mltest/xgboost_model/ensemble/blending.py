#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from padal.core.algo.clfs.stack.ensemble_selection import EnsembleSelection
import os
from sklearn.metrics import roc_auc_score
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


def train_weight_ensemble():
    train_dir = "../M1/pred/train"
    test_dir = "../M1/pred/test"
    train_models, train_model_names, train_labels = get_pred_data(train_dir)
    test_models, test_model_names, test_labels = get_pred_data(test_dir)

    ensemble = EnsembleSelection(model_names=train_model_names, mode="vote", ensemble_size=50, corr_select=True)
    train_label = train_labels[0]
    ensemble.fit(train_models, train_label)
    model_corr = ensemble.models_correlation
    select_model_names = ensemble.select_model_names
    print('model correlation is:', model_corr)
    print("select model names is:", select_model_names)
    preds = copy.deepcopy(test_models)
    pred_y = ensemble.predict_proba(preds)
    print("pred is:", pred_y)
    test_label = test_labels[0]
    auc_score = roc_auc_score(test_label, pred_y)
    print("ensemble train weight score is:", auc_score)

    avg_pred = np.mean(test_models, axis=0)
    print("ensemble avg score is:", roc_auc_score(test_label, avg_pred))

    train_scores, test_scores = get_single_score()
    sum_scores = sum([score for _, score in test_scores])
    test_weight = {name: score/sum_scores for name, score in test_scores}
    for i, pred in enumerate(test_models):
        model_name = test_model_names[i]
        weight = test_weight.get(model_name)
        test_models[i] = test_models[i] * weight
    pred = np.sum(test_models, axis=0)
    auc_score = roc_auc_score(test_label, pred)
    print("ensemble test weight score is:", auc_score)


def train_stack_ensemble():
    train_dir = "../M1/pred/train"
    test_dir = "../M1/pred/test"
    train_models, train_model_names, train_labels = get_pred_data(train_dir)
    test_models, test_model_names, test_labels = get_pred_data(test_dir)
    train_X = np.column_stack(train_models)
    train_y = train_labels[0]
    test_X = np.column_stack(test_models)
    test_y = test_labels[0]
    param_grid = {"C": np.arange(0.01, 1, 0.01)}
    clf = GridSearchCV(LogisticRegression(), param_grid=param_grid)
    # param_grid = {"n_estimators":range(100, 1000, 100), "learning_rate":np.arange(0.01, 0.1, 0.01)}
    # clf = xgb.XGBClassifier(n_estimators=500, learning_rate=0.02, max_depth=4)
    # param_grid = {"n_estimators":np.arange(10, 100, 10), "max_depth": range(1, 10)}
    # clf = GridSearchCV(RandomForestClassifier(), param_grid=param_grid)
    # clf = LogisticRegression(C=0.01)
    # clf = GridSearchCV(xgb.XGBClassifier(n_estimators=500, learning_rate=0.02, max_depth=4), param_grid=param_grid)
    clf.fit(train_X, train_y)
    best_params = clf.best_params_
    print("best params is:", best_params)
    pred_y = clf.predict_proba(test_X)[:, 1]
    auc_score = roc_auc_score(test_y, pred_y)
    print("stack ensemble score is:", auc_score)


def get_single_score():
    train_dir = "../M1/pred/train"
    test_dir = "../M1/pred/test"
    train_models, train_model_names, train_labels = get_pred_data(train_dir)
    test_models, test_model_names, test_labels = get_pred_data(test_dir)
    train_scores = {}
    test_scores = {}
    for i, test_pred in enumerate(train_models):
        train_score = roc_auc_score(train_labels[i], train_models[i])
        test_score = roc_auc_score(test_labels[i],  test_models[i])
        train_scores[train_model_names[i]] = train_score
        test_scores[test_model_names[i]] = test_score
    train_scores_ = sorted(train_scores.items(), key=lambda x: x[1], reverse=True)
    test_scores_ = sorted(test_scores.items(), key=lambda x: x[1], reverse=True)
    return train_scores_, test_scores_





def get_pred_data(path, clf_name=None, data_name=None):
    model_names = []
    models = []
    labels = []
    for fs in os.listdir(path):
        base_name = os.path.basename(fs)
        model_name, _ = os.path.splitext(base_name)
        mns = model_name.split("_")
        dn = mns[0]
        cn = mns[1]
        if clf_name and cn != clf_name:
            continue
        if data_name and dn != data_name:
            continue
        model_names.append(model_name)
        df = pd.read_csv(os.path.join(path, base_name))
        models.append(df["score"].values)
        labels.append(df["label"].values)
    return models, model_names, labels

if __name__ == "__main__":
    import pprint
    import json
    train_scores, test_scores = get_single_score()
    print("train score is:")
    print train_scores
    print("test score is:")
    print test_scores
    train_weight_ensemble()
    # train_stack_ensemble()


