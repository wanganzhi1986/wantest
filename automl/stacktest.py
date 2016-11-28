#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import prepocess
import feature_select
import StackingClassfier
import pandas as pd
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import preprocessing
from sklearn import svm
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc, roc_auc_score
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from feature_select import ReliefF
import  xgboost as xgb
import time
import datetime
# from ..optimise.baysopt import  BayesianOptimization
from sklearn import svm, grid_search, datasets
# from spark_sklearn import GridSearchCV
from sklearn.grid_search import GridSearchCV
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.cross_validation import StratifiedKFold
from evolutionary_search import EvolutionaryAlgorithmSearchCV
import scipy.optimize._lbfgsb


features = [ "MP0050048", "MP0041080","MP0045048", "MP0042110", "MS0050001", "mp_months", "MP0050003",
             "MP0110001", "MP0110008", "MP0110012","MP0110010", "mp_district_41", "MP0050047","MP0045110",
             "MP0050028", "MP0044026", "MP0050021", "MP0041053", "MP0045115", "MP0041060", "MP0050018",
             "MS0050011", "mp_district_13", "MP0042067", "MP0110029"]

def mobile_model():
    train_path = "/Users/wangwei/workplace/data_train_mp.csv"
    test_path = ""

    df = pd.read_csv(train_path)
    print(df.isnull())

    print("-----------数据预处理开始------------")
    start = time.time()
    print("start time is:", datetime.datetime.now().isoformat())
    pc = prepocess.PrepocessBase(
        train_path=train_path,
        test_path=test_path,
        split_value=0.33,
        include_columns=features,
        # exclude_columns=["applyNo", "overDueDays"]
        # exclude_columns=features

    )

    pc.fit()

    train_data = pc.train_data
    test_data = pc.test_data
    train_X = train_data.train_features
    train_y = train_data.train_label
    test_X = test_data.test_features
    test_y = test_data.test_label
    feature_names = train_data.feature_names

    print("orgin features num is:", len(feature_names))

    #
    # # 特征选择
    # fc = feature_select.FeatureSelection(
    #     train_data=train_data,
    #     test_data=test_data,
    #     feature_names=feature_names,
    #     filter_feature_strategy="xgboost",
    #     add_feature_strategy=[],
    #     feature_nums=300
    # )
    #
    # fc.fit(train_X, train_y)
    #
    # print("select feature num is:", len(fc.select_features))
    # print("select feature is:", fc.select_features)
    #
    # train_data, test_data = fc.transform()
    # train_X = train_data.train_features
    # train_y = train_data.train_label
    # test_X = test_data.test_features
    # test_y = test_data.test_label

    print("select data is:")
    print(test_X.shape)
    print(train_X.shape)


    xgboost = xgb.XGBClassifier()
    # params = {
    #     "n_estimators": range(10, 500, 10),
    #     "learning_rate": list(np.arange(0.001, 1, 0.005)),
    #     "max_depth": range(2, 10),
    # }

    rf = RandomForestClassifier()
    params = {
        "n_estimators": range(10, 500, 10),
        "max_depth": range(2, 10),
    }

    clf = EvolutionaryAlgorithmSearchCV(estimator=rf,
                                   params=params,
                                   scoring="roc_auc",
                                   cv=StratifiedKFold(train_y, n_folds=10),
                                   verbose=1,
                                   population_size=50,
                                   gene_mutation_prob=0.10,
                                   gene_crossover_prob=0.5,
                                   tournament_size=3,
                                   generations_number=5,
                                   n_jobs=4)
    clf.fit(train_X, train_y)

    pred_y = clf.predict_proba(test_X)[:, 1]
    score = roc_auc_score(test_y, pred_y)
    print("rf model score is:", score)

    lr = linear_model.LogisticRegression(C=0.01)
    lr.fit(train_X, train_y)
    # param_grid = {"C": np.arange(0.01, 1, 0.01)}
    # best_lr = GridSearchCV(lr, param_grid=param_grid, n_jobs=3)
    # best_lr.fit(train_X, train_y)
    # print("best param is:", best_lr.best_params_)
    score = roc_auc_score(test_y, lr.predict_proba(test_X)[:, 1])
    print("lr score is:", score)

    end = time.time()
    print("end time is:", datetime.datetime.now().isoformat())
    print("time interval is:", end-start)

    base_estimators = {
        "nb": naive_bayes.GaussianNB(),
        "lr": linear_model.LogisticRegression(C=0.01),
        # svm.SVC(kernel='linear', probability=True),
        "rf":RandomForestClassifier(n_estimators=300, max_depth=5,
            min_samples_split=200, min_samples_leaf=200),
        # DecisionTreeClassifier(max_depth=10, min_samples_split=1)
        "xgboost": xgb.XGBClassifier(n_estimators=300, learning_rate=5, nthread=5),

    }

    # probs = []
    # scores = []
    # for clf in base_estimators:
    #     print(clf.__class__.name)
    #     clf.fit(train_X, train_y)
    #     prob = clf.predict_proba(test_X)[:, 1]
    #     print("prob is:")
    #     print(prob)
    #     scores.append(roc_auc_score(test_y, prob))
    #     probs.append(prob)
    #
    # df = pd.DataFrame(np.array(probs)).T
    # print ("df is:")
    # print (df)
    # print("score is:")
    # print(scores)

   #  combiner = xgb.XGBClassifier(
   #     n_estimators=400,
   #     learning_rate=0.02,
   #     nthread=5,
   #     max_depth=7,
   #     subsample=0.5
   # )

    # combiner = linear_model.LogisticRegression(C=3)

    # combiner.fit(train_X, train_y)
    # prob = combiner.predict_proba(test_X)
    # pred = combiner.predict(test_X)
    #
    # score = roc_auc_score(test_y, prob[:, 1])
    # print("xgboost score is:", score)

    # print("************———————开始集成学习————————*******")
    # start = time.time()
    # print("start time is:", datetime.datetime.now().isoformat())
    # stc = StackingClassfier.StackingClassifier(
    #     extra_feature="origin",
    #     base_estimators=base_estimators,
    #     combiner=combiner,
    #     feature_names=feature_names
    # )
    # stc.fit(train_X, train_y)
    # scores = stc.evaluate_score(test_X, test_y)
    # print("classfier score is:")
    # print(scores)
    # end = time.time()
    # print("end time is:", datetime.datetime.now().isoformat())
    # print("model train time is :", end-start)


def transfer(X):
    from sklearn.preprocessing import OneHotEncoder
    step2 = ('OneHotEncoder', OneHotEncoder(categorical_features=[6], sparse=False))


def test_ensemble():
    from classfiers import GaussianNBClassfier, DecisionTreesClassfier, \
        LogistRegressionClassfier, AdaboostClassifier, XgboostClassfier, RandomForestClassfiers

    from datetime import datetime
    from sklearn.cross_validation import train_test_split
    from sklearn.preprocessing import MinMaxScaler, StandardScaler,OneHotEncoder
    from sklearn.preprocessing import MaxAbsScaler, Normalizer
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from datetime import datetime
    from util import get_time_interval

    base_clfs = {"nb": GaussianNBClassfier, "adaboost": AdaboostClassifier, "lr": LogistRegressionClassfier,
                 "dt": DecisionTreesClassfier
                 }

    clfs = dict()

    train_path = "/Users/wangwei/workplace/xgboost/test.txt"
    test_path = "/Users/wangwei/workplace/xgboost/test.txt"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    # df = pd.concat((df_train, df_test))
    train_y = df_train["label"].values
    # X = df[features].fillna(df.median()).values
    steps = [("impute")]
    feature_names = list(df_train[df_train.columns.drop("label")].columns)
    train_X = df_train[df_train.columns.drop("label")].fillna(df_train.median()).values
    # train_y = df["label"].values
    # train_X = df_train[df.columns.drop("label")].fillna(0.0).values
    # print("train X shape is:", X.shape)
    # train_X, test_X, train_y, test_y = train_test_split(X, y)
    test_y = df_test["label"].values
    test_X = df_test[df_test.columns.drop("label")].fillna(df_test.median()).values
    train_X_ = MinMaxScaler().fit_transform(train_X)
    test_X_ = MinMaxScaler().fit_transform(test_X)
    # train_X_ = OneHotEncoder(sparse=False).fit_transform(train_X_)
    # test_X_ = OneHotEncoder(sparse=False).fit_transform(test_X_)
    print("gaussian nb  train begin:")
    nb_clf = GaussianNBClassfier().fit(train_X_, train_y)
    nb_pred = nb_clf.predict_proba(test_X_)[:, 1]
    print("nb auc score is:", roc_auc_score(test_y, nb_pred))
    #
    lr_clf = LogistRegressionClassfier().fit(train_X_, train_y)
    # lr_best = lr_clf.best_estimator
    # lr_clf = LogisticRegression(C=2.4043).fit(train_X_, train_y)
    lr_pred = lr_clf.predict_proba(test_X_)[:, 1]
    print("lr auc score is:", roc_auc_score(test_y, lr_pred))

    # dt_clf = DecisionTreeClassifier(max_depth=5,  random_state=2).fit(train_X, train_y)
    # dt_pred = dt_clf.predict_proba(test_X_)[:, 1]
    # print("dt auc score is:", roc_auc_score(test_y, dt_pred))

    # bag_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=150).fit(train_X_, train_y)
    dt_clf = DecisionTreesClassfier().fit(train_X, train_y)
    dt_pred = dt_clf.predict_proba(test_X)[:, 1]
    print("bagging auc score is:", roc_auc_score(test_y, dt_pred))
    print("random forest is:")
    dt_train_df = pd.DataFrame(dt_clf.predict_proba(train_X), columns=["zero", "one"])

    dt_test_df = pd.DataFrame(dt_clf.predict_proba(test_X), columns=["zero", "one"])

    dt_train_df.to_csv("decision_tree_train.csv")
    start = datetime.now()
    dt_test_df.to_csv("decision_tree_test.csv")
    rf_clf = RandomForestClassfiers().fit(train_X, train_y)
    rf_pred = rf_clf.predict_proba(test_X)[:, 1]
    print("rf auc score is:", roc_auc_score(test_y, rf_pred))
    rf_train_df = pd.DataFrame(rf_clf.predict_proba(train_X), columns=["zero", "one"])
    rf_test_df = pd.DataFrame(rf_clf.predict_proba(test_X), columns=["zero", "one"])
    rf_train_df.to_csv("randomforest_train.csv")
    end = datetime.now()
    rf_test_df.to_csv("randomforest_test.csv")
    print("rf time is:", get_time_interval(start, end))

    ada_clf = AdaboostClassifier().fit(train_X, train_y)
    ada_train_df = pd.DataFrame(ada_clf.predict_proba(train_X), columns=["zero", "one"])
    ada_test_df = pd.DataFrame(ada_clf.predict_proba(test_X), columns=["zero", "one"])
    ada_train_df.to_csv("adaboost_train.csv")
    ada_test_df.to_csv("adaboost_test.csv")

    ada_pred = ada_clf.predict_proba(test_X)[:, 1]
    print("ada auc score is:", roc_auc_score(test_y, ada_pred))

    base_clfs = {
        "nb": nb_clf,
        # "adaboost": ada_clf,
        "lr": lr_clf,
        "dt": dt_clf,
        "rf": rf_clf
    }

    combiner = XgboostClassfier().fit(train_X, train_y)

    start = datetime.now()
    print("start time is:", datetime.now())
    stc = StackingClassfier.StackingClassifier(
        extra_feature="origin",
        base_estimators=base_clfs,
        combiner=combiner,
        feature_names=feature_names
    )
    stc.fit(train_X, train_y)
    scores = stc.evaluate_score(test_X, test_y)
    print("classfier score is:")
    print(scores)
    end = time.time()
    print("end time is:", datetime.datetime.now().isoformat())
    print("model train time is :", end-start)


if __name__ == "__main__":
    test_ensemble()






