#!/usr/bin/env python
# -*- coding: utf-8 -*-
import stackone
import stackfold
import time
import datetime
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.externals import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
import pandas as pd
import copy
import json
import os
import codecs
from sklearn import cross_validation
from sklearn.preprocessing import LabelBinarizer
import util
import logging


class BaseStackingModel(BaseEstimator):

    def __init__(self,
                 train_X,
                 train_y,
                 test_X,
                 test_y,
                 metadata_folder,
                 ensemble_folder,
                 stack_by_prob,
                 save_base_dump=True,
                 folds=10
                 , stratify=True,
                 metric="auc",
                 oob_flag=True,
                 n_classes=2

                 ):

        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.folds = folds
        self.stratify = stratify
        self.metric = metric
        self.oob_flag = oob_flag
        self.n_classes = n_classes
        self.save_base_dump = save_base_dump
        self.ensemble_folder = ensemble_folder
        self.stack_by_prob = stack_by_prob
        self.metadata_folder = metadata_folder
        self.models = [None for i in range(folds)]
        self.cv_time = None
        self.test_time = None
        self.cv_score_mean = None
        self.cv_score_std = None
        self.test_score = None
        self.training_predictor = None
        self.test_predictor = None


    # 获得模型预测产生的文件
    def get_model_metadata(self, model):

        try:
            train_prediction , test_prediction = self.load_model_meta(model)
        except Exception as e:
            if self.training_predictor is None or self.test_predictor is None:
                train_prediction = self.make_train_meta(model)
                test_prediction = self.make_test_metadata(model)
            else:

                train_prediction = self.training_predictor.values()
                test_prediction = self.test_predictor.values()

                # self.save_files(model)
        return train_prediction, test_prediction


    # 产生每个模型的预测概率
    def make_train_meta(self, model):

        if "predict_proba" in dir(model): model.predict = model.predict_proba

        eval_metric = self.get_eval_metric()

        # cp = util.CrossPartitioner(n=len(self.train_y) if not self.stratify else None,
        #                       y=self.train_y,
        #                       k=self.folds,
        #                       stratify=self.stratify,
        #                       shuffle=True,
        #                       random_state=655321)

        scores = []

        t1 = time.time()
        cv = cross_validation.KFold(len(self.train_y), n_folds=self.folds)
        # gen = cp.make_partitions(input=self.train_X, target=self.train_y,  append_indices=False)
        train_prediction = None
        for cv_index, (train_index, test_index) in enumerate(cv):
            train_X_cv = self.train_X[train_index]
            train_y_cv = self.train_y[train_index]
            test_X_cv = self.train_X[test_index]
            test_y_cv = self.train_y[test_index]
            new_model = clone(model)
            model_id = util.get_model_id(model)
            dump_file = util.get_cache_file(model_id, test_index, )
            # 判断是否有id
            if not hasattr(new_model, 'id'):
                    new_model.id = util.get_model_id(new_model)
            if self.save_base_dump and self.is_saved(model, test_index):
                new_model = joblib.load(dump_file)

            else:
                new_model.fit(train_X_cv, train_y_cv)
                if self.save_base_dump:
                    joblib.dump(new_model, dump_file, compress=True)

            if train_prediction is None:
                train_prediction = self.get_blend_init(new_model, self.train_y)

            test_prediction_cv = self.get_base_predict(new_model, test_X_cv, test_index)

            train_prediction[test_index] = test_prediction_cv

            # 获得验证集上的分数
            score = eval_metric(test_y_cv, test_prediction_cv)
            print("score is:", score)
            scores.append(score)

            # self.training_predictor = train_prediction

            # prediction_batches.extend(test_prediction_cv)
            # indices_batches.extend(test_id_cv)
            # assert len(prediction_batches) == len(indices_batches)
            self.models[cv_index] = copy.deepcopy(new_model)

        t2 = time.time()
        self.cv_time = t2 - t1
        self.cv_score_mean = np.mean(scores)
        self.cv_score_std = np.std(scores)
        self.training_predictor = pd.DataFrame(train_prediction)
        # training_predictor = pd.DataFrame({"target": prediction_batches}, index=indices_batches).ix[self.train_id]
        # assert len(training_predictor) == len(train_X)
        # self.training_predictor = training_predictor
        return train_prediction

    # 产生测试集的预测概率及运行状态
    def make_test_metadata(self, model):

        eval_metric = self.get_eval_metric()
        t1 = time.time()
        if self.oob_flag:
            dataset_blend = np.full(
                (np.shape(self.test_X)[0], self.folds),
                np.nan
            )
            for j in range(self.folds):
                dataset_blend[:, j] = self.models[j].predict_proba(self.test_X)[:, 1]
            test_prediction = dataset_blend.mean(1)

        else:
            model.fit(self.train_X, self.train_y)
            test_prediction = model.predict(self.test_X)

        self.test_score = eval_metric(self.test_y, test_prediction)
        t2 = time.time()
        self.test_time = t2 - t1
        self.test_prediction = test_prediction
        self.test_predictor = pd.DataFrame(test_prediction)
        return test_prediction
        # test_prediction = np.reshape(test_prediction, (len(test_id), test_prediction.ndim))  # this code
        # # forces having 2D
        # test_prediction = test_prediction[:, -1]  # Extract the last column
        # test_predictor = pd.DataFrame({"target": test_prediction}, index=test_id)
        # self.test_predictor = test_predictor
        # return test_predictor

    # 用来保存训练后的测试和训练文件和训练的基本信息
    def save_files(self, model, alias=None, metadata=None):
        model_id = util.get_model_id(model)
        folder = self.metadata_folder
        if not alias:
            train_file_path = os.path.join(folder, model_id+"train_predictor.csv")
            test_file_path = os.path.join(folder, model_id+"test_predictor.csv")
        else:
            train_file_path = os.path.join(folder, alias + model_id+"train_predictor.csv")
            test_file_path = os.path.join(folder, alias + model_id+"test_predictor.csv")
        #
        # train_file_path = make_sure_do_not_replace(train_file_path)
        # test_file_path= make_sure_do_not_replace(test_file_path)

        # self.training_predictor.to_csv(train_file_path, sep=",", encoding="utf-8")
        # self.test_predictor.to_csv(test_file_path, sep=",", encoding="utf-8")

        index = {
            "name": metadata["name"],
            "description": metadata["description"],
            "cv": {
                "score_mean": self.cv_score_mean,
                "score_std": self.cv_score_std,
                "score_metric": self.metric,
                "folds": self.folds,
                "stratify": self.stratify,
                "shuffle": True,
                "time": self.cv_time
            },
            "total_time": self.test_time + self.cv_time,
            "alias": alias,
            "test_file_path": test_file_path,
            "train_file_path": train_file_path,
            "datetime": datetime.datetime.now().isoformat(),
        }

        for key in metadata:
            if key == "name" or key == "description" or key in index:
                continue
            else:
                index[key] = metadata[key]
        json_dump = json.dumps(index)
        with codecs.open(os.path.join(folder, model_id+"index.jl"), "a", "utf-8") as f:
            f.write(json_dump + "\r\n")

    # 加载预测的数据
    def load_model_meta(self, model):
        model_id = util.get_model_id(model)
        try:
            train_predict_path = os.path.join(self.metadata_folder, model_id+"train_predictor.csv")
            test_predict_path = os.path.join(self.metadata_folder, model_id+"test_predictor.csv")
            df_train = pd.read_csv(train_predict_path)
            df_test = pd.read_csv(test_predict_path)
            return df_train.values, df_test.values
        except Exception as e:
            raise ValueError("read saved data failed")

    # 加载训练的信息数据
    def load_model_index(self):
        pass

    def is_saved(self, model, index):
        model_id = self.ensemble_folder + util.get_model_id(model)
        return os.path.isfile(util.get_cache_file(model_id, index))

    def get_base_predict(self, clf, X, index=None):
        if self.stack_by_prob and hasattr(clf, 'predict_proba'):
            # if self.save_base_dump and index is not None:
            #     proba = util.saving_predict_proba(clf, X, index)
            # else:
            proba = clf.predict_proba(X)
            return proba[:, 1:]
        elif hasattr(clf, 'predict'):
            predict_result = clf.predict(X)
            if isinstance(clf, ClassifierMixin):
                lb = LabelBinarizer()
                lb.fit(predict_result)
                return lb.fit_transform(predict_result)
            else:
                return predict_result.reshape((predict_result.size, 1))
        else:
            return clf.fit_transform(X)

    def get_blend_init(self, clf, train_y):
        if self.stack_by_prob and hasattr(clf, 'predict_proba'):
            width = self.n_classes - 1
        elif hasattr(clf, 'predict') and isinstance(clf, ClassifierMixin):
            width = self.n_classes
        elif hasattr(clf, 'predict'):
            width = 1
        elif hasattr(clf, 'n_components'):
            width = clf.n_components
        else:
            raise Exception('Unimplemented for {0}'.format(type(clf)))
        return np.zeros((train_y.size, width))

    # 获得预测的评价方法
    def get_eval_metric(self):
        if self.metric.lower() == "auc":
            eval_metric = roc_auc_score
        elif self.metric.lower() == "logloss":
            eval_metric = log_loss
        else:
            raise ValueError("Got a unrecognized metric name: %s" % self.metric)
        return eval_metric


class StackingClassifier(BaseEstimator):

    def __init__(self,
                 base_estimators,
                 combiner,
                 feature_names,
                 meta_optimser="grid",
                 metric="auc",
                 base_folds=10,
                 meta_folds=5,
                 extra_feature=None,
                 metadata_folder="metadata",
                 ensemble_folder="ensemble",
                 n_classes=2,
                 oob_flag=False
                 ):

        self.base_estimators = base_estimators
        self.combiner = combiner
        self.feature_names = feature_names
        self.meta_optimser = meta_optimser
        self.extra_feature = extra_feature
        self.metric = metric
        self.n_classes = n_classes
        self.base_folds = base_folds
        self.meta_folds = meta_folds
        self.oob_flag = oob_flag


        self.blend_train = None
        self.blend_test = None
        self.train_models = None
        self.train_infos = None
        self.train_probs = None
        self.train_preds = None
        self.train_cv_models = None
        self.test_infos = None
        self.evaluate_info = dict()
        self.blend_predict_prob = None

    def fit(self, X, y):
        meta_X = self._fit(X, y)
        if self.extra_feature:
            X = self.make_new_features(X, meta_X)
            self.combiner.fit(X, y)

        else:
            self.combiner.fit(meta_X, y)
        return self

    def _fit(self, X, y):
        train_preds = []
        train_probs = []
        train_infos = dict()
        train_models = dict()
        train_cv_models = dict()
        for clf_name, clf in self.base_estimators.items():
            train_pred, train_prob, train_info, train_cv_model, train_model = self.make_train_meta(X, y, clf_name)
            train_preds.append(train_pred)
            train_probs.append(train_prob)
            train_infos[clf_name] = train_info
            train_models[clf_name] = train_model
            train_cv_models[clf_name] = train_cv_model

        train_probs = np.hstack(train_probs)
        self.train_probs = train_probs
        self.train_preds = np.hstack(train_preds)
        self.train_infos = train_infos
        self.train_models = train_models
        self.train_cv_models = train_cv_models
        return train_probs

    # 获得训练集上的元数据
    def make_train_meta(self, X, y, clf_name):
        eval_metric = self.get_eval_metric()
        clf = self.base_estimators.get(clf_name)
        scores = []

        t1 = time.time()
        cv = cross_validation.KFold(len(y), n_folds=self.base_folds)
        train_prob = np.zeros((X.shape[0], ))
        train_pred = np.zeros((X.shape[0], ))
        cv_models = [None for i in range(self.base_folds)]
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
            test_prob = new_clf.predict_proba(test_X)[:, 1]
            train_prob[test_index] = test_prob
            train_pred[test_index] = new_clf.predict(test_X)

            # 获得验证集上的分数
            score = eval_metric(test_y, test_prob)
            print(clf_name + "___score is:", score)
            scores.append(score)
            cv_models[i] = copy.deepcopy(new_clf)

        train_prob = train_prob.reshape((X.shape[0], 1))
        train_pred = train_pred.reshape((X.shape[0], 1))
        clf.fit(X, y)
        # models.append(clf)
        t2 = time.time()
        train_info["feature_important"] = self.get_feature_important(clf, clf_name)
        train_info["cv_time"] = t2 - t1
        train_info["cv_score_mean"] = np.mean(scores)
        train_info["cv_score_std"] = np.std(scores)
        return train_pred, train_prob, train_info, cv_models, clf

    # 获得测试集上的元数据
    def make_test_meta(self, X):
        test_preds = []
        test_probs = []
        for clf_name, clf in self.base_estimators.items():
            evaluate_info = dict()
            if self.oob_flag:
                models = self.train_cv_models.get(clf_name)
                test_prob = np.zeros(X.shape[0], self.base_folds)
                test_pred = []

                for j in range(self.base_folds):
                    test_prob[:, j] = models[j].predict_proba(X)[:, 1]
                    test_pred.append(models[j].predict(X))
                test_prob = test_prob.mean(1)
                test_pred = np.array([np.argmax(np.bincount(x))for x in test_pred])
                test_probs.append(test_prob)
                test_preds.append(test_pred)
            else:
                model = self.train_models.get(clf_name)
                test_pred = model.predict(X)
                test_prob = model.predict_proba(X)[:, 1]
                test_preds.append(test_pred)
                test_probs.append(test_prob)

            evaluate_info["prob"] = test_prob
            evaluate_info["pred"] = test_pred
            self.evaluate_info[clf_name] = evaluate_info
        return np.array(test_preds).T

    # 使用集成学习预测概率
    def predict_proba(self, X):
        blend_test = self.make_test_meta(X)
        if self.extra_feature:
            blend_test = self.make_new_features(X, blend_test)
        prob = self.combiner.predict_proba(blend_test)[:, 1]
        if self.evaluate_info.get("stack"):
            self.evaluate_info["stack"].update({"prob": prob})
        else:
            self.evaluate_info["stack"] = {"prob": prob}

        return prob

    def predict(self, X):
        blend_test = self.make_test_meta(X)
        if self.extra_feature:
            blend_test = self.make_new_features(X, blend_test)
        pred = self.combiner.predict(blend_test)
        if self.evaluate_info.get("stack"):
            self.evaluate_info["stack"] = {"pred": pred}
        else:
            self.evaluate_info["stack"].update({"preb": pred})
        return pred

    #按照指定的策略将元数据和原来的特征进行合并构成新的特征
    def make_new_features(self, X, meta_X):
        allow_stragey = ["origin", "xgboost", "lasso", "rf"]
        if self.extra_feature not in allow_stragey:
            raise ValueError("select stragey not support")

        if X.shape[0] != meta_X.shape[0]:
            raise ValueError("orgin data shape diff meta data shape")

        new_X = X
        if self.extra_feature == "orgin":
            new_X = np.hstack([X, meta_X])

        if self.extra_feature == "lasso":
            important_features= self.train_infos.get("lr").get("feature_important")
            select_X = self.get_select_featurs(important_features, X)
            new_X = np.hstack([select_X, meta_X])

        if self.extra_feature == "rf":
            important_features = self.train_infos.get("rf").get("feature_important")
            select_X = self.get_select_featurs(important_features, X)
            new_X = np.hstack([select_X, meta_X])
        return new_X

    # 获得预测的评价方法
    def get_eval_metric(self):
        if self.metric.lower() == "auc":
            eval_metric = roc_auc_score
        elif self.metric.lower() == "logloss":
            eval_metric = log_loss
        else:
            raise ValueError("Got a unrecognized metric name: %s" % self.metric)
        return eval_metric


    # 获得特征的重要性排名
    def get_feature_important(self, clf, clf_name, top_n=10):
        important_weights = dict()
        if clf_name == "lr":
            important_weights = dict(sorted(zip(self.feature_names, list(clf.coef_)),
                                       key=lambda x:x[1], reverse=True)[:top_n])

        if clf_name == "rf":
            important_weights = dict(sorted(zip(self.feature_names, list(clf.feature_importances_)),
                                       key=lambda x:x[1], reverse=True)[:top_n])

        return important_weights

    # 获得选中的特征
    def get_select_featurs(self, important_features, X):

        if len(set(important_features).intersection(self.feature_names)) < len(important_features):
            raise ValueError("some select features  not exist in feature names")

        df = pd.DataFrame(X, columns=self.feature_names)
        select_featues = df[important_features].values
        return select_featues

    #基本分类器和集成分类器性能的比较
    def evaluate_score(self, X, y):
        if not self.evaluate_info:
            prob = self.predict_proba(X)
        scores = dict()
        for clf_name, info in self.evaluate_info.items():
            prob = info.get("prob")
            score = roc_auc_score(y, prob)
            self.evaluate_info[clf_name].update({"score": score})
            scores[clf_name] = score
        return scores



















