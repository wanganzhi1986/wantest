#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from collections import Counter
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier
import six
from sklearn.feature_selection import SelectFromModel
from scipy.stats import pearsonr
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import FunctionTransformer
import random


#集成模型的模型选择
class EnsembleSelection(object):
    def __init__(self,
                 model_names=None,
                 ensemble_size=None,
                 classfier=None,
                 epison=None,
                 sorted_initialization=False,
                 corr_epison=0.7,
                 bagging=False,
                 mode="stack",
                 corr_select=False
                 ):
        self.ensemble_size = ensemble_size
        self.sorted_initialization = sorted_initialization
        self.bagging = bagging
        # 模型提高的最低分数值
        self.epison = epison
        self.classfier = classfier
        self.mode = mode
        self.model_names = model_names
        self.corr_select = corr_select
        self.select_model_names= {}

        self.indices_ = None
        self.weights_ = None
        self.ensemble = None
        self.models_correlation = None
        self.corr_epison = corr_epison
        self.corr_indices = None
        self.df_mapper = None
        self.identifiers = None

        if self.mode not in ["stack", "vote"]:
            raise ValueError("ensemble mode is invalid")
        if self.mode == "stack" and self.classfier is None:
            raise ValueError("stack ensemble must be given classfier")

    def fit(self, predictions, labels):
        self.num_input_models_ = len(predictions)
        if self.model_names is None:
            self.model_names = ["model_prob"+str(i) for i in range(len(predictions))]

        if self.models_correlation is None:
            self.models_correlation = self.get_models_correlation(predictions)
        if self.corr_indices is None:
            self.corr_indices = self.get_correlation_indices(self.models_correlation, self.epison)
        if self.indices_ is None:
            self.model_selection(predictions, labels)

        if self.weights_ is None:
            self._calculate_weights()

        if not self.select_model_names:
            self.select_model_names = filter(lambda x: x[1]>0, sorted(zip(self.model_names, self.weights_),
                                             key=lambda x: x[1], reverse=True))

        self.identifiers = self.get_model_identifiers()

        # df = pd.DataFrame(np.column_stack(predictions), columns=self.model_names)
        # pipelines = self.get_piplines(self.mode)
        # self.df_mapper = DataFrameMapper(pipelines)
        # predictions = self.df_mapper.fit_transform(df)
        if self.mode == "stack":
            predictions = np.column_stack(predictions)
            self.ensemble = self.classfier.fit(predictions, labels)
        else:
            self.ensemble = self
        return self

    # 按照给定的训练方式得到集成学习的管道流程
    def get_piplines(self, mode):
        pipelines = []
        if mode == "stack":
            pipelines = [(self.model_names, FunctionTransformer(self.get_selected_prediction))
                         ]

        if mode == "vote":
            pipelines = [(self.model_names, FunctionTransformer(self.get_selected_prediction)),
                         (self.select_model_names, FunctionTransformer(self.voting_predict))
                         ]

        return pipelines

    # 获得选择的模型预测结果
    def get_selected_prediction(self, predictions):
        # if type(predictions) == list:
        #     predictions = np.column_stack(predictions)e)
        if self.ensemble_size is None:
            return predictions
        if self.weights_ is None:
            self._calculate_weights()
        selected = list(set(self.indices_))
        select_prediction = []
        for i in selected:
            self.select_model_names.append(self.model_names[i])
            select_prediction.append(predictions[i])
        # n_predictions = predictions.shape[1]
        # sel = np.zeros(n_predictions, dtype=bool)
        # sel[np.asarray(selected)] = True
        # select_prediction = predictions[sel]
        # print("selec shape:", select_prediction.shape)
        return select_prediction

    def model_selection(self, predictions, labels):
        """集成模型的选择，根据相关系数和爬山法进行选择"""
        ensemble = []
        trajectory = []
        order = []

        ensemble_size = self.ensemble_size
        if self.sorted_initialization:
            n_best = 20
            indices = self._sorted_initialization(predictions, labels, n_best)
            for idx in indices:
                if self.corr_select and idx not in self.corr_indices:
                    continue
                ensemble.append(predictions[idx])
                order.append(idx)
                ensemble_ = np.array(ensemble).mean(axis=0)
                ensemble_performance = roc_auc_score(labels, ensemble_)
                trajectory.append(ensemble_performance)
            ensemble_size -= n_best

        for i in range(ensemble_size):
            if self.corr_select and i not in self.corr_indices:
                continue
            scores = np.zeros((len(predictions)))
            s = len(ensemble)
            if s == 0:
                weighted_ensemble_prediction = np.zeros(predictions[0].shape)
            else:
                ensemble_prediction = np.mean(np.array(ensemble), axis=0)
                weighted_ensemble_prediction = (s / float(s + 1)) * \
                                               ensemble_prediction
            for j, pred in enumerate(predictions):
                if self.corr_select and j not in self.corr_indices:
                    continue
                fant_ensemble_prediction = weighted_ensemble_prediction + \
                                           (1. / float(s + 1)) * pred

                scores[j] = roc_auc_score(labels, fant_ensemble_prediction)
            best = np.nanargmax(scores)
            ensemble.append(predictions[best])
            trajectory.append(scores[best])
            order.append(best)
            # Handle special case
            if len(predictions) == 1:
                break
        self.indices_ = order

    def _calculate_weights(self):
        ensemble_members = Counter(self.indices_).most_common()
        weights = np.zeros((self.num_input_models_,), dtype=float)
        for ensemble_member in ensemble_members:
            weight = float(ensemble_member[1]) / self.ensemble_size
            weights[ensemble_member[0]] = weight

        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)
        self.weights_ = weights

    def _sorted_initialization(self, predictions, labels, n_best):
        perf = np.zeros([predictions.shape[0]])
        for i, p in enumerate(predictions):
            perf[i] = roc_auc_score(labels, predictions)
        indices = np.argsort(perf)[perf.shape[0] - n_best:]
        return indices

    def _bagging(self, predictions, labels, fraction=0.5, n_bags=20):
        """Rich Caruana's ensemble selection method with bagging."""
        # raise ValueError('Bagging might not work with class-based interface!')
        n_models = predictions.shape[0]
        bag_size = int(n_models * fraction)

        order_of_each_bag = []
        for j in range(n_bags):
            # Bagging a set of models
            indices = sorted(random.sample(range(0, n_models), bag_size))
            bag = predictions[indices, :, :]
            order, _ = self._fit(bag, labels)
            order_of_each_bag.append(order)
        return np.array(order_of_each_bag)

    #预测模型的概率
    def predict_proba(self, predictions):
        prob = self._predict_proba(predictions, mode=self.mode)
        return prob

    def _predict_proba(self, predictions, mode):
        if mode == "stack":

            return self.ensemble.predict_proba(predictions)
            # predictions = self.df_mapper.fit_transform(predictions)
            # if self.ensemble and hasattr(self.ensemble, "predict_proba"):
            #     return self.ensemble.predict_proba(predictions)
            # else:
            #     raise ValueError("stacking classfier is invalid")
        elif mode == "vote":
            if self.weights_ is None:
                self._calculate_weights()
            # return self.df_mapper.fit_transform(predictions)
            # predictions = self.get_selected_prediction(predictions)
            pred = self.voting_predict(predictions)
            return pred

        else:
            raise ValueError("ensemble mode: %s is invalid"%mode)

    def get_correlation_indices(self, corrlist, epsion):
        indices = []
        for corr in corrlist:
            if corr[2] < epsion:
                continue
            if corr[0] not in indices:
                indices.append(corr[0])
            if corr[1] not in indices:
                indices.append(corr[1])
        return indices

    # 得到模型的相关系数
    def get_models_correlation(self, predictions):
        corrlist = []
        for i in range(len(predictions)):
            for j in range(len(predictions)):
                if i < j:
                    corr = pearsonr(predictions[i], predictions[j])
                    corrlist.append((i, j, abs(corr[0])))
        return sorted(corrlist, key=lambda x: x[2])

    def predict(self, predictions):
        pass


    # 投票预测
    def voting_predict(self, predictions):
        for i, weight in enumerate(self.weights_):
            predictions[i] *= weight
        return np.sum(predictions, axis=0)

    def __str__(self):
        return 'Ensemble Selection:\n\tTrajectory: %s\n\tMembers: %s' \
               '\n\tWeights: %s\n\tIdentifiers: %s' % \
               (' '.join(['%d: %5f' % (idx, performance)
                         for idx, performance in enumerate(self.trajectory_)]),
                self.indices_, self.weights_,
                ' '.join([str(identifier) for idx, identifier in
                          enumerate(self.identifiers_)
                          if self.weights_[idx] > 0]))

    def pprint_ensemble_string(self, models):
        output = []
        sio = six.StringIO()
        for i, weight in enumerate(self.weights_):
            identifier = self.identifiers_[i]
            model = models[identifier]
            if weight > 0.0:
                output.append((weight, model))

        output.sort(reverse=True, key=lambda t: t[0])

        sio.write("[")
        for weight, model in output:
            sio.write("(%f, %s),\n" % (weight, model))
        sio.write("]")

        return sio.getvalue()

    def get_model_identifiers(self):
        if self.model_names is None:
            return None
        if self.indices_ is None:
            return None
        if type(self.indices_) == list:
            indices = np.unique(np.array(self.indices_))
        else:
            indices = np.unique(self.indices_)

        if type(self.model_names) == list:
            model_names = np.array(self.model_names)
        else:
            model_names = self.model_names
        print("indices is:", indices)
        return list(model_names[indices])



