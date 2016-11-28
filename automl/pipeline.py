#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import sklearn_pandas
from sklearn.pipeline import make_pipeline
from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import BaggingRegressor, BaggingClassifier
from sklearn.learning_curve import learning_curve


class PipelineModel(BaseEstimator):

    def __init__(self,
                 classfier,
                 pipeline_name,
                 origin_datamanager,
                 feature_datamanager,
                 feature_names=None,
                 label_name=None,
                 pipelines=None,
                 drop_columns=None,
                 datamanager=None

                 ):

        if label_name is None:
            raise ValueError("label name must be given")

        if self.feature_names is None:
            raise ValueError("feature names must be given")

        if label_name in feature_names:
            self.feature_names = feature_names.remove(label_name)
        else:
            self.feature_names = feature_names
        self.classfier = classfier
        self.drop_columns = drop_columns
        self.pipelines = pipelines
        self.pipeline_name = pipeline_name
        self.origin_datamanager = origin_datamanager
        self.feature_datamanager = feature_datamanager

        self.df_mapper = None
        self._estimator = None

    def fit(self, X, y):
        if type(X) == pd.DataFrame:
            df = X

        elif type(X) == pd.Series:
            df = pd.DataFrame(X, columns=self.feature_names)
        else:
            raise ValueError("X type: %s is not illegal" % str(type(X)))
        self._fit(df, y)

    def _fit(self, X, y):
        self.df_mapper = DataFrameMapper(self.pipelines)
        X = self.df_mapper.fit_transform(X)
        self._estimator = self.classfier.fit(X, y)

    def predict(self, X):
        X = self.df_mapper.fit_transform(X)
        return self._estimator.predict(X)

    def predict_proba(self, X):
        X = self.df_mapper.fit_transform(X)
        return self._estimator.predict_proba(X)

