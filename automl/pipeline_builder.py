#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from classfiers import AdaboostClassifier, LogistRegressionClassfier,\
    GradientBoostingClassifier, RandomForestClassfiers, XgboostClassfier, DecisionTreesClassfier
import feature_enigneer as fe
from prepocess import PreprocessBulider
from pipeline import PipelineModel
from sklearn.pipeline import Pipeline


class PipelineBuilder(object):

    def __init__(self,
                 origin_datamanager=None,
                 feature_datamanager=None,
                 origin_pipelines=None,
                 feature_pipelines=None,
                 feature_names=None,
                 lable_name=None
                 ):
        self.origin_datamanager = origin_datamanager
        self.feature_datamanager = feature_datamanager
        self.origin_pipelines = origin_pipelines
        self.feature_pipelines = feature_pipelines

        self.models = None
        self.df_mappers = None

    def run(self):
        pass

    def fit(self, X, y):
        if self.origin_pipelines:
            self._build_pipeline(self.origin_datamanager, self.origin_pipelines, X, y)

        if self.feature_pipelines:
            for stratery, piplines in self.feature_pipelines.items():
                datamanager = self.feature_datamanager.get(stratery)
                self._build_pipeline(datamanager, piplines, X, y)

    # 建立数据的管道流
    def _build_pipeline(self, X, y, datamanager, pipelines,):
        # 获得数据源的名称
        dataset_name = datamanager.dataset_name
        classfiers = self.get_classfiers()
        feature_names = datamanager.feature_names
        label_name = datamanager.label_name
        for clf_name, clf in classfiers.items():
            pipeline_name = dataset_name + "_" + clf_name
            pipe = PipelineModel(classfier=clf,
                                 pipeline_name=pipeline_name,
                                 pipelines=pipelines,
                                 feature_names=feature_names,
                                 label_name=label_name,
                                 datamanager=datamanager
                                 )
            pipe.fit(X, y)
            self.models[pipeline_name] = pipe
            self.df_mappers[pipeline_name] = pipe.df_mapper

    def transform(self, X, y):
        pass

    # 得到分类器
    def get_classfiers(self):
        clfs = {
            "ada": AdaboostClassifier(n_estimators=300, learning_rate=0.02),
            "lr": LogistRegressionClassfier(C=2.403),
            "gbdt": GradientBoostingClassifier(n_estimators=300),
            "dt": DecisionTreesClassfier(max_depth=4),
            "xgboost": XgboostClassfier(n_estimators=300, max_depth=7, learning_rate=0.02)
        }

        return clfs

