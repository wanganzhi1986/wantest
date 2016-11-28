#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from feature_enigneer import SortedFeature, DiscreteBinsFeature, FeatureData
from datamanager import DataManager
from sklearn.base import TransformerMixin
from prepocess import PreprocessBulider


class FeatureDataBuilder(TransformerMixin):

    def __init__(self, origin_datamanager=None, origin_pipelines=None, feature_stragety=None):
        # 输入的数据对象
        self.origin_datamanager = origin_datamanager
        # 指定生成的特征
        self.feature_stragety = feature_stragety
        # 原始数据的工作流
        self.origin_pipelines = origin_pipelines

        if self.origin_datamanager is None and self.origin_pipelines is None:
            raise ValueError("origin data must be given")
        # 生成的数据对象
        self.datamanagers = None

        self.feature_pipelines = None

    def run(self):
        pass

    def fit(self, X, y):
        if self.origin_pipelines is None:
            self.origin_pipelines = self._build_preprocess(self.origin_datamanager)
        # if self.feature_pipelines is None:
        #     self.feature_pipelines["origin"] = self.origin_pipelines
        fd = FeatureData(feature_stragety=self.feature_stragety,
                         )
        fd.fit(X, y)
        origin_feature_names = self.origin_datamanager.feature_names
        datamanagers = fd.datamanagers
        transforms = fd.transforms
        for strategy, datamanager in datamanagers.items():
            pipelines = self.origin_pipelines
            pipelines.append((origin_feature_names, transforms.get(strategy)))
            pipelines.extend(self._build_preprocess(datamanager))
            self.feature_pipelines[strategy] = pipelines
        self.datamanagers = datamanagers


    #建立预处理流程
    def _build_preprocess(self, datamanager):
        feature_names = datamanager.feature_names
        metadata = datamanager.metadata
        label_name = datamanager.label_name
        pb = PreprocessBulider(metadata=metadata,
                               feature_names=feature_names,
                               label_name=label_name
                               )
        pipelines = pb.pipelines
        return pipelines

    def transform(self, X):
        pass




    # # 组装数据
    # def _builder_data(self):
    #     origin_data = self.in_datamanager.data
    #     train_X = origin_data.get("train_X")
    #     test_X = origin_data.get("test_X")
    #     train_y = origin_data.get("train_y")
    #     test_y = origin_data.get("test_y")
    #     in_feature_names = origin_data.get("feature_names")
    #     label_name = origin_data.get('label_name')
    #     features = self.get_feature_data(in_feature_names)
    #     if self.include_features:
    #         for name, feature in features.items():
    #             if name in self.include_features:
    #                 train_feature = feature.fit_transform(train_X)
    #                 out_feaure_names = feature.out_feature_names
    #                 test_feature = feature.fit_transform(test_X)
    #                 out_datamanger = DataManager(train_X=train_feature,
    #                                              train_y=train_y,
    #                                              test_X=test_feature,
    #                                              test_y=test_y,
    #                                              feature_names=out_feaure_names,
    #                                              label_name=label_name
    #                                             )
    #                 self.out_datamanagers[name] = out_datamanger
