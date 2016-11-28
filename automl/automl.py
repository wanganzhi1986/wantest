#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from prepocess import PreprocessBulider
from datamanager import DataManager
from featuredata_builder import FeatureDataBuilder
from pipeline_builder import PipelineBuilder
from ensemble_builder import EnsembleBuilder
import backend
import sigopt
from sigopt_sklearn import search
import util


class AutoMlClassfier(BaseEstimator):

    def __init__(self,
                 datamanager=None,
                 feature_names=None,
                 label_name=None,
                 deploy_mode=None,
                 dataset_name=None,
                 output_dir=None,
                 refit=True,
                 deploy=True,
                 run_cluster=False,
                 base_classfiers=None,
                 ensemble_classfiers=None,
                 include_estimators=None,
                 include_preprocessors=None,
                 feature_stragety=None,
                 ensemble_mode="stack"


                 ):
        self.deploy_mode = deploy_mode
        # 模型结果输出目录
        self.output_dir = output_dir
        # 是否重新训练
        self.refit = refit
        # 是否进行模型部署
        self.deploy = deploy
        # 是否集群运行
        self.run_cluster = run_cluster
        # 包含的分类器
        self.include_estimators = include_estimators
        # 包含的预处理器
        self.include_preprocessors = include_preprocessors
        self.dataset_name = dataset_name
        self.feature_names = feature_names
        self.label_name = label_name
        self.datamanager = datamanager
        self.feature_stragety = feature_stragety
        self.base_classfiers = base_classfiers
        self.ensemble_classfiers = ensemble_classfiers
        self.ensemble_mode = ensemble_mode

        # 生成的特征数据集
        self.feature_data = None
        # 原始的训练数据
        self.origin_data = None
        # 训练的基本分类器转换器
        self.df_base_mappers = None
        # 训练的基本模型
        self.base_models = None
        # 训练的集成模型转换器
        self.df_ensemble_mapper = None
        # 训练的集成模型
        self.ensemble_model = None
        # 部署的模型
        self.deploy_models = None
        self.models_preb = None
        self.backend = backend.Backend(output_dir=self.output_dir)

        if self.output_dir is None:
            raise ValueError("save path must be given")

    # 运行程序
    def run(self):
        pass

    # 训练模型
    def fit(self, X, y):
        pass

    def _fit(self, X, y):
        if self.datamanager is None:
            self.datamanager = DataManager(dataset_name=self.dataset_name,
                                           X=X,
                                           y=y,
                                           feature_names=self.feature_names,
                                           label_name=self.label_name
                                            )
        # 建立预处理分析策略
        feature_names = self.datamanager.feature_names
        label_name = self.datamanager.label_name
        metadata = self.datamanager.metadata
        pb = PreprocessBulider(metadata=metadata,
                               feature_names=feature_names,
                               label_name=label_name
                               )
        origin_pipelines = pb.pipelines

        # 建立特征数据
        fdb = FeatureDataBuilder(origin_datamanager=self.datamanager,
                                 origin_pipelines=origin_pipelines,
                                 feature_stragety=self.feature_stragety
                                 )
        feature_pipelines = fdb.feature_pipelines
        feature_datamanager = fdb.datamanagers

        # 建立训练流水线
        plb = PipelineBuilder(origin_datamanager=self.datamanager,
                              origin_pipelines=origin_pipelines,
                              feature_datamanager=feature_datamanager,
                              feature_pipelines=feature_pipelines,
                              feature_names=feature_names,
                              lable_name=label_name
                              )
        plb.fit(X, y)
        self.base_models = plb.models
        self.df_base_mappers = plb.df_mappers

        eb = EnsembleBuilder(output_dir=self.output_dir,
                             ensemble_classfier=self.ensemble_classfiers
                             )
        self.ensemble_model = eb.ensemble_model
        self.df_ensemble_mapper = eb.df_ensemble_mapper

    def predict(self, X):
        if self.base_models is None or len(self.base_models) == 0:
            self._load_models()

    def predict_proba(self, X):
        if self.base_models is None or len(self.base_models) == 0:
            self._load_models()

        probs = []
        for name, model in self.base_models.items():
            X_ = X.copy()
            prob = model.predict_proba(X_)
            probs.append(prob)
            self.models_preb[name] = prob
        if len(probs) == 0:
            raise ValueError('Something went wrong generating the predictions. '
                             'The ensemble should consist of the following '
                             'models: %s, the following models were loaded: '
                             '%s' % (str(list(self.ensemble_model.keys())),
                                     str(list(self.base_models.keys()))))
        ensemble_prob = self.ensemble_model.predict_proba(np.array(probs))
        self.models_preb[self.ensemble_mode] = ensemble_prob
        return ensemble_prob

    # 加载基本模型
    def _load_models(self):
        ensemble_id = util.get_ensemble_identifier(self.dataset_name, self.ensemble_mode)
        self.ensemble_mode = self.backend.load_single_ensemble(identifier=ensemble_id)
        if self.ensemble_mode:
            identifiers = self.ensemble_mode.identifiers
        else:
            identifiers = None
        self.base_models = self.backend.load_many_models(identifiers)
        if self.base_models is None or len(self.base_models) == 0:
            raise ValueError('No models fitted!')

    # 重新训练模型
    def refit(self, X, y):
        pass

    # 线上部署模型
    def deploy(self):
        pass

    # 对训练结果可视化
    def visual(self):
        pass



