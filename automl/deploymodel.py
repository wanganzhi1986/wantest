#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sklearn2pmml
import backend


class DeployModelManager(object):

    def __init__(self, df_mapper, pmml_path,
                 model_path=None, model_name=None,
                 model=None,
                 ):
        # 数据预处理器
        self.df_mapper = df_mapper
        # pmml文件的保存路径
        self.pmml_path = pmml_path
        # 模型保存的路径
        self.model_path = model_path
        # 模型的名称
        self.model_name = model_name
        # 训练的模型
        self.model = model

    def run(self):
        pass



    # 加载模型
    def load_model(self, model_path):
        pass



