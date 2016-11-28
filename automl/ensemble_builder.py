#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from backend import Backend
import re
from ensemble_selection import EnsembleSelection
from sklearn.ensemble import RandomForestClassifier


class EnsembleBuilder(object):

    def __init__(self, output_dir,
                 ensemble_size=None,
                 ensemble_classfier=None,

                 ):
        self.output_dir = output_dir
        self.ensemble_classfier = ensemble_classfier
        self.ensemble_size = ensemble_size

        self.model_info = dict()
        # 模型的相关系数
        self.models_correlation = None
        self.df_ensemble_mapper = None
        self.ensemble_model = None

    def run(self):
        self._build_ensemble()

    # 集成模型建立
    def _build_ensemble(self):
        backend = Backend(self.output_dir)
        # 目录是"用户定义的暂存路径/auto-sklearn/predictions_ensemble"
        dir_train = os.path.join(self.output_dir,
                                    '.wangwei',
                                    'predictions_train')
        # 保存的是验证集的结果
        dir_valid = os.path.join(self.output_dir,
                                 '.wangwei',
                                 'predictions_valid')
        # 保存的是测试集的结果
        dir_test = os.path.join(self.output_dir,
                                '.wangwei',
                                'predictions_test')
        paths_ = [dir_train, dir_valid, dir_test]

        dir_ensemble_list_mtimes = []
        exists = [os.path.isdir(dir_) for dir_ in paths_]
        dir_train_list = sorted(os.listdir(dir_train))
        dir_valid_list = sorted(os.listdir(dir_valid)) if exists[1] else []
        dir_test_list = sorted(os.listdir(dir_test)) if exists[2] else []
        re_model_pattern = re.compile(r'_([a-z]*)_([a-z]*_[a-z]*_[0-9]*_[0-9]*)\.npy$')
        #
        # for dir_ensemble_file in dir_ensemble_list:
        #     if dir_ensemble_file.endswith("/"):
        #         dir_ensemble_file = dir_ensemble_file[:-1]
        #     # 获得文件的文件名
        #     basename = os.path.basename(dir_ensemble_file)
        #     dir_ensemble_file = os.path.join(dir_ensemble, basename)
        #     # 获得文件的最后的修改时间
        #     mtime = os.path.getmtime(dir_ensemble_file)
        #     dir_ensemble_list_mtimes.append(mtime)

        try:
            train_prediction = self.get_prediction(dir_train, dir_train_list, re_model_pattern)
            test_prediction = self.get_prediction(dir_test, dir_test_list, re_model_pattern)
            ensemble = EnsembleSelection(ensemble_size=self.ensemble_size,
                                         classfier=self.ensemble_classfier
                                         )
            ensemble.fit(train_prediction, test_prediction)
            self.models_correlation = ensemble.models_correlation
            self.df_ensemble_mapper = ensemble.df_mapper
            self.ensemble_model = ensemble.ensemble
            backend.save_ensemble(ensemble)
            ensemble_predicition = ensemble.predict_proba
            backend.save_predictions_as_npy(ensemble_predicition, "prediction_ensemble")
        except Exception as e:
            print("ensemble run error is:" + str(e))


    #获得预测结果
    def get_prediction(self, dir_path,  dir_file_list, re_model_pattern, precision="32"):
        result = []
        for i, model_name in enumerate(dir_file_list):
            match = re_model_pattern.search(model_name)
            automl_seed = int(match.group(1))
            num_run = int(match.group(2))

            if model_name.endswith("/"):
                model_name = model_name[:-1]
            basename = os.path.basename(model_name)

            if precision == "16":
                predictions = np.load(os.path.join(dir_path, basename)).astype(
                    dtype=np.float16)
            elif precision == "32":
                predictions = np.load(os.path.join(dir_path, basename)).astype(
                    dtype=np.float32)
            elif precision == "64":
                predictions = np.load(os.path.join(dir_path, basename)).astype(
                    dtype=np.float64)
            else:
                predictions = np.load(os.path.join(dir_path, basename))
            result.append(predictions)
        return np.hstack(result)










