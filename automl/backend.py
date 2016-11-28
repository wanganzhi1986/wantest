#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os, errno
import cPickle as pickle
import glob
import tempfile


class Backend(object):

    def __init__(self, output_dir=None):
        if output_dir is None:
            raise ValueError("no given output dir")

        try:
            os.makedirs(output_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        self.output_dir = output_dir

    def get_features_dir(self):
        return os.path.join(self.output_dir, "features")

    # 保存选择的特征
    def save_features(self, features, identifier):
        filepath = str(os.path.join(self.get_features_dir(), "%s.feature"%identifier))
        # filepath = self.get_features_dir() + ("%s.feature"%identifier)
        # with tempfile.NamedTemporaryFile('wb', dir=os.path.dirname(filepath),
        #         delete=False) as fh:
        with open(filepath, "w") as fh:
            pickle.dump(features, fh)

    # 加载所有的特征
    def load_all_features(self):
        result = []
        feature_directory = self.get_features_dir()
        filepath = feature_directory + '*.feature'
        feature_files = glob.glob(filepath)
        # feature_files = glob.glob(os.path.join(feature_directory, '*.feature'))
        for filepath in feature_files:
            with open(filepath, "r") as fh:
               result.append(pickle.load(fh))
        return result

    # 按照指定的标识符选择指定的特征
    def load_features_by_identifier(self, identifiers):
        features = []
        for indentifier in identifiers:
            filepath = os.path.join(self.get_features_dir(), "%.feature"%indentifier)
            with open(filepath, "r") as fh:
                features.append(pickle.load(fh))
        return features

    def get_hyperparams_dir(self):
        return os.path.join(self.output_dir, "hyperparams")

    # 保存参数选择后的参数
    def save_hyperparams(self, hyperparams, identifier):
        filepath = os.path.join(self.get_hyperparams_dir(), "%s.hyperparam"%identifier)
        with open(filepath, "w") as fh:
            pickle.dump(hyperparams, fh)

    #  加载所有选择的超参数
    def load_all_hyperparams(self):
        pass

    # 按照指定的标识符选择超参数
    def load_hyperparams_by_identifiers(self, identifiers):
        result = []
        for identifer in identifiers:
            filepath = os.path.join(self.get_hyperparams_dir(), "%s.hyperparam"%identifer)
            result.append(pickle.load(filepath))
        return result

    def get_models_dir(self):
        return os.path.join(self.output_dir, "models")

    # 保存训练好的模型
    def save_model(self, model, identifier):
        filepath = os.path.join(self.get_models_dir(), "%s.model"%identifier)
        with open(filepath, "w") as fp:
            pickle.dump(model, fp)

    def load_single_model(self, identifier):
        filepath = os.path.join(self.get_models_dir(), "%s.model" % identifier)
        filepath = self.prepare_path(filepath)
        with open(filepath, "r") as fp:
            return pickle.load(fp)

    def load_all_models(self):
        models = []
        model_dir = self.get_models_dir()
        model_files = glob.glob(os.path.join(model_dir, '*.model'))
        for model_file in model_files:
            with open(model_file, "rb") as mf:
                models.append(pickle.load(mf))
        return models

    # 按照指定的标识符选择模型
    def load_many_models(self, identifiers=None):
        models = []
        if identifiers is None:
            models = self.load_all_models()
        else:
            if isinstance(identifiers, list):
                models = [self.load_single_model(identifier) for identifier in identifiers]
        return models

    def get_ensemble_dir(self):
        return os.path.join(self.output_dir, "ensemble")

    # 保存集成模型
    def save_ensemble(self, ensemble,  identifier):
        filepath = os.path.join(self.get_ensemble_dir(), "%s.ensemble"%identifier)
        filepath = self.prepare_path(filepath)
        with open(filepath, "w") as fp:
            pickle.dump(ensemble, fp)

    # 加载所有的集成模型
    def load_all_ensembles(self):
        ensembles = []
        ensemble_dir = self.get_ensemble_dir()
        ensemble_files = glob.glob(os.path.join(ensemble_dir, '*.ensemble'))
        for ensemble_file in ensemble_files:
            with open(ensemble_file, "rb") as ef:
                ensembles.append(pickle.load(ef))
        return ensembles

    def load_single_ensemble(self, identifier):
        filepath = os.path.join(self.get_ensemble_dir(), "%s.ensemble" % identifier)
        filepath = self.prepare_path(filepath)
        with open(filepath, "r") as fp:
            result = pickle.load(fp)
        return result

    # 按照指定的标识符加载集成模型
    def load_many_ensembles(self, identifiers=None):
        results = []
        if identifiers is None:
            results = self.load_all_ensembles()
        else:
            if isinstance(identifiers, list):
                results = [self.load_single_ensemble(identifier) for identifier in identifiers]
        return results

    # 建立起文件的路径
    def prepare_path(self, filepath):
        """
        :param filepath:
        :rtype: str
        """
        filepath = os.path.expanduser(filepath)
        dirname = os.path.dirname(filepath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        return filepath

    # 得到预测结果输出的目录
    # subset = "train"/"test"
    def get_prediction_output_dir(self, subset):
        return os.path.join(self.output_dir,
                            'predictions_%s' % subset)

    # 保存预测结果作为.npy文件
    def save_predictions_as_npy(self, predictions, subset, identifiers):
        output_dir = self.get_prediction_output_dir(subset)
        # Make sure an output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filepath = os.path.join(output_dir, 'predictions_%s_%s.npy' %
                                            (subset, identifiers))
        filepath = self.prepare_path(filepath)
        with tempfile.NamedTemporaryFile('wb', dir=os.path.dirname(
                filepath), delete=False) as fh:
            pickle.dump(predictions.astype(np.float32), fh, -1)
            tempname = fh.name
        os.rename(tempname, filepath)

    def get_runhistory_dir(self):
        return os.path.join(self.output_dir, "runhistory")

    def save_runhistory(self, runhistroy, identifier):
        filepath = self.get_runhistory_dir()
        filepath = self.prepare_path(filepath)
        with open(filepath, "wb") as fp:
            pickle.dump(runhistroy, "%s.runhistory"%identifier)

    def load_runhistory(self, identifier):
        filepath = os.path.join(self.get_runhistory_dir(), "%s.runhistory" % identifier)
        filepath = self.prepare_path(filepath)
        with open(filepath, "r") as fp:
            result = pickle.load(fp)
        return result


if __name__  == "__main__":
    from sklearn.linear_model import LogisticRegression
    import hashlib
    import util
    features = ["MP0041053", "MP0045115", "MP0041060", "MP0050018",
             "MS0050011", "mp_district_13", "MP0042067", "MP0110029"]
    lr = LogisticRegression(C=0.01)
    lr_id = util.get_model_id(lr)

    m = hashlib.md5()
    m.update(lr_id + ",".join(features))
    identifier = m.hexdigest()
    backend = Backend(output_dir="/Users/wangwei/workplace/train")
    # backend.save_features(features, identifier)
    # features = backend.load_features_by_identifier([identifier])
    # print("features", features)
    features = backend.load_all_features()
    print("features is:", features)



