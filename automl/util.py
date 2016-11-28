#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import os
from sklearn.cross_validation import StratifiedKFold, KFold
import numpy as np
import hashlib
import json


def get_time_interval(start, end):
    if start > end:
        tmp = end
        end = start
        start = tmp
    interval = (end - start).total_seconds()
    if interval <= 60:
        return str(interval)+"s"

    elif interval < 3600:
        mins, sec = divmod(interval, 60)
        return str(mins)+"m"+str(sec)+"s"
    else:
        mins, sec = divmod(interval, 60)
        hour, minute = divmod(mins, 60)
        return str(hour)+"h"+str(minute)+"m"+str(sec)+"s"



# 获得模型的标识
# def get_model_id(model):
#     model_type = str(type(model))
#     print('model type is:', model_type)
#     model_type = model_type[model_type.rfind(".")+1: model_type.rfind("'")]
#     print('model type is:', model_type)
#     param_dict = model.get_params()
#     ignore_list = ('n_jobs', 'oob_score', 'verbose', 'warm_start')
#     new_param_dict = {}
#     for key, value in sorted(param_dict.items(), key=lambda x: x[0]):
#         i = 0
#         if key in ignore_list:
#             continue
#         while True:
#             new_key = key[0] + str(i)
#             if not new_key in new_param_dict:
#                 new_param_dict[new_key] = value
#                 break
#             i += 1
#     model_type += str(new_param_dict)
#     replace_dict = {'{': '_',
#                     '}': '',
#                     "'": "",
#                     '.': 'p',
#                     ',': '__',
#                     ':': '_',
#                     ' ': '',
#                     'True': '1',
#                     'False': '0',
#                     'None': 'N',
#                     '=': '_',
#                     '(': '_',
#                     ')': '_',
#                     '\n': '_'}
#     for key, value in replace_dict.items():
#         model_type = model_type.replace(key, value)
#     if len(model_type) > 150:
#         model_type = model_type[:150]
#     return model_type

 # 获得预测的评价方法
def get_eval_metric(metric):
    from sklearn.metrics import roc_auc_score, log_loss
    if metric.lower() == "auc":
        eval_metric = roc_auc_score
    elif metric.lower() == "logloss":
        eval_metric = log_loss
    else:
        raise ValueError("Got a unrecognized metric name: %s" % metric)
    return eval_metric


# 获得模型的标识
def get_model_id(model=None):
    if model is None:
        des = "?????*****((((()))))))>>>>><<<<<<<<<<<?????"
        m = hashlib.md5()
        m.update(des)
        model_id = m.hexdigest()
    else:
        model_type = str(type(model))
        param_dict = json.dumps(model.get_params())
        des = model_type + param_dict
        m = hashlib.md5()
        m.update(des)
        model_id = m.hexdigest()
    return model_id


def get_config_id(config):
    m = hashlib.md5()
    m.update(config)
    config_id = m.hexdigest()
    return config_id

def get_model_identifier(dataset_name, clf_name, model):
    model_id = get_model_id(model)
    return dataset_name + "_" + clf_name + "_" + model_id


def get_ensemble_identifier(dataset_name, mode, model=None):
    model_id = get_model_id(model)
    return dataset_name + "_" + mode + "_" + model_id

# 得到缓存的文件
def get_cache_file(model_id, index, cache_dir='', suffix='csv'):
    # Identify index trick.
    # If sum of first 20 index, recognize as the same index.
    if index is None:
        raise IOError
    if len(index) < 20:
        sum_index = sum(index)
    else:
        sum_index = sum(index[:20])
    return "{0}{1}_{2}.{3}".format(cache_dir,
                                   model_id,
                                   sum_index,
                                   suffix)


def saving_predict_proba(model, X, index, cache_dir=''):
    csv_file = get_cache_file(model.id, index, cache_dir)
    try:
        df = pd.read_csv(csv_file)
        proba = df.values[:, 1:]
        print("**** prediction is loaded from {0} ****".format(csv_file))
    except IOError:
        proba = model.predict_proba(X)
        df = pd.DataFrame({'index': index})
        for i in range(proba.shape[1]):
            df["prediction" + str(i)] = proba[:, i]
        df.to_csv(csv_file, index=False)
    return proba


def saving_predict(model, X, index, cache_dir=''):
    csv_file = get_cache_file(model.id, index, cache_dir)
    try:
        df = pd.read_csv(csv_file)
        prediction = df.values[:, 1:]
        prediction = prediction.reshape([prediction.size,])
        print("**** prediction is loaded from {0} ****".format(csv_file))
    except IOError:
        prediction = model.predict(X)
        df = pd.DataFrame({'index': index})
        prediction.reshape([prediction.shape[-1], ])
        df["prediction"] = prediction
        df.to_csv(csv_file, index=False)
    return prediction


def make_sure_do_not_replace(path):
    """
    This function has been created for those cases when a file has to be generated and we want to be sure that we don't
    replace any previous file. It generates a different filename if a file already exists with the same name
    :param path: path to be checked and adapted (str or unicode)
    :return: path (str or unicode)
    """
    base, fileName = os.path.split(path)
    file_name, file_ext = os.path.splitext(fileName)
    i = 1
    while os.path.exists(path):
        path = os.path.join(base, file_name + "_copy%s" % i + file_ext)
        i += 1
    return path

class CrossPartitioner():
    def __init__(self, n=None, y=None, k=10, stratify=False, shuffle=True, random_state=655321):
        """
        Function for creating the partitions for the Stacker by using CrossValidation.
        :param n: When stratify=False, n defines the length of the datasets (int of None, default None)
        :param y: When stratify=True, train_y is the class used to preserve the percentages (array-like or None,
        default=None)
        :param k: Number of folds (int, default=10).
        :param stratify: Whether to preserve the percentage of samples of each class (boolean, default=False).
        :param shuffle: Whether to shuffle the data before splitting into batches (boolean, default=True).
        :param random_state: When shuffle=True, pseudo-random number generator state used for shuffling. If None, use
        default numpy RNG for shuffling (None, int or RandomState, default=655321)
        :return: None
        """
        self.y = y
        self.k = k
        self.stratify = stratify
        self.shuffle = shuffle
        self.seed = random_state
        self.N = None
        if self.stratify:
            assert type(y) != None, "You must pass the  'train_y' parameter if you want to stratify."
            if n: assert len(
                y) == n, "The length of the parameter 'train_y' and the 'n' value don't mismatch. If you are " \
                         "stratifying, it is not necessary to specify n."
            self.cvIterator = StratifiedKFold(y = self.y,
                                              n_folds = self.k,
                                              shuffle = self.shuffle,
                                              random_state = self.seed)
            self.N = len(self.y)
        else:
            assert n != None, "You must specify the size of the data using the 'n' parameter if you don't stratify."
            self.cvIterator = KFold(n=n,
                                    n_folds = self.k,
                                    shuffle = self.shuffle,
                                    random_state = self.seed)
            self.N = n


    def make_partitions(self, append_indices = True, dict_format=False, **kwargs):
        """
        This function is feed by keyword arguments containing the data to be splitted in folds.
        :param append_indices: Returns the indices within the tuple containing the data split
        :param dict_format: Whether to return a dictionary. If False, it returns a list of tuples. (boolean,
        default=False)
        :param kwargs: keyword arguments which value is the data arrays to be partitioned. (array-like)
        :return: generator containing, for each data array, the data corresponding to the each fold in each case.
        Whether dict_format=True, it has de subsequent format:
        {
        "data1":    (train, test),
        "data2":    (train, test),
        ...
        }
        if dict_format=False, the format of the generator objects yielded are:
        [(train_d1, test_d1), (train_d2, test_d2), ...]
        """
        from scipy.sparse.csr import csr_matrix
        import pandas as pd
        for k, (train_index, test_index) in enumerate(self.cvIterator):
            partitioned_data = {} if dict_format else []

            for name, data in kwargs.items():
                if type(data) == list:
                    data = np.array(data)  # Converts lists to Numpy Array
                elif type(data) == pd.core.frame.DataFrame or type(data) == pd.core.series.Series:
                    data = np.array(data)


                if append_indices:
                    if type(data) == csr_matrix:
                        split = (data[train_index], data[test_index], train_index, test_index)  # Appends the indices
                    else:
                        split = (
                            data[[train_index]], data[[test_index]], train_index, test_index)  # Appends the indices
                else:
                    if type(data) == csr_matrix:
                        split = (data[train_index], data[test_index])  # Not appends the indices
                    else:
                        split = (data[[train_index]], data[[test_index]])  # Not appends the indices

                if dict_format:
                    partitioned_data[name] = split # Returns a dictionary
                else:
                    partitioned_data.append(split) # Returns a list
            if not dict_format and len(kwargs) == 1: partitioned_data = partitioned_data[0]
            yield partitioned_data


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from datetime import datetime
    import hashlib
    start = datetime.now()
    lr1 = LogisticRegression(C=0.01)
    lr2 = LogisticRegression(C=0.02)
    lr3 = LogisticRegression(C=0.01, fit_intercept=False)
    lr_id_1 = get_model_id(lr1)
    lr_id_2 = get_model_id(lr2)
    lr_id_3 = get_model_id(lr3)
    m = hashlib.md5()
    m.update(lr_id_1)
    t1 = m.hexdigest()
    m = hashlib.md5()
    m.update(lr_id_2)
    t2 = m.hexdigest()
    print t1 == t2
    print t2
    # print(lr_id_1 == lr_id_3)
    # print(lr_id_1)
    # print(lr_id_2)
    end = datetime.now()
    interval = get_time_interval(start, end)
    print("time interval is:", interval)





