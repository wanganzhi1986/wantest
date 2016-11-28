#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KDTree
from collections import namedtuple
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.feature_selection import chi2
# from minepy import MINE
from sklearn import metrics, preprocessing
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
import util
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree._tree import TREE_LEAF
from sklearn.base import TransformerMixin


TrainData = namedtuple("TrainData", ["train_features", "train_label", "feature_names", "label_name"])

TestData = namedtuple("TestData", ["test_features", "test_label", "feature_names", "label_name"])



class FeatureSelection(object):

    def __init__(self,
                 train_data, test_data,
                 feature_names,
                 filter_feature_strategy,
                 add_feature_strategy,
                 feature_nums
                 ):

        self.feature_names = feature_names
        self.train_data = train_data
        self.test_data = test_data
        self.filter_feature_strategy = filter_feature_strategy
        self.add_feature_strategy = add_feature_strategy
        self.feature_nums = feature_nums

        self.select_features = []
        self.df_train = pd.DataFrame(train_data.train_features, columns=train_data.feature_names)
        self.df_test = pd.DataFrame(test_data.test_features, columns=test_data.feature_names)

    def fit(self, X, y):
        allow_strategy = ["rf", "lasso", "relief", "xgboost"]
        if self.filter_feature_strategy:
            if self.filter_feature_strategy not in allow_strategy:
                raise ValueError("not strategy not allowed")
            if self.filter_feature_strategy == "lasso":
                self.select_features = self.get_lasso_features(X, y)

            if self.filter_feature_strategy == "rf":
                self.select_features = self.get_rf_features(X, y)

            if self.filter_feature_strategy == "relief":
                self.select_features = self.get_relief_features(X,y)

            if self.filter_feature_strategy == "xgboost":
                # self.select_features = self.feature_names.append(self.get_xgboost_features(X, y))
                self.select_features = self.get_xgboost_features(X, y)
        else:
            self.select_features = self.feature_names
        return self


    # 返回选择特征后的数据
    def transform(self):
        df_train = pd.DataFrame(self.train_data.train_features,
                                columns=self.train_data.feature_names)

        df_test = pd.DataFrame(self.test_data.test_features,
                               columns=self.test_data.feature_names)

        df_train_select = df_train[self.select_features]
        df_test_select = df_test[self.select_features]

        train_data = TrainData(train_features=df_train_select.values,
                               train_label=self.train_data.train_label,
                               feature_names=self.select_features,
                               label_name=self.train_data.label_name
                               )

        test_data = TestData(test_features=df_test_select.values,
                             test_label=self.test_data.test_label,
                             feature_names=self.select_features,
                             label_name=self.test_data.label_name
                               )

        return train_data, test_data


    # 使用lasso选择特征, alpha值越大,出现的零值越多
    def get_lasso_features(self, X, y):
        important_features = []
        alpha = 1.0
        feature_names = self.feature_names
        lasso = Lasso(alpha=alpha)
        lasso.fit(X, y)
        coefs = lasso.coef_
        feature_coefs = dict(sorted(zip(feature_names, list(coefs)), key=lambda x: x[1], reverse=True))
        important_features = [name for name, coef in feature_coefs.items() if coef != 0]
        #
        # while len(important_features) < self.feature_nums:
        #     feature_names = self.feature_names
        #     lasso = Lasso(alpha=alpha)
        #     lasso.fit(X, y)
        #     coefs = lasso.coef_
        #     feature_coefs = dict(sorted(zip(feature_names, list(coefs)), key=lambda x: x[1], reverse=True))
        #     important_features = [name for name, coef in feature_coefs.items() if coef != 0]
        #     if len(important_features) > self.feature_nums:
        #         important_features = important_features[: self.feature_nums]
        #         break
        #     alpha -= 0.1
        return important_features


    # 使用随机森林选择特征
    def get_rf_features(self, X, y):
        from sklearn.ensemble import RandomForestRegressor
        features_thresold = 25
        feature_sample_rate = 0.8
        feature_names = self.feature_names
        rf = RandomForestRegressor(n_estimators=200, oob_score=False, max_depth=5)
        rf.fit(X, y)
        print("hellllo")
        feature_coefs = rf.feature_importances_
        important_features = sorted(zip(feature_names, feature_coefs), key=lambda x: x[1], reverse=True)
        pre_features = important_features[:100]
        df = pd.DataFrame(X, columns=self.feature_names)
        rfs = []
        return dict(pre_features).keys()


        # # 对特征集进行采样,获得最好的oob_score的分数的特征集
        # while len(pre_features) > features_thresold:
        #     pre_index = int(len(pre_features) * feature_sample_rate)
        #     pre_features = dict(pre_features[:pre_index])
        #     pre_names = pre_features.keys()
        #     df = df[pre_names]
        #     rf = RandomForestRegressor(n_estimators=50, oob_score=True)
        #     rf.fit(df.values, y)
        #     rfs.append((pre_names, rf))
        #     print("select feature count is:", len(pre_features))
        #     print("oob score is:", rf.oob_score_)
        #
        #     pre_features = sorted(zip(pre_names, rf.feature_importances_), key=lambda x: x[1], reverse=True)
        #
        # best_features, best_oob_score_ = sorted([(pre_names, rf.oob_score_) for pre_names, rf in rfs], key=lambda x: x[1])[-1]
        # return best_features


    # 使用xgboost获得类别特征
    def get_xgboost_features(self, X, y):
        import xgboost as xgb
        combiner = xgb.XGBClassifier(
                                       n_estimators=300,
                                       learning_rate=0.02,
                                       nthread=5,
                                       max_depth=7,
                                       subsample=0.5
                                   )
        combiner.fit(X, y)

        coef = combiner.feature_importances_
        important_features = dict(sorted(zip(self.feature_names, coef), key=lambda x: x[1], reverse=True)[:self.feature_nums])
        return important_features.keys()


    #使用relief选择特征
    def get_relief_features(self, X, y):
        relief = ReliefF()
        return relief.fit_transform(X, y)


    # 选择最为重要的特征
    def feature_importances(self, data, top_n=None, feature_names=None):
        # data can be either a sklearn estimator or an iterator with
        # the actual importances values, try to get the values
        try:
            imp = data.feature_importances_
        except:
            imp = np.array(data)

        # in case the user passed an estimator, it may have an estimators_
        # attribute, which includes importnaces for every sub-estimator
        # get them if possible
        try:
            sub_imp = np.array([e.feature_importances_ for e in data.estimators_])
            # calculate std
            std = np.std(sub_imp, axis=0)
        except:
            std = None

        # get the number of features
        n_features = len(imp)

        # check that the data has the correct format
        if top_n and top_n > n_features:
            raise ValueError(('top_n ({}) cannot be greater than the number of'
                              ' features ({})'.format(top_n, n_features)))
        if top_n and top_n < 1:
            raise ValueError('top_n cannot be less than 1')
        if feature_names and len(feature_names) != n_features:
            raise ValueError(('feature_names ({}) must match the number of'
                              ' features ({})'.format(len(feature_names),
                                                      n_features)))

        # if the user did not pass feature names create generic names
        if feature_names is None:
            feature_names = ['Feature {}'.format(n) for n in range(1, n_features+1)]
            feature_names = np.array(feature_names)
        else:
            feature_names = np.array(feature_names)

        # order the data according to the importance for the feature
        idx = np.argsort(imp)[::-1]
        imp = imp[idx]
        feature_names = feature_names[idx]
        if std is not None:
            std = std[idx]

        # build the structured array
        if std is not None:
            names = 'feature_name,importance,std_'
            res = np.core.records.fromarrays([feature_names, imp, std],
                                             names=names)
        else:
            names = 'feature_name,importance'
            res = np.core.records.fromarrays([feature_names, imp],
                                             names=names)

        # get subset if top_n is not none
        if top_n:
            res = res[:top_n]
        return res



    def get_statis_feature(self, X):
       # 使用方差

       X = VarianceThreshold(threshold=3).fit_transform(X)
       return X


from sklearn.linear_model import LogisticRegression


class LR(LogisticRegression):
    def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):

        #权值相近的阈值
        self.threshold = threshold
        LogisticRegression.__init__(self, penalty='l1', dual=dual, tol=tol, C=C,
                 fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                 random_state=random_state, solver=solver, max_iter=max_iter,
                 multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
        #使用同样的参数创建L2逻辑回归
        self.l2 = LogisticRegression(penalty='l2', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight = class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None):
        from sklearn.feature_selection import SelectFromMode
        #训练L1逻辑回归
        super(LR, self).fit(X, y, sample_weight=sample_weight)
        self.coef_old_ = self.coef_.copy()

        #训练L2逻辑回归
        self.l2.fit(X, y, sample_weight=sample_weight)

        cntOfRow, cntOfCol = self.coef_.shape
        #权值系数矩阵的行数对应目标值的种类数目
        for i in range(cntOfRow):
            for j in range(cntOfCol):
                coef = self.coef_[i][j]
                #L1逻辑回归的权值系数不为0
                if coef != 0:
                    idx = [j]
                    #对应在L2逻辑回归中的权值系数
                    coef1 = self.l2.coef_[i][j]
                    for k in range(cntOfCol):
                        coef2 = self.l2.coef_[i][k]
                        #在L2逻辑回归中，权值系数之差小于设定的阈值，且在L1中对应的权值为0
                        if abs(coef1-coef2) < self.threshold and j != k and self.coef_[i][k] == 0:
                            idx.append(k)

                    #计算这一类特征的权值系数均值
                    mean = coef / len(idx)
                    self.coef_[i][idx] = mean
        return


class RandomForestSelectFeature(object):

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def transform(self, X):
        pass



# 贪心特征选择
class greedyFeatureSelection(object):

    def __init__(self,
                 feature_num,
                 save_dir,
                 estimator=None,
                 feature_names=None,
                 init_feature_names=None,
                 verbose=0,
                 early_stopping=True,
                 n_jobs=5
                 ):
        if feature_names is None:
            raise ValueError("feature names not given")
        if estimator is None:
            self.estimator = LogisticRegression(C=0.01)
        else:
            self.estimator = estimator
        self.feature_names = feature_names
        self.init_feature_names = init_feature_names
        self._verbose = verbose
        self.early_stopping = early_stopping
        self.feature_num = feature_num
        self.n_jobs = n_jobs
        self.save_dir = save_dir

        self.select_feaure_names = []
        self.good_features = set([])
        self.score_history = []
        self.features_info = dict()
        self.important_features = dict()
        self.interval = None
        self.feature_num_score = dict()

    def evaluateScore(self, X, y):
        from sklearn.cross_validation import cross_val_score

        return cross_val_score(self.estimator, X, y, 'roc_auc', cv=5).mean()
        # self.estimator.fit(X, y)
        # predictions = self.estimator.predict_proba(X)[:, 1]
        # auc = metrics.roc_auc_score(y, predictions)
        # return auc

    def fit(self, X, y):
        start = datetime.now()
        self._fit(X, y)
        end = datetime.now()
        self.interval = util.get_time_interval(start, end)


    # 特征选择的主函数
    def _fit(self, X, y):
        from sklearn.externals.joblib import Parallel, delayed
        score_history = []
        # num_features = X.shape[1]
        important_features = []
        feature_nums_score = []
        df = pd.DataFrame(X, columns=self.feature_names)

        if self.init_feature_names:
            if self.init_feature_names == self.feature_names:
                raise ValueError("no available feature used")
            common_names = list(set(self.feature_names).intersection(set(self.init_feature_names)))
            avail_feature_names = [name for name in self.feature_names if name not in common_names ]
            good_features = self.init_feature_names
        else:
            avail_feature_names = self.feature_names
            good_features = []

        while len(avail_feature_names) > 0 or score_history[-1][0] > score_history[-2][0]:
            scores = []
            # out = Parallel(n_jobs=self.n_jobs, verbose=self._verbose)(
            #         delayed(self._add_good_feature)(feature, X, y, scores)
            #         for feature in range(num_features))
            for feature in avail_feature_names:
                if feature not in good_features:

                    selected_features = good_features + [feature]
                    # Xts = np.column_stack(X[:, j] for j in selected_features)
                    xts = df[selected_features].values
                    score = self.evaluateScore(xts, y)
                    print("score is:", score)
                    scores.append((score, feature))
                    if self._verbose:
                        print "Current AUC : ", str(score)
            good = sorted(scores, lambda x: x[0], reverse=True)
            good_feature = good[0][1]
            good_score = good[0][0]
            last_good_score = score_history[-1][0]
            # 用于绘制特征个数与得分的关系
            feature_nums_score.append((str(len(score_history)), last_good_score))
            # 用于评价每个特征后提升的幅度,也就是重要性
            print("important features:",(good_feature, good_score-last_good_score))
            important_features.append((good_feature, good_score-last_good_score))
            # 添加好的特征
            good_features.append(good_feature)
            score_history.append(good[0])
            # 将选中的特征从待选集中移除
            avail_feature_names.remove(good_feature)

            if self._verbose:
                print "Current Features : ", sorted(list(good_features))

            if self.feature_num and len(good_features) >= self.feature_num:
                break

        # Remove last added feature
        good_features.remove(score_history[-1][1])
        good_features = sorted(list(good_features))
        if self._verbose:
            print "Selected Features : ", good_features

        self.important_features = dict(important_features)
        self.feature_num_score = dict(feature_nums_score)
        # self.features_info["feature_important"] = dict(important_features)
        # self.features_info["feature_num_score"] =
        self.good_features = good_features

    def transform(self, X):
        df = pd.DataFrame(X, self.feature_names)
        selected = df[self.important_features.keys()].values
        return selected
        # return X[:, self.good_features]

    # 绘制选择特征的曲线
    def plot_feature_curve(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        nums = [ int(i) for i in self.feature_num_score.keys()]
        scores = self.feature_num_score.values()
        plt.plot(nums, scores, 'darkblue', label='feature num and score relation')
        plt.legend(loc='upper right')
        plt.xlabel('feature num')
        plt.ylabel('auc scores')

    # 添加选择的特征
    def _add_good_feature(self, feature, X, y, scores):
        if feature not in self.good_features:
            selected_features = list(self.good_features) + [feature]

            Xts = np.column_stack(X[:, j] for j in selected_features)

            score = self.evaluateScore(Xts, y)
            scores.append((score, feature))

            if self._verbose:
                print "Current AUC : ", str(score)

    def dist_select_loop(self, X, y):
        pass

    # 保存训练的特征
    def save_feature(self):
        pass

    def load_feature(self):
        pass


class KmeansClusterFeature(TransformerMixin):

    def __init__(self, n_clusters, merge_origin=True):
        self.n_clusters = n_clusters
        self.merge_origin = merge_origin

        self.preprocessor = None

    def fit(self, X, y=None):
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import MinMaxScaler
        scale = MinMaxScaler()
        X = scale.fit_transform(X)
        self.preprocessor = KMeans(n_clusters=self.n_clusters)
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError
        features = []
        labels = self.preprocessor.predict(X)
        for label in labels:
            features.append(np.array([int(label == i) for i in range(self.n_clusters)]))
        if self.merge_origin:
            return np.hstack([X, np.vstack(features)])

        return np.vstack(features)


# 随机森林的权重特征
class RandomForestWeightFeature(object):
    def __init__(self, n_estimators, max_depth, feature_names):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_names = feature_names

        self.feature_weitghts = None
        self.weitghts = None

    def fit(self, X, y):
        rf = RandomForestClassifier(n_estimators=self.n_estimators,
                                     max_depth=self.max_depth
                                     )

        clf.fit(X, y)
        self.weitghts = rf.feature_importances_
        self.feature_weitghts = dict(zip(self.feature_names, rf.feature_importances_))
        return self

    def transform(self, X):
        if self.weitghts is None:
            raise NotImplementedError
        df = pd.DataFrame(X, columns=self.feature_names)
        df1 = df.mul(self.weitghts, axis=1)
        return df1.values


class GradientBoostingEmbedding(TransformerMixin):

    def __init__(self, n_estimators, max_depth, learning_rate,
                 n_jobs=3,
                 save_origin=True
                 ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.save_origin = save_origin

        self.preprocessor = None
        self.estimators = None

    def fit(self, X, y):
        from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
        self.preprocessor = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate
        )
        self.preprocessor.fit(X, y)
        self.estimators = self.preprocessor.estimators_
        return self

    def transform(self, X):
        if self.estimators is None:
            raise NotImplementedError
        categorys = []
        for tree in self.estimators[:, 0]:
            tt = tree.tree_
            leafs = list(np.where(tt.children_left == TREE_LEAF)[0])
            leaf_samples = tt.apply(np.asarray(X, dtype=np.float32))
            category = np.zeros((len(leaf_samples), len(leafs)))
            for i, sample in enumerate(leaf_samples):
                category[i, :] = np.array([1 if leaf == sample else 0 for leaf in leafs])
            categorys.append(category)

        category_feature = np.hstack(categorys)
        if self.save_origin:
            return np.hstack([X, category_feature])
        return category_feature


class RandomForestEmbedding(TransformerMixin):

    def __init__(self, n_estimators, max_depth, n_jobs,
                 sparse_output=True,
                 max_leaf_nodes=5,
                 min_samples_leaf=None

                 ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.sparse_output = sparse_output
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf

        self.preprocessor = None

    def fit(self, X, y=None):
        from sklearn.ensemble import RandomTreesEmbedding
        self.preprocessor = RandomTreesEmbedding(
                                  n_estimators=self.n_estimators,
                                  max_depth=self.max_depth,
                                  n_jobs=self.n_jobs,
                                  sparse_output=self.sparse_output,
                                  max_leaf_nodes=self.max_leaf_nodes
                                  )

        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError
        return self.preprocessor.transform(X)


# relief算法的实现
class ReliefF(object):

    def __init__(self, n_neighbors=100, n_features_to_keep=10):

        self.feature_scores = None
        self.top_features = None
        self.tree = None
        self.n_neighbors = n_neighbors
        self.n_features_to_keep = n_features_to_keep

    def fit(self, X, y):

        self.feature_scores = np.zeros(X.shape[1], dtype=np.int64)
        self.tree = KDTree(X)

        # Find nearest k neighbors of all points. The tree contains the query
        # points, so we discard the first match for all points (first column).
        indices = self.tree.query(X, k=self.n_neighbors+1,
                                  return_distance=False)[:, 1:]

        for (source, nn) in enumerate(indices):
            # Create a binary array that is 1 when the sample  and neighbors
            # match and -1 everywhere else, for labels and features.
            labels_match = np.equal(y[source], y[nn]) * 2 - 1
            features_match = np.equal(X[source], X[nn]) * 2 - 1

            # The change in feature_scores is the dot product of these  arrays
            self.feature_scores += np.dot(features_match.T, labels_match)

        # Compute indices of top features, cast scores to floating point.
        self.top_features = np.argsort(self.feature_scores)[::-1]
        self.feature_scores = self.feature_scores.astype(np.float64)

    def transform(self, X):

        return X[:, self.top_features[:self.n_features_to_keep]]


    def fit_transform(self, X, y):
        """Computes the feature importance scores from the training data, then
        reduces the feature set down to the top `n_features_to_keep` features.
        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels
        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_keep}
            Reduced feature matrix
        """
        self.fit(X, y)
        return self.transform(X)

if __name__ == "__main__":
    features = ["MP0050048", "MP0041080","MP0045048", "MP0042110", "MS0050001", "mp_months", "MP0050003",
             "MP0110001", "MP0110008", "MP0110012","MP0110010", "mp_district_41", "MP0050047","MP0045110",
             "MP0050028", "MP0044026", "MP0050021", "MP0041053", "MP0045115", "MP0041060", "MP0050018",
             "MS0050011", "mp_district_13", "MP0042067", "MP0110029"]

    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import roc_auc_score
    train_path = "/Users/wangwei/workplace/data_train_mp.csv"
    df = pd.read_csv(train_path)
    feature_names = list(df.drop(["applyNo", "overDueDays", "label"], axis=1).columns)
    y = df["label"].values
    X = df[feature_names].fillna(0.0).values
    train_X, test_X, train_y, test_y = train_test_split(X, y)
    xgb_params ={
        "n_estimators": (10, 500),
        "learning_rate": (0.001, 1, 0.005),
        "max_depth": (2, 10),
    }

    best_params = {
        "n_estimators": 300,
        "learning_rate":0.02,
        "max_depth":7
    }

    clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.02, max_depth=7)
    grfs = greedyFeatureSelection(
        feature_num=1000,
        save_dir="",
        estimator=clf,
        feature_names=feature_names,
        init_feature_names=features,
    )

    grfs.fit(train_X, train_y)
    print ("best features is: %s" %grfs.important_features)
    grfs.plot_feature_curve()













