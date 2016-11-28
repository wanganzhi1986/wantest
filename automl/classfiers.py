#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import BaseEnsemble
from ConfigSpace import ConfigurationSpace
from ConfigSpace import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from paramselect import BayesianOptimise, GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier


class BaseClassfierAlgorithm(object):

    def __init__(self, opt_name, opt_params):
        self.estimator = None
        if opt_name and opt_name not in ["bayopt", "smac", "grid"]:
            raise ValueError("optimiser not  support ")
        self.optimiser = None
        self.opt_name = opt_name
        self.opt_params = opt_params
        self.best_params_ = None
        self.best_estimator_ = None

    def fit(self, X, y):
        raise NotImplementedError

    def fit_optimiser(self, X, y):
        if self.estimator is None:
            raise

        if self.opt_name is None:
            self.optimiser = self.estimator

        if self.opt_name == "bayopt":
            pbounds = self.get_bayopt_hyperparameter()
            if self.opt_params:
                self.optimiser = BayesianOptimise(estimator=self.estimator,
                                                  pbounds=pbounds,
                                                  **self.opt_params
                                                  )
            else:
                self.optimiser = BayesianOptimise(estimator=self.estimator,
                                                  pbounds=pbounds
                                                  )

        if self.opt_name == "grid":
            param_grid = self.get_grid_hyperparameter()
            if self.opt_params:
                self.optimiser = GridSearchCV(estimator=self.estimator,
                                              param_grid=param_grid,
                                              **self.opt_params
                                              )
            else:
                self.optimiser = GridSearchCV(estimator=self.estimator,
                                              param_grid=param_grid
                                            )

        self.optimiser.fit(X, y)
        self.best_params_ = self.optimiser.best_params_
        self.best_estimator_ = self.optimiser.best_estimator_

    def predict(self, X):
        return self.optimiser.predict(X)

    def predict_proba(self, X):
        return self.optimiser.predict_proba(X)

    def transform(self):
        pass

    def get_bayopt_hyperparameter(self):
        raise NotImplementedError

    def get_grid_hyperparameter(self):
        raise NotImplementedError

    def get_important_features(self, X, y):
        pass

    def score(self, X, y):
        pass


class RandomForestClassfiers(BaseClassfierAlgorithm):

    def __init__(self,
                 opt_name="bayopt",
                 opt_params=None,
                 n_estimators=10,
                 criterion="gini",
                 max_depth=50,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_split=1e-7,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):

        super(RandomForestClassfiers, self).__init__(
            opt_name=opt_name, opt_params=opt_params
        )
        self.n_estimators = int(n_estimators)
        self.criterion = criterion
        self.max_depth = int(max_depth)
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_split = min_impurity_split
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight

        self.estimator = None

    def fit(self, X, y):
        self.estimator = RandomForestClassifier(n_estimators=self.n_estimators,
                                                criterion=self.criterion,
                                                max_depth=self.max_depth,
                                                min_samples_split=self.min_samples_split,
                                                min_samples_leaf=self.min_samples_leaf,
                                                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                                max_features=self.max_features,
                                                max_leaf_nodes=self.max_leaf_nodes,
                                                min_impurity_split=self.min_impurity_split,
                                                bootstrap=self.bootstrap,
                                                oob_score=self.oob_score,
                                                n_jobs=self.n_jobs,
                                                random_state=self.random_state,
                                                verbose=self.verbose,
                                                warm_start=self.warm_start,
                                                class_weight=self.class_weight
                                                )

        self.fit_optimiser(X, y)
        return self

    def get_bayopt_hyperparameter(self):
        # pbounds = {
        #     "n_estimators": (2, 150, "Integer"),
        #     "max_features": (0, 1, "Float"),
        #     "min_samples_leaf": (1, 20, "Integer")
        #
        # }
        pbounds = {"n_estimators": (1, 400, "Integer"),
                    # "max_depth": (1, 30, "Integer"),
                    # "min_samples_split": (20, 400, "Integer"),
                    "min_samples_leaf": (1, 200, "Integer"),
                    "max_features": (0.1, 0.99, "Float")
                   }
        return pbounds

    def get_grid_hyperparameter(self):
        pass


class XgboostClassfier(BaseClassfierAlgorithm):

    def __init__(self, opt_name="bayopt", opt_params=None,
                 n_estimators=300,
                 max_depth=7,
                 learning_rate=0.02
                 ):

        super(XgboostClassfier,self).__init__(opt_name=opt_name, opt_params=opt_params)

        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)

        self.estimator = None

    def fit(self, X, y):
        import xgboost as xgb

        self.estimator = xgb.XGBClassifier(n_estimators=self.n_estimators,
                                           max_depth=self.max_depth,
                                           learning_rate=self.learning_rate,

        )

        self.fit_optimiser(X, y)
        return self

    def get_bayopt_hyperparameter(self):
        pbounds = {
            "n_estimators": (50, 500, "Integer"),
            "learning_rate": (0.01, 0.5, "Float"),
            "max_depth": (3, 10, "Integer"),
            "colsample_bytree": (0.5, 0.9, "Float"),
            "gamma":(0.1, 0.5, "Float"),
            "min_child_weight": (2, 100, "Integer"),
            "reg_alpha": (0.01, 500, "Float"),
            "reg_lambda": (0.01, 500, "Float"),
            "subsample": (0.2, 0.9, "Float")
        }
        return pbounds

    def get_grid_hyperparameter(self):
        pass


class AdaboostClassifier(BaseClassfierAlgorithm):

    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 max_depth=3,
                 random_state=None,
                 opt_name="bayopt",
                 opt_params=None
                 ):
        super(AdaboostClassifier, self).__init__(opt_name=opt_name, opt_params=opt_params)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.algorithm = algorithm
        self.random_state = random_state
        self.max_depth = max_depth
        self.estimator = None

    def fit(self, X, y, sample_weight=None):
        import sklearn.ensemble
        import sklearn.tree

        base_estimator = sklearn.tree.DecisionTreeClassifier(max_depth=self.max_depth)

        self.estimator = sklearn.ensemble.AdaBoostClassifier(
            base_estimator=base_estimator,
            n_estimators=int(self.n_estimators),
            learning_rate=self.learning_rate,
            algorithm=self.algorithm,
            random_state=self.random_state
        )
        self.fit_optimiser(X, y)
        # if len(Y.shape) == 2 and Y.shape[1] > 1:
        #     # estimator = MultilabelClassifier(estimator, n_jobs=1)
        #     estimator.fit(X, Y, sample_weight=sample_weight)
        # else:
        #     estimator.fit(X, Y, sample_weight=sample_weight)

        return self

    def get_grid_hyperparameter(self):

        pass

    def get_bayopt_hyperparameter(self):
        # pbounds = {
        #     "n_estimators": (50, 500, "Integer"),
        #     "learning_rate": (0.001, 2, "Float"),
        #     # "max_depth": (1, 10)
        # }
        pbounds = {
            "n_estimators": (200, 1000, "Integer"),
            "learning_rate": (0.1, 0.5, "Float")
        }

        return pbounds


# gbdt分类器
class GradientBoostingClassifier(BaseClassfierAlgorithm):
    def __init__(self, loss, learning_rate, n_estimators, subsample,
                 min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf, max_depth, max_features,max_leaf_nodes,
                 init=None, random_state=None, verbose=0, save_origin=True, metric="roc_auc",
                 opt_name="bayopt",
                 opt_params=None
                 ):
        super(GradientBoostingClassifier, self).__init__(opt_name=opt_name, opt_params=opt_params)

        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.init = init
        self.random_state = random_state
        self.verbose = verbose
        self.estimator = None
        self.fully_fit_ = False
        self.save_origin = True
        self.metric = metric

    def fit(self, X, y, sample_weight=None, refit=False):
        self._fit(X, y)

    def _fit(self, X, y, sample_weight=None, n_iter=1, refit=False):
        import sklearn.ensemble

        # Special fix for gradient boosting!
        if isinstance(X, np.ndarray):
            X = np.ascontiguousarray(X, dtype=X.dtype)
        if refit:
            self.estimator = None

        if self.estimator is None:
            self.learning_rate = float(self.learning_rate)
            self.n_estimators = int(self.n_estimators)
            self.subsample = float(self.subsample)
            self.min_samples_split = int(self.min_samples_split)
            self.min_samples_leaf = int(self.min_samples_leaf)
            self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)
            if self.max_depth == "None":
                self.max_depth = None
            else:
                self.max_depth = int(self.max_depth)
            num_features = X.shape[1]
            max_features = int(
                float(self.max_features) * (np.log(num_features) + 1))
            # Use at most half of the features
            max_features = max(1, min(int(X.shape[1] / 2), max_features))
            if self.max_leaf_nodes == "None":
                self.max_leaf_nodes = None
            else:
                self.max_leaf_nodes = int(self.max_leaf_nodes)
            self.verbose = int(self.verbose)

            self.estimator = sklearn.ensemble.GradientBoostingClassifier(
                loss=self.loss,
                learning_rate=self.learning_rate,
                n_estimators=10,
                subsample=self.subsample,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_depth=self.max_depth,
                max_features=max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                init=self.init,
                random_state=self.random_state,
                verbose=self.verbose,
                warm_start=True,
            )
        self.fit_optimiser(X, y)

        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        elif not hasattr(self, 'fully_fit_'):
            return False
        else:
            return self.fully_fit_



    def score(self, X, y):
        from sklearn.metrics import roc_auc_score

        if self.metric == "roc_auc":
            pred = self.estimator.predict_proba(X)[:, 1]
            score = roc_auc_score(y, pred)
            return score

    # 用于提供贝叶斯调参需要的参数
    def get_bayopt_hyperparameter(self):
        pbound = {
            "n_estimators": (50, 500, "Integer"),
            "learning_rate": (0.001, 1, "Float"),
            "max_depth": (2, 10, "Integer"),
        }
        return pbound

    # 用于提供gridsearch需要的超参数
    def get_grid_hyperparameter(self):
        params = {
            "n_estimators": (50, 500),
            "learning_rate": (0.001, 1),
            "max_depth": (2, 10),
        }
        return params


# 逻辑回归分类器
class LogistRegressionClassfier(BaseClassfierAlgorithm):

    def __init__(self, penalty="l1",  threshold=0.01, dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1, refit=True,
                 opt_name="bayopt",
                 opt_params=None
                 ):
        super(LogistRegressionClassfier, self).__init__(opt_name=opt_name, opt_params=opt_params)
        #权值相近的阈值
        self.threshold = threshold
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.penalty = penalty

        #使用同样的参数创建L2逻辑回归
        # self.l2 = LogisticRegression(penalty='l2', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight = class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
        self.estimator = None
        self.l1 = None
        self.l2 = None
        self.coef_ = None

    def fit(self, X, y, sample_weight=None):
        from sklearn.linear_model import LogisticRegression
        self.estimator = LogisticRegression(
            penalty="l1",
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver=self.solver, max_iter=self.max_iter,
            multi_class=self.multi_class, verbose=self.verbose,
            warm_start=self.warm_start, n_jobs=self.n_jobs)

        self.fit_optimiser(X, y)

        return self

    def _fit(self, X, y):
        self.l1 = LogisticRegression(
                penalty="l1",
                dual=self.dual,
                tol=self.tol,
                C=self.C,
                fit_intercept=self.fit_intercept,
                intercept_scaling=self.intercept_scaling,
                class_weight=self.class_weight,
                random_state=self.random_state,
                solver=self.solver, max_iter=self.max_iter,
                multi_class=self.multi_class, verbose=self.verbose,
                warm_start=self.warm_start, n_jobs=self.n_jobs)

        self.l2 = LogisticRegression(
                penalty="l2",
                dual=self.dual,
                tol=self.tol,
                C=self.C,
                fit_intercept=self.fit_intercept,
                intercept_scaling=self.intercept_scaling,
                class_weight=self.class_weight,
                random_state=self.random_state,
                solver=self.solver, max_iter=self.max_iter,
                multi_class=self.multi_class, verbose=self.verbose,
                warm_start=self.warm_start, n_jobs=self.n_jobs)

        self.l1.fit(X, y)
        self.l2.fit(X, y)

    # 获得重要特征的名称
    def get_important_features(self, feature_names):
        pass

    # 获得特征的系数
    def get_feature_coefs(self):
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

    def get_bayopt_hyperparameter(self):
        pbound = {
            "C": (0.001, 10, "Float"),
        }
        return pbound

    def get_grid_hyperparameter(self):
        params = {"C": np.arange(0.001, 2, 0.01)}
        return params


class DecisionTreesClassfier(BaseClassfierAlgorithm):

    def __init__(self, opt_name="bayopt", opt_params=None,
                 criterion="gini",
                 splitter="best",
                 max_depth=4,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_split=1e-7,
                 class_weight=None,
                 presort=False
                 ):
        super(DecisionTreesClassfier, self).__init__(opt_name=opt_name, opt_params=opt_params)
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = int(min_weight_fraction_leaf)
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_split = min_impurity_split
        self.class_weight = class_weight
        self.presort = presort

        self.estimator = None

    def fit(self, X, y):
        from sklearn.tree import DecisionTreeClassifier

        self.estimator = DecisionTreeClassifier(
                                criterion=self.criterion,
                                splitter=self.splitter,
                                max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                min_samples_leaf=self.min_samples_leaf,
                                max_leaf_nodes=self.max_leaf_nodes,
                                class_weight=self.class_weight,
                                random_state=self.random_state
                                )

        self.fit_optimiser(X, y)
        return self

    def get_bayopt_hyperparameter(self):
        pbounds = {
            "max_depth": (2, 10, "Integer"),
            # "min_samples_split": (2, 20),
            "min_samples_leaf": (1, 20, "Integer")

        }
        return pbounds

    def get_grid_hyperparameter(self):
        pass


class GaussianNBClassfier(BaseClassfierAlgorithm):

    def __init__(self, opt_name=None, opt_params=None):

        super(GaussianNBClassfier, self).__init__(opt_name, opt_params)

        self.estimator = None

    def fit(self, X, y):
        from sklearn.naive_bayes import GaussianNB

        self.estimator = GaussianNB()

        self.fit_optimiser(X, y)
        return self

    def get_grid_hyperparameter(self):
        pass

    def get_bayopt_hyperparameter(self):
        pass









if __name__ == "__main__":
    features = ["MP0050048", "MP0041080","MP0045048", "MP0042110", "MS0050001", "mp_months", "MP0050003",
             "MP0110001", "MP0110008", "MP0110012","MP0110010", "mp_district_41", "MP0050047","MP0045110",
             "MP0050028", "MP0044026", "MP0050021", "MP0041053", "MP0045115", "MP0041060", "MP0050018",
             "MS0050011", "mp_district_13", "MP0042067", "MP0110029"]
    from sklearn.cross_validation import train_test_split
    from sklearn.linear_model import LogisticRegression
    train_path = "/Users/wangwei/workplace/data_train_mp.csv"
    df = pd.read_csv(train_path)
    df = df.fillna(0)
    y = df["label"].values
    X = df[features].values
    # X = df.drop(["applyNo", "overDueDays", "label"], axis=1)
    train_X, test_X, train_y, test_y = train_test_split(X, y)

    l1 = LogisticRegression(C=0.01, penalty="l1")
    l1.fit(train_X, train_y)
    l1_coef = l1.coef_
    print("l1 coef is:", list(l1_coef[0]))

    l2 = LogisticRegression(C=0.01, penalty="l2")
    l2.fit(train_X, train_y)
    l2_coef = l2.coef_
    print("l2 coef is:", l2_coef.shape)




