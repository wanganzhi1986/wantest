#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class AbstractAcquisitionFunction(object):

    def __init__(self, model,
                 verbose=True,
                 acq_kind="ei",
                 y_max=None,
                 ei_par=0.01,
                 ucb_par=2.576,
                 derivative=False

                 ):
        self.model = model
        self.verbose = verbose
        self.acq_kind = acq_kind
        self.y_max = y_max
        self.ei_par = ei_par
        self.ucb_par = ucb_par
        self.derivative = derivative

        self.new_y = None
        self.acq_value = None
        self.der_value = None
        self.new_model = None
        if self.acq_kind not in ["ei", "ucb", "poi"]:
            raise ValueError("acquistion function type: %s is invalid"%self.acq_kind)

    def fit(self, X, y):
        if hasattr(self.model, "fit"):
            self.model.fit(X, y)
        if self.y_max is None:
            self.y_max = np.max(y)
        return self


    def predict(self, X):
        m, v = self._predict(X)

        acq_value = None
        der_value = None
        if self.acq_kind == "ei":
            acq_value, der_value = self._ei(self.y_max, m, v)

        if self.acq_kind == "ucb":
            acq_value, der_value = self._ucb(self.y_max, m, v)

        if self.acq_kind == "poi":
            acq_value, der_value = self._poi(self.y_max, m, v)
        return acq_value, der_value

    def _predict(self, X):
        raise NotImplementedError

    def predict_gradients(self, X):
        raise NotImplementedError

    def _ei(self, y_max, m, v):
        assert m.shape[1] == 1
        assert v.shape[1] == 1
        s = np.sqrt(v)
        z = (y_max - m - self.ei_par) / s
        f = (y_max- m - self.ei_par) * norm.cdf(z) + s * norm.pdf(z)
        f[s == 0.0] = 0.0

        df = None
        if self.derivative:
            dmdx, ds2dx = self.predict_gradients(self.new_X)
            dmdx = dmdx[0]
            ds2dx = ds2dx[0][:, None]
            dsdx = ds2dx / (2 * s)
            df = (-dmdx * norm.cdf(z) + (dsdx * norm.pdf(z))).T
            df[s == 0.0] = 0.0

        if self.derivative:
            return f, df
        else:
            return f, None

    def _ucb(self, y_max, m, v):
        assert m.shape[1] == 1
        assert v.shape[1] == 1
        s = np.sqrt(v)
        f = m + self.ucb_par * s
        f[s == 0.0] = 0.0

        df = None
        if self.derivative:
            pass

        if self.derivative:
            return f, df
        else:
            return f, None

    def _poi(self, y_max, m, v):
        assert m.shape[1] == 1
        assert v.shape[1] == 1
        s = np.sqrt(v)
        z = (m - y_max - self.ei_par) / s
        f = norm.cdf(z)
        f[s == 0.0] = 0.0

        df = None
        if self.derivative:
            pass

        if self.derivative:
            return f, df
        else:
            return f, None


class GaussianAcquisitionFunction(AbstractAcquisitionFunction):

    def __init__(self,model,
                 verbose=True,
                 acq_kind="ei",
                 y_max=None,
                 ei_par=0.01,
                 ucb_par=2.576,
                 derivative=False):
        super(GaussianAcquisitionFunction, self).__init__(
            model=model,
            acq_kind=acq_kind,
            y_max=y_max,
            ei_par=ei_par,
            ucb_par=ucb_par,
            derivative=derivative
        )
        if self.model is None:
            self.model = GaussianProcessRegressor(kernel=Matern(), n_restarts_optimizer=25,)

    def _predict(self, X):
        mean, std = self.model.predict(X, return_std=True)
        return mean, std

    def predict_gradients(self, X):
        pass


class RandomForestAcquisitionFunction(AbstractAcquisitionFunction):
    def __init__(self, model=None,
                 verbose=True,
                 acq_kind="ei",
                 y_max=None,
                 ei_par=0.01,
                 ucb_par=2.576,
                 derivative=False,
                 var_threshold=0.01,
                 instance_features=None
    ):
        super(RandomForestAcquisitionFunction, self).__init__(
            model=model,
            acq_kind=acq_kind,
            y_max=y_max,
            ei_par=ei_par,
            ucb_par=ucb_par,
            derivative=derivative
        )
        self.instance_features = instance_features
        self.var_threshold = var_threshold
        if self.model is None:
            self.model = RandomForestRegressor(n_estimators=20)

    def _predict(self, X):
        if self.instance_features is None or \
                len(self.instance_features) == 0:
            mean, var = self._predict_single_instance(X)
        else:
            mean, var = self._predict_marginalized_over_instances(X, self.instance_features)
        return mean, var

    def _predict_single_instance(self, X):
        if len(X.shape) != 2:
            raise ValueError(
                'Expected 2d array, got %dd array!' % len(X.shape))
        if X.shape[1] != self.types.shape[0]:
            raise ValueError('Rows in X should have %d entries but have %d!' %
                             (self.types.shape[0], X.shape[1]))

        means = np.ndarray((X.shape[0], 1))
        vars = np.ndarray((X.shape[0], 1))
        for i, x in enumerate(X):
            m, v = self.model.predict(X)
            means[i] = m
            vars[i] = v
        return means, vars

    def predict_gradients(self, X):
        pass

    def _predict_marginalized_over_instances(self, X, instance_features):
        """Predict mean and variance marginalized over all instances.

        Returns the predictive mean and variance marginalised over all
        instances for a set of configurations.

        Parameters
        ----------
        X : np.ndarray of shape = [n_features (config), ]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """
        n_instance_features = instance_features.shape[1]
        n_instances = len(instance_features)

        if len(X.shape) != 2:
            raise ValueError(
                'Expected 2d array, got %dd array!' % len(X.shape))
        # if X.shape[1] != self.types.shape[0] - n_instance_features:
        #     raise ValueError('Rows in X should have %d entries but have %d!' %
        #                      (self.types.shape[0] - n_instance_features,
        #                       X.shape[1]))

        mean = np.zeros(X.shape[0])
        var = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            X_ = np.hstack(
                (np.tile(x, (n_instances, 1)), instance_features))
            means, vars = self._predict_single_instance(X_)
            # use only mean of variance and not the variance of the mean here
            # since we don't want to reason about the instance hardness distribution
            var_x = np.mean(vars) # + np.var(means)
            if var_x < self.var_threshold:
                var_x = self.var_threshold

            var[i] = var_x
            mean[i] = np.mean(means)

        var[var < self.var_threshold] = self.var_threshold
        var[np.isnan(var)] = self.var_threshold
        if len(mean.shape) == 1:
            mean = mean.reshape((-1, 1))
        if len(var.shape) == 1:
            var = var.reshape((-1, 1))
        return mean, var










