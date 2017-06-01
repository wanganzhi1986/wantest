# coding: utf8

import pandas, numpy, time, gc, sys, random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse


class DataProcessingFactory(object):
    def __init__(self,
                 training_file,
                 test_file,
                 target_label,
                 excluded_features=[],
                 rows_limit_train=None,
                 rows_limit_test=None,
                 verbose=True,
                 card_threshold=10):

        self.training_file = training_file
        self.test_file = test_file
        self.target_label = target_label
        self.excluded_features = excluded_features
        self.rows_limit_train = rows_limit_train
        self.rows_limit_test = rows_limit_test
        self.card_threshold = card_threshold
        self.verbose = verbose
        self.encoded = False
        self.scaled = False
        self.imputed = False
        self.loaded = False
        self.sparse = False
        self.train_set = pandas.DataFrame()
        self.test_set = pandas.DataFrame()
        self.factors = []
        self.quant_feat = []
        self.bool_feat = []
        self.unknown_cat = {}

    # method: load_data
    #
    # - loads the training/test sets from training file and test file
    # - each feature is stored with the cheapest data type that suits
    # - the factors categories are encoded as integers (cheaper than object type)
    # - set the lists factors, quant_feat & bool_feat and the dictionary unknown_cat
    # - new categories in the test file are replaced with the mode observed in the training set
    #
    def load_data(self):

        # few consistency checks between the training and the test set
        first_row_train = pandas.read_csv(self.training_file, nrows=1)
        first_row_test = pandas.read_csv(self.test_file, nrows=1)
        label_diff = list(set(first_row_train.columns) - set(first_row_test.columns))
        train_drivers = sorted(set(list(set(first_row_train.columns) - set([self.target_label]))))
        test_drivers = sorted(set(first_row_test.columns))
        target_index = list(first_row_train.dtypes.keys()).index(self.target_label)
        train_drivers_types = list(first_row_train.dtypes)
        train_drivers_types.pop(target_index)
        test_drivers_types = list(first_row_test.dtypes)
        error_msg = []
        train_size = 0
        test_size = 0
        if train_drivers != test_drivers:
            error_msg.append("feature labelling in train and test data don't match")
        if len(label_diff) == 0:
            error_msg.append("training file should have one more column than the test file")
        if label_diff[0] != self.target_label:
            error_msg.append("target variable name doesn't match the definition")
        if len(label_diff) > 1:
            error_msg.append("more than on column difference between train and test data")
        if set(train_drivers_types) != set(test_drivers_types):
            error_msg.append("feature types in train and test data don't match")
        if len(error_msg) > 0:
            sys.exit(", ".join(error_msg))

        begin = time.time()
        LabelEnc = LabelEncoder()

        test_index = 0
        for train_index in range(first_row_train.shape[1]):

            start = time.time()

            # load the current feature from the training file an retrieve its name and type
            if self.rows_limit_train is not None:
                train_feature = pandas.read_csv(self.training_file, usecols=[train_index], nrows=self.rows_limit_train)
            else:
                train_feature = pandas.read_csv(self.training_file, usecols=[train_index])

            feature_type = str(train_feature.dtypes[0])
            feature_name = str(train_feature.columns.values[0])

            if feature_type.find("int") < 0 and feature_type.find("float") < 0 and feature_type.find("bool"):
                self.factors.append(feature_name)
            elif feature_type.find("int") >= 0 or feature_type.find("float") >= 0:
                self.quant_feat.append(feature_name)
            elif feature_type.find("bool") >= 0:
                self.bool_feat.append(feature_name)

            # contrary to the drivers the target variable is absent in the test file
            if feature_name != self.target_label:
                if self.rows_limit_test is not None:
                    test_feature = pandas.read_csv(self.test_file, usecols=[test_index], nrows=self.rows_limit_test)
                else:
                    test_feature = pandas.read_csv(self.test_file, usecols=[test_index])
                test_index += 1

                # if the current feature is a factor we turn it into an integer feature because it is cheaper
            if feature_name not in self.excluded_features and feature_name in self.factors:
                # label encoding, basically string are replaced by integers
                train_series = pandas.Series(LabelEnc.fit_transform(train_feature.values.ravel()))
                # process the feature in the test file only if the current feature is not the target variable
                if test_feature is not None:
                    test_series = test_feature[feature_name]
                    # we check that there is no new category in the test file
                    # if so there are replace by the mode in the training set
                    # NOTE: we could think about a better workaround but it is a start
                    diff = numpy.setdiff1d(test_series.unique(), LabelEnc.classes_)
                    diff_size = len(diff)
                    if diff_size > 0:
                        self.unknown_cat.update({feature_name: diff})
                        # NOTE: don't use the replace function from pandas (too long)!
                        known_cat = numpy.in1d(test_feature[feature_name].values.ravel(), LabelEnc.classes_)
                        test_feature[feature_name][known_cat == False] = train_feature[feature_name].mode()[0]
                    test_series = pandas.Series(LabelEnc.transform(test_feature.values.ravel()))
                # missing values have been encoded as integer during the label encoding, we put back missing values
                missing_in_train = any(train_series.isnull().values)
                missing_in_test = False
                if test_feature is not None:
                    missing_in_test = any(test_series.isnull().values)
                if missing_in_train or missing_in_test:
                    na_category = list(LabelEnc.classes_).index(numpy.nan)
                if missing_in_train:
                    train_series.replace(na_category, numpy.nan, inplace=True)
                if missing_in_test:
                    test_series.replace(na_category, numpy.nan, inplace=True)
            else:
                train_series = train_feature[feature_name]
                if test_feature is not None:
                    test_series = test_feature[feature_name]
                    missing_in_test = any(test_series.isnull().values)

            # boolean type is already the cheapest encoding type
            encoding_type = 'bool_'
            if feature_type.find('bool') < 0:
                # add the feature to the data frames without specifying encoding type
                self.train_set[feature_name] = pandas.Series(train_series)
                if test_feature is not None:
                    self.test_set[feature_name] = pandas.Series(test_series)
                encoding_type = self.get_feat_cheap_type(feature_name)
                # this time add the feature to the data frames with the correct encoding type
            self.train_set[feature_name] = pandas.Series(self.train_set[feature_name], dtype=encoding_type)

            if test_feature is not None:
                self.test_set[feature_name] = pandas.Series(self.test_set[feature_name], dtype=encoding_type)

            if self.verbose:
                train_feat_size = self.train_set[feature_name].values.nbytes / 10 ** 2
                test_feat_size = self.test_set[feature_name].values.nbytes / 10 ** 2
                train_size += train_feat_size
                test_size += test_feat_size
                print feature_name + ": train set size " + str(train_size) + ", test set size " + str(
                    test_size) + ", feature processing time: " + str(
                    int(round((time.time() - start) / 60))) + " minutes"

            test_feature = None
            gc.collect()

        gc.collect()
        self.loaded = True
        print "data has been loaded"
        print "total processing time:", str(int(round((time.time() - begin) / 60))), "minutes"

    # method: encode_data()
    #
    # - factors are replaced with binary vectors (the so called one-hot encoding)
    #
    def encode_data(self):

        if self.encoded:
            sys.exit("data is already encoded")
        if not self.loaded:
            sys.exit("data has not been loaded yet")
        if len(self.factors) < 1:
            sys.exit("there is no factor feature")

        begin = time.time()
        OneHotEnc = OneHotEncoder()
        to_remove = []

        for feature_name in self.factors:
            if feature_name not in self.excluded_features and feature_name != self.target_label:
                # if there are too many categories we don't encode the factor
                if self.train_set[feature_name].max() <= self.card_threshold:
                    dummies_train = OneHotEnc.fit_transform(self.train_set[[feature_name]]).toarray()
                    dummies_test = OneHotEnc.transform(self.test_set[[feature_name]]).toarray()
                    for i in range(dummies_train.shape[1]):
                        # TO DO : insert category/missing in the name
                        self.train_set[feature_name + '_' + str(i)] = pandas.Series(dummies_train[:, i], dtype='bool_')
                        self.test_set[feature_name + '_' + str(i)] = pandas.Series(dummies_test[:, i], dtype='bool_')
                        self.bool_feat.append(feature_name + '_' + str(i))
                self.train_set.drop(feature_name, axis=1, inplace=True)
                self.test_set.drop(feature_name, axis=1, inplace=True)
                to_remove.append(feature_name)
                gc.collect()

        for fact in to_remove:
            self.factors.remove(fact)

        gc.collect()
        self.encoded = True
        print "factor features have been encoded (one-hot strategy)"
        print "total processing time:", str(int(round((time.time() - begin) / 60))), "minutes"

    # method: scale_data()
    #
    # - each quantitative feature is rescaled between 0 and 1
    # - scaling is based on the range observed in the training file
    #
    def scale_data(self):

        if self.scaled:
            sys.exit("data is already scaled")
        if not self.loaded:
            sys.exit("data has not been loaded yet")
        if len(self.quant_feat) < 1:
            sys.exit("there is no quantitative feature")

        begin = time.time()
        Scaler = MinMaxScaler()

        for feature_name in self.quant_feat:
            if feature_name not in self.excluded_features and feature_name != self.target_label:
                # NOTE: we need a default value for the little hack (see below)
                default_val = self.train_set[feature_name].median()
                # keep track of missing values so as to put again numpy.nan after the scaling if need be
                na_training = self.train_set[feature_name].isnull()
                na_test = self.test_set[feature_name].isnull()
                # NOTE: there is a little hack here because Scaler() can only deal with matrices and non missing values
                temp_matrix = numpy.column_stack((self.train_set[feature_name].fillna(default_val).values.astype(float),
                                                  self.train_set[feature_name].fillna(default_val).values.astype(
                                                      float)))
                # give a two columns matrix of rank 1 to the scaler and take only one column..
                self.train_set[feature_name] = pandas.Series(Scaler.fit_transform(temp_matrix)[:, 0])
                # same hack but for the test data
                temp_matrix = numpy.column_stack((self.test_set[feature_name].fillna(default_val).values.astype(float),
                                                  self.test_set[feature_name].fillna(default_val).values.astype(float)))
                self.test_set[feature_name] = pandas.Series(Scaler.transform(temp_matrix)[:, 0])
                # put back the missing values if need be
                if any(na_training.values):
                    self.train_set[feature_name][na_training.values.ravel()] = numpy.nan
                if any(na_test.values):
                    self.test_set[feature_name][na_test.values.ravel()] = numpy.nan
                # ask for a cheaper encoding type
                encoding_type = self.get_feat_cheap_type(feature_name)
                self.train_set[feature_name] = pandas.Series(self.train_set[feature_name], dtype=encoding_type)
                self.test_set[feature_name] = pandas.Series(self.test_set[feature_name], dtype=encoding_type)
                gc.collect()

        gc.collect()
        self.scaled = True
        print "quantitative features have been scaled (min-max strategy)"
        print "total processing time:", str(int(round((time.time() - begin) / 60))), "minutes"

    # method: impute_data()
    #
    # - each quantitative feature is rescaled between 0 and 1
    # - scaling is based on the range observed in the training file
    #
    def impute_data(self):

        if self.imputed:
            sys.exit("data is already imputed")
        if not self.loaded:
            sys.exit("data has not been loaded yet")

        begin = time.time()

        for feature_name in self.quant_feat + self.factors:
            if feature_name not in self.excluded_features and feature_name != self.target_label:
                if feature_name in self.quant_feat:
                    default_val = self.train_set[feature_name].median()
                    if self.verbose:
                        print feature_name, "median imputation with", default_val
                elif feature_name in self.factors:
                    default_val = self.train_set[feature_name].mode()[0]
                    if self.verbose:
                        print feature_name, "mode imputation with", default_val
                self.train_set[feature_name].fillna(default_val, inplace=True)
                self.test_set[feature_name].fillna(default_val, inplace=True)
                gc.collect()

        gc.collect()
        self.imputed = True
        print "features have been imputed (median or mode)"
        print "total processing time:", str(int(round((time.time() - begin) / 60))), "minutes"

    # method: get_feat_cheap_type(feature_name)
    #
    # - returns the cheapest type that suits the given feature name
    # - calculation is based on the current train_set and test_set
    #
    def get_feat_cheap_type(self, feature_name):

        # we need to know the values range to find the best encoding type
        m = self.train_set[feature_name].min()
        M = self.train_set[feature_name].max()

        from sklearn.preprocessing import Imputer

        # check if there are missing values in the training set or the test set
        missing_in_train = any(self.train_set[feature_name].isnull().values)
        missing_in_test = False
        if feature_name in list(self.test_set.columns.values):
            test_series = self.test_set[feature_name]
            missing_in_test = any(test_series.isnull().values)
            m = min(m, self.test_set[feature_name].min())
            M = max(M, self.test_set[feature_name].max())

        # if there are missing values a float encoding is required (cf pandas gotchas)
        if missing_in_train or missing_in_test:
            # check which is the cheapest float type that could suit the current feature
            # NOTE: float8 doesn't exit
            if m >= numpy.finfo(numpy.float16).min and M <= numpy.finfo(numpy.float16).max:
                encoding_type = 'float16'
            elif m >= numpy.finfo(numpy.float32).min and M <= numpy.finfo(numpy.float32).max:
                encoding_type = 'float32'
            else:
                encoding_type = 'float64'

        # otherwise we can encode the feature with an integer type
        else:
            if m >= numpy.iinfo(numpy.int8).min and M <= numpy.iinfo(numpy.int8).max:
                encoding_type = 'int8'
            elif m >= numpy.iinfo(numpy.int16).min and M <= numpy.iinfo(numpy.int16).max:
                encoding_type = 'int16'
            elif m >= numpy.iinfo(numpy.int32).min and M <= numpy.iinfo(numpy.int32).max:
                encoding_type = 'int32'
            else:
                encoding_type = 'int64'

        return encoding_type

    def calculate_size(self):
        pass

    def make_sparse(self, ):
        if self.sparse:
            sys.exit("data sets are already sparse")
        if not self.loaded:
            sys.exit("data has not been loaded yet")
        if not self.encoded:
            sys.exit("data has not been encoded yet")

        begin = time.time()

        seq = range(self.train_set.shape[0])
        random.shuffle(seq)
        self.sparse_seq = seq
        cutoff = 2 * int(round(len(seq) / 3))
        self.sparse_cutoff = cutoff

        drivers = set(self.train_set.columns) - set(list(self.target_label) + self.excluded_features)
        drivers = list(drivers)
        feature_name = drivers[0]

        self.sparse_train = sparse.csr_matrix(self.train_set[feature_name].iloc[seq[0:cutoff]].values,
                                              dtype=self.train_set[feature_name].dtypes)
        self.sparse_train = self.sparse_train.transpose()

        self.sparse_test = sparse.csr_matrix(self.train_set[feature_name].iloc[seq[(cutoff + 1):]].values,
                                             dtype=self.train_set[feature_name].dtypes)
        self.sparse_test = self.sparse_test.transpose()

        for feature_name in drivers[1:]:
            sparse_vector = sparse.csr_matrix(self.train_set[feature_name].iloc[seq[0:cutoff]].values,
                                              dtype=self.train_set[feature_name].dtypes)
            sparse_vector = sparse_vector.transpose()
            self.sparse_train = sparse.hstack([self.sparse_train, sparse_vector])

            sparse_vector = sparse.csr_matrix(self.train_set[feature_name].iloc[seq[(cutoff + 1):]].values,
                                              dtype=self.train_set[feature_name].dtypes)
            sparse_vector = sparse_vector.transpose()
            self.sparse_test = sparse.hstack([self.sparse_test, sparse_vector])

            self.train_set.drop(feature_name, axis=1, inplace=True)
            gc.collect()

        gc.collect()
        self.sparse = True

        print "train & test set are sparse now"
        print "total processing time:", str(int(round((time.time() - begin) / 60))), "minutes"

    def encode_and_make_sparse(self):

        # TODO: check that error ValueError: X needs to contain only non-negative integers (rows_limit_train=1000)
        if self.encoded:
            sys.exit("data is already encoded")
        if self.sparse:
            sys.exit("data sets are already sparse")
        if not self.loaded:
            sys.exit("data has not been loaded yet")
        if len(self.factors) < 1:
            sys.exit("there is no factor feature")

        begin = time.time()
        to_remove = []

        seq = range(self.train_set.shape[0])
        random.shuffle(seq)
        self.sparse_seq = seq
        cutoff = 2 * int(round(len(seq) / 3))
        self.sparse_cutoff = cutoff

        factor_drivers = self.factors
        other_drivers = set(self.train_set.columns) - set(
            list(self.target_label) + self.excluded_features + self.factors)

        l = 0
        feature_name = factor_drivers[l]
        categories = self.train_set[feature_name].unique()
        n_cat = categories.shape[0]
        while n_cat > self.card_threshold:
            l += 1
            feature_name = factor_drivers[l]
            categories = self.train_set[feature_name].unique()
            n_cat = categories.shape[0]

        # TO DO: what if there is no driver that match the criterion??
        OneHotEnc = OneHotEncoder(n_values=n_cat)
        dummies_train = OneHotEnc.fit_transform(self.train_set.iloc[seq[0:cutoff]][[feature_name]])
        dummies_test = OneHotEnc.transform(self.train_set.iloc[seq[(cutoff + 1):]][[feature_name]])
        self.sparse_train = sparse.csr_matrix(dummies_train, dtype="bool_")
        self.sparse_test = sparse.csr_matrix(dummies_test, dtype="bool_")
        print feature_name, n_cat
        l += 1

        for feature_name in factor_drivers[l:]:
            categories = self.train_set[feature_name].unique()
            n_cat = categories.shape[0]
            OneHotEnc = OneHotEncoder(n_values=n_cat)
            if n_cat <= self.card_threshold:
                print feature_name, n_cat
                # OneHotEnc returns sparse matrix by default, oh yeah
                dummies_train = OneHotEnc.fit_transform(self.train_set.iloc[seq[0:cutoff]][[feature_name]])
                dummies_test = OneHotEnc.transform(self.train_set.iloc[seq[(cutoff + 1):]][[feature_name]])
                sparse_vector = sparse.csr_matrix(dummies_train, dtype="bool_")
                self.sparse_train = sparse.hstack([self.sparse_train, sparse_vector])
                sparse_vector = sparse.csr_matrix(dummies_test, dtype="bool_")
                self.sparse_test = sparse.hstack([self.sparse_test, sparse_vector])

            gc.collect()

            self.train_set.drop(feature_name, axis=1, inplace=True)
            self.test_set.drop(feature_name, axis=1, inplace=True)
            to_remove.append(feature_name)

        for fact in to_remove:
            self.factors.remove(fact)

        # TODO: do the same as above
        for feature_name in list(other_drivers):
            sparse_vector = sparse.csr_matrix(self.train_set[feature_name].iloc[seq[0:cutoff]].values,
                                              dtype=self.train_set[feature_name].dtypes)
            sparse_vector = sparse_vector.transpose()
            self.sparse_train = sparse.hstack([self.sparse_train, sparse_vector])
            sparse_vector = sparse.csr_matrix(self.train_set[feature_name].iloc[seq[(cutoff + 1):]].values,
                                              dtype=self.train_set[feature_name].dtypes)
            sparse_vector = sparse_vector.transpose()
            self.sparse_test = sparse.hstack([self.sparse_test, sparse_vector])
            self.train_set.drop(feature_name, axis=1, inplace=True)
            gc.collect()

        gc.collect()
        self.encoded = True
        self.sparse = True
        print "factor features have been encoded (one-hot strategy)"
        print "train & test set are sparse now"
        print "total processing time:", str(int(round((time.time() - begin) / 60))), "minutes"


import numpy
import types
from abc import ABCMeta, abstractmethod
from bisect import bisect_left
from collections import Counter
from pandas import DataFrame
from .core import check


class FeaturesHandler(object):
    __metaclass__ = ABCMeta

    def __init__(self, in_feature_names):
        self.in_feature_names = in_feature_names
        self.out_feature_names = None

    def size(self):
        return len(self.out_feature_names)

    @abstractmethod
    def apply(self, input_generator):
        """
        :param input_generator: sequence of feature values corresponding to in_feature_names
        :return sequence of (index,value)
        """
        return enumerate(input_generator)

    def learn(self, df):
        """
        Train the handler from data
        :param df: data frame
        """
        assert isinstance(df, DataFrame)


class NoHandler(FeaturesHandler):
    def __init__(self, in_feature_names):
        super(NoHandler, self).__init__(in_feature_names)
        self.out_feature_names = in_feature_names

    def apply(self, input_generator):
        return enumerate(input_generator)


class BoolHandler(NoHandler):
    """
    This is nothing but a NoHandler with a different name
    """


# Mapper handler maps input features to output features via a function (e.g. a lambda)
class MapperHandler(FeaturesHandler):
    def __init__(self, in_feature_names, out_feature_names, mapper):
        """
        :param in_feature_names: list of input feature names
        :param out_feature_names: list of output feature names
        :param mapper: a function that takes a list and return another list
        """
        super(self.__class__, self).__init__(in_feature_names)
        self.out_feature_names = out_feature_names
        self._mapper = mapper

    def apply(self, input_generator):
        return enumerate(self._mapper(list(input_generator)))


class OneToOneMapperHandler(FeaturesHandler):
    def __init__(self, in_feature_name, out_feature_name, mapper):
        self.in_feature_names = [in_feature_name]
        self.out_feature_names = [out_feature_name]
        self._mapper = mapper

    def apply(self, input_generator):
        yield 0, self._mapper(input_generator.next())


class CategoricalHandler(FeaturesHandler):
    def __init__(self, in_feature_names, cats={}, preprocessor=None):
        for in_feature_name in cats:
            if in_feature_name not in in_feature_names:
                raise ValueError('cats contains keys ({0}) not in feature names'.format(in_feature_name))
        super(self.__class__, self).__init__(in_feature_names)

        maps = [None] * len(in_feature_names)
        count = 0  # total number of unique values
        out_feature_names = []
        for in_feature_name in cats:
            _map = {}
            for value in cats[in_feature_name]:
                _map[value] = count
                out_feature_names.append(in_feature_name + '=' + str(value))
                count += 1
            maps[in_feature_names.index(in_feature_name)] = _map

        self.out_feature_names = out_feature_names
        self._maps = maps
        self._count = count
        self._preprocessor = lambda x: x if preprocessor is None else preprocessor

    def learn(self, df):
        maps = self._maps
        preprocessor = self._preprocessor
        out_feature_names = self.out_feature_names
        for i, in_feature_name in enumerate(self.in_feature_names):
            if maps[i] is not None:
                continue
            _map = {}
            for value in df[in_feature_name]:
                value = preprocessor(value)
                if value not in _map:
                    _map[value] = self._count
                    out_feature_names.append(in_feature_name + '=' + str(value))
                    self._count += 1
            maps[i] = _map

    def apply(self, input_generator):
        preprocessor = self._preprocessor
        for i, value in enumerate(input_generator):
            index = self._maps[i].get(preprocessor(value))
            if index is not None:
                yield index, 1


class BinNormalizer(FeaturesHandler):
    def __init__(self, in_feature_names, n_bins=32, thresholds_groups=None):
        """
        :param in_feature_names: names of input features
        :param n_bins: number of bins for each input feature
            A small number of bins (10-20) is typically sufficient for tree learners.
            For other learners (e.g. linear or NN), use a larger number of bins (100-1000).
        :param thresholds_groups: thresholds for each group
            If provided, n_bins will be ignored.
        """
        super(self.__class__, self).__init__(in_feature_names)
        self.out_feature_names = [name + '_bin' for name in in_feature_names]
        check(thresholds_groups is None
              or (len(thresholds_groups) == len(in_feature_names) and n_bins > 1))
        self._threshold_groups = thresholds_groups
        self._n_bins = n_bins

    def learn(self, df):
        if self._threshold_groups is not None:
            return

        self._threshold_groups = []
        for col in self.in_feature_names:
            # 在计算时首先舍去空值
            f_values = df[col].dropna()
            assert numpy.issubdtype(f_values, numpy.number)
            # 统计每个值出现的次数
            histogram = sorted(Counter(f_values).items())
            # 划分的数据的个数要大于区间的个数，如果区间个数大于点的个数，则至少有一个区间将会是空区间
            if len(histogram) <= self._n_bins:
                self._threshold_groups.append(None)
                continue

            thresholds = []
            n_bins = self._n_bins
            # 要划分的值，将这些值分到区间中去
            n_items = float(len(f_values))
            ih = 0
            ih_lim = len(histogram)
            while n_bins > 1:
                # 均分的区间的大小
                expected_bin_size = n_items / n_bins
                count = 0
                while count < expected_bin_size:

                    count += histogram[ih][1]
                    ih += 1
                # 添加到上个区间的最后一个值
                last_addition = histogram[ih-1][1]
                # 如果添加了某个值使得这个区间内的个数超过了区间的范围，那么则去衡量添加这个数后超过的大小与
                # 不添加这个数后再需要添加多少个数才能得到范围进行衡量对比，如果超过的程度大于不足的程度。那么
                # 这个数就添加到下一个区间内否则添加到上一个区间
                if count - expected_bin_size > expected_bin_size - (count - last_addition) \
                        and count != last_addition:
                    ih -= 1
                    count -= last_addition
                # 如果得到了范围的边界，那么就跳出
                if ih == ih_lim:
                    break
                # 上一个区间的最后一个值和下一个区间的第一个值得中间值添加到划分点的集合中
                thresholds.append((histogram[ih-1][0] + histogram[ih][0])/2)
                n_items -= count
                n_bins -= 1
            self._threshold_groups.append(thresholds)

    # 获得这个值得索引
    def apply(self, input_generator):
        for i, value in enumerate(input_generator):
            thresholds = self._threshold_groups[i]
            yield i, (bisect_left(thresholds, value) if thresholds else value)


class MinMaxNormalizer(FeaturesHandler):
    def __init__(self, in_feature_names):
        super(MinMaxNormalizer, self).__init__(in_feature_names)
        raise RuntimeError('Not yet implemented')

    def learn(self, df):
        pass

    def apply(self, input_generator):
        pass


class PredicatesHandler(FeaturesHandler):
    """
    The predicates handler computes predicate features
    Those predicates are defined as normal functions in a module
    For each predicate:
        - The function name is treated as output feature name
        - The function argument variables are treated as input feature names
    """

    def __init__(self, module):
        predicates = []
        features_set = set()
        out_feature_names = []
        for f in dir(module):
            if (not isinstance(module.__dict__.get(f), types.FunctionType)) or f.startswith('_'):
                continue  # filter out elements that are not public functions
            predicate = module.__dict__.get(f)
            predicates.append(predicate)
            out_feature_names.append(f)
            features_set |= set(predicate.func_code.co_varnames)
        in_feature_names = list(features_set)
        name_to_idx = {name: idx for idx, name in enumerate(in_feature_names)}
        self._indices = [[name_to_idx[name] for name in predicate.func_code.co_varnames]
                         for predicate in predicates]
        self._predicates = predicates
        self.in_feature_names = in_feature_names
        self.out_feature_names = out_feature_names

    def apply(self, input_generator):
        values = list(input_generator)
        indices = self._indices
        return enumerate(predicate(*[values[j] for j in indices[i]])
                         for i, predicate in enumerate(self._predicates))