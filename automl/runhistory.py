#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json
import collections
from collections import namedtuple
import util

RunKey = collections.namedtuple(
    'RunKey', ['config_id', 'instance_id'])

RunValue = collections.namedtuple(
    'RunValue', ['cost', 'time', 'status', 'additional_info'])


class RunHistory(object):
    def __init__(self):
        self.run_info = collections.OrderedDict()

        self.config_to_id = {}  # config -> id
        self.id_to_config = {}  # id -> config
        self.cost_per_config = {}  # config_id -> cost

    def add(self, config, cost, time,
            status, instance_id=None,
            additional_info=None):

        config_id = self.config_ids.get(config)
        if config_id is None:
            config_id = util.get_config_id(config)
            self.config_to_id[json.dumps(config)] = config_id
            self.id_to_config[config_id] = config

        k = RunKey(config_id, instance_id)
        v = RunValue(cost, time, status, additional_info)

        self.run_info[k] = v
        self.update_cost(config)

    def update(self, config, cost, ):

    def update_cost(self, config, result):
        inst_seeds = set(self.get_runs_for_config(config))
        # perf = self.aggregate_func(config, self, inst_seeds)
        config_id = self.config_ids[config]
        self.cost_per_config[config_id] = result

    def get_cost(self, config):
        config_id = self.config_to_id[config]
        return self.cost_per_config[config_id]

        # 对于每一个参数配置获得其运行的实例
    def get_runs_for_config(self, config):
        config_id = util.get_config_id(config)
        instances_ = []
        for k in self.run_info:
            # TA will return ABORT if config. budget was exhausted and
            # we don't want to collect such runs to compute the cost of a configuration
            if config_id == k.config_id and self.data[k].status not in [StatusType.ABORT]:
                instances_.append(k.instance_id)
        return instances_


    def empty(self):
        """
        Check whether or not the RunHistory is empty.

        Returns
        ----------
            bool: True if runs have been added to the RunHistory,
                  False otherwise
        """
        return len(self.data) == 0

    def update_from_json(self, fn, cs):
        """Update the current runhistory by adding new runs from a json file.

        Parameters
        ----------
        fn : str
            file name to load from
        cs : ConfigSpace
            instance of configuration space
        """

        new_runhistory = RunHistory(self.aggregate_func)
        new_runhistory.load_json(fn, cs)

        # Configurations might be already known, but by a different ID. This
        # does not matter here because the add() method handles this
        # correctly by assigning an ID to unknown configurations and re-using
        #  the ID
        for key, value in new_runhistory.data.items():
            config_id, instance_id, seed = key
            cost, time, status, additional_info = value
            config = new_runhistory.ids_config[config_id]
            self.add(config=config, cost=cost, time=time,
                     status=status, instance_id=instance_id,
                     seed=seed, additional_info=additional_info)


    #
    def transform_configuration(self):
        pass


    def _select_instance(self, config):
        pass


