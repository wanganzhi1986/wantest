#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from backend import Backend
import os


class RunHistoryBuilder(object):

    def __init__(self, dataset_name=None,
                 output_dir=None,
                 runhistory_dir=None,

                 ):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.runhistory_dir = runhistory_dir

        self.runhistory = None

        if self.output_dir is None:
            raise ValueError("output dir is not given")
        self.backend = Backend(output_dir=self.output_dir)

    def run(self):
        self._build_runhistory()

    def _build_runhistory(self):
        identifier = self.dataset_name
        if self.runhistory is None:
            self.runhistory = self.backend.load_runhistory(identifier)


    # 加载历史信息
    def load_runhistory(self):
        if self.runhistory_dir is None:
            self.runhistory_dir = os.path.join(self.output_dir, "runhistory")

        instance_info = self.backend.load_instance_info(self.runhistory_dir)
        configure_info = self.backend.load_configure_info(self.runhistory_dir)

