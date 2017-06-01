#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(["label", "uid"], axis=1).values
    y = df["label"].values
    uid = df["uid"].values
    return X, y, uid


def get_all_data():
    data_path = "../data/extend"
    trains = {}
    tests = {}
    for fp in os.listdir(data_path):
        base_name = os.path.basename(fp)
        fp_name, _ = os.path.splitext(base_name)
        kind = fp_name.split("_")[0]
        name = fp_name.split("_")[1]
        if kind == "train":
            trains[name] = load_data(os.path.join(data_path, base_name))
        if kind == "test":
            tests[name] = load_data(os.path.join(data_path, base_name))
    return trains, tests



