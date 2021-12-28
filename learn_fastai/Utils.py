from tqdm import tqdm
import json
import cv2
import os
import re
import pickle
import math
from pathlib import *
import multiprocessing
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from collections import *
from itertools import *
from functools import *
from sklearn.metrics import *
from scipy.stats import *
import pandas as pd
import seaborn as sns
import hashlib
from PIL import Image
from timeit import default_timer


def isnan(x):
    if not isinstance(x, float):
        return False
    return math.isnan(x)

def to_dataset_mapping(ids, n_fold, salt=''):
    result = {}
    for one_id in ids:
        result[one_id] = int(hashlib.sha256((str(one_id)+salt).encode('utf-8')).hexdigest(), 16) % n_fold
    return result

def str_hash(s, salt=''):
    return int(hashlib.sha256((str(s)+salt).encode('utf-8')).hexdigest(), 16)

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def json_dump(obj, path):
    ensure_file(path)
    with open(path, 'w', encoding='utf8') as f:
        json.dump(obj, f, indent=4, ensure_ascii=False, sort_keys=True, cls=SetEncoder)


def json_load(path):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)


def pkl_dump(obj, path):
    ensure_file(path)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def pkl_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def np_save(obj, path):
    ensure_file(path)
    with open(path, 'wb') as f:
        np.save(f, obj)


def np_load(path):
    with open(path, 'rb') as f:
        return np.load(f)


def chunk(list, n):
    result = []
    for i in range(n):
        result.append(list[math.floor(i / n * len(list)):math.floor((i + 1) / n * len(list))])
    return result


def df_split(list, ratios):
    results = []
    sum_value = sum(ratios)
    ratios = [x / sum_value for x in ratios]
    current = 0
    for ratio in ratios:
        results.append(list[int(len(list) * current):int(len(list) * (current + ratio))])
        current += ratio
    return results


def list_to_str(list):
    return [str(x) for x in list]


def chunk_sample(list, n):
    result = []
    for i in range(1, n):
        result.append(list[math.floor(i / n * len(list))])
    return result


def chunk_to_batches(list, batch_size):
    result = []
    for i in range(0, len(list), batch_size):
        result.append(list[i:i + batch_size])
    return result


def ensure_path(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def ensure_file(filepath):
    Path(os.path.dirname(filepath)).mkdir(parents=True, exist_ok=True)


def run_multi_process(item_list, n_proc, func, with_proc_num=False):
    tasks = chunk(item_list, n_proc)
    if with_proc_num:
        for i in range(len(tasks)):
            tasks[i] = (i, tasks[i])
    with multiprocessing.Pool(processes=n_proc) as pool:
        results = pool.map(func, tasks)
    return results


def bootstrap(func, y_true, y_pred, n=100, random_state=42, ci=(0.025, 0.975), index=None, with_ci=True):
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    val = func(y_true, y_pred)
    if index is not None:
        val = val[index]
    if not with_ci:
        return val
    bootstrapped_scores = []
    rng = np.random.RandomState(random_state)
    for i in range(n):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = func(y_true[indices], y_pred[indices])
        if index is not None:
            score = score[index]
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    ci_lower = sorted_scores[int(ci[0] * len(sorted_scores))]
    ci_upper = sorted_scores[int(ci[1] * len(sorted_scores))]
    return val, ci_lower, ci_upper


def print_df(df, row=2):
    cols = df.columns.tolist()
    pd.set_option('display.max_columns', len(cols))
    pd.set_option('display.max_rows', row)
    print(cols)
    print(len(df))
    display(df)
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows')
    

def df2map(df,col_key,col_val):
    return df.drop_duplicates(col_key).set_index(col_key)[col_val]


def isnan(x):
    return isinstance(x, float) and math.isnan(x)


def vc(series, to_dict=True, dropna=True):
    result = series.value_counts(dropna=dropna)
    if to_dict:
        return print(result.to_dict())
    print(result)
    

def bp():
    raise Exception()
    

class Benchmark(object):
    def __init__(self, msg, print=True):
        self.msg = msg
        self.print = print
        
    def print_elapsed(self, add_msg):
        t = default_timer() - self.start
        if self.print:
            print((f"{self.msg}, {add_msg}: {t:.2f} seconds"))

    def __enter__(self):
        self.start = default_timer()
        if self.print:
            print((f"{self.msg}: begin"))
        return self

    def __exit__(self, *args):
        t = default_timer() - self.start
        if self.print:
            print((f"{self.msg}: {t:.2f} seconds"))
        self.time = t
    
    
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(1)
