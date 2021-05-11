import nest_asyncio
nest_asyncio.apply()

import os, sys, io
sys.path.append(os.getcwd())

import json
import collections
import functools
import six
import time
import string
from tqdm import tqdm

import numpy as np
# import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_federated as tff

import pandas as pd



train_data, test_data = tff.simulation.datasets.shakespeare.load_data(cache_dir = './file')
print(type(train_data))

# print(type(train_data), type(test_data))