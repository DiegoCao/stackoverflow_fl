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
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_federated as tff

from src import (dataset, metrics, embeddings, model, validation,
    federated, generate_text, transfer_learning)


def load_tag_counts(cache_dir=None):
  """Loads the tag counts for the Stack Overflow dataset.
  Args:
    cache_dir: (Optional) directory to cache the downloaded file. If `None`,
      caches in Keras' default cache directory.
  Returns:
    A collections.OrderedDict where the keys are string tags, and the values
    are the counts of unique users who have at least one example in the training
    set containing with that tag. The dictionary items are in decreasing order
    of tag frequency.
  """
  path = tf.keras.utils.get_file(
      'stackoverflow.tag_count.tar.bz2',
      origin='https://storage.googleapis.com/tff-datasets-public/stackoverflow.tag_count.tar.bz2',
      file_hash='6fe281cec490d9384a290d560072438e7e2b377bbb823876ce7bd6f82696772d',
      hash_algorithm='sha256',
      extract=True,
      archive_format='tar',
      cache_dir=cache_dir)

  dir_path = os.path.dirname(path)
  file_path = os.path.join(dir_path, 'stackoverflow.tag_count')
  with open(file_path) as f:
    tag_counts = json.load(f)
  return collections.OrderedDict(
      sorted(tag_counts.items(), key=lambda item: item[1], reverse=True))

sav = 'taginfo'
if not os.path.exists(sav):
    os.makedirs(sav)

with open("params.json", "r") as read_file:
    params = json.load(read_file)

# Set Parameters
VOCAB_SIZE = params['VOCAB_SIZE']
BATCH_SIZE = params['BATCH_SIZE']
CLIENTS_EPOCHS_PER_ROUND = params['CLIENTS_EPOCHS_PER_ROUND']
MAX_SEQ_LENGTH = params['MAX_SEQ_LENGTH']
MAX_ELEMENTS_PER_USER = params['MAX_ELEMENTS_PER_USER']
CENTRALIZED_TRAIN = params['CENTRALIZED_TRAIN']
SHUFFLE_BUFFER_SIZE = params['SHUFFLE_BUFFER_SIZE']
NUM_VALIDATION_EXAMPLES = params['NUM_VALIDATION_EXAMPLES']
NUM_TEST_EXAMPLES = params['NUM_TEST_EXAMPLES']
NUM_PRETRAINING_ROUNDS = params['NUM_PRETRAINING_ROUNDS']
NUM_ROUNDS = params['NUM_ROUNDS']
NUM_TRAIN_CLIENTS = params['NUM_TRAIN_CLIENTS']
EMBEDDING_DIM = params['EMBEDDING_DIM']
RNN_UNITS = params['RNN_UNITS']
EMBEDDING_LAYER = params['EMBEDDING_LAYER']

train_data, val_data, test_data = dataset.construct_word_level_datasets(
    vocab_size=VOCAB_SIZE,
    batch_size=BATCH_SIZE,
    client_epochs_per_round=CLIENTS_EPOCHS_PER_ROUND,
    max_seq_len=MAX_SEQ_LENGTH,
    max_elements_per_user=MAX_ELEMENTS_PER_USER,
    centralized_train=CENTRALIZED_TRAIN,
    shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
    num_validation_examples=NUM_VALIDATION_EXAMPLES,
    num_test_examples=NUM_TEST_EXAMPLES)




import time 

alltags = load_tag_counts()
users_tag_matrix = []

index_dict = {}
idx = 0

NUM_OF_CHOSEN_WORDS = 100

for key, val in alltags.items():
    index_dict[key] = idx
    idx += 1
    if idx >NUM_OF_CHOSEN_WORDS:
        break



for i in range(1, 100):
    t1 = time.time()
    train_dataset = train_data.create_tf_dataset_for_client(train_data.client_id[i])
    t2 = time.time()
    print('the time of generate one dataset is ', t2 - t1)
    user_map = np.zeros((NUM_OF_CHOSEN_WORDS, 1))
    for x in train_dataset:
        tag = x['tags']
        if tag in index_dict:
            idx = index_dict[tag]
            user_map[idx] += 1

    
    users_tag_matrix.append(user_map)

np.save('taginfo/mat.npy', users_tag_matrix)
print(users_tag_matrix)
