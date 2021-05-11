import tensorflow as tf
import json
import tensorflow_federated as tff 
from src import federated
from src import dataset

import pickle
import pandas as pd

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

with open("params.json", "r") as read_file:
    params = json.load(read_file)

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

LOAD_FROM_PREVIOUS = params['LOAD_FROM_PREVIOUS']

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

train_clients = federated.get_sample_clients(
    dataset=train_data, num_clients=NUM_TRAIN_CLIENTS)

clientids = train_data.client_ids
data_lengths = []

for cid in clientids:
    
    train_dataset = train_data.create_tf_dataset_for_client(cid)
    itr = 0
    for data in train_dataset:
        itr += 1

    data_lengths.append(itr)

    print('processing cid ',cid,' with len ', itr)

pickle.dump(data_lengths, open('id_datalength.txt', 'wb'))
df = pd.DataFrame({'client_id':clientids, 'data_length': data_lengths})
df.to_csv('cid_data.csv')