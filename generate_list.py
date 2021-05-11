import json
import numpy as np
import pickle

import tensorflow as tf
import tensorflow_federated as tff


sample_Data, _, test_data = tff.simulation.datasets.stackoverflow.load_data(cache_dir = './stfdata')

idlist = sample_Data.client_ids
# lis = pickle.load(open('list.txt', 'rb'))
# fp = open('list.json', 'w')
# data = []
# dic = {}
# strid = "00000000"
# id = 0
dic = {}
fp = open('graph_id.json', 'w')
for client_id in idlist:
    print(type(client_id))
    dic[client_id] = int(client_id)


# for i in range(342477):
#     id += 1

    
# json_info = json.dump(dic)
json.dump(dic, fp, sort_keys= True)