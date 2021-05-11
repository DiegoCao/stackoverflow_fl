

import tensorflow as tf
import tensorflow_federated as tff
import time 
from tensorflow_federated.python.simulation import from_tensor_slices_client_data
from tensorflow_federated.python.simulation import hdf5_client_data
import collections
from src import (dataset, metrics, embeddings, model, validation,
    federated, generate_text, transfer_learning)

import matplotlib.pyplot as plt

# graph = tf.Graph()
import numpy as np
# tf.io.write_graph('file.txt', graph)
import networkx as nx 
import pickle

def loadGraph(G, file):
    G = pickle.load(open(file, 'rb'))

def saveGraph(G, file):
    print('-------------savegraph!--------------------')
    G = pickle.dump(G, open(file, 'wb'))
    
def printNode(G):
    for node in G.nodes(data = True):
        print(node)

def initGraph(G, client_ids):
    G.add_nodes_from(client_ids)

def buildGraph(G, title_answer_dict, title_question_dict):
    print('------------buildinggraph!-----')
    for title_name, anser_id_list in title_answer_dict.items():
        
        if title_name in title_question_dict:
            quest_id_list = title_answer_dict[title_name]
            for qid in quest_id_list:
                for ansid in anser_id_list:
                    if (ansid, qid) not in G.edges():
                        G.add_edge(ansid, qid, weight = 1)
                    else:
                        G[ansid][qid]['weight'] += 1

    print('-----------build graph done!-----------------')


def federatedGraph(G, large_dataset, title_answer_dict, title_question_dict):
    initGraph(G, large_dataset.client_ids)
    # the dataset should be the corresponding large dataset
    for cid in large_dataset.client_ids:
        data = large_dataset.create_tf_dataset_for_client(i)

        for sample in data.take:
            title_name = sample['title'].ref()

            if(sample['type'] == "question"):
                type = 0
            else:
                type = 1
            '''
            pair refers to client_ids and 
            '''
            id = cid 
            if type == 1:
                if title_name in title_answer_dict:
                    title_answer_dict[title_name].append(id)
                else:
                    tlist = []
                    tlist.append(id)
                    title_answer_dict[title_name] = tlist
            elif type == 0:
                if title_name in title_question_dict:
                    title_question_dict[title_name].append(id)
                else:
                    tlist = []   
                    tlist.append(id)
                    title_question_dict[title_name] = tlist   


                    
def sortnodebyDegree(G):
    '''
        return sorted nodes set, first idx degree, second idx edge
    '''
    lis = sorted(G.degree, key=lambda x: x[1], reverse=True)

# def loadGraph(G, file):
#     G = pickle.load(open(file, 'rb'))

def saveIdxlist(lis):
    with open('list.txt', 'wb') as fp:
        pickle.dump(lis, fp)

def readIdxlist():
    with open('list.txt', 'wb') as fp:
        pickle.load(fp)

def mysynthetic():
  """Returns a small synthetic dataset for testing.
  Provides two clients, each client with only 3 examples. The examples are
  derived from a fixed set of examples in the larger dataset, but are not exact
  copies.
  Returns:
     A `tff.simulation.ClientData` object that matches the characteristics
     (other than size) of those provided by
     `tff.simulation.datasets.stackoverflow.load_data`.
  """
  return from_tensor_slices_client_data.FromTensorSlicesClientData(
      _SYNTHETIC_STACKOVERFLOW_DATA)


_SYNTHETIC_STACKOVERFLOW_DATA = {
    'synthetic_1':
        collections.OrderedDict(
            creation_date=[
                b'2010-01-08 09:34:05 UTC',
                b'2008-08-10 08:28:52.1 UTC',
                b'2008-08-10 08:28:52.1 UTC',
            ],
            title=[
                b'function to calculate median in sql server',
                b'creating rounded corners using css',
                b'creating rounded corners using css',
            ],
            score=np.asarray([
                172,
                80,
                80,
            ]).astype(np.int64),
            tags=[
                b'sql|sql-server|aggregate-functions|median',
                b'css|cross-browser|rounded-corners|css3',
                b'css|cross-browser|rounded-corners|css3',
            ],
            tokens=[
                b"if you're using sql 2005 or better this is a nice , simple-ish median calculation for a single column in a table :",
                b'css3 does finally define the',
                b"which is exactly how you'd want it to work .",
            ],
            type=[
                b'answer',
                b'question',
                b'answer',
            ]),
    'synthetic_2':
        collections.OrderedDict(
            creation_date=[
                b'2008-08-05 19:01:55.2 UTC',
                b'2010-07-15 18:15:58.5 UTC',
                b'2010-07-15 18:15:58.5 UTC',
            ],
            title=[
                b'creating rounded corners using css',
                b'writing to / system / framework in emulator',
                b'writing to / system / framework in emulator',
            ],
            score=np.asarray([
                3,
                12,
                -1,
            ]).astype(np.int64),
            tags=[
                b'git|svn|version-control|language-agnostic|dvcs',
                b'android|android-emulator|monkey',
                b'android|android-emulator|monkey',
            ],
            tokens=[
                b'if you are on mac osx , i found <URL> " > versions to be an incredible ( free ) gui front-end to svn .',
                b'edit :',
                b'thanks .',
            ],
            type=[
                b'answer',
                b'question',
                b'question',
            ],
        ),
    'synthetic_3':
        collections.OrderedDict(
            creation_date=[
                b'2008-10-30 16:49:26.9 UTC',
                b'2008-10-30 16:49:26.9 UTC',
            ],
            title=[
                b'iterator pattern in vb . net ( c # would use yield ! )',
                b'iterator pattern in vb . net ( c # would use yield ! )',
            ],
            score=np.asarray([
                1,
                1,
            ]).astype(np.int64),
            tags=[
                b'vb . net|design-patterns|iterator|yield',
                b'vb . net|design-patterns|iterator|yield',
            ],
            tokens=[
                b'edit :',
                b'the spec is available here .',
            ],
            type=[
                b'answer',
                b'answer',
            ],
        )
}    


if __name__ == "__main__":
    
    G = nx.Graph()
    print(G.is_directed())


    start_t = time.time()
    sample_Data = mysynthetic()
    
    sample_Data, _, test_data = tff.simulation.datasets.stackoverflow.load_data(cache_dir = './stfdata')
    end_t = time.time()
    print('time to load the data: ', end_t - start_t)
    print('start processing data: ')

    # G.add_node("name", feature = [], label = 0)
    # G.add_node(1)
    # G.add_node(2)
    # G.add_edge(1,2, weight = 1)
    # G[1][2]['weight']+=1
    # print('the weight is ', G[1][2]['weight'])
    # print(G.graph)
    # g = pickle.load(open('graph.txt', 'rb'))
    # printNode(g)
    # title_list= tff.simulation.datasets.stackoverflow.load_tag_counts()
    # print(len(title_list))


    title_answer_dict = {}
    title_question_dict = {}
    # print(type(sample_Data.client_ids)

    print(sample_Data.client_ids)
    TRAINNUM = len(sample_Data.client_ids)
    trainlist = sample_Data.client_ids
    G.add_nodes_from(trainlist, ans_num = 0, ques_num = 0, avgscore = 0)
    # nx.set_node_attributes(G, 0, "ans_num")
    # nx.set_node_attributes(G, 0, "ques_num")
    itr = 0
    for i in sample_Data.client_ids:
        # print(i)
        # print('precessing the ', itr ,' client: ')
        itr += 1
        if(itr > TRAINNUM):
            break

        if( (itr%1000)==0):
            print('processing rounds: ', itr)
        data = sample_Data.create_tf_dataset_for_client(i)
        user_ans , user_ques = (0, 0)
        score_list = []
        for sample in data:
            # print(sample)
            title_name = sample['title']
            title_name = title_name.numpy()
            title_score = sample['score']
            score_list.append(title_score)
            typ = 0

            if(sample['type'] == "question"):
                typ = 0
            else:
                typ = 1
            '''
            pair refers to client_ids and 
            '''
            pair = (i, typ)
            id = i
            if sample['type']=="answer":
                user_ans += 1
                # answer 
                if title_name in title_answer_dict:
                    title_answer_dict[title_name].append(id)
                else:
                    tlist = []
                    tlist.append(id)
                    title_answer_dict[title_name] = tlist

            else:
                # question 
                user_ques += 1
                if title_name in title_question_dict:
                    title_question_dict[title_name].append(id)
                else:
                    tlist = []   
                    tlist.append(id)
                    title_question_dict[title_name] = tlist    
        
        
        G.nodes[id]['ans_num'] = user_ans
        G.nodes[id]['ques_num'] = user_ques
        G.nodes[id]['score'] = np.average(score_list)
        
    # print('-------question-----------')
    # for key, item in title_question_dict.items():
    #     print(key) 
    #     print(item)
    # print('------------ans------------')
    # for key, item in title_answer_dict.items():
    #     print(key)
    #     print(item)
    buildGraph(G, title_answer_dict, title_question_dict)      

    node_list = sorted(G.degree, key=lambda x: x[1], reverse=True)


    file = 'graph_large_v2.txt'   
    saveGraph(G, file)
    
    saveIdxlist(node_list)

  
    print('the number of edges: ', G.number_of_edges())

