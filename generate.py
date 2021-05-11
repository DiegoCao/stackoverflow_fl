import numpy as np
import networkx as nx 

import pickle


# def loadGraph(G, file):
#     G = pickle.loads(open(file, 'rb'))

# G = nx.Graph()
# loadGraph(G, 'graph_large.txt')

import json
import os
src = './GraphSAGE/fljson/'
def generateFeats(G, id_map):
    N = G.number_of_nodes()
    print('the num of nodes', G.number_of_nodes())
    feats_array = np.zeros((N, 2))  # Use features,
    itr = 0
    for key, _ in id_map.items():
        ans_num = G.nodes[key]['ans_num']
        ques_num = G.nodes[key]['ques_num']
        array = np.array([ans_num, ques_num])
        feats_array[itr][:] = array
        itr += 1

    save_dir = src + 'sto-feats.npy'
    # if os.path.isfile(save_dir) is False:
        
    #     open(save_dir, 'w')
    np.save(save_dir, feats_array)

import random
def generateGraph(G, id_map):
    dic = {}
    dic["directed"] = False
    graphdic = {'name': 'disjoint_union( ,  )'}
    dic["graph"] = graphdic
    dic["multigraph"] = False
    
    nodes_list = []
    links_list = []
    N = G.number_of_nodes()
    val_pro = 0.1
    test_pro = 0.15
    train_pro  = 1 - test_pro - test_pro

    itr = 0
    val_num = int(N*val_pro)
    test_num = int (N*test_pro)
    train_num = N - val_num - test_num
    
    shuffle_list = []
    for key, _id in id_map.items():
        shuffle_list.append((key,_id))
    random.shuffle(shuffle_list)
    
    val_set = set()
    train_set = set()
    test_set = set()

    for _key, _id in shuffle_list:
        itr += 1
        node_dict = {}
        node_dict['test']= False
        node_dict['val'] = False
        node_dict['id'] = _id
        node_dict['key'] = _key
        if itr < val_num:
            node_dict['val'] = True
            val_set.add(_id)

        elif itr >= val_num and itr < val_num + test_num:
            node_dict['test'] = True
            test_set.add(_id)
        else:
            train_set.add(_id)
            pass
        nodes_list.append(node_dict)

    
    for edge in G.edges():
        p1 = id_map[edge[0]]
        p2 = id_map[edge[1]]
        # p1 = edge[0]
        # p2 = edge[1]
        p = [p1, p2]
        source = min(p)
        # if id_map[p1]<id_map[p2]:
        #     source = p1
        #     target = p2
        # else:
        #     source = p2
        #     tar
        target = max(p)
        edgeDict = {}
        edgeDict["source"] = source
        edgeDict["target"] = target
        edgeDict["test_removed"] = False
        edgeDict["train_removed"] = False
        if source in test_set and (target in test_set):
            edgeDict["test_removed"] = True
            edgeDict["train_removed"] = True

        if source in val_set and (target in val_set):
            edgeDict["test_removed"] = True

        links_list.append(edgeDict)    

    dic["nodes"] = nodes_list
    dic["links"] = links_list
    
    save_dir = src + 'sto-G.json'   
    with open(save_dir, 'w') as fp:
        json.dump(dic, fp)



    

with open('./GraphSAGE/fljson/sto-id_map_old.json', 'r') as fp:
    id_map = json.load(fp)


G = nx.read_gpickle('graph_large_v2.txt')


generateFeats(G, id_map)
generateGraph(G, id_map)

for edge in G.edges():
    source = edge[0]
    target = edge[1]
    sid = id_map[source]
    tid = id_map[target]




