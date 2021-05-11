
import tensorflow as tf
import tensorflow_federated as tff
import time 

from src import (dataset, metrics, embeddings, model, validation,
    federated, generate_text, transfer_learning)

import matplotlib.pyplot as plt

import networkx as nx 
import pickle


def loadGraph(file):
    G = pickle.load(open(file, 'rb'))
    return G

def printGraph(G):
    print(G.nodes())


if __name__ == "__main__":
    G = loadGraph('graph.txt')
    itr = 0
    list(G.edges())
    