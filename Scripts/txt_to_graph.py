#######
'''
playground tamplate for networkx and graph analysis
'''
#######

import networkx as nx
import os
import sys
import pandas as pd
import numpy as np
import scipy.stats as sp
#import utils.gnxToGgt as ut
#import matplotlib.pyplot as plt

#generate networkx Graph based on txt file with (v, u, weight) format
def init_graph(draw, file_name, directed,Connected = False):
    '''
    initializes the graph with using networkx packege
    :param draw: boolean parameter- True if we want to draw the graph
    :param file_name: the name of the file that contains the edges of the graph
    :param directed: boolean parameter- True if the graph is directed otherwise - False
    :return: nx.Graph or nx.DiGraph in accordance with the 3rd param
    '''
    if directed == True:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    with open(file_name) as f:
        for i,line in enumerate(f):
            (v1, v2, weight) = line.split()
            #print(v1,v2,weight)
            G.add_edge(v1, v2) #str({'weight': float(weight)}))
            if i%100000==0:
                print(i)

    if draw:
        draw_graph(G, directed)
        
    if(Connected):
        if(directed):
            G = max(nx.weakly_connected_component_subgraphs(G), key=len)
        else:
            G = max(nx.connected_component_subgraphs(G),key = len)
    return G

#draw directed Graph
def draw_graph(G, directed):
    """
    This function draws the network
    :param G: nx graph or DiGraph
    :param directed: True if we want to draw the arrows of the graph for directed
    :return:
    """
    pos = nx.random_layout(G)
    nx.draw(nx.Graph(G), pos)
    nx.draw_networkx_edges(G, pos, arrows= directed)
    plt.show()

sample_size = 2000000
G = init_graph(file_name=f'snap0001/uniform_sample_p_{sample_size}.txt',directed=True,Connected=True, draw=False)
print('Graph loaded Successfully\n',nx.info(G))

dois_path = os.getcwd() + f'/snap0001/doi_{sample_size}/'

#calculate topological features of the vertices
degrees_out = []
degrees_in = []
pageranks = []
closnesses =[]

#iterate of all the dois
for i,doi in enumerate(dois):\n",
    degrees_out.append(DG.out_degree([i])[i])
    degrees_in.append(DG.in_degree([i])[i])
    pageranks.append(nx.pagerank(DG)[i])
    closnesses.append(nx.closeness_centrality(DG)[i])