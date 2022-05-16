#!/usr/bin/env python
# coding: utf-8

# In[2]:


# -*- coding: utf-8 -*-
"""
Simulation of extracting network topology correlation matrix as presented on fig.1 a-c
"""
import networkx as nx
from scipy import stats 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
#plt.rcParams.update({'font.size': 24})

dois = ['Music','Movies', 'Reading','Writing','Friends','Art','Photography','Books']

#generate a random directed graph seed 25
DG = nx.gn_graph(len(dois),seed=25)
label_dict = {}
for i,doi in enumerate(dois):
    DG.node[i]['name']= doi
    label_dict[i] = doi
    
nx.draw_circular(DG,labels = label_dict, with_labels=True, arrowsize=22, node_color='bisque',node_size=3000 ,font_size=22)
plt.savefig('Results/simulated_DOIs_directed_network.png')
print(label_dict)


# In[17]:


d_closeness = nx.algorithms.centrality.closeness_centrality(DG)
ud_closeness = nx.algorithms.centrality.closeness_centrality(DG.to_undirected())
                                                             
c = ["{0:0.1f}\n({1:0.1f})".format(d_closeness[i],ud_closeness[j]) for i,j in zip(d_closeness,ud_closeness)]

label_dict = {}
for i,doi in enumerate(dois):
    label_dict[i] = c[i]
                                                               
import numpy as np
np.set_printoptions(precision=1)
nx.draw_circular(DG,labels = label_dict, with_labels=True,verticalalignment='center', node_color='bisque', arrowsize=16, node_size=3000 ,font_size=12)
#plt.title('closeness centraility - directed (undirected)')


# In[26]:


import networkx.algorithms.isomorphism as iso

#directed motif3_8
fig, ax = plt.subplots(figsize=(14,10), nrows=1, ncols=3)    

G2 = nx.DiGraph()
G1 = nx.DiGraph()
nx.add_path(G1, [3,2,1], weight=1)
nx.add_path(G1, [3,1], weight=1)

nx.add_path(G2, [3,2,1], weight=1)
nx.add_path(G2, [1,2,3], weight=1)

#plt.title('motif3_8')
nx.draw_circular(G2,arrowsize=22, node_color='bisque',node_size=3000 ,font_size=22, ax=ax[0])
nx.draw_circular(G2.to_undirected(),arrowsize=22, node_color='bisque',node_size=3000 ,font_size=22,ax=ax[1])
nx.draw_circular(G1.to_undirected(),arrowsize=22, node_color='bisque',node_size=3000 ,font_size=22,ax=ax[2])

'''
#undirected motif
G1 = DG.to_undirected()
G2 = nx.DiGraph()
nx.add_path(G1, [1,2,3,4], weight=1)
nx.add_path(G2, [10,20,30,40], weight=2)
em = iso.numerical_edge_match('weight', 1)
nx.is_isomorphic(G1, G2)  # no weights considered
'''


# In[24]:


'''
#calculate topological features of the vertices
degrees_out = []
degrees_in = []
pageranks = []
closnesses =[]
k_core = []

#iterate of all the dois
for i,doi in enumerate(dois):
    degrees_out.append(DG.out_degree([i])[i])
    degrees_in.append(DG.in_degree([i])[i])
    pageranks.append(nx.pagerank(DG)[i])
    closnesses.append(nx.closeness_centrality(DG)[i])
    k_core.append(nx.algorithms.core.core_number(DG)[i])
    

c = ["{0:0.1f}".format(i) for i in degrees_out]
label_dict = {}
for i,doi in enumerate(dois):
    DG.node[i]['name']= doi
    label_dict[i] = c[i]
    
import numpy as np
np.set_printoptions(precision=1)
nx.draw_circular(DG,labels = label_dict, with_labels=True, node_color='bisque')

#doi tags representation  
vertex=0 # DOI 'Love' for example
edges = list(DG.edges(data=True))
tag_to_vertex = np.zeros((len(dois)))

for j,edge in enumerate(edges):
    src, trg , w = edge
    if trg == vertex:
        tag_to_vertex[src]=1
     
#calculate spearman's correlation between DOI&Features
y = tag_to_vertex
x = np.array([degrees_out,degrees_in,pageranks,closnesses])
print(f'Doi:{label_dict[vertex]}\n\ntags-->{y}\n\nNAV-->deg_out,deg_in,pagerank,closeness\n{x}') 

corr, _ = stats.spearmanr(x.T,y)

print(f'\nspermans rho:{corr[1:,0]}')
'''


# In[25]:





# In[27]:





# In[ ]:




