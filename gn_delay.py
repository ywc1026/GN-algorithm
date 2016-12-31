# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 10:12:56 2016
1.代码优化了一部分：切片更专业化
2.延迟相关性
@author: Yu
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

#read the data and select one day for test
#use the data align as much as possible
def clean(df):
    #set the index first
    df.index = pd.DatetimeIndex(df['QTime'])
    #slice
    df_morning = df['1/04/2013 09:30:00': '1/04/2013 11:30:00']
    df_resample = df_morning.resample('5S', fill_method='ffill')
    return df_resample
    

def modularity(G, deg_, m_):
    New_A = nx.adj_matrix(G)
    New_deg = {}
    New_deg = UpdateDeg(New_A, G.nodes())
    # Let's compute the Q
    comps = nx.connected_components(G)    # list of components
    print('no of comp: %d' % nx.number_connected_components(G))
    Mod = 0    # Modularity of a given partitionning
    for c in comps:
        EWC = 0    # no of edges within a community
        RE = 0    # no of random edges
        for u in c:
            EWC += New_deg[u]
            RE += deg_[u]        # count the probability of a random edge
        Mod += ( float(EWC) - float(RE*RE)/float(2*m_) )
    Mod = Mod/float(2*m_)
    #print "Modularity: %f" % Mod
    return Mod


def CmtyGirvanNewmanStep(G):
    # print "call CmtyGirvanNewmanStep"
    init_ncomp = nx.number_connected_components(G)    # no of components
    ncomp = init_ncomp
    while ncomp <= init_ncomp:
        bw = nx.edge_betweenness_centrality(G, weight='weight')    # edge betweenness for G
        # find the edge with max centrality
        max_ = max(bw.values())
        # find the edge with the highest centrality and remove all of them if there is more than one!
        for k, v in bw.items():
            if float(v) == max_:
                G.remove_edge(k[0],k[1])    # remove the central edge
        ncomp = nx.number_connected_components(G)    # recalculate the no of components


def runGirvanNewman(G, Orig_deg, m_):
    BestQ = 0.0
    Q = 0.0
    for i in range(5):
        CmtyGirvanNewmanStep(G)
        Q = modularity(G, Orig_deg, m_)
        print ("current modularity: %f" % Q)
        if Q > BestQ:
            BestQ = Q
            Bestcomps = nx.connected_components(G)    # Best Split
            print("comps:")
            print(Bestcomps)
    #draw the networkx
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.axis('off')
    plt.title('delay10min')    
    plt.show()
        

#calculate the degree of each node        
def UpdateDeg(A, nodes):
    deg_dict = {}
    n = len(nodes)  # len(A) ---> some ppl get issues when trying len() on sparse matrixes!
    B = A.sum(axis = 1)# rows
    for i in range(n):
        deg_dict[nodes[i]] = B[i, 0]
    return deg_dict
    

if __name__ == '__main__':    
    fileList = []
    fileList = os.listdir("E:/Data/sample")
    path = "E:/Data/sample/"
    corrMat = np.eye(len(fileList))
    distMat = np.zeros((len(fileList),len(fileList)))
    G = nx.Graph()
    for i in range((len(fileList)-1)):
        for j in range(i+1,len(fileList)):
            df1 = pd.read_csv(path+fileList[i], usecols=(5,11))
            df2 = pd.read_csv(path+fileList[j], usecols=(5,11))
            df1_resample = clean(df1)
            df2_resample = clean(df2)
            corr1 = df1_resample['TPrice'].corr(df2_resample['TPrice'].shift(-10, freq='1T'))
            corr2 = df2_resample['TPrice'].corr(df1_resample['TPrice'].shift(-10, freq='1T'))
            corrMat[i][j] = corr1
            corrMat[j][i] = corr2
            dist1 = (2*(1-abs(corr1)))**0.5
            dist2 = (2*(1-abs(corr2)))**0.5
            distMat[i][j] = dist1
            distMat[j][i] = dist2
            if dist1 < dist2 and dist1 < 0.90:
                G.add_edge(fileList[i], fileList[j], weight = distMat[i][j])
            elif dist1 > dist2 and dist2 < 0.90:
                G.add_edge(fileList[j], fileList[i], weight = distMat[j][i])
        

    T = nx.minimum_spanning_tree(G)            
    n = T.number_of_nodes()
    A = nx.adj_matrix(T)
    m_ = 0.0    # the weighted version for number of edges
    for i in range(0, n):
        for j in range(0, n):
            m_ += A[i,j]
    m_ = m_/2.0
    Orig_deg = {}
    Orig_deg = UpdateDeg(A, T.nodes())
    runGirvanNewman(T, Orig_deg, m_)

'''          
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos,family='Graph')
plt.axis('off')    
plt.show()
'''