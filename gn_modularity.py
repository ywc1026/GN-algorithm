# -*- coding: utf-8 -*-

__author__ = "yuweicheng"


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os


PATH = "/media/ywc1026/Myself/Data/sample/"
fileList = os.listdir(PATH)

columns1 = ['QTime', 'TPrice', 'TVolume_accu', 'TDeals_accu',
 'Bidpr1', 'Bidpr2', 'Bidpr3', 'Askpr1', 'Askpr2', 'Askpr3', 'Trdirec']
columns2 = ['TPrice', 'TVolume_accu', 'TDeals_accu',
 'Bidpr1', 'Bidpr2', 'Bidpr3', 'Askpr1', 'Askpr2', 'Askpr3']
columns3 = ['TPrice', 'TVolume_accu', 'TDeals_accu',
 'Bidpr1', 'Bidpr2', 'Bidpr3', 'Askpr1', 'Askpr2', 'Askpr3', 'Trdirec']


def get_data(file):
	# preprocess the raw data.
	reader = pd.read_csv(os.path.join(PATH,file),encoding='gbk',iterator=True)
	df = reader.get_chunk(10000)
	df = df[df.Qdate == '1/04/2013']
	stock = df[columns1]
	stock.index = pd.DatetimeIndex(stock['QTime'])
	stk = stock.between_time(time(9,30), time(15,0))
	ret_index = retindex(stk)
	dic = {'F': 0, 'B': 1, 'S': 2}
	trdirec = stk['Trdirec'].map(dic)
	ret_index['Trdirec'] = trdirec
	resamp = ret_index.resample('2S', fill_method='ffill')
	return resamp


def retindex(df, cols=columns2):
	# calculate the returns of all features.
	returns = df[cols].pct_change()
	ret_index = (1 + returns).cumprod()
	ret_index.ix[0,cols] = 1
	return ret_index


def eucldist(df1, df2):
	# calculate the euclidean distance of two stock.
	resamp1 = get_data(df1)
	resamp2 = get_data(df2)
	array1 = np.array(resamp1.reset_index()[columns3])
	array2 = np.array(resamp2.reset_index()[columns3])
	length = min(len(array1), len(array2))
	arr1 = array1[:length, :]
	arr2 = array2[:length, :]
	eucdist = np.linalg.norm(arr1-arr2) / length
	return eucdist
	

def modularity(G, deg_, m_):
	# calculate the modularity Q.
	New_A = nx.adj_matrix(G)
	New_deg = {}
	New_deg = UpdateDeg(New_A, G.nodes())
	comps = nx.connected_components(G)
	print('no of comp: %d' % nx.number_connected_components(G))
	Mod = 0    
	for c in comps:
	    EWC = 0    
	    RE = 0    
	    for u in c:
	        EWC += New_deg[u]
	        RE += deg_[u]        
	    Mod += ( float(EWC) - float(RE*RE)/float(2*m_) )
	Mod = Mod/float(2*m_)
	return Mod


def UpdateDeg(A, nodes):
	# calculate the degree of each node.
	deg_dict = {}
	n = len(nodes)
	B = A.sum(axis = 1)
	for i in range(n):
		deg_dict[nodes[i]] = B[i, 0]
	return deg_dict


def CmtyGirvanNewmanStep(G):
	# remove the edge which has biggest betweenness.
	init_ncomp = nx.number_connected_components(G)
	ncomp = init_ncomp
	while ncomp <= init_ncomp:
	    bw = nx.edge_betweenness_centrality(G, weight='weight') 
	    max_ = max(bw.values())
	    for k, v in bw.items():
	        if float(v) == max_:
	        	G.remove_edge(k[0],k[1])
	    ncomp = nx.number_connected_components(G)


def runGirvanNewman(G, Orig_deg, m_):
	# run the GN algorithm
	BestQ = 0.0
	Q = 0.0
	for i in range(5):
	    CmtyGirvanNewmanStep(G)
	    Q = modularity(G, Orig_deg, m_)
	    print ("current modularity: %f" % Q)
	    if Q > BestQ:
	        BestQ = Q
	        Bestcomps = nx.connected_components(G)    
	        print("comps:")
	        print(Bestcomps)

	pos = nx.spring_layout(G)
	nx.draw_networkx_nodes(G, pos)
	nx.draw_networkx_edges(G, pos)
	nx.draw_networkx_labels(G, pos)
	plt.axis('off')
	plt.title('delay10min')    
	plt.show()




if __name__ == '__main__':
	corr = []



