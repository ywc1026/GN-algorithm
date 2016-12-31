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
	array1 = np.array(df1.reset_index()[columns3])
	array2 = np.array(df2.reset_index()[columns3])
	length = min(len(array1), len(array2))
	arr1 = array1[:length, :]
	arr2 = array2[:length, :]
	eucdist = np.linalg.norm(arr1-arr2) / length
	return eucdist
	


