# -*- coding: utf-8 -*-

__author__ = "yuweicheng"


import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os


PATH = "/media/ywc1026/Myself/Data/sample/"
fileList = os.listdir(PATH)

def get_data(file):
	# preprocess the raw data.
	reader = pd.read_csv(os.path.join(PATH,file),encoding='gbk',iterator=True)
	df = reader.get_chunk(10000)
	df['Qdate'] = '1/04/2013'
	cols = ['QTime', 'TPrice', 'TVolume_accu', 'TDeals_accu', 'Bidpr1', 'Bidpr2', 'Bidpr3',
        'Askpr1', 'Askpr2', 'Askpr3', 'Trdirec']
    stock = df[cols]
	stock.index = pd.DatetimeIndex(stock['QTime'])
	stk = stock.between_time(time(9,30), time(15,0))
	resamples = df.resample('2S', fill_method='ffill')


def corr(df1, df2):
	# calculate the correlation of two stock.
	pass


