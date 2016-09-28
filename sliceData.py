# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 10:29:45 2016

@author: iki
"""
import csv as csv 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
#
df_row = pd.read_csv('train_numeric.csv',nrows=1000, header=0)
#df_row.to_csv('Overview_row.csv', index=False, header=False)

overviwe_row = df_row.values
#
#df_col = pd.read_csv('train_numeric.csv',usecols=[444,615,616,617,704], header=0)
##df_row.to_csv('Overview_row.csv', index=False, header=False)
#overviwe_col = df_col.values
#
#dfTrainDate_row = pd.read_csv('train_date.csv',nrows=10, header=0)
##df_row.to_csv('Overview_row.csv', index=False, header=False)
#
#overviewDate_row = dfTrainDate_row.values

df_cat = pd.read_csv('train_categorical.csv',nrows=5000, header=0)
#df_row.to_csv('Overview_row.csv', index=False, header=False)

overview_cat = df_cat.values
df_cat.to_csv('slice_categorical.csv', index=False, header=True)
df_row.to_csv('slice_numeric.csv', index=False, header=True)




df_date = pd.read_csv('train_date.csv',nrows=1000, header=0)

overview_dat = df_date.values
df_date.to_csv('slice_date.csv', index=False, header=True)

df_response = pd.read_csv('train_numeric.csv',usecols=[969], header=0)
response_col = df_response.values

col_sum = np.sum(response_col)