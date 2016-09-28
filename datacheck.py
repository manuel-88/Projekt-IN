# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 09:47:27 2016

@author: iki
"""

import csv as csv 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 

#filename = 'train_categorical.csv'
#chunksize = 10 
#for chunk in pd.read_csv(filename, chunksize=chunksize):
#    process(chunk)
    
# For .read_csv, always use header=0 when you know row 0 is the header row
#dfcat = pd.read_csv('train_categorical.csv',ncolumns=1000, header=0)

#a = []
#    
#for i in range(0,10):
#    
#   a.append(i)    
#
#a = np.asarray(a)   
    

#df_row = pd.read_csv('train_numeric.csv',nrows=2, header=0)
dfnum = pd.read_csv('train_numeric.csv',usecols=[969,1,2,3,4,5], header=0)
dfnum_test = pd.read_csv('test_numeric.csv',usecols=[1,2,3,4,5], header=0)
#dfsample = pd.read_csv('sample_submission.csv', header=0)

dfID = pd.read_csv('test_numeric.csv',usecols=[0], header=0)

dfnum = dfnum.fillna(0)
dfnum_test = dfnum_test.fillna(0)

train_data = dfnum.values
test_data = dfnum_test.values
id = dfID.values
id  = id.reshape((1183748,)) 

#dfcat.to_csv('Bosch_test.csv', index=False, header=False)
#dfnum.to_csv('Bosch_test_num.csv', index=False, header=False)





# Create the random forest object which will include all the parameters
## for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::,0:-1],train_data[0::,-1])

# Take the same decision trees and run it on the test data
output = forest.predict(test_data)

predictArr = array([id, output]).T
predictArr = predictArr.astype(int)

predictHead = array([['Id','Response']])
predictArr = r_[predictHead, predictArr]

o_df = pd.DataFrame(predictArr)
o_df.to_csv('Bosch_Out.csv', index=False, header=False)





#
#smallDataset =read_in_chunks('train_numeric.csv')
#
#
#
#
#def read_in_chunks(file_path):
#    chunksize = 25000
#    chunks = []
#    for chunk in pd.read_csv(file_path, chunksize=chunksize):
#        # do some dimensionality reduction...
#        chunks.append(chunk)
#    reduced_data = pd.concat(chunks, axis=0)
#    return reduced_data