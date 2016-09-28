# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:43:19 2016

@author: iki
"""

import csv as csv 
import numpy as np
import pandas as pd

import sys
sys.path.append('/home/iki/Software/xgboost/python-package')

from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.neighbors.nearest_centroid import NearestCentroid
import xgboost as xgb

#df_train = pd.read_csv('Trainset1_500000_5000.csv', header=0)
#df_test = pd.read_csv('Testset1_198121_1879.csv', header=0)
#
#testRes = df_test['Response'].values
#testRes = testRes.astype(int)
#
#df_test = df_test.drop(['Response'], axis=1)
#
#df_train = df_train.fillna(2)
#df_test = df_test.fillna(2)
#
#
#train_data = df_train.values
#test_data = df_test.values
#
#
## Create the random forest object which will include all the parameters
### for the fit
#forest = RandomForestClassifier(n_estimators = 100)
#
## Fit the training data to the Survived labels and create the decision trees
#forest = forest.fit(train_data[0::,0:-1],train_data[0::,-1])
#
##forest = xgb.XGBClassifier(max_depth=10, n_estimators=100, learning_rate=0.05).fit(train_data[0::,0:-1],train_data[0::,-1])
## Take the same decision trees and run it on the test data
#output = forest.predict(test_data)
##output = array([output]).T
#output = output.astype(int)
#
#Result = matthews_corrcoef(testRes, output)  


##############################################################################
#############################################################################
#Bosch submission try


feature_names = ['L3_S38_F3960', 'L3_S33_F3865', 'L3_S38_F3956', 'L3_S33_F3857',
       'L3_S29_F3321', 'L1_S24_F1846', 'L3_S32_F3850', 'L3_S29_F3354',
       'L3_S29_F3324', 'L3_S35_F3889', 'L0_S1_F28', 'L1_S24_F1844',
       'L3_S29_F3376', 'L0_S0_F22', 'L3_S33_F3859', 'L3_S38_F3952', 
       'L3_S30_F3754', 'L2_S26_F3113', 'L3_S30_F3759', 'L0_S5_F114']    
    
#885,884,883,851

dfnum_test = pd.read_csv('test_numeric.csv',usecols=featureColl, header=0)

dfID = pd.read_csv('test_numeric.csv',usecols=[0], header=0)
dfnum_test = dfnum_test.fillna(2)
test_data = dfnum_test.values
id = dfID.values
id  = id.reshape((1183748,)) 

############################################################################
##############################################################################


output = forest.predict(test_data)


predictArr = array([id, output]).T
predictArr = predictArr.astype(int)

predictHead = array([['Id','Response']])
predictArr = r_[predictHead, predictArr]

o_df = pd.DataFrame(predictArr)
o_df.to_csv('Bosch_Out.csv', index=False, header=False)
