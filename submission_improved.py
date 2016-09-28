# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:30:11 2016

@author: iki
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 09:29:36 2016

@author: iki
"""


import csv as csv 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
#import xgboost as xgb

    
feature_names1 = ['L3_S38_F3960', 'L3_S33_F3865', 'L3_S38_F3956', 'L3_S33_F3857',
       'L3_S29_F3321', 'L1_S24_F1846', 'L3_S32_F3850', 'L3_S29_F3354',
       'L3_S29_F3324', 'L3_S35_F3889', 'L0_S1_F28', 'L1_S24_F1844',
       'L3_S29_F3376', 'L0_S0_F22', 'L3_S33_F3859', 'L3_S38_F3952', 
       'L3_S30_F3754', 'L2_S26_F3113', 'L3_S30_F3759', 'L0_S5_F114','Response']   
       
feature_names2 = ['L3_S38_F3960', 'L3_S33_F3865', 'L3_S38_F3956', 'L3_S33_F3857',
       'L3_S29_F3321', 'L1_S24_F1846', 'L3_S32_F3850', 'L3_S29_F3354',
       'L3_S29_F3324', 'L3_S35_F3889', 'L0_S1_F28', 'L1_S24_F1844',
       'L3_S29_F3376', 'L0_S0_F22', 'L3_S33_F3859', 'L3_S38_F3952', 
       'L3_S30_F3754', 'L2_S26_F3113', 'L3_S30_F3759', 'L0_S5_F114']         

feat = []
    
with open('ImportantFeatures(150).csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
         feat = row
        
feat_train = list(feat)
feat_train.append('Response')
#df_part1 = pd.read_csv('train_numeric.csv',usecols=feat,  header = 0)




dfnum = pd.read_csv('train_numeric.csv',usecols=feat_train, header=0)

#dfsample = pd.read_csv('sample_submission.csv', header=0)



dfnum = dfnum.fillna(2)


train_data = dfnum.values
del dfnum


#dfcat.to_csv('Bosch_test.csv', index=False, header=False)
#dfnum.to_csv('Bosch_test_num.csv', index=False, header=False)





# Create the random forest object which will include all the parameters
## for the fit
#forest = xgb.XGBClassifier(max_depth=10, n_estimators=100, learning_rate=0.05).fit(train_data[0::,0:-1],train_data[0::,-1])
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::,0:-1],train_data[0::,-1])

train_data = []



#Predict
dfnum_test = pd.read_csv('test_numeric.csv',usecols=feat, header=0)

dfnum_test = dfnum_test.fillna(2)

test_data = dfnum_test.values

del dfnum_test

# Take the same decision trees and run it on the test data
output = forest.predict(test_data)

test_data = []











#Submission Output
dfID = pd.read_csv('test_numeric.csv',usecols=[0], header=0)
id = dfID.values
id  = id.reshape((1183748,)) 

predictArr = array([id, output]).T
predictArr = predictArr.astype(int)

predictHead = array([['Id','Response']])
predictArr = r_[predictHead, predictArr]

o_df = pd.DataFrame(predictArr)
o_df.to_csv('Bosch_Out_3.csv', index=False, header=False)