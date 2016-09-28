# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 13:44:45 2016

@author: iki
"""
import csv as csv 
import numpy as np
import pandas as pd
import csv

#Select all trainingsexamples which go through L0_S00 to Lo_S12

df_bad = pd.DataFrame()

if df_bad.empty:
    
     #Test on most important feautures   
    
#    feature_names = ['L3_S38_F3960', 'L3_S33_F3865', 'L3_S38_F3956', 'L3_S33_F3857',
#       'L3_S29_F3321', 'L1_S24_F1846', 'L3_S32_F3850', 'L3_S29_F3354',
#       'L3_S29_F3324', 'L3_S35_F3889', 'L0_S1_F28', 'L1_S24_F1844',
#       'L3_S29_F3376', 'L0_S0_F22', 'L3_S33_F3859', 'L3_S38_F3952', 
#       'L3_S30_F3754', 'L2_S26_F3113', 'L3_S30_F3759', 'L0_S5_F114','Response']    
      
#    use_col = feature_names
#    skipData = []    
    
#    featureColl = featureColl.append('Response')
#    use_col = featureColl  
#    skipData = [] 

    feat = []
    
    with open('ImportantFeatures(300).csv', 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
             feat = row
        

    feat.append('Response')
    df_part1 = pd.read_csv('train_numeric.csv',usecols=feat,  header = 0)





    #Product1
    #df_col1 = pd.read_csv('train_numeric.csv',usecols = [1],header=0)
    
    #Product 2
    #df_col1 = pd.read_csv('train_numeric.csv',usecols = [82],header=0)
    
    #Product 3
    #df_col1 = pd.read_csv('train_numeric.csv',usecols = [169],header=0)
    #df_col2 = pd.read_csv('train_numeric.csv',usecols = [398],header=0)
    
    
    
    
    
    #Product 1
    #df_nan = df_col1[np.isnan(df_col1['L0_S0_F0'])] 
    #df_nan.index = df_nan.index +1
    #skipData = df_nan.index.tolist()

    #Product2
#    df_nan = df_col1[np.isnan(df_col1['L0_S12_F330'])] 
#    df_nan.index = df_nan.index +1
#    skipData = df_nan.index.tolist()

    #Product3
#    df_num = df_col2[np.isfinite(df_col2['L1_S25_F1855'])]
#    df_num.index = df_num.index +1
#    list1 = df_num.index.tolist()
#    df_nan = df_col1[np.isnan(df_col1['L1_S24_F679'])] 
#    df_nan.index = df_nan.index +1
#    list2 = df_nan.index.tolist()
#    skipData = list1 + list2
#    skipData = list(set(skipData))




#    #Part 1 Product
    #use_col1 = range(0,83)
    #use_col2 = range(683,970)
    #use_col = use_col1 + use_col2

    #Part 2 Product
    #use_col = range(82,970)
    
    #Part 3 Product
    #use_col = range(169,970)

    

    #df_part1 = pd.read_csv('train_numeric.csv',usecols=use_col, skiprows=skipData, header = 0)
    #df_part1 = pd.read_csv('train_numeric.csv',usecols=a,  header = 0)

    failNum = df_part1['Response'].sum()

    #Select all bad products
    df_bad= df_part1.loc[df_part1['Response'] == 1]
    bad_idx= df_bad.index.tolist()

    #Select all good products
    df_good = df_part1.drop(bad_idx)

    del df_part1


##############################################################################
#Generate datasets
##############################################################################


#Initialize parameters
#############################################################################
goodTrainSamples = 20000
badTrainSamples = 5000
sizeTestSamples = 10000 
trainDatasetName = 'Trainset1'
testDatasetName = 'Testset1'
#############################################################################


#Size of testset
badTestSamples = df_bad.shape[0]-badTrainSamples
goodTestSamples = sizeTestSamples - badTestSamples 

#Shuffle bad data
df_dataBad = df_bad.iloc[np.random.permutation(len(df_bad))]

#Slice bad data to test and train
df_trainBad = df_dataBad[0:badTrainSamples]
df_testBad = df_dataBad[badTrainSamples:df_bad.shape[0]]



#Choose random samples from the datasets
df_trainGood = df_good.sample(n=goodTrainSamples)
df_testGood = df_good.sample(n=goodTestSamples)

#Put good and bad samples together
df_trainSet = pd.concat([df_trainGood,df_trainBad])
df_testSet = pd.concat([df_testGood,df_testBad])


#Suffle the datasets
df_trainSet = df_trainSet.iloc[np.random.permutation(len(df_trainSet))]
df_testSet = df_testSet.iloc[np.random.permutation(len(df_testSet))]



#output as csv
df_trainSet.to_csv(trainDatasetName + '_' + str(goodTrainSamples) + '_' + str(badTrainSamples) + '.csv', index=False, header=True)
df_testSet.to_csv(testDatasetName + '_' + str(goodTestSamples) + '_' + str(badTestSamples) + '.csv', index=False, header=True)











