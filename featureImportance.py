# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 16:19:31 2016

@author: iki
"""


# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

##############################################################################
#Parameter
##############################################################################

#Number of the first n most important features
NumberOfFeatures = 150

#True to comptute the series of important features
compute = False

##############################################################################

if compute:

    def process(dataframe):
        
        dataframe = dataframe.fillna(2)
        array = dataframe.values
        X = array[:,1:969]
        Y = array[:,969]
        # feature extraction
        model = ExtraTreesClassifier()
        model.fit(X, Y)
        importances = model.feature_importances_
          
        
        std = np.std([tree.feature_importances_ for tree in model.estimators_],
                     axis=0)
    #    indices = np.argsort(importances)[::-1]
    #    
    #     Print the feature ranking
    #    print("Feature ranking:")
    #    
    #    for f in range(X.shape[1]):
    #        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    #    
    #    # Plot the feature importances of the forest
    #    plt.figure()
    #    plt.title("Feature importances")
    #    plt.bar(range(X.shape[1]), importances[indices],
    #           color="r", yerr=std[indices], align="center")
    #    plt.xticks(range(X.shape[1]), indices)
    #    plt.xlim([-1, X.shape[1]])
    #    plt.show()
        
        n = X.shape[1]
        return importances,n
    
    feature_importance = zeros((1,968))
    count = 0
    n = 0
    
    chunksize = 10 ** 5
    for chunk in read_csv('train_numeric.csv', chunksize=chunksize):
        imp,n = process(chunk)
        feature_importance = feature_importance + imp
        count = count + 1
       
    
    indices = np.argsort(feature_importance)
    indices = indices[0][::-1]
    
    #Print the feature ranking
    print("Feature ranking:")
    
    for f in range(n):
        print("%d. feature %d (%f)" % (f + 1, indices[f], feature_importance[0][indices[f]]))
    
    df = pd.read_csv('train_numeric.csv',nrows=1, header=None)
    featureNames = df.values
    featureNames = featureNames[:,1:969]

featureColl = []

for f in range(NumberOfFeatures):
    featureColl.append(featureNames[0][indices[f]])

resultFile = open("ImportantFeatures("+ str(NumberOfFeatures) + ").csv",'wb')
wr = csv.writer(resultFile)
wr.writerow(featureColl)


#np.savetxt("ImportantFeatures("+ str(NumberOfFeatures) + ").csv", featureColl, delimiter=",")    
    