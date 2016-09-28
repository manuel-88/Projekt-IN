# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:28:47 2016

@author: iki
"""

import csv as csv 
import numpy as np
import pandas as pd


#Read a chunk of the trainingdata
df = pd.read_csv('train_numeric.csv',nrows=10000, header=0)

#Convert pandas dataframe to a numpy object for better working with matrices 
data = df.values

#Check for all failure parts in the chunk (Response = 1)
failure_idx = np.where(data[0:,969])
failure_idx = failure_idx[0]

#Number of failed parts
fail = failure_idx.size


#Initialize Matrix
failMat = np.empty((0,970), float)


#Collect all failed parts
for i in xrange(0,fail):
    
    #take a complete row of a failed part
    fail_row = np.array([data[failure_idx[i],0:]])
    
    #Append alle rows with response 1
    failMat = np.append(failMat, fail_row, axis=0)

#
failMatBool = np.isnan(failMat[0:,0:])

    
dataBool = np.isnan(data[0:,0:]) 





productFamilies = []



for i in xrange(0,failMatBool.shape[0]):
    
    eqlProducts=[]
    
    comp_row = array([failMatBool[i,0:]])
    
    eqlProducts.append(failure_idx[i])    

    for k in  xrange(0,dataBool.shape[0]):
    
        eqlPart = np.array_equal(comp_row, array([dataBool[k,0:]]))

        if k == failure_idx[i]:
             continue       
        
        if eqlPart==True: 
            eqlProducts.append(k)
    
    productFamilies.append(eqlProducts)


#for i in xrange(0,failMatBool.shape[0])   :
#    
#    comp_row = failMatBool[i,0:]
#    
#    a = np.array_equal(comp_row, array([dataBool[539,0:]]))
    