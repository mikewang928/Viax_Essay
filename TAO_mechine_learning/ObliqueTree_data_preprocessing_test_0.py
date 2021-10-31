# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 21:06:21 2021

@author: wsycx
"""

#%% imports 
import scipy.io as sio
#import NeuralFeatures as NeuralFeature
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

#%% load the data
TIME_WINDOW = 10
DATA_FS = 1000
DG_FS = 25
WINDOW_SIZE = 1000 # as the data is sampled in 1000Hz
TIME_WINDOW = 10
train_data_1 = sio.loadmat("train_data_1")["train_data"]
train_dg_1 = sio.loadmat("train_dg_1")["train_dg"]
test_data_1 = sio.loadmat("test_data_1")["test_data"]
test_dg_1 = sio.loadmat("test_dg_1")["test_dg"]


#plt.plot(test_dg_1[:10000,:])



#%% plot the data
plt.figure()
plt.plot(test_data_1[:10000,55])
plt.title("test_data_1")
plt.show()
plt.figure()
plt.title("test_dg_1")
plt.plot(test_dg_1[:10000,:])
plt.show()
#%%
# reshape data (windows,channel,sample)
test_data_reshaped_1 = []
for i in range(0,len(test_data_1)-WINDOW_SIZE,WINDOW_SIZE):
    test_data_reshaped_1.append(np.transpose(train_data_1[i:i+WINDOW_SIZE]))
test_data_reshaped_1 = np.array(test_data_reshaped_1)
print(test_data_reshaped_1.shape)


# band powerv 0.15-200Hz
from NeuralFeatures import NeuralFeature
NF = NeuralFeature()
bandpower=NF.bandpower(test_data_reshaped_1,[13],[50])
print(bandpower.shape)
plt.figure()
plt.plot(bandpower[:10,55])
plt.show()
