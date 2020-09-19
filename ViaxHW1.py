#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:52:14 2020

@author: mikewang
"""
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.signal import butter, lfilter


#path='Desktop/Study/Undergraduate/McGill/Semester_7/Viax_Essay/Viax_1/'
'''
The .mat file contains a (1372633, 47) martix of data (1372633 rows 47 colomns)
This data contains the 2 mins peior and 2 mins after the seizure period
'''

# loading the data set
rawDictData = scipy.io.loadmat('ViaxHW1_data.mat')
#print(rawMatData)


'''
#####Some attempts on turning the mat files to svm
rawMatData = {k:v for k, v in rawMatData.items() if k[0] != '_'}
rawData = pd.DataFrame({k: pd.Series(v[0]) for k, v in rawMatData.items()})
rawData.to_csv("ViaxHW1_data.csv")

rawCsvData = pd.read_csv("ViaxHW1_data.csv")
'''
'''
##### PART I: finding the top 5 most informative channels 
'''

'''
##### Constant definations
'''
#print(rawMatData)
sampleFrequency = 5000
numberOfChannel = 47
peiorSeizureDataIndex = sampleFrequency*60
postSeizureDataIndex = sampleFrequency*60
WINDOW_SIZE = sampleFrequency *1         # which is one sec 

'''
##### Data processing
'''
#.mat files are in dict 
rawData = rawDictData["data"]
#print(rawData.shape)

#deleting the prior and post dataset form the raw data
seizureDataWithPost = np.delete(rawData,slice(0,peiorSeizureDataIndex,1),0) 
seizureDataWithPostShape = seizureDataWithPost.shape
#print(seizureDataWithPostShape)
seizureData = np.delete(seizureDataWithPost,slice((seizureDataWithPostShape[0]-postSeizureDataIndex),
                                          (seizureDataWithPostShape[0]),1),0)
#print(seizureData.shape)
#print(seizureData)

seizureData = seizureData.reshape(seizureData.shape[1],seizureData.shape[0])
#print(seizureData.shape)



'''
##### ploting all 47 channels data
'''
'''
for i in range (0,47):
    x = np.linspace (0, seizureData.shape[1], seizureData.shape[1])
    plt.figure(i+1)
    plt.figure(figsize = (20,2))
    plt.plot(x,seizureData[i,:],label = ('Seizure plot for Channel '+ str(i+1)))
    plt.xlabel('samples')
    plt.xlabel('voltage')
    plt.title('Channel '+ str(i+1))
    plt.axis([0,772633,-1000,1000])
    plt.legend()
    #plt.savefig(('Channel_' + str(i+1) + '_plot.png'))
    plt.show()
'''
    
'''
##### extracting the 5 most informative channel
From the plt above we can see that Channel 21,22,23,24,25 are most informative
we will extract the data information from these channels 
'''
channel21Data = seizureData[20,:]
channel22Data = seizureData[21,:] 
channel23Data = seizureData[22,:]
channel24Data = seizureData[23,:] 
channel25Data = seizureData[24,:]

'''
##### some testing on slicing in python
testArray = np.arange(0,30)
print("the shape of the testArray is: " + str(testArray.shape))
testArray = testArray.reshape(1,30)
print(testArray)
deletedArray = np.delete(testArray,slice(25,30,1),1)
print(deletedArray)
'''

'''
###### some fail attemps on extracting information from arrays
for colomns in range(0,peiorSeizureData):
    SeizureData = np.delete(rawData,colomns,1)
        
print(SeizureData)
'''

'''
Part II: Line-length, Energy, Spectral Power (Beta: 12-30 Hz, HFO:100-600 Hz)
'''
'''
##### Line-length
'''
##### channel 21
channel21LineLength = []

for k in range(0,772633-WINDOW_SIZE,5000):
    channel21DiffSumN = 0
    for n in range(0,WINDOW_SIZE):
        channel21Diff = abs(channel21Data[k+n+1]-channel21Data[k+n])
        channel21DiffSumN = channel21Diff + channel21DiffSumN
    channel21LineLength.append(channel21DiffSumN)
channel21LineLength = np.array(channel21LineLength)
print(channel21LineLength)
xChannel21 = np.linspace(0,len(channel21LineLength),len(channel21LineLength))
plt.figure(48)
plt.figure(figsize = (20,2))
plt.plot(xChannel21,channel21LineLength,label = ('Channel 21 Line Length plot'))
plt.xlabel('seconds')
plt.ylabel('Line Length')
plt.title('Channel 21 Line Length plot')
#plt.axis([0,772633,-1000,1000])
plt.legend()
plt.savefig('Channel_21_LineLength.png')
plt.show()


##### channel 22
channel22LineLength = []

for k in range(0,772633-WINDOW_SIZE,5000):
    channel22DiffSumN = 0
    for n in range(1,WINDOW_SIZE):
        channel22Diff = abs(channel22Data[k+n]-channel22Data[k+n-1])
        channel22DiffSumN = channel22Diff + channel22DiffSumN
    channel22LineLength.append(channel22DiffSumN)
channel22LineLength = np.array(channel22LineLength)
print(channel22LineLength)
xChannel22 = np.linspace(0,len(channel22LineLength),len(channel22LineLength))
plt.figure(49)
plt.figure(figsize = (20,2))
plt.plot(xChannel22,channel22LineLength,label = ('Channel 22 Line Length plot'))
plt.xlabel('seconds')
plt.ylabel('Line Length')
plt.title('Channel 22 Line Length plot')
#plt.axis([0,772633,-1000,1000])
plt.legend()
plt.savefig('Channel_22_LineLength.png')
plt.show()


##### channel 23
channel23LineLength = []

for k in range(0,772633-WINDOW_SIZE,5000):
    channel23DiffSumN = 0
    for n in range(1,WINDOW_SIZE):
        channel23Diff = abs(channel23Data[k+n]-channel23Data[k+n-1])
        channel23DiffSumN = channel23Diff + channel23DiffSumN
    channel23LineLength.append(channel23DiffSumN)
channel23LineLength = np.array(channel23LineLength)
print(channel23LineLength)
xChannel23 = np.linspace(0,len(channel23LineLength),len(channel23LineLength))
plt.figure(50)
plt.figure(figsize = (20,2))
plt.plot(xChannel23,channel23LineLength,label = ('Channel 23 Line Length plot'))
plt.xlabel('seconds')
plt.ylabel('Line Length')
plt.title('Channel 23 Line Length plot')
#plt.axis([0,772633,-1000,1000])
plt.legend()
plt.savefig('Channel_23_LineLength.png')
plt.show()


##### channel 24
channel24LineLength = []

for k in range(1,772633-WINDOW_SIZE,5000):
    channel24DiffSumN = 0
    for n in range(0,WINDOW_SIZE):
        channel24Diff = abs(channel24Data[k+n]-channel24Data[k+n-1])
        channel24DiffSumN = channel24Diff + channel24DiffSumN
    channel24LineLength.append(channel24DiffSumN)
channel24LineLength = np.array(channel24LineLength)
print(channel24LineLength)
xChannel24 = np.linspace(0,len(channel24LineLength),len(channel24LineLength))
plt.figure(51)
plt.figure(figsize = (20,2))
plt.plot(xChannel24,channel24LineLength,label = ('Channel 24 Line Length plot'))
plt.xlabel('seconds')
plt.ylabel('Line Length')
plt.title('Channel 24 Line Length plot')
#plt.axis([0,772633,-1000,1000])
plt.legend()
plt.savefig('Channel_24_LineLength.png')
plt.show()


##### channel 25
channel25LineLength = []

for k in range(0,772633-WINDOW_SIZE,5000):
    channel25DiffSumN = 0
    for n in range(1,WINDOW_SIZE):
        channel25Diff = abs(channel25Data[k+n]-channel25Data[k+n-1])
        channel25DiffSumN = channel25Diff + channel25DiffSumN
    channel25LineLength.append(channel25DiffSumN)
channel25LineLength = np.array(channel25LineLength)
print(channel25LineLength)
xChannel25 = np.linspace(0,len(channel25LineLength),len(channel25LineLength))
plt.figure(52)
plt.figure(figsize = (20,2))
plt.plot(xChannel25,channel25LineLength,label = ('Channel 25 Line Length plot'))
plt.xlabel('seconds')
plt.ylabel('Line Length')
plt.title('Channel 25 Line Length plot')
#plt.axis([0,772633,-1000,1000])
plt.legend()
plt.savefig('Channel_25_LineLength.png')
plt.show()



'''
##### Energy 
'''
##### channel 21
channel21Energy = []
for k in range(0,772633-WINDOW_SIZE,5000):
    channel21EnergySquareSumN = 0
    for n in range(1,WINDOW_SIZE):
        channel21EnergySquare = (channel21Data[k+n-1])**2
        channel21EnergySquareSumN = channel21EnergySquare + channel21EnergySquareSumN
    channel21Energy.append(channel21EnergySquareSumN)
channel21Energy = np.array(channel21Energy)
print(channel21Energy)
xChannelEnergy21 = np.linspace(0,len(channel21Energy),len(channel21Energy))
plt.figure(53)
plt.figure(figsize = (20,2))
plt.plot(xChannelEnergy21,channel21Energy,label = ('Channel 21 Energy plot'))
plt.xlabel('seconds')
plt.ylabel('Energy')
plt.title('Channel 21 Energy plot')
#plt.axis([0,772633,-1000,1000])
plt.legend()
plt.savefig('Channel_21_Energy.png')
plt.show()



##### channel 22
channel22Energy = []
for k in range(0,772633-WINDOW_SIZE,5000):
    channel22EnergySquareSumN = 0
    for n in range(1,WINDOW_SIZE):
        channel22EnergySquare = (channel22Data[k+n-1])**2
        channel22EnergySquareSumN = channel22EnergySquare + channel22EnergySquareSumN
    channel22Energy.append(channel22EnergySquareSumN)
channel22Energy = np.array(channel22Energy)
print(channel22Energy)
xChannelEnergy22 = np.linspace(0,len(channel22Energy),len(channel22Energy))
plt.figure(54)
plt.figure(figsize = (20,2))
plt.plot(xChannelEnergy22,channel22Energy,label = ('Channel 22 Energy plot'))
plt.xlabel('seconds')
plt.ylabel('Energy')
plt.title('Channel 22 Energy plot')
#plt.axis([0,772633,-1000,1000])
plt.legend()
plt.savefig('Channel_22_Energy.png')
plt.show()

##### channel 23
channel23Energy = []
for k in range(0,772633-WINDOW_SIZE,5000):
    channel23EnergySquareSumN = 0
    for n in range(1,WINDOW_SIZE):
        channel23EnergySquare = (channel23Data[k+n-1])**2
        channel23EnergySquareSumN = channel23EnergySquare + channel23EnergySquareSumN
    channel23Energy.append(channel23EnergySquareSumN)
channel23Energy = np.array(channel23Energy)
print(channel23Energy)
xChannelEnergy23 = np.linspace(0,len(channel23Energy),len(channel23Energy))
plt.figure(55)
plt.figure(figsize = (20,2))
plt.plot(xChannelEnergy23,channel23Energy,label = ('Channel 23 Energy plot'))
plt.xlabel('seconds')
plt.ylabel('Energy')
plt.title('Channel 23 Energy plot')
#plt.axis([0,772633,-1000,1000])
plt.legend()
plt.savefig('Channel_23_Energy.png')
plt.show()

##### channel 24
channel24Energy = []
for k in range(0,772633-WINDOW_SIZE,5000):
    channel24EnergySquareSumN = 0
    for n in range(1,WINDOW_SIZE):
        channel24EnergySquare = (channel24Data[k+n-1])**2
        channel24EnergySquareSumN = channel24EnergySquare + channel24EnergySquareSumN
    channel24Energy.append(channel24EnergySquareSumN)
channel24Energy = np.array(channel24Energy)
print(channel24Energy)
xChannelEnergy24 = np.linspace(0,len(channel24Energy),len(channel24Energy))
plt.figure(56)
plt.figure(figsize = (20,2))
plt.plot(xChannelEnergy24,channel24Energy,label = ('Channel 24 Energy plot'))
plt.xlabel('seconds')
plt.ylabel('Energy')
plt.title('Channel 24 Energy plot')
#plt.axis([0,772633,-1000,1000])
plt.legend()
plt.savefig('Channel_24_Energy.png')
plt.show()

##### channel 25
channel25Energy = []
for k in range(0,772633-WINDOW_SIZE,5000):
    channel25EnergySquareSumN  = 0
    for n in range(1,WINDOW_SIZE):
        channel25EnergySquare = (channel25Data[k+n-1])**2
        channel25EnergySquareSumN = channel25EnergySquare + channel25EnergySquareSumN
    channel25Energy.append(channel25EnergySquareSumN)
channel25Energy = np.array(channel25Energy)
print(channel25Energy)
xChannelEnergy25 = np.linspace(0,len(channel25Energy),len(channel25Energy))
plt.figure(57)
plt.figure(figsize = (20,2))
plt.plot(xChannelEnergy25,channel25Energy,label = ('Channel 25 Energy plot'))
plt.xlabel('seconds')
plt.ylabel('Energy')
plt.title('Channel 25 Energy plot')
#plt.axis([0,772633,-1000,1000])
plt.legend()
plt.savefig('Channel_25_Energy.png')
plt.show()

'''
##### spectral power 
'''
'''
input:
    lowcut frequancy 
    highcut frequancy 
    frequancy sampled

output: 
    b: The numerator coefficient vector in a 1-D sequence.
    a: The denominator coefficient vector in a 1-D sequence. 
    If a[0] is not 1, then both a and b are normalized by a[0].
'''
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs 
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

'''
input:
    lowcut frequancy 
    highcut frequancy 
    frequancy sampled

output: 
    IRR FIR filtered signal
'''
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

##### Beta (12-30 Hz)
'''    
FilteredBetaChannel21Data = butter_bandpass_filter(channel21Data,12,30,5000,order=5)
x_FilteredBetaChannel21= np.linspace(0,len(FilteredBetaChannel21Data),len(FilteredBetaChannel21Data))
plt.figure(58)
plt.figure(figsize = (20,2))
plt.plot(x_FilteredBetaChannel21,FilteredBetaChannel21Data, label =('channel 21 beta filtered data'))
plt.ylabel('voltage')
plt.xlabel('samples')
plt.legend()
plt.show()
'''

##### Channel 21
channel21PowerSpectralBeta = []
FilteredBetaChannel21Data = butter_bandpass_filter(channel21Data,12,30,5000,order=5)
#print(FilteredBetaChannel21Data)
for k in range(0,len(FilteredBetaChannel21Data)-WINDOW_SIZE,5000):
    channel21PowerSpectralSumN = 0
    for n in range(1,WINDOW_SIZE):
        channel21PowerSpectralN = (FilteredBetaChannel21Data[k+n-1])**2
        channel21PowerSpectralSumN = channel21PowerSpectralN + channel21PowerSpectralSumN
    channel21PowerSpectral=channel21PowerSpectralSumN / WINDOW_SIZE
    channel21PowerSpectralBeta.append(channel21PowerSpectral)
channel21PowerSpectralBeta = np.array(channel21PowerSpectralBeta)
print(channel21PowerSpectralBeta)
xChannel21PowerSpectralBeta = np.linspace(0,len(channel21PowerSpectralBeta),len(channel21PowerSpectralBeta))
plt.figure(58)
plt.figure(figsize = (20,2))
plt.plot(xChannel21PowerSpectralBeta,channel21PowerSpectralBeta,label = ('Channel 21 Beta Power Spectural plot'))
plt.xlabel('seconds')
plt.ylabel('Beta Power')
plt.title('Channel 21 Beta Power Spectural plot')
#plt.axis([0,772633,-1000,1000])
plt.legend()
plt.savefig('Channel_21_Beta_Power.png')
plt.show()


##### Channel 22
channel22PowerSpectralBeta = []
FilteredBetaChannel22Data = butter_bandpass_filter(channel22Data,12,30,5000,order=5)
#print(FilteredBetaChannel21Data)
for k in range(0,len(FilteredBetaChannel22Data)-WINDOW_SIZE,5000):
    channel22PowerSpectralSumN = 0
    for n in range(1,WINDOW_SIZE):
        channel22PowerSpectralN = (FilteredBetaChannel22Data[k+n-1])**2
        channel22PowerSpectralSumN = channel22PowerSpectralN + channel22PowerSpectralSumN
    channel22PowerSpectral=channel22PowerSpectralSumN / WINDOW_SIZE
    channel22PowerSpectralBeta.append(channel22PowerSpectral)
channel22PowerSpectralBeta = np.array(channel22PowerSpectralBeta)
print(channel22PowerSpectralBeta)
xChannel22PowerSpectralBeta = np.linspace(0,len(channel22PowerSpectralBeta),len(channel22PowerSpectralBeta))
plt.figure(59)
plt.figure(figsize = (20,2))
plt.plot(xChannel22PowerSpectralBeta,channel22PowerSpectralBeta,label = ('Channel 22 Beta Power Spectural plot'))
plt.xlabel('seconds')
plt.ylabel('Beta Power')
plt.title('Channel 22 Beta Power Spectural plot')
#plt.axis([0,772633,-1000,1000])
plt.legend()
plt.savefig('Channel_22_Beta_Power.png')
plt.show()


##### Channel 23
channel23PowerSpectralBeta = []
FilteredBetaChannel23Data = butter_bandpass_filter(channel23Data,12,30,5000,order=5)
#print(FilteredBetaChannel21Data)
for k in range(0,len(FilteredBetaChannel23Data)-WINDOW_SIZE,5000):
    channel23PowerSpectralSumN = 0
    for n in range(1,WINDOW_SIZE):
        channel23PowerSpectralN = (FilteredBetaChannel23Data[k+n-1])**2
        channel23PowerSpectralSumN = channel23PowerSpectralN + channel23PowerSpectralSumN
    channel23PowerSpectral=channel23PowerSpectralSumN / WINDOW_SIZE
    channel23PowerSpectralBeta.append(channel23PowerSpectral)
channel23PowerSpectralBeta = np.array(channel23PowerSpectralBeta)
print(channel23PowerSpectralBeta)
xChannel23PowerSpectralBeta = np.linspace(0,len(channel23PowerSpectralBeta),len(channel23PowerSpectralBeta))
plt.figure(60)
plt.figure(figsize = (20,2))
plt.plot(xChannel23PowerSpectralBeta,channel23PowerSpectralBeta,label = ('Channel 23 Beta Power Spectural plot'))
plt.xlabel('seconds')
plt.ylabel('Beta Power')
plt.title('Channel 23 Beta Power Spectural plot')
#plt.axis([0,772633,-1000,1000])
plt.legend()
plt.savefig('Channel_23_Beta_Power.png')
plt.show()


##### Channel 24
channel24PowerSpectralBeta = []
FilteredBetaChannel24Data = butter_bandpass_filter(channel24Data,12,30,5000,order=5)
#print(FilteredBetaChannel21Data)
for k in range(0,len(FilteredBetaChannel24Data)-WINDOW_SIZE,5000):
    channel24PowerSpectralSumN = 0
    for n in range(1,WINDOW_SIZE):
        channel24PowerSpectralN = (FilteredBetaChannel24Data[k+n-1])**2
        channel24PowerSpectralSumN = channel24PowerSpectralN + channel24PowerSpectralSumN
    channel24PowerSpectral=channel24PowerSpectralSumN / WINDOW_SIZE
    channel24PowerSpectralBeta.append(channel24PowerSpectral)
channel24PowerSpectralBeta = np.array(channel24PowerSpectralBeta)
print(channel24PowerSpectralBeta)
xChannel24PowerSpectralBeta = np.linspace(0,len(channel24PowerSpectralBeta),len(channel24PowerSpectralBeta))
plt.figure(61)
plt.figure(figsize = (20,2))
plt.plot(xChannel24PowerSpectralBeta,channel24PowerSpectralBeta,label = ('Channel 24 Beta Power Spectural plot'))
plt.xlabel('seconds')
plt.ylabel('Beta Power')
plt.title('Channel 24 Beta Power Spectural plot')
#plt.axis([0,772633,-1000,1000])
plt.legend()
plt.savefig('Channel_24_Beta_Power.png')
plt.show()


##### Channel 25
channel25PowerSpectralBeta = []
FilteredBetaChannel25Data = butter_bandpass_filter(channel25Data,12,30,5000,order=5)
#print(FilteredBetaChannel21Data)
for k in range(0,len(FilteredBetaChannel25Data)-WINDOW_SIZE,5000):
    channel25PowerSpectralSumN = 0
    for n in range(1,WINDOW_SIZE):
        channel25PowerSpectralN = (FilteredBetaChannel25Data[k+n-1])**2
        channel25PowerSpectralSumN = channel25PowerSpectralN + channel25PowerSpectralSumN
    channel25PowerSpectral=channel25PowerSpectralSumN / WINDOW_SIZE
    channel25PowerSpectralBeta.append(channel25PowerSpectral)
channel25PowerSpectralBeta = np.array(channel25PowerSpectralBeta)
print(channel25PowerSpectralBeta)
xChannel25PowerSpectralBeta = np.linspace(0,len(channel25PowerSpectralBeta),len(channel25PowerSpectralBeta))
plt.figure(62)
plt.figure(figsize = (20,2))
plt.plot(xChannel25PowerSpectralBeta,channel25PowerSpectralBeta,label = ('Channel 25 Beta Power Spectural plot'))
plt.xlabel('seconds')
plt.ylabel('Beta Power')
plt.title('Channel 25 Beta Power Spectural plot')
#plt.axis([0,772633,-1000,1000])
plt.legend()
plt.savefig('Channel_25_Beta_Power.png')
plt.show()


'''
##### HFO (100-600Hz)
'''
##### channel 21
channel21PowerSpectralHFO = []
FilteredHFOChannel21Data = butter_bandpass_filter(channel21Data,100,600,5000,order=5)
#print(FilteredHFOChannel21Data)
for k in range(0,len(FilteredHFOChannel21Data)-WINDOW_SIZE,5000):
    channel21HFOPowerSpectralSumN = 0
    for n in range(1,WINDOW_SIZE):
        channel21HFOPowerSpectralN = (FilteredHFOChannel21Data[k+n-1])**2
        channel21HFOPowerSpectralSumN = channel21HFOPowerSpectralN + channel21HFOPowerSpectralSumN
    channel21HFOPowerSpectral=channel21HFOPowerSpectralSumN / WINDOW_SIZE
    channel21PowerSpectralHFO.append(channel21HFOPowerSpectral)
channel21PowerSpectralHFO = np.array(channel21PowerSpectralHFO)
print(channel21PowerSpectralHFO)
xChannel21PowerSpectralHFO = np.linspace(0,len(channel21PowerSpectralHFO),len(channel21PowerSpectralHFO))
plt.figure(63)
plt.figure(figsize = (20,2))
plt.plot(xChannel21PowerSpectralHFO,channel21PowerSpectralHFO,label = ('Channel 21 HFO Power Spectural plot'))
plt.xlabel('seconds')
plt.ylabel('HFO Power')
plt.title('Channel 21 HFO Power Spectural plot')
#plt.axis([0,772633,-1000,1000])
plt.legend()
plt.savefig('Channel_21_HFO_Power.png')
plt.show()


##### channel 22
channel22PowerSpectralHFO = []
FilteredHFOChannel22Data = butter_bandpass_filter(channel22Data,100,600,5000,order=5)
#print(FilteredHFOChannel21Data)
for k in range(0,len(FilteredHFOChannel22Data)-WINDOW_SIZE,5000):
    channel22HFOPowerSpectralSumN = 0
    for n in range(1,WINDOW_SIZE):
        channel22HFOPowerSpectralN = (FilteredHFOChannel22Data[k+n-1])**2
        channel22HFOPowerSpectralSumN = channel22HFOPowerSpectralN + channel22HFOPowerSpectralSumN
    channel22HFOPowerSpectral=channel22HFOPowerSpectralSumN / WINDOW_SIZE
    channel22PowerSpectralHFO.append(channel22HFOPowerSpectral)
channel22PowerSpectralHFO = np.array(channel22PowerSpectralHFO)
print(channel22PowerSpectralHFO)
xChannel22PowerSpectralHFO = np.linspace(0,len(channel22PowerSpectralHFO),len(channel22PowerSpectralHFO))
plt.figure(64)
plt.figure(figsize = (20,2))
plt.plot(xChannel22PowerSpectralHFO,channel22PowerSpectralHFO,label = ('Channel 22 HFO Power Spectural plot'))
plt.xlabel('seconds')
plt.ylabel('HFO Power')
plt.title('Channel 22 HFO Power Spectural plot')
#plt.axis([0,772633,-1000,1000])
plt.legend()
plt.savefig('Channel_22_HFO_Power.png')
plt.show()


##### channel 23
channel23PowerSpectralHFO = []
FilteredHFOChannel23Data = butter_bandpass_filter(channel23Data,100,600,5000,order=5)
#print(FilteredHFOChannel21Data)
for k in range(0,len(FilteredHFOChannel23Data)-WINDOW_SIZE,5000):
    channel23HFOPowerSpectralSumN = 0
    for n in range(1,WINDOW_SIZE):
        channel23HFOPowerSpectralN = (FilteredHFOChannel23Data[k+n-1])**2
        channel23HFOPowerSpectralSumN = channel23HFOPowerSpectralN + channel23HFOPowerSpectralSumN
    channel23HFOPowerSpectral=channel23HFOPowerSpectralSumN / WINDOW_SIZE
    channel23PowerSpectralHFO.append(channel23HFOPowerSpectral)
channel23PowerSpectralHFO = np.array(channel23PowerSpectralHFO)
print(channel21PowerSpectralHFO)
xChannel23PowerSpectralHFO = np.linspace(0,len(channel23PowerSpectralHFO),len(channel23PowerSpectralHFO))
plt.figure(65)
plt.figure(figsize = (20,2))
plt.plot(xChannel23PowerSpectralHFO,channel23PowerSpectralHFO,label = ('Channel 23 HFO Power Spectural plot'))
plt.xlabel('seconds')
plt.ylabel('HFO Power')
plt.title('Channel 23 HFO Power Spectural plot')
#plt.axis([0,772633,-1000,1000])
plt.legend()
plt.savefig('Channel_23_HFO_Power.png')
plt.show()


##### channel 24
channel24PowerSpectralHFO = []
FilteredHFOChannel24Data = butter_bandpass_filter(channel24Data,100,600,5000,order=5)
#print(FilteredHFOChannel21Data)
for k in range(0,len(FilteredHFOChannel24Data)-WINDOW_SIZE,5000):
    channel24HFOPowerSpectralSumN = 0
    for n in range(1,WINDOW_SIZE):
        channel24HFOPowerSpectralN = (FilteredHFOChannel24Data[k+n-1])**2
        channel24HFOPowerSpectralSumN = channel24HFOPowerSpectralN + channel24HFOPowerSpectralSumN
    channel24HFOPowerSpectral=channel24HFOPowerSpectralSumN / WINDOW_SIZE
    channel24PowerSpectralHFO.append(channel24HFOPowerSpectral)
channel24PowerSpectralHFO = np.array(channel24PowerSpectralHFO)
print(channel24PowerSpectralHFO)
xChannel24PowerSpectralHFO = np.linspace(0,len(channel24PowerSpectralHFO),len(channel24PowerSpectralHFO))
plt.figure(66)
plt.figure(figsize = (20,2))
plt.plot(xChannel24PowerSpectralHFO,channel24PowerSpectralHFO,label = ('Channel 24 HFO Power Spectural plot'))
plt.xlabel('seconds')
plt.ylabel('HFO Power')
plt.title('Channel 24 HFO Power Spectural plot')
#plt.axis([0,772633,-1000,1000])
plt.legend()
plt.savefig('Channel_24_HFO_Power.png')
plt.show()


##### channel 25
channel25PowerSpectralHFO = []
FilteredHFOChannel25Data = butter_bandpass_filter(channel25Data,100,600,5000,order=5)
#print(FilteredHFOChannel21Data)
for k in range(0,len(FilteredHFOChannel25Data)-WINDOW_SIZE,5000):
    channel25HFOPowerSpectralSumN = 0
    for n in range(1,WINDOW_SIZE):
        channel25HFOPowerSpectralN = (FilteredHFOChannel25Data[k+n-1])**2
        channel25HFOPowerSpectralSumN = channel25HFOPowerSpectralN + channel25HFOPowerSpectralSumN
    channel25HFOPowerSpectral=channel25HFOPowerSpectralSumN / WINDOW_SIZE
    channel25PowerSpectralHFO.append(channel25HFOPowerSpectral)
channel25PowerSpectralHFO = np.array(channel25PowerSpectralHFO)
print(channel25PowerSpectralHFO)
xChannel25PowerSpectralHFO = np.linspace(0,len(channel25PowerSpectralHFO),len(channel25PowerSpectralHFO))
plt.figure(67)
plt.figure(figsize = (20,2))
plt.plot(xChannel25PowerSpectralHFO,channel25PowerSpectralHFO,label = ('Channel 25 HFO Power Spectural plot'))
plt.xlabel('seconds')
plt.ylabel('HFO Power')
plt.title('Channel 25 HFO Power Spectural plot')
#plt.axis([0,772633,-1000,1000])
plt.legend()
plt.savefig('Channel_25_HFO_Power.png')
plt.show()

