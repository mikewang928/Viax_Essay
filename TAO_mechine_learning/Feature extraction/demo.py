import numpy as np
import scipy.io as sio 
from preprocessing_funcs import get_spikes_with_history
if __name__ == '__main__':
    
    for Idx_subject in list([10,11,12]): # 3 subjects index 10-12
        for Finger in list([0,1,2,3,4]): # 5 fingers for each subject. 0:thumb, 1:index, 2:middle ...

            #load training data (TrainX: feature vectors, TrainY: labels)
            matData = sio.loadmat('data/BCImoreData_Subj_'+str(Idx_subject)+'_200msLMP.mat')
            TrainX = matData['Data_Feature'].transpose()
            TrainY = matData['SmoothedFinger']
            TrainY = TrainY [:,Finger]
            TrainY = TrainY.reshape(TrainY.shape[0],1)
            #load testing data (TestX: feature vectors, TestY: labels)
            matData = sio.loadmat('data/BCImoreData_Subj_'+str(Idx_subject)+'_200msLMPTest.mat')
            TestX = matData['Data_Feature'].transpose()
            TestY = matData['SmoothedFinger']
            TestY = TestY[:,Finger]
            TestY = TestY.reshape(TestY.shape[0],1)
            
            
            
            # from here, we reconstruct the input by "looking back" a few steps
            bins_before= 20 # How many bins of neural data prior to the output are used for decoding
            bins_current=1 # Whether to use concurrent time bin of neural data
            bins_after=0 # How many bins of neural data after the output are used for decoding
            
            TrainX = get_spikes_with_history(TrainX,bins_before,bins_after,bins_current)
            TrainX, TrainY = TrainX[bins_before:,:,:], TrainY[bins_before:,]
            
            TestX=get_spikes_with_history(TestX,bins_before,bins_after,bins_current)
            TestX, TestY = TestX[bins_before:,:,:], TestY[bins_before:,]
            
            # Now, we reconstructed TrainX/TestX to have a shape (num_of_samples, sequence_length, input_size)
            # You can fit this to the LSTM
            
            # Preprocess the data may leed to better performance. e.g. StandardScaler 


		
      
              
       
    






























