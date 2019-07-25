#**
# @file     cmnedata.py
# @author   Christoph Dinh <chdinh@nmr.mgh.harvard.edu>
# @version  1.0
# @date     September, 2017
#
# @section  LICENSE
#
# Copyright (C) 2017, Christoph Dinh. All rights reserved.
#
# @brief    CMNEData contains, e.g., data loader
#**

#%%
import os
import numpy as np
import random
import mne
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator

from cmnesettings import CMNESettings


###################################################################################################
# Standardize
###################################################################################################

def standardize(mat, mean=None, std=None):
	"""
	0 center and scale data
	Standardize an np.array to the array mean and standard deviation or specified parameters
	See https://en.wikipedia.org/wiki/Feature_scaling
	"""
	if mean is None:
		mean = np.mean(mat, axis=1)
	
	if std is None:
		std = np.std(mat, axis=1)

	data_normalized = (mat.transpose() - mean).transpose()
	data = (data_normalized.transpose() / std).transpose()

	return data


###################################################################################################
# Reshape with lookback value for LSTM
###################################################################################################

# creates lookback values using helper, then reshapes for LSTM
def reshape_stc_data(stc, look_back, start=0, step_size=1):
    
    samples,n = stc.shape

    stop = samples - look_back - 1

    feature_parts, label_parts = [], []

    for i in range(start,stop,step_size):
        feature_parts.append(stc[i:(i + look_back), :])
        # Take just the last estimate as label
        label_parts.append(stc[i + look_back, :])

    return feature_parts, label_parts


###################################################################################################
# Reshape with lookback value for LSTM
###################################################################################################

# creates lookback values using helper, then reshapes for LSTM
def reshape_epoch_stc_data(epoch, stc, look_back):
    
    samples_epoch,n = epoch.shape
    samples_stc,n = stc.shape

    epoch_parts, stc_parts = [], []

    # Make sure that samples are equaly long
    if samples_epoch != samples_stc:
        return epoch_parts, stc_parts

    for i in range(samples_epoch - look_back):
        epoch_parts.append(epoch[i:(i + look_back), :])
        # Take the whole estimate sequence as label
        #stc_parts.append(stc[i:(i + look_back), :])
        # Take just the last estimate as label
        stc_parts.append(stc[i + look_back - 1, :])

    return epoch_parts, stc_parts


###################################################################################################
# Reshape with lookback and look into future value for LSTM
###################################################################################################

def reshape_future_data(stc, look_back, future_steps=1, start=0, step_size=1):
    
    samples,n = stc.shape

    stop = samples - look_back - 1 - future_steps + 1

    feature_parts, label_parts = [], []

    for i in range(start,stop,step_size):
        feature_parts.append(stc[i:(i + look_back), :])
        # Take just the last estimate as label
        label_parts.append(stc[i + look_back : i + look_back + future_steps:1, :].flatten())

    return feature_parts, label_parts


###################################################################################################
# Create Sequence Parts
###################################################################################################

def create_sequence_parts(stc, look_back, start=0, step_size=1):
    samples,n = stc.shape

    stop = samples - look_back - 1

    stc_sequence_parts = []

    for i in range(start,stop,step_size):
        stc_sequence_parts.append(stc[i:(i + look_back), :])

    stc_sequences = np.array(stc_sequence_parts)

    return stc_sequences


###################################################################################################
# CMNEData class
###################################################################################################

class CMNEData(object):
    """the cmne data object

    Attributes:
        _cmne_settings: CMNE settings object.
        _inv_op: The loaded inverse operator.
        _epochs: The loaded epochs.
        _num_epochs: Number of available epochs.
        _train_idcs: Indeces which should be used for training
        _test_idcs: Indeces which should be used for testing
    """

    _mag_th = 4e-12 #Simulation: 4e-11
    _grad_th = 4000e-13 #Simulation: 4000e-12
    # Using the same inverse operator when inspecting single trials Vs. evoked
    _snr = 3.0  # Standard assumption for average data but using it for single trial
    _lambda2 = 1.0 / _snr ** 2
    _method = "dSPM"
    
	###############################################################################################
	# Constructor
	###############################################################################################
    def __init__(self, cmne_settings):
        """Return a new CMNEData object."""
        self._cmne_settings = cmne_settings

	###############################################################################################
	# Load Data
	###############################################################################################
    def load_data(self, event_id=1, tmin=-0.2, tmax=0.5):
		# Load data
        inverse_operator = read_inverse_operator(self._cmne_settings.fname_inv())
        raw = mne.io.read_raw_fif(self._cmne_settings.fname_raw())
        events = mne.read_events(self._cmne_settings.fname_event())
        
        # Set up pick list
        include = []
        
        # set EEG average reference
#        raw.set_eeg_reference()
        
        # Add a bad channel
        #    raw.info['bads'] += ['EEG 053']  # bads + 1 more
        
        # pick MEG channels
        picks = mne.pick_types( raw.info, meg=True, eeg=self._cmne_settings.meg_and_eeg(), stim=False, eog=False, include=include, exclude='bads')
        
        # Read epochs
        epochs = mne.Epochs( raw, events, event_id, tmin, tmax, baseline=(None, 0),
                            picks=picks, preload=self._cmne_settings.large_memory(), reject=dict(mag=self._mag_th, grad=self._grad_th))#, eog=150e-5))#eog=150e-6))
        
        epochs.drop_bad()
        
        self._inv_op = inverse_operator
        self._epochs = epochs
        
        #Count epochs - since they are not preloaded it has to be done with a for loop
        num_epochs = 0
        for epoch in epochs:
            num_epochs = num_epochs + 1
        self._num_epochs = num_epochs
        
        whole_list = list(range(num_epochs))
        
        if os.path.isfile(self._cmne_settings.fname_test_idcs()):
            self._test_idcs = []
            with open(self._cmne_settings.fname_test_idcs(), "r") as f:
              for line in f:
                self._test_idcs.append(int(line.strip()))
                
            self._train_idcs = [item for item in whole_list if item not in self._test_idcs]
        else:
            #split train and test
            random.seed(42)
            self._train_idcs = random.sample(range(num_epochs), (int)(num_epochs*0.85))
            self._test_idcs = [item for item in whole_list if item not in self._train_idcs]
            with open(self._cmne_settings.fname_test_idcs(), "w") as f:
                for idx in self._test_idcs:
                    f.write(str(idx) +"\n")

    ###############################################################################################
    # Generate Input
    ###############################################################################################
    
    def generate_normalized_input(self, look_back=40, batch_size=20):
        """
        Create the normalized input
        """
        nave = 1

        #Count epochs
        num_epochs = 0
        for epoch in self._epochs:
            num_epochs = num_epochs + 1

        while True:
            # select random epochs
            idx = random.sample(range(num_epochs), batch_size)
            sel_epochs = self.epochs[idx]

            stcs = apply_inverse_epochs(sel_epochs, self._inv_op, self._lambda2, self._method, pick_ori="normal", nave=nave)

            # Attention - just an approximation, since not all stc are considered for the mean and the std
            stc_data = np.hstack([stc.data for stc in stcs])
            stc_mean = np.mean(stc_data, axis=1)
            stc_std = np.std(stc_data, axis=1)
            stc_data = None
            #Attention end

            for stc in stcs:
                stc_normalized = standardize(stc.data, mean=stc_mean, std=stc_std)
                stc_normalized_T = stc_normalized.transpose()
				
                feature_list, label_list = reshape_stc_data(stc = stc_normalized_T, look_back = look_back)

                features = np.array(feature_list)
                labels = np.array(label_list)
                
                yield features, labels
		
    ###############################################################################################
    # Getters and setters
    ###############################################################################################
    def epochs(self, idx=None):
        """
        Returns selected epochs, if selection is None then all epochs are returned
        """
        if idx == None:
            return self._epochs
        else:
            return self._epochs[idx]
            
    def test_idcs(self):
        """
        Returns selected test indeces
        """
        return self._test_idcs
            
    def test_epochs(self, idx=None):
        """
        Returns selected test epochs, if selection is None then all test epochs are returned
        """
        if idx == None:
            return self._epochs[self._test_idcs]
        else:
            return self._epochs[self._test_idcs][idx]

    def train_idcs(self):
        """
        Returns selected test indeces
        """
        return self._train_idcs
    
    def train_epochs(self, idx=None):
        """
        Returns selected test epochs, if selection is None then all test epochs are returned
        """
        if idx == None:
            return self._epochs[self._train_idcs]
        else:
            return self._epochs[self._train_idcs][idx]
            
    def inv_op(self):
        """
        Returns the loaded inverse operator
        """
        return self._inv_op
    
    def method(self):
        """
        Returns the inverse method
        """
        return self._method
    
    def snr(self):
        """
        Returns the datas snr
        """
        return self._snr
    
    def lambda2(self):
        """
        Returns the datas snr
        """
        return self._lambda2
        
    def num_epochs(self):
        """
        Returns the datas snr
        """
        return self._num_epochs
        
        

###################################################################################################
# Create all data at once
###################################################################################################

def create_lstm_data(epochs, inverse_operator, lambda2, method, look_back = 1):
    """
    Create the dataset for testing regression models in the CNTK format
    Y = GQ + E -> features = stc, labels = stc
    """
    nave = 2
    
    # Compute inverse solution and stcs for each epoch
    # Use the same inverse operator as with evoked data (i.e., set nave)
    # If you use a different nave, dSPM just scales by a factor sqrt(nave)
    stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method, pick_ori="normal", nave=nave)

    ###############################################################################
    # Standardize Label and Mean Data

    feature_data = np.hstack([stc.data for stc in stcs])
    feature_mean = np.mean(feature_data, axis=1)
    feature_std = np.std(feature_data, axis=1)

    features_normalized = []
    labels_normalized = []
    
    for stc in stcs:
        stc_normalized = standardize(stc.data,mean=feature_mean,std=feature_std)
        stc_normalized_T = stc_normalized.transpose()
        
        feature_parts, label_parts = reshape_stc_data(stc = stc_normalized_T, look_back = look_back)

        features_normalized.extend(feature_parts)
        labels_normalized.extend(label_parts)

    features= np.array(features_normalized)
    labels = np.array(labels_normalized)

    return features, labels        


def create_epoch_stc_data(epochs, inverse_operator, lambda2, method, look_back = 1):
    """
    Create the dataset for testing regression models in the CNTK format
    Y = GQ + E
    """
    nave = 2
    
    # Compute inverse solution and stcs for each epoch
    # Use the same inverse operator as with evoked data (i.e., set nave)
    # If you use a different nave, dSPM just scales by a factor sqrt(nave)
    stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method, pick_ori="normal", nave=nave)

    ###############################################################################
    # Standardize Label and Mean Data

    label_data = np.hstack([stc.data for stc in stcs])
    label_mean = np.mean(label_data, axis=1)
    label_std = np.std(label_data, axis=1)

    feature_data = np.hstack([epoch.data for epoch in epochs])
    feature_mean = np.mean(feature_data, axis=1)
    feature_std = np.std(feature_data, axis=1)

    epochs_normalized = []
    stcs_normalized = []
    
    for epoch,stc in zip(epochs,stcs):
        stc_normalized = standardize(stc.data,mean=label_mean,std=label_std)
        stc_normalized_T = stc_normalized.transpose()
        
        epoch_normalized = standardize(epoch,mean=feature_mean,std=feature_std)
        epoch_normalized_T = epoch_normalized.transpose()

        epoch_parts, stc_parts = reshape_epoch_stc_data(epoch = epoch_normalized_T, stc = stc_normalized_T, look_back = look_back)

        epochs_normalized.extend(epoch_parts)
        stcs_normalized.extend(stc_parts)

    features = np.array(epochs_normalized)
    labels = np.array(stcs_normalized)

    return features, labels

      
###################################################################################################
# Generate DNN Batches
###################################################################################################

def generate_dnn_batches(epochs, inverse_operator, lambda2, method, batch_size=20):    
    """    
    Create the DNN Training Batches     
    """    
    nave = 1
    
    # Compute inverse solution and stcs for each epoch    
    # Use the same inverse operator as with evoked data (i.e., set nave)    
    # If you use a different nave, dSPM just scales by a factor sqrt(nave)    
    stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method, pick_ori="normal", nave=nave, verbose=None)
    
    #Count epochs    
    num_epochs = 0    
    for epoch in epochs:    
        num_epochs = num_epochs + 1

    while True:    
        # select random epochs    
        idx = random.sample(range(num_epochs), batch_size)    
        sel_epochs = epochs[idx]    
        sel_stcs = [stcs[i] for i in idx]

        ###############################################################################    
        # Standardize Label and Mean Data

        feature_data = np.hstack([epoch.data for epoch in sel_epochs])    
        feature_mean = np.mean(feature_data, axis=1)    
        feature_std = np.std(feature_data, axis=1)

        label_data = np.hstack([stc.data for stc in sel_stcs])    
        label_mean = np.mean(label_data, axis=1)    
        label_std = np.std(label_data, axis=1)

        features_normalized = standardize(feature_data, mean=feature_mean, std=feature_std)    
        features = features_normalized.transpose()
        
        labels_normalized = standardize(label_data, mean=label_mean, std=label_std)    
        labels = labels_normalized.transpose()
        
        yield features, labels
        
        
###################################################################################################
# Generate DNN Eval Batches
###################################################################################################

def generate_dnn_eval_batches(epochs, inverse_operator, lambda2, method):    
    """    
    Create the DNN Evaluation Batches     
    """    
    nave = 1        
    
    # Compute inverse solution and stcs for each epoch    
    # Use the same inverse operator as with evoked data (i.e., set nave)    
    # If you use a different nave, dSPM just scales by a factor sqrt(nave)    
    stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method, pick_ori="normal", nave=nave, verbose=None)
        
    ###############################################################################    
    # Standardize Label and Mean Data    
    feature_data = np.hstack([epoch.data for epoch in epochs])    
    feature_mean = np.mean(feature_data, axis=1)    
    feature_std = np.std(feature_data, axis=1)
        
    label_data = np.hstack([stc.data for stc in stcs])    
    label_mean = np.mean(label_data, axis=1)    
    label_std = np.std(label_data, axis=1)
        
    features_normalized = standardize(feature_data, mean=feature_mean, std=feature_std)    
    features = features_normalized.transpose()
    
    labels_normalized = standardize(label_data, mean=label_mean, std=label_std)    
    labels = labels_normalized.transpose()
    
    total_num_samples = features.shape[1]
    epoch_num_samples = stcs[0].shape[1]
    # for i in range(0, total_num_samples, epoch_num_samples):
    i = 0
    while True:
        yield features[i:i+epoch_num_samples:1], labels[i:i+epoch_num_samples:1]
        i = i + epoch_num_samples
        i = i % total_num_samples


###################################################################################################
# Generate Future Batches
###################################################################################################

def generate_lstm_batches(epochs, inverse_operator, lambda2, method, look_back=40, batch_size=20):
    """
    Create the LSTM Training Batches 
    """
    nave = 2

    #Count epochs
    num_epochs = 0
    for epoch in epochs:
        num_epochs = num_epochs + 1
    #take 0.75 of the epochs for training
    train_max = num_epochs

    while True:
        # select random epochs
        idx = random.sample(range(train_max), batch_size)
        sel_epochs = epochs[idx]
        # Compute inverse solution and stcs for each epoch
        # Use the same inverse operator as with evoked data (i.e., set nave)
        # If you use a different nave, dSPM just scales by a factor sqrt(nave)
        sel_epochs = mne.set_eeg_reference(sel_epochs, ref_channels=None, copy=True)[0]
        sel_epochs.apply_proj()
        
        stcs = apply_inverse_epochs(sel_epochs, inverse_operator, lambda2, method, pick_ori="normal", nave=nave)

        # Attention - just an approximation, since not all stc are considered for the mean and the std
        stc_data = np.hstack([stc.data for stc in stcs])
        stc_mean = np.mean(stc_data, axis=1)
        stc_std = np.std(stc_data, axis=1)
        stc_data = None
        #Attention end

        for stc in stcs:
            stc_normalized = standardize(stc.data,mean=stc_mean,std=stc_std)
            stc_normalized_T = stc_normalized.transpose()
            
            feature_list, label_list = reshape_stc_data(stc = stc_normalized_T, look_back = look_back)

            features = np.array(feature_list)
            labels = np.array(label_list)

            yield features, labels


###################################################################################################
# Generate LSTM Eval Batches
###################################################################################################

def generate_lstm_eval_batches(epochs, inverse_operator, lambda2, method, look_back=40):
    """
    Create the LSTM Training Batches 
    """
    nave = 1

    #Count epochs
    num_epochs = 0
    for epoch in epochs:
        num_epochs = num_epochs + 1

    # # select subset of epochs since numpy std ran into MemoryError
    # if num_epochs > 100:
    #     idx = random.sample(range(num_epochs), 100)
    #     sel_epochs = epochs[idx]
    #     sel_stcs = apply_inverse_epochs(sel_epochs, inverse_operator, lambda2, method, pick_ori="normal", nave=nave)
    stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method, pick_ori="normal", nave=nave)

    # # select subset of epochs since numpy std ran into MemoryError
    # if num_epochs > 100:
    #     # Attention - just an approximation, since not all stc are considered for the mean and the std
    #     stc_data = np.hstack([stc.data for stc in sel_stcs])
    #     #Attention end
    # else:
    stc_data = np.hstack([stc.data for stc in stcs])
    stc_mean = np.mean(stc_data, axis=1)
    stc_std = np.std(stc_data, axis=1)
    stc_data = None

    count = 0
    while True:
        for stc in stcs:
            stc_normalized = standardize(stc.data,mean=stc_mean,std=stc_std)
            stc_normalized_T = stc_normalized.transpose()
            
            feature_list, label_list = reshape_stc_data(stc = stc_normalized_T, look_back = look_back)

            features = np.array(feature_list)
            labels = np.array(label_list)

            print(">>>> LSTM Eval iteration ",count)

            count = count + 1

            yield features, labels


###################################################################################################
# Generate LSTM Future Batches
###################################################################################################

def generate_lstm_future_batches(epochs, inverse_operator, lambda2, method, look_back=40, future_steps=1, batch_size=20):
    """
    Create the Batches
    """

    #%%    epochs.drop(range(3,len(epochs)))#debug purpose keep only the first 3 epochs
    nave = 2 #len(epochs)

    #Count epochs
    num_epochs = 0
    for epoch in epochs:
        num_epochs = num_epochs + 1
    #take 0.75 of the epochs for training
    train_max = num_epochs

    #%%
    while True:
        #%% select random epochs
        idx = np.random.randint(train_max, size=batch_size)
        sel_epochs = epochs[idx]
        sel_epochs = mne.set_eeg_reference(sel_epochs, ref_channels=None, copy=True)[0]
        sel_epochs.apply_proj()

        # Compute inverse solution and stcs for each epoch
        # Use the same inverse operator as with evoked data (i.e., set nave)
        # If you use a different nave, dSPM just scales by a factor sqrt(nave)
        stcs = apply_inverse_epochs(sel_epochs, inverse_operator, lambda2, method, pick_ori="normal", nave=nave)

        # Attention - just an approximation, since not all stc are considered for the mean and the std
        stc_data = np.hstack([stc.data for stc in stcs])
        stc_mean = np.mean(stc_data, axis=1)
        stc_std = np.std(stc_data, axis=1)
        stc_data = None
        #Attention end
        #%%
        for stc in stcs:
            #%%
            stc_normalized = standardize(stc.data,mean=stc_mean,std=stc_std)
            stc_normalized_T = stc_normalized.transpose()
            
            #%%
            feature_list, label_list = reshape_future_data(stc=stc_normalized_T, look_back=look_back, future_steps=future_steps)

            features = np.array(feature_list)
            labels = np.array(label_list)

            yield features, labels
