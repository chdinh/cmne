#**
# @file     Option_4_bio_mne_comparison.py
# @author   Christoph Dinh <chdinh@nmr.mgh.harvard.edu>;
#           John GW Samuelsson <johnsam@mit.edu>;
#           Matti Hamalainen <msh@nmr.mgh.harvard.edu>
# @version  1.0
# @date     May, 2017
#
# @section  LICENSE
#
# Copyright (C) 2017, Christoph Dinh. All rights reserved.
#
# @brief    Model inverse operator with Deep Learning Model
#           to estimate a MNE-dSPM inverse solution on single epochs
#
#**

#==================================================================================
#%%
import sys
sys.path.append("../..") #Add relative path to include modules

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')# remove for plt.show()
import matplotlib.pyplot as plt

import pandas as pd

import datetime

from mne.minimum_norm import apply_inverse

from keras.models import load_model

import helpers.cmnedata as bd
from helpers.cmnesettings import CMNESettings
from helpers.cmnedata import CMNEData



###################################################################################################
# The Script
###################################################################################################

event_id, tmin, tmax = 1, -0.5, 1.0
# 1 - ASSR
fname_model = 'D:/Data/Models/bio/best_models/Model_Opt_5_sim_meg-eeg_nu_1280_lb_80_2018-02-03_114734.h5'
settings = CMNESettings(repo_path='D:/Users/Christoph/Git/bio/', data_path='D:/Data/Simulation/',
                       fname_raw='SpikeSim250_processed_fs900_raw.fif',
                       fname_inv='SpikeSim250_processed_fs900_raw-ico-4-meg-eeg-inv.fif',
                       fname_eve='SpikeSim250_processed_fs900_raw-eve.fif',
                       fname_test_idcs='SpikeSim250_processed_fs900_raw-test-idcs.txt')

data = CMNEData(cmne_settings=settings)
data.load_data(event_id=event_id, tmin=tmin, tmax=tmax)


###################################################################################################
#%% Select random subset of epochs - Averaging
#idx = random.sample(range(num_epochs), nave)
#idx = [1238, 283, 1133, 1433, 70, 1292, 1064, 982, 1025, 1405, 1371, 805, 427, 693, 144, 1185, 152, 618, 1374, 1281]

random.seed(42)
idx_list = []
idx_list.append(random.sample(range(len(data.test_epochs())), 2)) # two averages as an input to the trained network
idx_list.append(random.sample(range(len(data.test_epochs())), 5)) # five averages
idx_list.append(random.sample(range(len(data.test_epochs())), 20)) # 20 averages
#idx_list.append(random.sample(range(len(data.test_epochs())), 50)) # 50 averages
idx_list.append(random.sample(range(len(data.test_epochs())), len(data.test_epochs())))


#%% Evaluation
for idx in idx_list:
    print(">>>> Starting evaluation (number of epochs = %d) <<<<\n" % (len(idx)))
    nave = len(idx)
    
    sel_epochs = data.test_epochs(idx)
    
    evoked = sel_epochs.average()
        
    ###################################################################################################
    #%% dSPM reference estimation
    stc_dSPM = apply_inverse(evoked, data.inv_op(), data.lambda2(), data.method(), pick_ori="normal")
    
    # Abs Max Normalization
    stc_dSPM._data = np.absolute(stc_dSPM.data)
    stc_dSPM._data = stc_dSPM.data / stc_dSPM.data.max()
    
    ###################################################################################################
    #%% LSTM estimation
    # load model
    lstm_model = load_model(fname_model)
    
    stc_data = stc_dSPM._data
    stc_mean = np.mean(stc_data, axis=1)
    stc_std = np.std(stc_data, axis=1)
    stc_normalized = bd.standardize(stc_data, mean=stc_mean, std=stc_std)
    stc_normalized_T = stc_normalized.transpose()
    
    ###################################################################################################
    #%% CMNE estimation
    stc_sens = stc_normalized_T.copy()
    
    lb = 80
    steps = stc_sens.shape[0] - lb
    
    stc_result = np.zeros((stc_sens.shape[0],5124))
    stc_result[0:lb:1,:] = stc_sens[0:lb:1,:] # fill beginning with sensing results
    
    stc_result_predict_lstm = np.zeros((stc_sens.shape[0],5124))
    stc_result_predict_lstm[0:lb:1,:] = stc_sens[0:lb:1,:] # fill beginning with lstm prediction results
    
    for i in range(steps):
        stc_prior = np.expand_dims(stc_result[i:i+lb:1,:], axis=0)
        stc_predict = lstm_model.predict(stc_prior)
        stc_result_predict_lstm[i+lb,:] = stc_predict
        
        #do the bayesian step
        stc_posterior = stc_sens[i+lb,:] * stc_predict
        stc_result[i+lb,:] = stc_posterior
        
        print('Step %d/%d'%(i+1, steps))
        
        # Network Interpretation
        #dir(lstm_model)

        #for weight in lstm_model.get_weights(): # weights from Dense layer omitted
        #    print(weight.shape)  


        for layer in lstm_model.layers:
            if 'LSTM' in str(layer):
                weights = layer.get_weights()
        
                print('Previous memory state states[0] = {}'.format(K.get_value(layer.states[0])))
                print('Previous carry state states[1] = {}'.format(K.get_value(layer.states[1])))
                
                
                #see if we can get ft, ot etc here as welk
        
#                print('Input')
#                print('bias_i = {}'.format(K.get_value(layer.cell.bias_i)))
#                print('kernel_i = {}'.format(K.get_value(layer.cell.kernel_i)))
#                print('recurrent_kernel_i = {}'.format(K.get_value(layer.cell.recurrent_kernel_i)))
#        
#                print('Forget')
#                print('bias_f = {}'.format(K.get_value(layer.cell.bias_f)))
#                print('kernel_f = {}'.format(K.get_value(layer.cell.kernel_f)))
#                print('recurrent_kernel_f = {}'.format(K.get_value(layer.cell.recurrent_kernel_f)))
#        
#                print('Cell')
#                print('bias_c = {}'.format(K.get_value(layer.cell.bias_c)))
#                print('kernel_c = {}'.format(K.get_value(layer.cell.kernel_c)))
#                print('recurrent_kernel_c = {}'.format(K.get_value(layer.cell.recurrent_kernel_c)))
#        
#                print('Output')
#                print('bias_o = {}'.format(K.get_value(layer.cell.bias_o)))
#                print('kernel_o = {}'.format(K.get_value(layer.cell.kernel_o)))
#                print('recurrent_kernel_o = {}'.format(K.get_value(layer.cell.recurrent_kernel_o)))
        
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    #%% LSTM prediction
    stc_LSTM_predict = stc_dSPM.copy()
    stc_LSTM_predict._data = stc_result_predict_lstm.transpose();
    
    # Abs Max Normalization
    stc_LSTM_predict._data = np.absolute(stc_LSTM_predict.data)
    stc_LSTM_predict._data = stc_LSTM_predict.data / stc_LSTM_predict.data.max()
    
    print('stc_LSTM_predict._data.shape', stc_LSTM_predict._data.shape)
    
    #%% CMNE
    stc_CMNE = stc_dSPM.copy()
    stc_CMNE._data = stc_result.transpose();
    
    # Abs Max Normalization
    stc_CMNE._data = np.absolute(stc_CMNE.data)
    stc_CMNE._data = stc_CMNE.data / stc_CMNE.data.max()
    
    print('stc_CMNE._data.shape', stc_CMNE._data.shape)
    
    ###################################################################################################
    #%% Save Results
    ###################################################################################################
    
    date_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    