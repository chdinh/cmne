#**
# @file     Option_4_bio_mne_comparison.py
# @author   Christoph Dinh <chdinh@nmr.mgh.harvard.edu>;
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
import os
import sys

import config as cfg

sys.path.append(cfg.repo_path + 'I_implementation/I_cmne/II_training') #Add relative path to include modules
sys.path.append(cfg.repo_path + 'I_implementation/helpers')
sys.path.append(cfg.repo_path + 'I_implementation/I_cmne/I_hyperparameter_evaluation')

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')# remove for plt.show()
import matplotlib.pyplot as plt

import pandas as pd

import datetime

from mne.minimum_norm import apply_inverse

import mne
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator

from keras.models import load_model

from helpers.cmnesettings import CMNESettings
from helpers.cmnedata import CMNEData, standardize, reshape_future_data


###################################################################################################
# The Script
###################################################################################################
## assr_270LP_fs900 fs_1_nu_10_lb_80
# look_back = 80

# # 0
# fname_model = 'C:/Users/chris/Dropbox/CMNE/Results/I_models/lb80/eval_hyper_model_meg-eeg_fs_1_nu_10_lb_80_2020-07-26_062129.h5'
# fname_cross_validation_idcs = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_idcs_job4_it_0.txt'
# fname_cross_validation_mse = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_mse_job4_it_0_fs_1_nu_10_lb_80.txt'

# # 1
# fname_model = 'C:/Users/chris/Dropbox/CMNE/Results/I_models/lb80/eval_hyper_model_meg-eeg_fs_1_nu_10_lb_80_2020-07-29_001638.h5'
# fname_cross_validation_idcs = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_idcs_job4_it_1.txt'
# fname_cross_validation_mse = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_mse_job4_it_1_fs_1_nu_10_lb_80.txt'

# # 2
# fname_model = 'C:/Users/chris/Dropbox/CMNE/Results/I_models/lb80/eval_hyper_model_meg-eeg_fs_1_nu_10_lb_80_2020-07-29_051144.h5'
# fname_cross_validation_idcs = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_idcs_job4_it_2.txt'
# fname_cross_validation_mse = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_mse_job4_it_2_fs_1_nu_10_lb_80.txt'

# # 3
# fname_model = 'C:/Users/chris/Dropbox/CMNE/Results/I_models/lb80/eval_hyper_model_meg-eeg_fs_1_nu_10_lb_80_2020-07-29_095246.h5'
# fname_cross_validation_idcs = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_idcs_job4_it_3.txt'
# fname_cross_validation_mse = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_mse_job4_it_3_fs_1_nu_10_lb_80.txt'

# # 4
# fname_model = 'C:/Users/chris/Dropbox/CMNE/Results/I_models/lb80/eval_hyper_model_meg-eeg_fs_1_nu_10_lb_80_2020-07-29_144903.h5'
# fname_cross_validation_idcs = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_idcs_job4_it_4.txt'
# fname_cross_validation_mse = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_mse_job4_it_4_fs_1_nu_10_lb_80.txt'

# # 5
# fname_model = 'C:/Users/chris/Dropbox/CMNE/Results/I_models/lb80/eval_hyper_model_meg-eeg_fs_1_nu_10_lb_80_2020-07-29_195717.h5'
# fname_cross_validation_idcs = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_idcs_job4_it_5.txt'
# fname_cross_validation_mse = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_mse_job4_it_5_fs_1_nu_10_lb_80.txt'

# # 6
# fname_model = 'C:/Users/chris/Dropbox/CMNE/Results/I_models/lb80/eval_hyper_model_meg-eeg_fs_1_nu_10_lb_80_2020-07-30_010414.h5'
# fname_cross_validation_idcs = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_idcs_job4_it_6.txt'
# fname_cross_validation_mse = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_mse_job4_it_6_fs_1_nu_10_lb_80.txt'

## assr_270LP_fs900 fs_1_nu_160_lb_80
# look_back = 80

# # 0
# fname_model = 'C:/Users/chris/Dropbox/CMNE/Results/I_models/lb80/eval_hyper_model_meg-eeg_fs_1_nu_160_lb_80_2020-07-24_211559.h5'
# fname_cross_validation_idcs = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_idcs_job5_it_0.txt'
# fname_cross_validation_mse = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_mse_job5_it_0_fs_1_nu_160_lb_80.txt'

# # 1
# fname_model = 'C:/Users/chris/Dropbox/CMNE/Results/I_models/lb80/eval_hyper_model_meg-eeg_fs_1_nu_160_lb_80_2020-07-25_023115.h5'
# fname_cross_validation_idcs = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_idcs_job5_it_1.txt'
# fname_cross_validation_mse = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_mse_job5_it_1_fs_1_nu_160_lb_80.txt'

# # 2
# fname_model = 'C:/Users/chris/Dropbox/CMNE/Results/I_models/lb80/eval_hyper_model_meg-eeg_fs_1_nu_160_lb_80_2020-07-25_074010.h5'
# fname_cross_validation_idcs = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_idcs_job5_it_2.txt'
# fname_cross_validation_mse = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_mse_job5_it_2_fs_1_nu_160_lb_80.txt'

# # 3
# fname_model = 'C:/Users/chris/Dropbox/CMNE/Results/I_models/lb80/eval_hyper_model_meg-eeg_fs_1_nu_160_lb_80_2020-07-25_125038.h5'
# fname_cross_validation_idcs = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_idcs_job5_it_3.txt'
# fname_cross_validation_mse = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_mse_job5_it_3_fs_1_nu_160_lb_80.txt'

# # 4
# fname_model = 'C:/Users/chris/Dropbox/CMNE/Results/I_models/lb80/eval_hyper_model_meg-eeg_fs_1_nu_160_lb_80_2020-07-25_180344.h5'
# fname_cross_validation_idcs = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_idcs_job5_it_4.txt'
# fname_cross_validation_mse = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_mse_job5_it_4_fs_1_nu_160_lb_80.txt'

# # 5 <<<
# fname_model = 'C:/Users/chris/Dropbox/CMNE/Results/I_models/lb80/eval_hyper_model_meg-eeg_fs_1_nu_160_lb_80_2020-07-25_235531.h5'
# fname_cross_validation_idcs = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_idcs_job5_it_5.txt'
# fname_cross_validation_mse = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_mse_job5_it_5_fs_1_nu_160_lb_80.txt'

# # 6
# fname_model = 'C:/Users/chris/Dropbox/CMNE/Results/I_models/lb80/eval_hyper_model_meg-eeg_fs_1_nu_160_lb_80_2020-07-26_061135.h5'
# fname_cross_validation_idcs = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_idcs_job5_it_6.txt'
# fname_cross_validation_mse = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_mse_job5_it_6_fs_1_nu_160_lb_80.txt'

## assr_270LP_fs900 fs_1_nu_1280_lb_80
# look_back = 80

# # 0
# fname_model = 'C:/Users/chris/Dropbox/CMNE/Results/I_models/lb80/eval_hyper_model_meg-eeg_fs_1_nu_1280_lb_80_2020-07-24_211925.h5'
# fname_cross_validation_idcs = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_idcs_job6_it_0.txt'
# fname_cross_validation_mse = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_mse_job6_it_0_fs_1_nu_1280_lb_80.txt'

# # 1
# fname_model = 'C:/Users/chris/Dropbox/CMNE/Results/I_models/lb80/eval_hyper_model_meg-eeg_fs_1_nu_1280_lb_80_2020-07-25_023622.h5'
# fname_cross_validation_idcs = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_idcs_job6_it_1.txt'
# fname_cross_validation_mse = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_mse_job6_it_1_fs_1_nu_1280_lb_80.txt'

# # 2
# fname_model = 'C:/Users/chris/Dropbox/CMNE/Results/I_models/lb80/eval_hyper_model_meg-eeg_fs_1_nu_1280_lb_80_2020-07-25_074933.h5'
# fname_cross_validation_idcs = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_idcs_job6_it_2.txt'
# fname_cross_validation_mse = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_mse_job6_it_2_fs_1_nu_1280_lb_80.txt'

# # 3
# fname_model = 'C:/Users/chris/Dropbox/CMNE/Results/I_models/lb80/eval_hyper_model_meg-eeg_fs_1_nu_1280_lb_80_2020-07-25_130259.h5'
# fname_cross_validation_idcs = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_idcs_job6_it_3.txt'
# fname_cross_validation_mse = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_mse_job6_it_3_fs_1_nu_1280_lb_80.txt'

# # 4
# fname_model = 'C:/Users/chris/Dropbox/CMNE/Results/I_models/lb80/eval_hyper_model_meg-eeg_fs_1_nu_1280_lb_80_2020-07-25_182002.h5'
# fname_cross_validation_idcs = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_idcs_job6_it_4.txt'
# fname_cross_validation_mse = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_mse_job6_it_4_fs_1_nu_1280_lb_80.txt'

# # 5
# fname_model = 'C:/Users/chris/Dropbox/CMNE/Results/I_models/lb80/eval_hyper_model_meg-eeg_fs_1_nu_1280_lb_80_2020-07-26_001853.h5'
# fname_cross_validation_idcs = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_idcs_job6_it_5.txt'
# fname_cross_validation_mse = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_mse_job6_it_5_fs_1_nu_1280_lb_80.txt'

# # 6
# fname_model = 'C:/Users/chris/Dropbox/CMNE/Results/I_models/lb80/eval_hyper_model_meg-eeg_fs_1_nu_1280_lb_80_2020-07-26_063512.h5'
# fname_cross_validation_idcs = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_idcs_job6_it_6.txt'
# fname_cross_validation_mse = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb80/assr_270LP_fs900_cross_mse_job6_it_6_fs_1_nu_1280_lb_80.txt'

## assr_270LP_fs900 fs_1_nu_1280_lb_10
# look_back = 10

# # 0
# fname_model = 'C:/Users/chris/Dropbox/CMNE/Results/I_models/lb10/eval_hyper_model_meg-eeg_fs_1_nu_1280_lb_10_2020-07-24_160109.h5'
# fname_cross_validation_idcs = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb10/assr_270LP_fs900_cross_idcs_job3_it_0.txt'
# fname_cross_validation_mse = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb10/assr_270LP_fs900_cross_mse_job3_it_0_fs_1_nu_1280_lb_10.txt'

## assr_270LP_fs900 fs_1_nu_1280_lb_160
look_back = 160

# 0
fname_model = 'C:/Users/chris/Dropbox/CMNE/Results/I_models/lb160/eval_hyper_model_meg-eeg_fs_1_nu_1280_lb_160_2020-07-25_101246.h5'
fname_cross_validation_idcs = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb160/assr_270LP_fs900_cross_idcs_job9_it_0.txt'
fname_cross_validation_mse = 'C:/Users/chris/Dropbox/CMNE/Results/III_training/lb160/assr_270LP_fs900_cross_mse_job9_it_0_fs_1_nu_1280_lb_160.txt'

#%% Data Settings
data_settings = CMNESettings(   repo_path=cfg.repo_path, data_path=cfg.data_path,
                                fname_raw=cfg.fname_raw,
                                fname_inv=cfg.fname_inv,
                                fname_eve=cfg.fname_eve,
                                fname_test_idcs=cfg.fname_test_idcs,
                                meg_and_eeg=cfg.meg_and_eeg
                            )

event_id, tmin, tmax = 1, -0.2, 0.5
train_percentage = 0.85
cross_validation_percentage = 0.85

data = CMNEData(cmne_settings=data_settings)
data.load_data(event_id=event_id, tmin=tmin, tmax=tmax, train_percentage=train_percentage)

#################################
#%%
num_train_idcs = len(data.train_idcs())

whole_list = list(range(num_train_idcs))

if os.path.isfile(fname_cross_validation_idcs):
    cross_validation_train_idcs = []
    with open(fname_cross_validation_idcs, "r") as f:
        for line in f:
            cross_validation_train_idcs.append(int(line.strip()))
    cross_validation_test_idcs = [item for item in whole_list if item not in cross_validation_train_idcs]


    sel_epochs = data.train_epochs(cross_validation_test_idcs)

    nave = 2 #len(epochs)
    
    # Compute inverse solution and stcs for each epoch
    # Use the same inverse operator as with evoked data (i.e., set nave)
    # If you use a different nave, dSPM just scales by a factor sqrt(nave)
    #sel_epochs = mne.set_eeg_reference(sel_epochs, ref_channels=None, copy=True)[0]
    sel_epochs.apply_proj()

    # Compute inverse solution and stcs for each epoch
    # Use the same inverse operator as with evoked data (i.e., set nave)
    # If you use a different nave, dSPM just scales by a factor sqrt(nave)

    stcs = apply_inverse_epochs(sel_epochs, inverse_operator=data.inv_op(), lambda2=data.lambda2(), method=data.method(), pick_ori="normal", nave=nave)

    # Attention - just an approximation, since not all stc are considered for the mean and the std
    stc_data = np.hstack([stc.data for stc in stcs])
    stc_mean = np.mean(stc_data, axis=1)
    stc_std = np.std(stc_data, axis=1)
    stc_data = None
    #Attention end

    # load model
    lstm_model = load_model(fname_model)

    future_steps = 1

    count_stcs = 1;
    #################################
    # %%
    with open(fname_cross_validation_mse, "w") as f:
        for stc in stcs:
            print('STC %d'%(count_stcs))
            stc_normalized = standardize(stc.data,mean=stc_mean,std=stc_std)
            stc_normalized_T = stc_normalized.transpose()
            
            feature_list, label_list = reshape_future_data(stc=stc_normalized_T, look_back=look_back, future_steps=future_steps)

            features = np.array(feature_list)
            labels = np.array(label_list)
            
            #%% LSTM estimation
            step = 1;
            for feature, label in (zip(features, labels)):
                stc_prior = np.expand_dims(feature, axis=0)
                stc_predict = lstm_model.predict(stc_prior)
                stc_mse = ((stc_predict - label)**2).mean(axis=1)
                
                #print('STC %d, Step %d, Error %f'%(count_stcs, step, stc_mse))

                f.write(str(stc_mse) +"\n")
                step = step + 1;
            
            count_stcs = count_stcs + 1;

            if count_stcs == 11:
                break    # break here