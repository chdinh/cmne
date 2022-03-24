#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# @authors   Christoph Dinh <christoph.dinh@brain-link.de>
#            John G Samuelson <johnsam@mit.edu>
# @version   1.0
# @date      May, 2017
# @copyright Copyright (c) 2017-2022, authors of CMNE. All rights reserved.
# @license   MIT
# @brief     CMNE example
# ---------------------------------------------------------------------------

#==================================================================================
#%%
import datetime
import random
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')# remove for plt.show()
import matplotlib.pyplot as plt

from mne.minimum_norm import apply_inverse
from keras.models import load_model

import cmne
import config as cfg


###################################################################################################
# The Script
###################################################################################################

#%% Settings
settings = cmne.Settings(results_path=cfg.result_path, data_path=cfg.data_path,
                    fname_raw=cfg.fname_raw,
                    fname_inv=cfg.fname_inv,
                    fname_eve=cfg.fname_eve,
                    fname_test_idcs=cfg.fname_test_idcs
                    )

#%% Data
event_id, tmin, tmax = 1, -0.2, 0.5
data = cmne.Data(settings=settings)
data.load_data(event_id=event_id, tmin=tmin, tmax=tmax)

#%% train
fname_model = cmne.train(settings, data, num_epochs=1, steps_per_ep=1)


###################################################################################################
#%% Select random subset of epochs
#idx = random.sample(range(num_epochs), nave)
#idx = [1238, 283, 1133, 1433, 70, 1292, 1064, 982, 1025, 1405, 1371, 805, 427, 693, 144, 1185, 152, 618, 1374, 1281]

random.seed(42)
idx_list = []
idx_list.append(random.sample(range(len(data.test_epochs())), 20))
#idx_list.append(random.sample(range(len(data.test_epochs())), len(data.test_epochs())))


#%%
for idx in idx_list:
    print(">>>> Starting evaluation (number of epochs = %d) <<<<\n" % (len(idx)))
    nave = len(idx)
    
    sel_epochs = data.test_epochs(idx)
    
    evoked = sel_epochs.average()
        
    ###################################################################################################
    #%% dSPM reference estimation
    stc_dSPM = apply_inverse(evoked, data.inv_op(), data.lambda2(), data.method(), pick_ori="normal")
    
    # Abs Max Normalization
    stc_dSPM._data = np.absolute(stc_dSPM.data) # TBD: Remove this line in a future version - think about the reason
    stc_dSPM._data = stc_dSPM.data / stc_dSPM.data.max() # TBD: Remove this line in a future version - since it is z scored down
    
        
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
    #%% MCMC estimation
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
    
    #%% LSTM prediction
    stc_LSTM_predict = stc_dSPM.copy()
    stc_LSTM_predict._data = stc_result_predict_lstm.transpose();
    
    # Abs Max Normalization
    stc_LSTM_predict._data = np.absolute(stc_LSTM_predict.data)
    stc_LSTM_predict._data = stc_LSTM_predict.data / stc_LSTM_predict.data.max()
    
    print('stc_LSTM_predict._data.shape', stc_LSTM_predict._data.shape)
    
    #%% MCMC
    stc_MCMC = stc_dSPM.copy()
    stc_MCMC._data = stc_result.transpose();
    
    # Abs Max Normalization
    stc_MCMC._data = np.absolute(stc_MCMC.data)
    stc_MCMC._data = stc_MCMC.data / stc_MCMC.data.max()
    
    print('stc_MCMC._data.shape', stc_MCMC._data.shape)
    
    ###################################################################################################
    #%% Control
    stc_control_result = np.zeros((stc_sens.shape[0],5124))
    stc_control_result[0:lb:1,:] = stc_sens[0:lb:1,:] # fill beginning with sensing results
    
    for i in range(steps):
        stc_pre_ave = np.mean(stc_sens[i:i+lb:1,:], axis = 0)
        
        #do the bayesian step
        stc_posterior = stc_sens[i+lb,:] * stc_pre_ave
        stc_control_result[i+lb,:] = stc_posterior
        
        print('Step %d/%d'%(i+1, steps))
    
    
    #%%
    stc_CTRL = stc_dSPM.copy()
    stc_CTRL._data = stc_control_result.transpose();
    
    # Abs Max Normalization
    stc_CTRL._data = np.absolute(stc_CTRL.data)
    stc_CTRL._data = stc_CTRL.data / stc_CTRL.data.max()
    
    print('stc_CTRL._data.shape', stc_CTRL._data.shape)


    ###################################################################################################
    #%% Save Results
    ###################################################################################################
    
    date_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    fname_stc_dSPM = settings.results_stc_path() + '/dSPM_nave_'  + str(nave) + '_' + date_stamp
    fname_stc_dSPM_resultfig = settings.results_stc_path() + '/dSPM_nave_'  + str(nave) + '_' + date_stamp + '.png'
    
    fname_stc_LSTM_predict = settings.results_stc_path() + '/LSTM_predict_nave_'  + str(nave) + '_' + date_stamp
    fname_stc_LSTM_predict_resultfig = settings.results_stc_path() + '/LSTM_predict_nave_'  + str(nave) + '_' + date_stamp + '.png'
    
    fname_stc_MCMC = settings.results_stc_path() + '/CMNE_nave_'  + str(nave) + '_' + date_stamp
    fname_stc_MCMC_resultfig = settings.results_stc_path() + '/CMNE_nave_'  + str(nave) + '_' + date_stamp + '.png'
    
    fname_stc_CTRL = settings.results_stc_path() + '/CTRL_nave_'  + str(nave) + '_' + date_stamp
    fname_stc_CTRL_resultfig = settings.results_stc_path() + '/CTRL_nave_'  + str(nave) + '_' + date_stamp + '.png'
    
    fname_idx_list = settings.results_stc_path() + '/idx_list_nave_'  + str(nave) + '_' + date_stamp + '.csv'
    
    
    #%% Save idxs
    print(">>>> Safe IDXs <<<<")
    df = pd.DataFrame(idx)
    df.to_csv(path_or_buf=fname_idx_list, index=False, header=False)
    
    #%% Save stcs
    print(">>>> Safe STCs <<<<")
    
    stc_dSPM.save(fname_stc_dSPM)
    stc_LSTM_predict.save(fname_stc_LSTM_predict)
    stc_MCMC.save(fname_stc_MCMC)
    stc_CTRL.save(fname_stc_CTRL)
    
    
    # Save figures
    print(">>>> Safe Figures <<<<")
    
    # plot dSPM STC results
    plt.figure(1)
    plt.plot(1e3 * stc_dSPM.times, stc_dSPM.data[::100, :].T)
    plt.xlabel('time (ms)')
    plt.ylabel('%s value' % data.method)
    plt.title('dSPM: STC time courses')
    # #axes = plt.gca()
    # #axes.set_xlim([xmin,xmax])
    # #axes.set_ylim([0,1.2])
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.savefig(fname_stc_dSPM_resultfig, dpi=300)
    #plt.show()
    

    # plot LSTM Prediction STC results
    plt.figure(3)
    plt.plot(1e3 * stc_LSTM_predict.times, stc_LSTM_predict.data[::100, :].T)
    plt.xlabel('time (ms)')
    plt.ylabel('LSTM Prediction value')
    plt.title('LSTM Prediction: STC time courses')
    # #axes = plt.gca()
    # #axes.set_xlim([xmin,xmax])
    # #axes.set_ylim([0,1.2])
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.savefig(fname_stc_LSTM_predict_resultfig, dpi=300)
    #plt.show()
    
    # plot MCMC STC results
    plt.figure(4)
    plt.plot(1e3 * stc_MCMC.times, stc_MCMC.data[::100, :].T)
    plt.xlabel('time (ms)')
    plt.ylabel('MCMC value')
    plt.title('MCMC: STC time courses')
    # #axes = plt.gca()
    # #axes.set_xlim([xmin,xmax])
    # #axes.set_ylim([0,1.2])
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.savefig(fname_stc_MCMC_resultfig, dpi=300)
    #plt.show()
    
    
    # plot CTRL STC results
    plt.figure(5)
    plt.plot(1e3 * stc_CTRL.times, stc_CTRL.data[::100, :].T)
    plt.xlabel('time (ms)')
    plt.ylabel('CTRL value')
    plt.title('CTRL: STC time courses')
    # #axes = plt.gca()
    # #axes.set_xlim([xmin,xmax])
    # #axes.set_ylim([0,1.2])
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.savefig(fname_stc_CTRL_resultfig, dpi=300)
    #plt.show()
    