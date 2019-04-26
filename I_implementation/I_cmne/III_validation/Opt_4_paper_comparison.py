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
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')# remove for plt.show()
import matplotlib.pyplot as plt

import pandas as pd

import datetime

from mne.minimum_norm import apply_inverse

from keras.models import load_model

import modules.biodata as bd
from modules.biosettings import BioSettings
from modules.biodata import BioData

###################################################################################################
# The Script
###################################################################################################

event_id, tmin, tmax = 1, -0.5, 1.0
# 0 - Sample Data
#settings = BioSettings(repo_path='D:/GitHub/bio/', data_path='D:/GitHub/mne-cpp/bin/MNE-sample-data/',
#                       fname_raw='sample_audvis_filt-0-40_raw.fif',
#                       fname_inv='sample_audvis-meg-eeg-oct-6-meg-eeg-inv.fif',
#                       fname_eve='sample_audvis_filt-0-40_raw-eve.fif',
#                       fname_test_idcs='sample_audvis-test-idcs.txt')
# 1 - Azure Windows
#settings = BioSettings(repo_path='C:/Git/bio/', data_path='Z:/MEG/jgs/170505/processed/')
# 2 - Azure Linux
#settings = BioSettings(repo_path='/home/chdinh/Git/bio/', data_path='/cloud/datasets/MNE-sample-data/')
# 3 - Azure Windows Simulation
#settings = BioSettings(repo_path='C:/Git/bio/', data_path='Z:/Simulation/',
#                       fname_raw='SpikeSim2000_fs900_raw.fif',
#                       fname_inv='SpikeSim2000_fs900_raw-ico-4-meg-eeg-inv.fif',
#                       fname_eve='SpikeSim2000_fs900_raw-eve.fif',
#                       fname_test_idcs='SpikeSim2000_fs900_raw-test-idcs.txt')
# 4 - Local
#fname_model = 'D:/Data/Models/bio/Model_Opt_3b_final_meg-eeg_nu_1280_lb_80_-0.2-0.8_2017-10-06_020221.h5' #'Z:/Shared Storage/Models/Model_Opt_3b_nu_1280_lb_80_2017-08-19_050712.h5'
##fname_model = 'D:/Data/Models/bio/Model_Opt_3b_nu_1280_lb_80_2017-08-19_050712.h5'
#settings = BioSettings(repo_path='D:/Users/Christoph/Git/bio/', data_path='D:/Data/MEG/jgs/170505/processed/')
#data = BioData(bio_settings=settings)
#data.load_data(event_id=event_id, tmin=tmin, tmax=tmax)

# 5 - Local Simulation
#fname_model = 'D:/Data/Models/bio/Model_Opt_5_sim_meg-eeg_nu_1280_lb_80_2017-10-08_042856.h5'
#settings = BioSettings(repo_path='D:/Users/Christoph/Git/bio/', data_path='D:/Data/Simulation/',
#                       fname_raw='SpikeSim2000_fs900_raw.fif',
#                       fname_inv='SpikeSim2000_fs900_raw-ico-4-meg-eeg-inv.fif',
#                       fname_eve='SpikeSim2000_fs900_raw-eve.fif',
#                       fname_test_idcs='SpikeSim2000_fs900_raw-test-idcs.txt')

fname_model = 'D:/Data/Models/bio/best_models/Model_Opt_5_sim_meg-eeg_nu_1280_lb_80_2018-02-03_114734.h5'
settings = BioSettings(repo_path='D:/Users/Christoph/Git/bio/', data_path='D:/Data/Simulation/',
                       fname_raw='SpikeSim250_processed_fs900_raw.fif',
                       fname_inv='SpikeSim250_processed_fs900_raw-ico-4-meg-eeg-inv.fif',
                       fname_eve='SpikeSim250_processed_fs900_raw-eve.fif',
                       fname_test_idcs='SpikeSim250_processed_fs900_raw-test-idcs.txt')

data = BioData(bio_settings=settings)
data.load_data(event_id=event_id, tmin=tmin, tmax=tmax)


###################################################################################################
#%% Select random subset of epochs
#idx = random.sample(range(num_epochs), nave)
#idx = [1238, 283, 1133, 1433, 70, 1292, 1064, 982, 1025, 1405, 1371, 805, 427, 693, 144, 1185, 152, 618, 1374, 1281]

random.seed(42)
idx_list = []
idx_list.append(random.sample(range(len(data.test_epochs())), 2))
idx_list.append(random.sample(range(len(data.test_epochs())), 5))
idx_list.append(random.sample(range(len(data.test_epochs())), 20))
#idx_list.append(random.sample(range(len(data.test_epochs())), 50))
idx_list.append(random.sample(range(len(data.test_epochs())), len(data.test_epochs())))


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
    stc_dSPM._data = np.absolute(stc_dSPM.data)
    stc_dSPM._data = stc_dSPM.data / stc_dSPM.data.max()
    
    
#    ###################################################################################################
#    #%% DNN estimation
#    # load model
#    dnn_model = load_model('Z:/Shared Storage/Models/Model_Opt_1_DNN_in_365_out_5124_2017-08-17_035213.h5')
#    
#    feature_data = evoked.data
#    feature_mean = np.mean(feature_data, axis=1)
#    feature_std = np.std(feature_data, axis=1)
#    features_normalized = bd.standardize(feature_data, mean=feature_mean, std=feature_std)
#    features = features_normalized.transpose()
#    
#    stc_tmp = dnn_model.predict(features) #evoked.data.transpose())
#    stc_DNN = stc_dSPM.copy()
#    stc_DNN._data = stc_tmp.transpose();
#    
#    # Abs Max Normalization
#    stc_DNN._data = np.absolute(stc_DNN.data)
#    stc_DNN._data = stc_DNN.data / stc_DNN.data.max()
#    
#    print('stc_DNN._data.shape', stc_DNN._data.shape)
    
    ###################################################################################################
    #%% LSTM estimation
    # load model
    lstm_model = load_model(fname_model)
    
    stc_data = stc_dSPM._data
    stc_mean = np.mean(stc_data, axis=1)
    stc_std = np.std(stc_data, axis=1)
    stc_normalized = bd.standardize(stc_data, mean=stc_mean, std=stc_std)
    stc_normalized_T = stc_normalized.transpose()
    
    #stc_sequences = create_sequence_parts(stc_normalized_T, 80)
    #
    #stc_tmp = lstm_model.predict(stc_sequences)
    #stc_LSTM = stc_dSPM.copy()
    #stc_LSTM._data = np.concatenate((np.zeros((5124,81), dtype=float), stc_tmp.transpose()),axis=1)
    #
    ## Abs Max Normalization
    #stc_LSTM._data = np.absolute(stc_LSTM.data)
    #stc_LSTM._data = stc_LSTM.data / stc_LSTM.data.max()
    #
    #print('stc_LSTM._data.shape', stc_LSTM._data.shape)
    
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
    #%% MCMC Control
    
    #    stc_mcmc_control_result = np.zeros((stc_sens.shape[0],5124))
    #    stc_mcmc_control_result[0:lb:1,:] = stc_sens[0:lb:1,:] # fill beginning with sensing results
    #    
    #    for i in range(steps):
    #        stc_pre_ave = np.mean(stc_mcmc_control_result[i:i+lb:1,:], axis = 0)
    #        
    #        #do the bayesian step
    #        stc_posterior = stc_sens[i+lb,:] * stc_pre_ave
    #        
    #        stc_posterior_abs = np.absolute(stc_posterior)
    #        stc_posterior = stc_posterior / stc_posterior_abs.max()
    #        
    #        stc_mcmc_control_result[i+lb,:] = stc_posterior
    #        
    #        print('Step %d/%d'%(i+1, steps))
    #    
    #    
    #    #%%
    #    stc_MCMC_CTRL = stc_dSPM.copy()
    #    stc_MCMC_CTRL._data = stc_mcmc_control_result.transpose();
    #    
    #    # Abs Max Normalization
    #    stc_MCMC_CTRL._data = np.absolute(stc_MCMC_CTRL.data)
    #    stc_MCMC_CTRL._data = stc_MCMC_CTRL.data / stc_MCMC_CTRL.data.max()
    #    
    #    print('stc_MCMC_CTRL._data.shape', stc_MCMC_CTRL._data.shape)
    
    ###################################################################################################
    #%% Save Results
    ###################################################################################################
    
    date_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    fname_stc_dSPM = settings.repo_path() + 'Results/STCs/Opt_4_STC_dSPM_nave_'  + str(nave) + '_' + date_stamp
    fname_stc_dSPM_resultfig = settings.repo_path() + 'Results/STCs/Opt_4_STC_dSPM_nave_'  + str(nave) + '_' + date_stamp + '.png'
    
#    fname_stc_DNN = settings.repo_path() + 'Results/STCs/Opt_4_STC_DNN_nave_'  + str(nave) + '_' + date_stamp
#    fname_stc_DNN_resultfig = settings.repo_path() + 'Results/STCs/Opt_4_STC_DNN_nave_'  + str(nave) + '_' + date_stamp + '.png'
    
    #fname_stc_LSTM = repo_path + 'Results/STCs/Opt_4_STC_LSTM_nave_'  + str(nave) + '_' + date_stamp
    #fname_stc_LSTM_resultfig = repo_path + 'Results/STCs/Opt_4_STC_LSTM_nave_'  + str(nave) + '_' + date_stamp + '.png'
    
    fname_stc_LSTM_predict = settings.repo_path() + 'Results/STCs/Opt_4_STC_LSTM_predict_nave_'  + str(nave) + '_' + date_stamp
    fname_stc_LSTM_predict_resultfig = settings.repo_path() + 'Results/STCs/Opt_4_STC_LSTM_predict_nave_'  + str(nave) + '_' + date_stamp + '.png'
    
    fname_stc_MCMC = settings.repo_path() + 'Results/STCs/Opt_4_STC_MCMC_nave_'  + str(nave) + '_' + date_stamp
    fname_stc_MCMC_resultfig = settings.repo_path() + 'Results/STCs/Opt_4_STC_MCMC_nave_'  + str(nave) + '_' + date_stamp + '.png'
    
    fname_stc_CTRL = settings.repo_path() + 'Results/STCs/Opt_4_STC_CTRL_nave_'  + str(nave) + '_' + date_stamp
    fname_stc_CTRL_resultfig = settings.repo_path() + 'Results/STCs/Opt_4_STC_CTRL_nave_'  + str(nave) + '_' + date_stamp + '.png'
    
#    fname_stc_MCMC_CTRL = settings.repo_path() + 'Results/STCs/Opt_4_STC_MCMC_CTRL_nave_'  + str(nave) + '_' + date_stamp
#    fname_stc_MCMC_CTRL_resultfig = settings.repo_path() + 'Results/STCs/Opt_4_STC_MCMC_CTRL_nave_'  + str(nave) + '_' + date_stamp + '.png'
    
    fname_idx_list = settings.repo_path() + 'Results/STCs/Opt_4_idx_list_nave_'  + str(nave) + '_' + date_stamp + '.csv'
    
    
    #%% Save idxs
    print(">>>> Safe IDXs <<<<")
    df = pd.DataFrame(idx)
    df.to_csv(path_or_buf=fname_idx_list, index=False, header=False)
    
    #%% Save stcs
    print(">>>> Safe STCs <<<<")
    
    stc_dSPM.save(fname_stc_dSPM)
#    stc_DNN.save(fname_stc_DNN)
    #stc_LSTM.save(fname_stc_LSTM)
    stc_LSTM_predict.save(fname_stc_LSTM_predict)
    stc_MCMC.save(fname_stc_MCMC)
    stc_CTRL.save(fname_stc_CTRL)
    #stc_MCMC_CTRL.save(fname_stc_MCMC_CTRL)
    
    
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
    
#    # plot DNN STC results
#    plt.figure(2)
#    plt.plot(1e3 * stc_DNN.times, stc_DNN.data[::100, :].T)
#    plt.xlabel('time (ms)')
#    plt.ylabel('DNN value')
#    plt.title('DNN: STC time courses')
#    # #axes = plt.gca()
#    # #axes.set_xlim([xmin,xmax])
#    # #axes.set_ylim([0,1.2])
#    fig = plt.gcf()
#    fig.set_size_inches(8, 6)
#    plt.savefig(fname_stc_DNN_resultfig, dpi=300)
#    #plt.show()
    
    ## plot LSTM STC results
    #plt.figure(3)
    #plt.plot(1e3 * stc_LSTM.times, stc_LSTM.data[::100, :].T)
    #plt.xlabel('time (ms)')
    #plt.ylabel('LSTM value')
    #plt.title('LSTM: STC time courses')
    ## #axes = plt.gca()
    ## #axes.set_xlim([xmin,xmax])
    ## #axes.set_ylim([0,1.2])
    #fig = plt.gcf()
    #fig.set_size_inches(8, 6)
    #plt.savefig(fname_stc_LSTM_resultfig, dpi=300)
    ##plt.show()
    
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
    
    
    # plot MCMC CTRL STC results
    #    plt.figure(6)
    #    plt.plot(1e3 * stc_MCMC_CTRL.times, stc_MCMC_CTRL.data[::100, :].T)
    #    plt.xlabel('time (ms)')
    #    plt.ylabel('MCMC CTRL value')
    #    plt.title('MCMC CTRL: STC time courses')
    #    # #axes = plt.gca()
    #    # #axes.set_xlim([xmin,xmax])
    #    # #axes.set_ylim([0,1.2])
    #    fig = plt.gcf()
    #    fig.set_size_inches(8, 6)
    #    plt.savefig(fname_stc_MCMC_CTRL_resultfig, dpi=300)
    #    #plt.show()
