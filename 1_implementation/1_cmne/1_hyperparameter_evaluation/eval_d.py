#**
# @file     Option_3b_bio_mne_LSTM_single_estimation_nu_eval.py
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import datetime

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.callbacks import TensorBoard

from modules.biosettings import BioSettings
from modules.biodata import BioData
from modules.biodata import generate_lstm_batches

###################################################################################################
# Configuration
###################################################################################################

minibatch_size = 30
steps_per_ep = 20 #30
num_epochs = 250 #300
lstm_look_back = 80 #40 #100 #40

num_units = [10,20,40,80,160,320,640,1280]

###################################################################################################
# The Script
###################################################################################################

event_id, tmin, tmax = 1, 0.0, 0.5
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
settings = BioSettings(repo_path='D:/Users/Christoph/GitHub/bio/', data_path='D:/Data/MEG/jgs/170505/processed/')

data = BioData(bio_settings=settings)
data.load_data(event_id=event_id, tmin=tmin, tmax=tmax)

num_features_in = data.inv_op()['nsource']
num_labels_out = num_features_in

# TensorBoard Callback
tbCallBack = TensorBoard(log_dir=settings.tb_log_dir(), histogram_freq=1, write_graph=True, write_images=True)

history_losses = []
for num_unit in num_units:
    print(">>>> Starting next iteration (Number of Units back = %d) <<<<\n" % (num_unit))
    #time_steps_in = lstm_look_back
    # create the Data Generator
    data_generator = generate_lstm_batches(epochs=data.epochs(), inverse_operator=data.inv_op(), lambda2=data.lambda2(), method=data.method(), look_back=lstm_look_back, batch_size=minibatch_size)

    # create LSTM model
    model = None
    model = Sequential()
    model.add(LSTM(num_unit, activation='tanh', return_sequences=False, input_shape=(lstm_look_back,num_features_in)))
    model.add(Dense(num_labels_out, activation='linear'))

    # compile the model
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')

    # Train - fit the model :D
    fitting_result = model.fit_generator(data_generator, steps_per_epoch = steps_per_ep, epochs = num_epochs, verbose=1, callbacks=[tbCallBack], validation_data=None, class_weight=None, workers=1)

    # # let's get some predictions
    # test_predict = model.predict(test_features)

    ###################################################################################################
    # Save Results
    ###################################################################################################
    date_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')

    fname_model = settings.repo_path() + 'Results/Models/Model_Opt_3b_' + settings.modality() + '_nu_' + str(num_unit) +'_lb_' + str(lstm_look_back) + '_' + date_stamp + '.h5'
    fname_training_loss = settings.repo_path() + 'Results/Training/Loss_Opt_3b_' + settings.modality() + '_nu_' + str(num_unit) +'_lb_' + str(lstm_look_back) + '_' + date_stamp + '.txt'
    fname_resultfig = settings.repo_path() + 'Results/img/Loss_Opt_3b_' + settings.modality() + '_nu_' + str(num_unit) +'_lb_' + str(lstm_look_back) + '_' + date_stamp + '.png'

    history_losses.append(fitting_result.history['loss'])

    # save model
    model.save(fname_model)

    # # plot the data
    # print('Testing Prediction',test_predict)
    # print('Testing Reference',test_labels)

    # save loss
    np.savetxt(fname_training_loss, fitting_result.history['loss'])

    # save plot the data
    plt.figure()
    plt.plot(fitting_result.history['loss'])
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
    #axes = plt.gca()
    #axes.set_xlim([xmin,xmax])
    #axes.set_ylim([0,1.2])
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.savefig(fname_resultfig, dpi=300)
    #plt.show()


# save overall plot
date_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
fname_overall_fig = settings.repo_path() + 'Results/img/Loss_Opt_3b_' + settings.modality() + '_overall_nu_' + date_stamp + '.png'
plt.figure()
plt.xlabel('Minibatch number')
plt.ylabel('Loss')
plt.title('Minibatch run vs. Training loss')
for i in range(len(history_losses)):
    plt.plot(history_losses[i], label='NU %s'%num_units[i])
plt.legend()
#axes = plt.gca()
#axes.set_xlim([xmin,xmax])
#axes.set_ylim([0,1.2])
fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.savefig(fname_overall_fig, dpi=300)
#plt.show()
