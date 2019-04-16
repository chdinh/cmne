#**
# @file     bio_mne_LSTM.py
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

import datetime

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.callbacks import TensorBoard

from modules.biosettings import BioSettings
from modules.biodata import BioData
from modules.biodata import generate_dnn_batches, generate_dnn_eval_batches, generate_lstm_batches, generate_lstm_eval_batches

###################################################################################################
# Configuration
###################################################################################################
# DNN Training
dnn_minibatch_size = 30
dnn_steps_per_ep = 20 #30
dnn_num_epochs = 250 #300

# LSTM Training
minibatch_size = 30
steps_per_ep = 20 #30
num_epochs = 250 #300
lstm_look_back = 80 #40 #100 #40

num_units = 1280 #320 #640

###################################################################################################
# The Script
###################################################################################################

event_id, tmin, tmax = 1, -0.2, 0.8
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

# Divide Epoch data in training and test set (cross validation will be done later)
test_size = round(0.15 * data.num_epochs())

# select random epochs for test set
test_idx = random.sample(range(data.num_epochs()), test_size)
test_epoch_data = data.epochs()[test_idx]

train_epoch_data = data.epochs()
train_epoch_data.drop(test_idx)

print(">>>> Total Number of Epochs: ",data.num_epochs())
print(">>>> Test Size: ",len(test_epoch_data))
print(">>>> Train Size: ",len(train_epoch_data))

num_samples = data.epochs()[0]._data.shape[2]
print(">>>> num_samples: ",num_samples)

num_ch_features_in = data.epochs().info['nchan']
num_src_features_in = data.inv_op()['nsource']
num_src_labels_out = num_src_features_in

# TensorBoard Callback
tbCallBack = TensorBoard(log_dir=settings.tb_log_dir(), histogram_freq=1, write_graph=True, write_images=True)

#history_losses = []

###################################################################################################
# DNN Model
dnn_data_generator = generate_dnn_batches(epochs=train_epoch_data, inverse_operator=data.inv_op(), lambda2=data.lambda2(), method=data.method(), batch_size=dnn_minibatch_size)

dnn_evaluate_generator = generate_dnn_eval_batches(epochs=test_epoch_data, inverse_operator=data.inv_op(), lambda2=data.lambda2(), method=data.method())

# # Debug
# for f,l in dnn_evaluate_generator:
#     print('f.shape: ', f.shape)
#     print('l.shape: ', l.shape)
#     print(f)

# create model
dnn_model = Sequential()
dnn_model.add(Dense(num_src_labels_out, activation='linear', input_shape=(num_ch_features_in,)))

# compile the model
#sgd = optimizers.SGD(lr=learning_rate)
dnn_model.compile(loss='mean_squared_error', optimizer='rmsprop')#optimizer = sgd)

# Train - fit the model :D
dnn_fitting_result = dnn_model.fit_generator(dnn_data_generator, steps_per_epoch=dnn_steps_per_ep, epochs=dnn_num_epochs, verbose=1, callbacks=[tbCallBack], validation_data=None, class_weight=None, workers=1)

# Evaluate - evaluate the model
dnn_evaluation_result = dnn_model.evaluate_generator(dnn_evaluate_generator, steps=len(test_epoch_data))

# # # let's get some predictions
# # test_predict = model.predict(test_features)

###################################################################################################
# LSTM Model
#time_steps_in = lstm_look_back
# create the LSTM Data Generator
lstm_data_generator = generate_lstm_batches(epochs=train_epoch_data, inverse_operator=data.inv_op(), lambda2=data.lambda2(), method=data.method(), look_back=lstm_look_back, batch_size=minibatch_size)

lstm_evaluate_generator = generate_lstm_eval_batches(epochs=test_epoch_data, inverse_operator=data.inv_op(), lambda2=data.lambda2(), method=data.method(), look_back=lstm_look_back)

# # Debug
# for f,l in lstm_evaluate_generator:
#     print('f.shape: ', f.shape)
#     print('l.shape: ', l.shape)
#     print(f)

# create LSTM model
lstm_model = None
lstm_model = Sequential()
lstm_model.add(LSTM(num_units, activation='tanh', return_sequences=False, input_shape=(lstm_look_back,num_src_features_in)))
lstm_model.add(Dense(num_src_labels_out, activation='linear'))

# compile the model
lstm_model.compile(loss = 'mean_squared_error', optimizer = 'adam')

# Train - fit the lstm_model :D
lstm_fitting_result = lstm_model.fit_generator(lstm_data_generator, steps_per_epoch = steps_per_ep, epochs = num_epochs, verbose=1, callbacks=[tbCallBack], validation_data=None, class_weight=None, workers=1)

# Evaluate - evaluate the model
lstm_evaluation_result = lstm_model.evaluate_generator(lstm_evaluate_generator, steps=len(test_epoch_data))#takes a long time

# # # let's get some predictions
# # test_predict = model.predict(test_features)

###################################################################################################
# Save Results
###################################################################################################
date_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
print(">>>> Safe Results <<<<")
fname_dnn_model = settings.repo_path() + 'Results/Models/DNN_Model_Opt_4_in_' + str(num_ch_features_in) + '_out_' + str(num_src_labels_out) + '_' + date_stamp + '.h5'
fname_dnn_training_loss = settings.repo_path() + 'Results/Training/DNN_Loss_Opt_4_in_' + str(num_ch_features_in) + '_out_' + str(num_src_labels_out) + '_' + date_stamp + '.txt'
fname_dnn_resultfig = settings.repo_path() + 'Results/img/DNN_Loss_Opt_4_in_' + str(num_ch_features_in) + '_out_' + str(num_src_labels_out) + '_' + date_stamp + '.png'

fname_lstm_model = settings.repo_path() + 'Results/Models/LSTM_Model_Opt_4_nu_' + str(num_units) +'_lb_' + str(lstm_look_back) + '_' + date_stamp + '.h5'
fname_lstm_training_loss = settings.repo_path() + 'Results/Training/LSTM_Loss_Opt_4_nu_' + str(num_units) +'_lb_' + str(lstm_look_back) + '_' + date_stamp + '.txt'
fname_lstm_resultfig = settings.repo_path() + 'Results/img/LSTM_Loss_Opt_4_nu_' + str(num_units) +'_lb_' + str(lstm_look_back) + '_' + date_stamp + '.png'

# Save models
print(">>>> Safe Models <<<<")
dnn_model.save(fname_dnn_model)
lstm_model.save(fname_lstm_model)

# Save losses
print(">>>> Safe Losses <<<<")
np.savetxt(fname_dnn_training_loss, dnn_fitting_result.history['loss'])
np.savetxt(fname_lstm_training_loss, lstm_fitting_result.history['loss'])

# Save figures
print(">>>> Safe Figures <<<<")
# plot DNN training results
plt.figure(1)
#    plt.subplot(211)
plt.plot(dnn_fitting_result.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('DNN: Training Epoch vs. Loss\n(Evaluation Result = %.4f)' %dnn_evaluation_result)
# #axes = plt.gca()
# #axes.set_xlim([xmin,xmax])
# #axes.set_ylim([0,1.2])
fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.savefig(fname_dnn_resultfig, dpi=300)
#plt.show()

# plot LSTM training results
plt.figure(2)
plt.plot(lstm_fitting_result.history['loss'])
plt.xlabel('Minibatch number')
plt.ylabel('Loss')
plt.title('LSTM: Minibatch vs. Loss\n(Evaluation Result = %.4f)' %lstm_evaluation_result)
# #axes = plt.gca()
# #axes.set_xlim([xmin,xmax])
# #axes.set_ylim([0,1.2])
fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.savefig(fname_lstm_resultfig, dpi=300)
#plt.show()
