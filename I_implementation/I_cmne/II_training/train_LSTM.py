#**
# @file     train_LSTM.py
# @author   Christoph Dinh <chdinh@nmr.mgh.harvard.edu>
# @version  1.0
# @date     April, 2018
#
# @section  LICENSE
#
# Copyright (C) 2018, Christoph Dinh. All rights reserved.
#
# @brief    Train the LSTM model
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

from helpers.cmnesettings import CMNESettings
from helpers.cmnedata import CMNEData
from helpers.cmnedata import generate_lstm_batches

def train_LSTM(settings, data):
    ###################################################################################################
    # Configuration
    ###################################################################################################
    
    minibatch_size = 30
    steps_per_ep = 25 #30
    num_epochs = 100 #300
    lstm_look_back = 80 #40 #100 #40
    
    num_unit = 1280
    
    
    ###################################################################################################
    # The Script
    ###################################################################################################
    
    num_features_in = data.inv_op()['nsource']
    num_labels_out = num_features_in
    
    # TensorBoard Callback
    tbCallBack = TensorBoard(log_dir=settings.tb_log_dir(), histogram_freq=1, write_graph=True, write_images=True)
    
    #time_steps_in = lstm_look_back
    # create the Data Generator
    data_generator = generate_lstm_batches(epochs=data.train_epochs(), inverse_operator=data.inv_op(), lambda2=data.lambda2(), method=data.method(), look_back=lstm_look_back, batch_size=minibatch_size)
    
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
    
    fname_model = settings.repo_path() + 'Results/Models/Model_Opt_5_sim_' + settings.modality() + '_nu_' + str(num_unit) +'_lb_' + str(lstm_look_back) + '_' + date_stamp + '.h5'
    fname_training_loss = settings.repo_path() + 'Results/Training/Loss_Opt_5_sim_' + settings.modality() + '_nu_' + str(num_unit) +'_lb_' + str(lstm_look_back) + '_' + date_stamp + '.txt'
    fname_resultfig = settings.repo_path() + 'Results/img/Loss_Opt_5_sim_' + settings.modality() + '_nu_' + str(num_unit) +'_lb_' + str(lstm_look_back) + '_' + date_stamp + '.png'
    
    history_losses = fitting_result.history['loss']
    
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
