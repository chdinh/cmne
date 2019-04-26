#**
# @file     Option_3b_bio_mne_LSTM_single_estimation_mh_eval.py
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
import sys
sys.path.append("../..") #Add relative path to include modules

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import datetime

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#from keras.callbacks import TensorBoard

from helpers.cmnesettings import CMNESettings
from helpers.cmnedata import CMNEData
from helpers.cmnedata import generate_lstm_batches

def eval_topo_multi_hidden(data_settings, data, training_settings):
    ###################################################################################################
    # Configuration
    ###################################################################################################

    lstm_look_back = 80 #40 #100 #40
    
    num_units = [[320,320,320],[640,640,640],[1280,1280,1280]]


    ###################################################################################################
    # The Script
    ###################################################################################################
    
    num_features_in = data.inv_op()['nsource']
    num_labels_out = num_features_in
    
    # TensorBoard Callback
    # tbCallBack = TensorBoard(log_dir=data_settings.tb_log_dir(), histogram_freq=1, write_graph=True, write_images=True)
    
    history_losses = []
    for num_unit in num_units:
        print('>>>> Starting next iteration (Number of Units = [%s]) <<<<\n'%', '.join(map(str, num_unit)))
            
        #time_steps_in = lstm_look_back
        # create the Data Generator
        data_generator = generate_lstm_batches(epochs=data.epochs(), inverse_operator=data.inv_op(), lambda2=data.lambda2(), method=data.method(), look_back=lstm_look_back, batch_size=training_settings['minibatch_size'])
    
        # create LSTM model
        model = None
        model = Sequential()
        model.add(LSTM(num_unit[0], activation='tanh', return_sequences=True, input_shape=(lstm_look_back,num_features_in)))
        if len(num_unit) > 2:
            model.add(LSTM(num_unit[1], activation='tanh', return_sequences=True))
            model.add(LSTM(num_unit[2], activation='tanh'))
        elif len(num_unit) > 1:
            model.add(LSTM(num_unit[1], activation='tanh'))
        model.add(Dense(num_labels_out, activation='linear'))
    
        # compile the model
        model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    
        # Train - fit the model :D
        fitting_result = model.fit_generator(data_generator, steps_per_epoch=training_settings['steps_per_ep'], epochs=training_settings['num_epochs'], verbose=1, validation_data=None, class_weight=None, workers=1)# callbacks=[tbCallBack], validation_data=None, class_weight=None, workers=1)
    
        # # let's get some predictions
        # test_predict = model.predict(test_features)
    
        ###################################################################################################
        # Save Results
        ###################################################################################################
        date_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    
        fname_model = data_settings.repo_path() + 'Results/Models/Model_Opt_3b_' + data_settings.modality() + '_mh_%s'%'_'.join(map(str, num_unit)) + '_lb_' + str(lstm_look_back) + '_' + date_stamp + '.h5'
        fname_training_loss = data_settings.repo_path() + 'Results/Training/Loss_Opt_3b_' + data_settings.modality() + '_mh_%s'%'_'.join(map(str, num_unit)) + '_lb_' + str(lstm_look_back) + '_' + date_stamp + '.txt'
        fname_resultfig = data_settings.repo_path() + 'Results/img/Loss_Opt_3b_' + data_settings.modality() + '_mh_%s'%'_'.join(map(str, num_unit)) + '_lb_' + str(lstm_look_back) + '_' + date_stamp + '.png'
    
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
    fname_overall_fig = data_settings.repo_path() + 'Results/img/Loss_Opt_3b_' + data_settings.modality() + '_overall_mh_' + date_stamp + '.png'
    plt.figure()
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
    for i in range(len(history_losses)):
        plt.plot(history_losses[i], label='NU [%s]'%', '.join(map(str, num_units[i])))
        
    plt.legend()
    #axes = plt.gca()
    #axes.set_xlim([xmin,xmax])
    #axes.set_ylim([0,1.2])
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.savefig(fname_overall_fig, dpi=300)
    #plt.show()
