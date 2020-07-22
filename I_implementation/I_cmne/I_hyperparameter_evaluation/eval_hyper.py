#**
# @file     eval_d.py
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

from keras.callbacks import TensorBoard

from cmnesettings import CMNESettings
from cmnedata import CMNEData
from cmnedata import generate_lstm_batches
from cmnedata import generate_lstm_future_batches


def eval_hyper(data_settings, data, training_settings, idx=None, idx_test=None):

    ###################################################################################################
    # The Script
    ###################################################################################################

    num_features_in = data.inv_op()['nsource']
    num_labels_out = num_features_in

    num_tests = len(idx_test)

    # TensorBoard Callback
    tbCallBack = TensorBoard(log_dir=data_settings.tb_log_dir(), histogram_freq=1, write_graph=True, write_images=True)

    history_losses = []
    
    for lstm_look_back in training_settings['lstm_look_backs']:
        for num_unit in training_settings['num_units']:
            if 'future_steps' in training_settings:
                for fs in training_settings['future_steps']:
                    print(">>>> Starting next iteration (Future steps = %d) <<<<\n" % (fs))
                    
                    num_labels_out = num_features_in * fs
                    
                    #time_steps_in = lstm_look_back
                    # create the Data Generator
                    #data_generator = generate_lstm_future_batches(epochs=data.epochs(idx=idx), inverse_operator=data.inv_op(), lambda2=data.lambda2(), method=data.method(), look_back=lstm_look_back, future_steps=fs, batch_size=training_settings['minibatch_size'])
                    data_generator = generate_lstm_future_batches(epochs=data.train_epochs(idx=idx), inverse_operator=data.inv_op(), lambda2=data.lambda2(), method=data.method(), look_back=lstm_look_back, future_steps=fs, batch_size=training_settings['minibatch_size'])

                    test_data_generator = generate_lstm_future_batches(epochs=data.train_epochs(idx=idx_test), inverse_operator=data.inv_op(), lambda2=data.lambda2(), method=data.method(), look_back=lstm_look_back, future_steps=fs, batch_size=training_settings['minibatch_size'])

                    # create LSTM model
                    model = None
                    model = Sequential()
                    model.add(LSTM(num_unit, activation='tanh', return_sequences=False, input_shape=(lstm_look_back,num_features_in)))
                    model.add(Dense(num_labels_out, activation='linear'))
                
                    # compile the model
                    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
                
                    # Train - fit the model :D
                    fitting_result = model.fit_generator(data_generator, steps_per_epoch=training_settings['steps_per_ep'], epochs=training_settings['num_epochs'], verbose=1, validation_data=test_data_generator, validation_steps=num_tests, class_weight=None, workers=1, use_multiprocessing=True) #, callbacks=[tbCallBack])
                
                    # # let's get some predictions
                    # test_predict = model.predict(test_features)
                
                    history_losses.append(fitting_result.history['loss'])
                    
                    ###################################################################################################
                    # Save Results
                    ###################################################################################################
                    date_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
                                
                    # save model
                    fname_model = data_settings.results_cmne_dir() + '/I_models/eval_hyper_model_' + data_settings.modality() + '_fs_' + str(fs) + '_nu_' + str(num_unit) + '_lb_' + str(lstm_look_back) + '_' + date_stamp + '.h5'
                    model.save(fname_model)
                
                    # # plot the data
                    # print('Testing Prediction',test_predict)
                    # print('Testing Reference',test_labels)
                
                    # save loss
                    fname_training_loss = data_settings.results_cmne_dir() + '/III_training/eval_hyper_loss_' + data_settings.modality() + '_fs_' + str(fs) + '_nu_' + str(num_unit) +'_lb_' + str(lstm_look_back) + '_' + date_stamp + '.txt'
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
                    
                    fname_resultfig = data_settings.results_cmne_dir() + '/IV_img/eval_hyper_loss_' + data_settings.modality() + '_fs_' + str(fs) + '_nu_' + str(num_unit) +'_lb_' + str(lstm_look_back) + '_' + date_stamp + '.png'
                    plt.savefig(fname_resultfig, dpi=300)
                    #plt.show()
            else:
                print("\n\n>>>> Starting next iteration (Look back = %d, Number of Units back = %d) <<<<\n" % (lstm_look_back, num_unit))
                #time_steps_in = lstm_look_back
                # create the Data Generator
                data_generator = generate_lstm_batches(epochs=data.epochs(), inverse_operator=data.inv_op(), \
                                                       lambda2=data.lambda2(), method=data.method(), look_back=lstm_look_back, \
                                                       batch_size=training_settings['minibatch_size'])
    
                # create LSTM model
                model = None
                model = Sequential()
                model.add(LSTM(num_unit, activation='tanh', return_sequences=False, input_shape=(lstm_look_back,num_features_in)))
                model.add(Dense(num_labels_out, activation='linear'))
    
                # compile the model
                model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    
                # Train - fit the model :D
                fitting_result = model.fit_generator(data_generator, steps_per_epoch=training_settings['steps_per_ep'], \
                                                     epochs=training_settings['num_epochs'], verbose=1, validation_data=None, \
                                                     class_weight=None, workers=1) #, callbacks=[tbCallBack])
    
                # # let's get some predictions
                # test_predict = model.predict(test_features)

                history_losses.append(fitting_result.history['loss'])
    
                ###################################################################################################
                # Save Results
                ###################################################################################################
                date_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    
                # save model
                fname_model = data_settings.results_cmne_dir() + '/I_models/eval_hyper_model_' + data_settings.modality() + \
                '_nu_' + str(num_unit) +'_lb_' + str(lstm_look_back) + '_' + date_stamp + '.h5'
                model.save(fname_model)
    
                # # plot the data
                # print('Testing Prediction',test_predict)
                # print('Testing Reference',test_labels)
    
                # save loss
                fname_training_loss = data_settings.results_cmne_dir() + '/III_training/eval_hyper_loss_' + data_settings.modality() + '_nu_' + str(num_unit) +'_lb_' + str(lstm_look_back) + '_' + date_stamp + '.txt'
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
                
                fname_resultfig = data_settings.results_cmne_dir() + '/IV_img/eval_hyper_loss_' + data_settings.modality() + '_nu_' + str(num_unit) +'_lb_' + str(lstm_look_back) + '_' + date_stamp + '.png'
                plt.savefig(fname_resultfig, dpi=300)
                #plt.show()


    # save overall plot
    date_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    
    plt.figure()
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
    c=0
    for k in range(len(training_settings['num_units'])):
        for l in range(len(training_settings['num_units'])):
            for m in range(len(training_settings['num_units'])):
                leg = 'num_units = ' + str(training_settings['num_units'][k]) + \
                'lstm_look_backs = ' + str(training_settings['lstm_look_backs'][l]) + \
                'future_steps = ' + str(training_settings['future_steps'][m])
                plt.plot(history_losses[c], label=leg)
                c = c+1
                                
    plt.legend()
    #axes = plt.gca()
    #axes.set_xlim([xmin,xmax])
    #axes.set_ylim([0,1.2])
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    
    fname_overall_fig = data_settings.results_cmne_dir() + '/IV_img/eval_hyper_loss_' + data_settings.modality() + '_overall_nu_' + date_stamp + '.png'
    plt.savefig(fname_overall_fig, dpi=300)
    plt.show()
