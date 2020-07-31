#**
# @file     1_run_hyper_evaluation.py
# @author   Christoph Dinh <christoph.dinh@mne-cpp.org>
# @version  1.0
# @date     April, 2019
#
# @section  LICENSE
#
# Copyright (C) 2019, Christoph Dinh. All rights reserved.
#
# @brief    Run hyper evaluation
#**

#%% Imports
import os
import sys

import config_md as cfg

sys.path.append(cfg.repo_path + 'I_implementation/I_cmne/II_training') #Add relative path to include modules
sys.path.append(cfg.repo_path + 'I_implementation/helpers')
sys.path.append(cfg.repo_path + 'I_implementation/I_cmne/I_hyperparameter_evaluation')

from helpers.cmnesettings import CMNESettings
from helpers.cmnedata import CMNEData

import random

from eval_hyper import eval_hyper
from eval_topo_multi_hidden import eval_topo_multi_hidden


#%% Data Settings
data_settings = CMNESettings(   repo_path=cfg.repo_path, data_path=cfg.data_path,
                                fname_raw=cfg.fname_raw,
                                fname_inv=cfg.fname_inv,
                                fname_eve=cfg.fname_eve,
                                fname_test_idcs=cfg.fname_test_idcs,
                                meg_and_eeg=cfg.meg_and_eeg
                            )


#%% Data
event_id, tmin, tmax = 1, -0.2, 0.5
train_percentage = 0.85
cross_validation_percentage = 0.9 #0.85

data = CMNEData(cmne_settings=data_settings)
data.load_data(event_id=event_id, tmin=tmin, tmax=tmax, train_percentage=train_percentage)

# num units d
#training_settings = {'minibatch_size': 30, 'steps_per_ep': 20, 'num_epochs': 10, 'lstm_look_backs': [10,80,160], 'num_units': [640]}#[10,20,40,80,160,320,640,1280]}
#eval_hyper(data_settings, data, training_settings)

# look back k
#training_settings = {'minibatch_size': 30, 'steps_per_ep': 20, 'num_epochs': 250, 'lstm_look_backs': [10,20,40,80,160,320,480], 'num_units': [1280]}
#eval_hyper(data_settings, data, training_settings)

# future steps
training_settings = {'minibatch_size': 30, 'steps_per_ep': 20, 'num_epochs': 250, 'lstm_look_backs': [10], 'num_units': [10], 'future_steps': [1]}
#training_settings = {'minibatch_size': 30, 'steps_per_ep': 20, 'num_epochs': 250, 'lstm_look_backs': [10,80,160], 'num_units': [10,160,1280], 'future_steps': [1]}
#training_settings = {'minibatch_size': 30, 'steps_per_ep': 20, 'num_epochs': 250, 'lstm_look_backs': [10,80,160], 'num_units': [10,320,1280], 'future_steps': [1, 4, 8]}
#training_settings = {'minibatch_size': 30, 'steps_per_ep': 20, 'num_epochs': 25, 'lstm_look_backs': [10,20], 'num_units': [10,20], 'future_steps': [1, 4]}

num_train_idcs = len(data.train_idcs())

num_cross_iterations = 10
fname_cross_validation_idcs_prefix = 'assr_270LP_fs900_cross_idcs_job1_it_'

#%% Evaluate

for iteration in range(num_cross_iterations):
    fname_cross_validation_idcs = data_settings.data_path() + fname_cross_validation_idcs_prefix + str(iteration) + '.txt'

    whole_list = list(range(num_train_idcs))

    if os.path.isfile(fname_cross_validation_idcs):
        cross_validation_train_idcs = []
        with open(fname_cross_validation_idcs, "r") as f:
            for line in f:
                cross_validation_train_idcs.append(int(line.strip()))
        cross_validation_test_idcs = [item for item in whole_list if item not in cross_validation_train_idcs]
    else:
        #split train and test
        random.seed(42)
        cross_validation_train_idcs = random.sample(range(num_train_idcs), (int)(num_train_idcs*cross_validation_percentage))
        cross_validation_test_idcs = [item for item in whole_list if item not in cross_validation_train_idcs]
        with open(fname_cross_validation_idcs, "w") as f:
            for idx in cross_validation_train_idcs:
                f.write(str(idx) +"\n")

    eval_hyper(data_settings, data, training_settings, idx=cross_validation_train_idcs, idx_test=cross_validation_test_idcs)

    # topology
    #eval_topo_multi_hidden(data_settings, data, training_settings)
