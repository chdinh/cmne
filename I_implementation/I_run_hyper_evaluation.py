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
import sys
sys.path.append("./I_cmne/I_hyperparameter_evaluation") #Add relative path to include modules

from helpers.cmnesettings import CMNESettings
from helpers.cmnedata import CMNEData

import config as cfg

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

data = CMNEData(cmne_settings=data_settings)
data.load_data(event_id=event_id, tmin=tmin, tmax=tmax)


#%% Evaluate

# num units d
training_settings = {'minibatch_size': 30, 'steps_per_ep': 20, 'num_epochs': 250, 'lstm_look_backs': [80], 'num_units': [10,20,40,80,160,320,640,1280]}
eval_hyper(data_settings, data, training_settings)

# look back k
training_settings = {'minibatch_size': 30, 'steps_per_ep': 20, 'num_epochs': 250, 'lstm_look_backs': [10,20,40,80,160,320,480], 'num_units': [1280]}
eval_hyper(data_settings, data, training_settings)

# future steps
training_settings = {'minibatch_size': 30, 'steps_per_ep': 20, 'num_epochs': 250, 'lstm_look_backs': [80], 'num_units': [640], 'future_steps': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
eval_hyper(data_settings, data, training_settings)

# topology
#eval_topo_multi_hidden(data_settings, data, training_settings)
