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
sys.path.append('D:/Users/Christoph/Git/cmne/I_implementation/helpers')
sys.path.append('D:/Users/Christoph/Git/cmne/I_implementation/I_cmne/I_hyperparameter_evaluation')
from cmnesettings import CMNESettings
from cmnedata import CMNEData

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
#training_settings = {'minibatch_size': 30, 'steps_per_ep': 20, 'num_epochs': 10, 'lstm_look_backs': [10,80,160], 'num_units': [640]}#[10,20,40,80,160,320,640,1280]}
#eval_hyper(data_settings, data, training_settings)

# look back k
#training_settings = {'minibatch_size': 30, 'steps_per_ep': 20, 'num_epochs': 250, 'lstm_look_backs': [10,20,40,80,160,320,480], 'num_units': [1280]}
#eval_hyper(data_settings, data, training_settings)

# future steps
training_settings = {'minibatch_size': 30, 'steps_per_ep': 20, 'num_epochs': 250, 'lstm_look_backs': [10,80,160], 'num_units': [10,320,1280], 'future_steps': [1, 4, 8]}
training_settings = {'minibatch_size': 30, 'steps_per_ep': 20, 'num_epochs': 25, 'lstm_look_backs': [10,20], 'num_units': [10,20], 'future_steps': [1, 4]}

eval_hyper(data_settings, data, training_settings)

# topology
#eval_topo_multi_hidden(data_settings, data, training_settings)
