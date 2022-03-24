#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# @authors   Christoph Dinh <christoph.dinh@brain-link.de>
#            John G Samuelson <johnsam@mit.edu>
# @version   1.0
# @date      September, 2017
# @copyright Copyright (c) 2017-2022, authors of CMNE. All rights reserved.
# @license   MIT
# @brief     Settings
# ---------------------------------------------------------------------------

#%%
import os

###################################################################################################
# Settings class
###################################################################################################

class Settings(object):
    """the settings object

    Attributes:
        _data_path: Path to the MEG data.
        _results_path: Result path
        _fname_raw: Raw file
        _fname_inv: Inverse file
        _fname_event: Event file
        _fname_test_idcs: Test indeces file
        _tb_log_dir: Tensor board directory.
        _modality: The selected modality meg or meg-eeg
    """
    
    _large_memory = True # If all epoch data fit into the Ram
    
    ###############################################################################################
    # Constructor
    ###############################################################################################
    def __init__(self, results_path='/workspace/results/', data_path='/workspace/data/', 
                 fname_raw='assr_270LP_fs900_raw.fif',
                 fname_inv='assr_270LP_fs900_raw-ico-4-meg-eeg-inv.fif',
                 fname_eve='assr_270LP_fs900_raw-eve.fif',
                 fname_test_idcs='assr_270LP_fs900_raw-test-idcs.txt',
                 meg_and_eeg=True):
        """Return a new Settings object."""
        
        self._meg_and_eeg = meg_and_eeg
        if meg_and_eeg:
            self._modality = 'meg-eeg'
        else: 
            self._modality = 'meg'
        
        self._results_path = results_path
        self._data_path = data_path
        
        self._fname_raw = self._data_path + fname_raw
        self._fname_inv = self._data_path + fname_inv
        self._fname_event = self._data_path + fname_eve
        self._fname_test_idcs = self._data_path + fname_test_idcs
        
        # CMNE
        if not os.path.isdir(self._results_path + 'cmne'):
            os.mkdir(self._results_path + 'cmne') 
        self._results_cmne_path = self._results_path + 'cmne'

        if not os.path.isdir(self._results_path + 'cmne/models'):
            os.mkdir(self._results_path + 'cmne/models')
        self._results_models_path = self._results_path + 'cmne/models'

        if not os.path.isdir(self._results_path + 'cmne/logs'):
            os.mkdir(self._results_path + 'cmne/logs')
        self._results_logs_path = self._results_path + 'cmne/logs'
        self._tb_log_dir = self._results_path + 'cmne/logs'

        if not os.path.isdir(self._results_path + 'cmne/training'):
            os.mkdir(self._results_path + 'cmne/training')
        self._results_training_path = self._results_path + 'cmne/training'

        if not os.path.isdir(self._results_path + 'cmne/img'):
            os.mkdir(self._results_path + 'cmne/img')
        self._results_img_path = self._results_path + 'cmne/img'
            
        # source_estimation
        if not os.path.isdir(self._results_path + 'source_estimations'):    
            os.mkdir(self._results_path + 'source_estimations')
        self._results_stc_path = self._results_path + 'source_estimations'
                    
    
    
    ###############################################################################################
    # Getters and setters
    ###############################################################################################
    def large_memory(self):
        """
        Returns whether large memory configuration is choosen
        """
        return self._large_memory
    
    def meg_and_eeg(self):
        """
        Returns the selected modality
        """
        return self._meg_and_eeg
        
    def modality(self):
        """
        Returns the selected modality
        """
        return self._modality
    
    def data_path(self):
        """
        Returns the data path
        """
        return self._data_path
        
    def results_path(self):
        """
        Returns the repository path
        """
        return self._results_path
            
    def fname_raw(self):
        """
        Returns the raw file name
        """
        return self._fname_raw
        
    def fname_inv(self):
        """
        Returns the inverse operator file name
        """
        return self._fname_inv
    
    def fname_event(self):
        """
        Returns the event file name
        """
        return self._fname_event
    
    def fname_test_idcs(self):
        """
        Returns the file name containing the test indeces
        """
        return self._fname_test_idcs
    
    def tb_log_dir(self):
        """
        Returns the tensor board log dir
        """
        return self._tb_log_dir
    
    def results_cmne_path(self):
        """
        Returns the cmne results path
        """
        return self._results_cmne_path
    
    def results_models_path(self):
        """
        Returns the models results path
        """
        return self._results_models_path

    def results_logs_path(self):
        """
        Returns the logs results path
        """
        return self._results_logs_path

    def results_training_path(self):
        """
        Returns the training results path
        """
        return self._results_training_path

    def results_img_path(self):
        """
        Returns the img results path
        """
        return self._results_img_path

    def results_stc_path(self):
        """
        Returns the stc results path
        """
        return self._results_stc_path
