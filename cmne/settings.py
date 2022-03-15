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
        _result_path: Result path
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
    def __init__(self, result_path='D:/Users/Christoph/Git/cmne/', data_path='D:/Data/MEG/jgs/170505/processed/', 
                 fname_raw='assr_270LP_fs900_raw.fif',
                 fname_inv='assr_270LP_fs900_raw-ico-4-meg-eeg-inv.fif',
                 fname_eve='assr_270LP_fs900_raw-eve.fif',
                 fname_test_idcs='assr_270LP_fs900_raw-test-idcs.txt',
                 meg_and_eeg=True):
        """Return a new CMNEConfiguration object."""
        
        self._meg_and_eeg = meg_and_eeg
        if meg_and_eeg:
            self._modality = 'meg-eeg'
        else: 
            self._modality = 'meg'
        
        self._result_path = result_path
        self._data_path = data_path
        
        self._fname_raw = self._data_path + fname_raw
        self._fname_inv = self._data_path + fname_inv
        self._fname_event = self._data_path + fname_eve
        self._fname_test_idcs = self._data_path + fname_test_idcs
                
        # Create directories for cmne results
        if not os.path.isdir(self._result_path + 'III_results'):
            os.mkdir(self._result_path + 'III_results')
            
        # CMNE
        if not os.path.isdir(self._result_path + 'III_results/I_cmne'):
            os.mkdir(self._result_path + 'III_results/I_cmne') 
        self._results_cmne_dir = self._result_path + 'III_results/I_cmne'
        if not os.path.isdir(self._result_path + 'III_results/I_cmne/I_models'):
            os.mkdir(self._result_path + 'III_results/I_cmne/I_models')
        if not os.path.isdir(self._result_path + 'III_results/I_cmne/II_logs'):
            os.mkdir(self._result_path + 'III_results/I_cmne/II_logs')
        if not os.path.isdir(self._result_path + 'III_results/I_cmne/III_training'):
            os.mkdir(self._result_path + 'III_results/I_cmne/III_training')
        if not os.path.isdir(self._result_path + 'III_results/I_cmne/III_training'):
            os.mkdir(self._result_path + 'III_results/I_cmne/III_training')
        if not os.path.isdir(self._result_path + 'III_results/I_cmne/IV_img'):
            os.mkdir(self._result_path + 'III_results/I_cmne/IV_img')
            
        
        # II_source_estimation
        if not os.path.isdir(self._result_path + 'III_results/II_source_estimation'):    
            os.mkdir(self._result_path + 'III_results/II_source_estimation')
                    
        self._tb_log_dir = self._result_path + 'III_results/I_cmne/II_logs'
    
    
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
        
    def result_path(self):
        """
        Returns the repository path
        """
        return self._result_path
            
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
    
    def results_cmne_dir(self):
        """
        Returns the cmne results dir
        """
        return self._results_cmne_dir
    
    