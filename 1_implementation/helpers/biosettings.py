#**
# @file     biosettings.py
# @author   Christoph Dinh <chdinh@nmr.mgh.harvard.edu>
# @version  1.0
# @date     September, 2017
#
# @section  LICENSE
#
# Copyright (C) 2017, Christoph Dinh. All rights reserved.
#
# @brief    Bio Settings
#**

#==================================================================================
#%%
import os

###################################################################################################
# BioSettings class
###################################################################################################

class BioSettings(object):
    """the settings object

    Attributes:
        _data_path: Path to the MEG data.
        _repo_path: Repository path
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
    def __init__(self, repo_path='C:/Git/bio/', data_path='Z:/MEG/jgs/170505/processed/', 
                 fname_raw='assr_270LP_fs900_raw.fif',
                 fname_inv='assr_270LP_fs900_raw-ico-4-meg-eeg-inv.fif',
                 fname_eve='assr_270LP_fs900_raw-eve.fif',
                 fname_test_idcs='assr_270LP_fs900_raw-test-idcs.txt',
                 meg_and_eeg=True):
        """Return a new BioConfiguration object."""
        
        self._meg_and_eeg = meg_and_eeg
        if meg_and_eeg:
            self._modality = 'meg-eeg'
        else: 
            self._modality = 'meg'
        
        self._repo_path = repo_path
        self._data_path = data_path
        
        self._fname_raw = self._data_path + fname_raw
        self._fname_inv = self._data_path + fname_inv
        self._fname_event = self._data_path + fname_eve
        self._fname_test_idcs = self._data_path + fname_test_idcs
        
        self._tb_log_dir = self._repo_path + 'Results/Logs'
        
        # Create directories for results
        if not os.path.isdir(self._repo_path + 'Results/Models'):
            os.mkdir(self._repo_path + 'Results/Models')
        if not os.path.isdir(self._repo_path + 'Results/Training'):
            os.mkdir(self._repo_path + 'Results/Training')
        if not os.path.isdir(self._repo_path + 'Results/STCs'):
            os.mkdir(self._repo_path + 'Results/STCs')
    
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
        
    def repo_path(self):
        """
        Returns the repository path
        """
        return self._repo_path
            
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
    