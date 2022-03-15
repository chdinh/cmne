#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# @authors   Christoph Dinh <christoph.dinh@brain-link.de>
#            John G Samuelson <johnsam@mit.edu>
# @version   1.0
# @date      April, 2019
# @copyright Copyright (c) 2017-2022, authors of CMNE. All rights reserved.
# @license   MIT
# @brief     Train CMNE
# ---------------------------------------------------------------------------

#%% Imports
import sys
import cmne
import config as cfg

#%% Settings
settings = cmne.Settings(result_path=cfg.result_path, data_path=cfg.data_path,
                    fname_raw=cfg.fname_raw,
                    fname_inv=cfg.fname_inv,
                    fname_eve=cfg.fname_eve,
                    fname_test_idcs=cfg.fname_test_idcs
                    )

#%% Data
event_id, tmin, tmax = 1, -0.2, 0.5
data = cmne.Data(settings=settings)
data.load_data(event_id=event_id, tmin=tmin, tmax=tmax)

#%% train
cmne.train(settings, data)
