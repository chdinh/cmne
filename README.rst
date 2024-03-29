.. -*- mode: rst -*-

|PyPI|_ |GH-CI|_

.. |PyPI| image:: https://badge.fury.io/py/cmne.svg?label=PyPI%20downloads
.. _PyPI: https://pypi.org/project/cmne/

.. |GH-CI| image:: https://github.com/chdinh/cmne/actions/workflows/ci.yml/badge.svg?branch=main
.. _GH-CI: https://github.com/chdinh/cmne/actions/workflows/ci.yml


Contextual Minimum-Norm Estimates (CMNE): A Deep Learning Method for Source Estimation in Neuronal Networks
===========================================================================================================

For more information on CMNE, please read the following papers:

  Dinh C, Samuelsson JGW*, Hunold A, Hämäläinen MS, Khan S. Contextual MEG and EEG Source Estimates Using Spatiotemporal LSTM Networks. Front. Neurosci 2021;15:119-134; doi: https://doi.org/10.3389/fnins.2021.552666

  Dinh C, Samuelsson JGW*, Hunold A, Hämäläinen MS, Khan S. Contextual Minimum-Norm Estimates (CMNE): A Deep Learning Method for Source Estimation in Neuronal Networks. arXiv:1909.02636; doi: https://doi.org/10.48550/arXiv.1909.02636


Installation
^^^^^^^^^^^^

To install the latest stable version of CMNE, you can use pip_ in a terminal:

.. code-block:: bash

    pip install -U cmne


Usage of the Docker Container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Build the docker image with

.. code-block:: bash

    docker build -t brain-link/cmne:v0.01 .

and run it with

.. code-block:: bash

    docker run -ti -v <YOUR DATA DIR>:/workspace/data -v <YOUR CMNE RESULTS DIR>:/workspace/results -v <YOUR CMNE GIT DIR>:/workspace/cmne --name CMNE brain-link/cmne:v0.01

It is convinient to install CMNE for development directly from the local repository. Change the directory to '/workspace/cmne' in the CLI of the Docker Container and run

.. code-block:: bash

    pip install -e .


Licensing
^^^^^^^^^
CMNE is **MIT-licensed**:

    Copyright (c) 2017-2022, authors of CMNE.
    All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    **THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.**


.. _pip: https://pip.pypa.io/en/stable/
