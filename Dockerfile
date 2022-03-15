# Example build:
#   docker build -t brain-link/cmne:v0.01 .
#
# Example usage:
#   docker run -ti -v <YOUR DATA DIR>:/workspace/data -v <YOUR CMNE RESULTS DIR>:/workspace/results -v <YOUR CMNE GIT DIR>:/workspace/cmne --name CMNE brain-link/cmne:v0.01
#   docker run -ti -v D:/Data/1_Studies/2017_09_30_MEG_jgs/jgs/170505/processed:/workspace/data -v D:/Data/2_Processed/cmne:/workspace/results -v D:/Git/cmne:/workspace/cmne --name CMNE brain-link/cmne:v0.01

# Start with tensorflow enabled gpu version
FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update

RUN mkdir /workspace
RUN mkdir /workspace/data
RUN mkdir /workspace/results
RUN mkdir /workspace/cmne

# Configure CMNE
RUN pip install cmne