# Example build:
#   docker build -t brain-link/cmne:v0.01 .
#
# Example usage:


# Start with ubuntu
FROM tensorflow/tensorflow:latest-gpu


RUN pip install mne