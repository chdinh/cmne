# CPU image (small):   docker build -t cmne .
# GPU image:           docker build -t cmne:gpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 .
#
# Run:  docker run --rm -v $HOME/cmne_data:/data -v $PWD/results:/results cmne \
#           cmne train --raw /data/assr_270LP_fs900_raw.fif --inv /data/assr_270LP_fs900_raw-ico-4-meg-eeg-inv.fif \
#                      --events /data/assr_270LP_fs900_raw-eve.fif -o /results/cmne.pt
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg

WORKDIR /opt/cmne
COPY pyproject.toml README.md LICENSE ./
COPY cmne ./cmne
RUN pip install --extra-index-url "${TORCH_INDEX}" ".[onnx,viz]" \
    && useradd --create-home cmne

USER cmne
WORKDIR /results
CMD ["cmne", "--help"]
