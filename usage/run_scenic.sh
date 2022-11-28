#!/bin/bash

git clone --branch=main https://github.com/google-research/t5x
cd t5x

python3 -m pip install -e '.[tpu]' -f \
  https://storage.googleapis.com/jax-releases/libtpu_releases.html
export PATH=~/.local/bin:$PATH

git clone https://github.com/YOUR_GITHUB_NAME/scenic.git
cd scenic
pip install .

export WORK_DIR=gs://${GOOGLE_CLOUD_BUCKET_NAME}/scenic/imagenet
python scenic/main.py \
  --config=scenic/projects/baselines/configs/imagenet/imagenet_vit_config.py \
  --workdir=$WORK_DIR

