## Boundary Attention: Learning to Find Faint Boundaries at Any Resolution

![Boundary Attention](rm.png)

### [Project Page](https://boundaryattention.github.io) | [arXiv](https://arxiv.org/abs/2401.00935) | [Dataset](kaleidoshapes)

Boundary Attention is a differentiable model that explicitly models
boundaries—including contours, corners and junctions—using a new mechanism
that we call boundary attention. Our model provides accurate results
even when the boundary signal is very weak or is swamped by noise.

> [**Boundary Attention: Learning to Find Faint Boundaries at Any Resolution**](https://arxiv.org/abs/2401.00935),
> Mia Gaia Polansky, Charles Herrmann, Junhwa Hur, Deqing Sun, Dor Verbin, Todd Zickler

### Quick Start

Boundary Attention is written in JAX and uses Scenic framework for training.
For more information on how to install JAX with GPU support,
see [here](https://github.com/google/jax#installation).

To begin, we recommend installing scenic to a new conda virtual environment. If necessary, install anaconda or [miniconda](https://docs.conda.io/projects/miniconda/en/latest/).

```shell
# Create virtual environment with python 3.10 and activate
conda create -n boundary_attention python=3.10 -y
conda activate boundary_attention
# Clone the scenic github repository
git clone https://github.com/google-research/scenic.git
cd ~/scenic
# Install scenic-wide packages
pip install .
# Install Boundary Attention specific packages
pip install -r scenic/projects/boundary_attention/requirements.txt

# (Optional) For GPU support:
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Download the [pretrained weights](#pretrained-checkpoints) so that they are available locally.

Then, you can use the following script to test Boundary Attention on new images by replacing `CHECKPOINT_DIRECTORY`, `PATH_TO_IMAGE`, and `PATH_TO_SAVE_OUTPUT` with local paths to the model's checkpoint, a new input image, and where to save the network's output.

```shell
python scenic/projects/boundary_attention/helpers/test_new_images.py \
  --ckpt_dir=${CHECKPOINT_DIRECTORY} \
  --img_path=${PATH_TO_IMAGE} \
  --save_path=${PATH_TO_SAVE_OUTPUT} \
  --height=216 \
  --width=216 \
  --save_raw_output=False
```

The height and width options resize the input image. The option "save_raw_output" toggles whether the entire output from the network is saved to a pickle file.

Alternatively, you can modify this simple script for Jupyter or Colab.

```python
import PIL
import jax.numpy as jnp
from tensorflow.io import gfile
from scenic.projects.boundary_attention.configs import base_config
from scenic.projects.boundary_attention.helpers import train_utils
from scenic.projects.boundary_attention.helpers import viz_utils

######## MODIFY THE OPTIONS BELOW #########

im_height = 216 # Replace with height to resize input to
im_width = 216  # Replace with width to resize input to

img_path = '' # Replace with path to new input
ckpt_dir = '' # Add path to ckpt directory here

############################################

input_img = jnp.array(PIL.Image.open(gfile.GFile(img_path, 'rb')).resize((im_width, im_height)))/255.0
input_img = jnp.expand_dims(input_img.transpose(2,0,1)[:3,:,:], axis=0)

config = base_config.get_config(model_name='boundary_attention',
                                  dataset_name='testing',
                                  input_size=(im_height, im_width, 3))

apply_jitted, trained_params = train_utils.make_apply(config, ckpt_dir)

outputs = apply_jitted(trained_params, input_img)
viz_utils.visualize_outputs(input_img, outputs)
```

### Pretrained Checkpoints
The pretrained checkpoint for boundary attention is available
[here](https://github.com/google/jax#installation) TODO: Upload and release weights..

### File Structure

A few important model files in this projects are:

- `models/model_lib/boundary_attention_model_base.py` is our base model, which is called by wrapper `models/boundary_attention.py`
- `helpers/junction_functions.py` defines a class to manipulate the model's output junctions and calls `helpers/render_junctions.py` to render junction patches
- `helpers/params2maps.py` is a wrapper for `helpers/junction_functions.py`

### Training


Below is an example command-line script to train Boundary Attention on Kaleidoshapes with this [base config](configs/boundary_attention_model_config).

There are two ways to specify dataset and checkpoint locations.
The first is to modify the [base config](configs/boundary_attention_model_config.py) so that the parameters defined at the top point to the correct locations.

```python
_CHECKPOINT_PATH = ''  # Add path here or set to None
_CHECKPOINT_STEP = -1
_DATASET_DIR = '' # Add path here
```

Modify `_CHECKPOINT_PATH` to point to the provided weights if starting training from the pretrained model. Otherwise, set `_CHECKPOINT_PATH` to `None`.

Then, create a workdir and train with the following terminal command:

```shell
WORKDIR="SET WORKDIR PATH HERE"
python -m scenic.projects.boundary_attention.main \
  --config=scenic/projects/boundary_attention/configs/base_config.py \
  --workdir=${WORKDIR}/
```

Alternatively, specify the checkpoint path, step, and dataset directory at train time (this will override changes to `base_config`):

```shell
WORKDIR='ADD PATH TO WORKDIR HERE'
DATASET_DIR='ADD DATASET DIRECTORY HERE'
CHECKPOINT_PATH='ADD CHECKPOINT LOCATION HERE'
CHECKPOINT_STEP=-1

python -m scenic.projects.boundary_attention.main \
  --config=scenic/projects/boundary_attention/configs/base_config.py \
  --workdir=${WORKDIR}/ \
  --dataset_dir=${DATASET_DIR}/ \
  --checkpoint_path=${CHECKPOINT_PATH}/ \
  --checkpoint_step=${CHECKPOINT_STEP}
```

### Evaluation

Below is an example command-line script to evaluate Boundary Attention on Kaleidoshapes.

```shell
WORKDIR='ADD PATH TO WORKDIR'
DATASET_DIR='ADD PATH TO DATASET'
CHECKPOINT_PATH='ADD PATH TO CHECKPOINT'
CHECKPOINT_STEP=-1

python -m scenic.projects.boundary_attention.eval_main \
  --config=scenic/projects/boundary_attention/configs/base_config.py \
  --workdir=${WORKDIR}/ \
  --dataset_dir=${DATASET_DIR}/ \
  --checkpoint_path=${CHECKPOINT_PATH}/ \
  --checkpoint_step=${CHECKPOINT_STEP}
```


### Citation
```
@article{mia2023boundaries,
  author    = {Polansky, Mia Gaia and Herrmann, Charles and Hur, Junhwa and Sun, Deqing
              and Verbin, Dor and Zickler, Todd},
  title     = {Boundary Attention: Learning to Find Faint Boundaries at Any Resolution},
  journal   = {arXiv},
  year      = {2023},
  }
```
