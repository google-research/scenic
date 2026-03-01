import jax
import jax.numpy as jnp
import numpy as np
import os
from flax.training import checkpoints
#from scenic.projects.vivit import configs
from scenic.projects.vivit.model import SpaceTimeViViT
# Step 1: Import config
from scenic.projects.vivit.configs.kinetics400 import vivit_base_factorised_encoder as config_lib

import csv
import subprocess
import tempfile
import tensorflow as tf
from PIL import Image
import io
import shutil

config = config_lib.get_config()

import cv2
import numpy as np

NUM_FRAMES = 32
FPS = 8

def load_kinetics400_label_map(label_csv):
    label_to_index = {}
    with open(label_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_to_index[row["name"]] = int(row["id"])

    print("Loaded official Kinetics-400 labels:", len(label_to_index))
    return label_to_index

def extract_frames(video_path):
    tmpdir = tempfile.mkdtemp()
    try:
        min_resize = 256

        vf = (
            f"fps={FPS},"
            "scale='if(gt(iw,ih),-2,256)':'if(gt(iw,ih),256,-2)',"
            "crop=224:224:(iw-224)/2:(ih-224)/2"
        )

        cmd = [
            "ffmpeg", "-loglevel", "error",
            "-i", video_path,
            "-vf", vf,
            "-vframes", str(NUM_FRAMES),
            f"{tmpdir}/%05d.jpg"
        ]
        subprocess.run(cmd, check=False)

        frames = []
        sizes = []

        for fname in sorted(os.listdir(tmpdir)):
            with open(os.path.join(tmpdir, fname), "rb") as f:
                jpeg = f.read()
                frames.append(tf.io.decode_jpeg(jpeg))
                #frames.append(jpeg)

                img = Image.open(io.BytesIO(jpeg))
                sizes.append((img.height, img.width))
        print("get frames")
        return frames, sizes
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# Step 2: Extract dataset information
num_frames = config.dataset_configs.num_frames        # 32
crop_size  = config.dataset_configs.crop_size         # 224
num_classes = config.dataset_configs.num_classes      # 400
spatial_mlp_dim = config.model.spatial_transformer.mlp_dim
spatial_num_layers = config.model.spatial_transformer.num_layers
spatial_num_heads = config.model.spatial_transformer.num_heads
temporal_mlp_dim = config.model.temporal_transformer.mlp_dim
temporal_num_layers = config.model.temporal_transformer.num_layers
temporal_num_heads = config.model.temporal_transformer.num_heads
num_classes = config.dataset_configs.num_classes
patches = config.model.patches
hidden_size = config.model.hidden_size
temporal_encoding_config = config.model.temporal_encoding_config
attention_config = config.model.attention_config

fake_video = jnp.zeros(
    (1, num_frames, crop_size, crop_size, 3),
    dtype=jnp.float32
)

# Step 3: Build ViViT model
model_kwargs = dict(config.model)
model_kwargs["num_classes"] = num_classes

model = SpaceTimeViViT(
    spatial_mlp_dim = config.model.spatial_transformer.mlp_dim,
    spatial_num_layers = config.model.spatial_transformer.num_layers,
    spatial_num_heads = config.model.spatial_transformer.num_heads,
    temporal_mlp_dim = config.model.temporal_transformer.mlp_dim,
    temporal_num_layers = config.model.temporal_transformer.num_layers,
    temporal_num_heads = config.model.temporal_transformer.num_heads,
    num_classes = 101,
    patches = config.model.patches,
    hidden_size = config.model.hidden_size,
    temporal_encoding_config = config.model.temporal_encoding_config,
    attention_config = config.model.attention_config,
)


# Step 4: Init
rng = jax.random.PRNGKey(0)
variables = model.init(rng, fake_video, train=False)
print("Model initialized successfully!")

# ---------- 載入 checkpoint ----------
checkpoint_dir = r"/users/undergraduate/rjchen25/share/checkpoint_mhl"

variables = model.init(rng, fake_video, train=False)
ckpt_dict  = checkpoints.restore_checkpoint(
    checkpoint_dir,
    target=None
)
def squeeze_leading_dim(tree):
    """
    對整個 PyTree 做：
    如果第一維 == 1 → squeeze 掉
    否則保持不變
    """
    def squeeze_if_needed(x):
        if isinstance(x, jnp.ndarray):
            if x.ndim > 0 and x.shape[0] == 1:
                return jnp.squeeze(x, axis=0)
        return x

    return jax.tree_util.tree_map(squeeze_if_needed, tree)


print(type(ckpt_dict))
print(ckpt_dict.keys() if ckpt_dict is not None else "ckpt_dict is None")
#print(ckpt_dict["optimizer"]["target"].keys())
#if ckpt_dict  is None or "target" not in ckpt_dict :
#    print("Checkpoint not loaded correctly!")

# params = ckpt_dict['params']
params = ckpt_dict['opt_state']['0']['trace']
print(params.keys())
params = squeeze_leading_dim(params)
""" for debug
print(params['TemporalTransformer']['posembed_input'].keys())
print(type(params['SpatialTransformer']['posembed_input']['pos_embedding']))
print(params['SpatialTransformer']['posembed_input']['pos_embedding'].shape)
"""

## fix pos_embedding problem

params['SpatialTransformer']['posembed_input']['pos_embedding'] = params['SpatialTransformer']['posembed_input']['pos_embedding'][:,1:,:]
params['TemporalTransformer']['posembed_input']['pos_embedding'] = params['TemporalTransformer']['posembed_input']['pos_embedding'][:,1:,:]


print("Loading success!")
# ************** path setting ******************* 
class2int = ""
vid_dir = "/users/undergraduate/rjchen25/k400/train_vid"
TRAINLIST = "/users/undergraduate/rjchen25/k400/csv_files/filted_test_set.csv"
UCF_ROOT = "/users/undergraduate/rjchen25/k400/train_vid"
# ***********************************************

vid_set = {}

success = 0
fail = 0
total = 0
error_cnt = 0
label_mp = load_kinetics400_label_map("/users/undergraduate/rjchen25/k400/csv_files/kinetics_400_labels.csv")

with open(TRAINLIST) as f:
    reader = csv.DictReader(f)
    for row in reader:
        video_path = row["video_path"]
        video_label = row["label"]
        label = label_mp[video_label]
        #rel_path = video_path
        #video_path = os.path.join(UCF_ROOT, rel_path)
        video_ori, size = extract_frames(video_path)
        #print("list: ", video)
        video = np.array(video_ori)
        video = video.astype(np.float32)
        #video = tf.convert_to_tensor(video, tf.float32)
        print("video shape: ", video.shape ,type(video))
        # error = 0
        # for i in range(32):
        #     for j in range(224):
        #         for k in range(224):
        #             error += (video_ori[i][j][k] - video[i][j][k])**2
        #print("error: ", error)
        #print(video)
        # ************ pre-processing video *****************
        # normalize
        #video = tf.cast(video, tf.float32)
        video /= 255.0
        # 
        # ***************************************************
        video = video[None, ...]
        print("video shape: ", video.shape)
        

        print("video processing successfully")
        #print(video.shape)
        # if (video.shape[1] != 32):
        #     print("video decoding error: less frames")
        #     error_cnt += 1
        #     continue
        # Step 5: Forward
        logits = model.apply({'params': params}, video, train=False)
        pred = jnp.argmax(logits, axis=-1)
        # print("Forward OK! logits shape:", logits.shape)
        # print(logits)
        print("video name: ", video_path)
        print("video label: ",int(pred[0]))
        print("real  label: ", label)
        real_label = label
        if (int(pred[0]) == label):
            print("success !", success)
            success += 1
        else:
            fail += 1
            print("fail !", fail)
        total += 1

print("info summary")
print(success, fail, error_cnt, total)
print("success: %d, fail: %d, total: %d"%(success, fail, total))