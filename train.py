import tensorflow as tf
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn.model import log
from bowl_dataset import BowlDataset

import numpy as np
import imgaug
from bowl_config import bowl_config

cfg = tf.ConfigProto()
cfg.gpu_options.allow_growth = True
session = tf.Session(config=cfg)


tf.logging.set_verbosity(tf.logging.ERROR)

def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


plt.rcParams["figure.figsize"] = [40,40]

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    
    
# Training dataset
dataset_train = BowlDataset()
dataset_train.load_bowl('stage1_train_fixed')
#dataset_train.load_bowl('stage1_train')
dataset_train.prepare()
print("Images: {} nClasses: {}".format(len(dataset_train.image_ids), dataset_train.class_names))
# Validation dataset
dataset_val= BowlDataset()
dataset_val.load_bowl('extra_data')
dataset_val.prepare()
print("Images: {} nClasses: {}".format(len(dataset_val.image_ids), dataset_val.class_names))


### augmentation
import imgaug
seq = imgaug.augmenters.Sequential([
    imgaug.augmenters.Fliplr(0.5), # horizontally flip 50% of all images
    imgaug.augmenters.Flipud(0.5), # vertically flip 20% of all images
    imgaug.augmenters.Affine(
        scale=(1, 2), # scale images to 80-120% of their size, individually per axis
        rotate=(-90, 90), # rotate by -45 to +45 degrees
        shear=(-16, 16), # shear by -16 to +16 degrees
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        order=[0, 1])  
])

for image_id in dataset_train.image_ids[:4]:
    image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
        dataset_train, bowl_config, image_id, augment=False, augmentation=seq, use_mini_mask=True)
    mask = utils.expand_mask(bbox, mask, image.shape)


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=bowl_config, model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)
    


model.train(dataset_train, dataset_val,
            learning_rate=bowl_config.LEARNING_RATE, 
            epochs=100,
            augmentation=seq,
            layers="all")