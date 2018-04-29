from mrcnn.config import Config
from bowl_dataset import BowlDataset
import numpy as np



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



    

#10th try
###config
class BowlConfig(Config):    
    # Give the configuration a recognizable name
    NAME = "bowl"
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # background + nuclei
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])    
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    TRAIN_ROIS_PER_IMAGE = 600   
    STEPS_PER_EPOCH = len(dataset_train.image_ids)/(GPU_COUNT*IMAGES_PER_GPU)
    VALIDATION_STEPS = len(dataset_val.image_ids)/(GPU_COUNT*IMAGES_PER_GPU)
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    MAX_GT_INSTANCES = 256
    DETECTION_MAX_INSTANCES = 512    
    RESNET_ARCHITECTURE = "resnet50"    
    LEARNING_RATE = 0.0001
    LEARNING_MOMENTUM = 0.9
    # Decrease the LR if it has not decreased in a given number of epochs
    # See keras.callbacks.ReduceLROnPlateau
    LR_PLATEAU = {
        "monitor": 'val_loss',
        "factor": 0.5,
        "patience": 5,
        "verbose": 1,
        "mode": 'auto',
        "epsilon": 0.0001,
        "cooldown": 5,
        "min_lr": 1e-7
    }

""" 
#9th try
###config
class BowlConfig(Config):    
    # Give the configuration a recognizable name
    NAME = "bowl"
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # background + nuclei
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])    
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    TRAIN_ROIS_PER_IMAGE = 600   
    STEPS_PER_EPOCH = len(dataset_train.image_ids)/(GPU_COUNT*IMAGES_PER_GPU)
    VALIDATION_STEPS = len(dataset_val.image_ids)/(GPU_COUNT*IMAGES_PER_GPU)
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    MAX_GT_INSTANCES = 256
    DETECTION_MAX_INSTANCES = 512    
    RESNET_ARCHITECTURE = "resnet101"    
    LEARNING_RATE = 0.0001
    LEARNING_MOMENTUM = 0.9
    # Decrease the LR if it has not decreased in a given number of epochs
    # See keras.callbacks.ReduceLROnPlateau
    LR_PLATEAU = {
        "monitor": 'val_loss',
        "factor": 0.5,
        "patience": 5,
        "verbose": 1,
        "mode": 'auto',
        "epsilon": 0.0001,
        "cooldown": 5,
        "min_lr": 1e-7
    }


#8th try: first step: heads only, 25 epochs, second step: all layers, 50 epochs, 1/10*learning_rate, coco, score:
#7th try: first step: heads only, 50 epochs, second step: all layers, 100 epochs, 1/10*learning_rate, coco, score: 0.377
#6th try: all layers, 100 epochs, coco, score: 0.392

class BowlConfig(Config): 
    NAME = "bowl" 
    GPU_COUNT = 1 
    IMAGES_PER_GPU = 2 
    NUM_CLASSES = 1 + 1 
    IMAGE_MIN_DIM = 256 
    IMAGE_MAX_DIM = 512 
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)
    TRAIN_ROIS_PER_IMAGE = 500 
    STEPS_PER_EPOCH = 600 // (IMAGES_PER_GPU * GPU_COUNT) 
    VALIDATION_STEPS = 70 // (IMAGES_PER_GPU * GPU_COUNT) 
    MEAN_PIXEL = [0, 0, 0] 
    LEARNING_RATE = 0.01 
    USE_MINI_MASK = True 
    MAX_GT_INSTANCES = 500
    

    
#5th try: all layers, 100 epochs, coco, score: 0.062
class BowlConfig(Config): 
    NAME = "bowl" 
    GPU_COUNT = 1 
    IMAGES_PER_GPU = 2 
    NUM_CLASSES = 1 + 1 
    IMAGE_MIN_DIM = 256 
    IMAGE_MAX_DIM = 512 
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)
    TRAIN_ROIS_PER_IMAGE = 500 
    STEPS_PER_EPOCH = 600 // (IMAGES_PER_GPU * GPU_COUNT) 
    VALIDATION_STEPS = 70 // (IMAGES_PER_GPU * GPU_COUNT) 
    MEAN_PIXEL = [0, 0, 0] 
    LEARNING_RATE = 0.01 
    USE_MINI_MASK = True 
    MAX_GT_INSTANCES = 500
    RESNET_ARCHITECTURE = "resnet50"
"""


#4th try: all layers, 100 epochs, imagenet, score: 0.360
#3rd try: heads only, 25 epochs, imagenet, score: 0.315
#2nd try: heads only, 100 epochs, imagenet, score:0.336
"""
class BowlConfig(Config): 
    NAME = "bowl" 
    GPU_COUNT = 1 
    IMAGES_PER_GPU = 2 
    NUM_CLASSES = 1 + 1 
    IMAGE_MIN_DIM = 256 
    IMAGE_MAX_DIM = 512 
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)
    TRAIN_ROIS_PER_IMAGE = 500 
    STEPS_PER_EPOCH = 600 // (IMAGES_PER_GPU * GPU_COUNT) 
    VALIDATION_STEPS = 70 // (IMAGES_PER_GPU * GPU_COUNT) 
    MEAN_PIXEL = [0, 0, 0] 
    LEARNING_RATE = 0.01 
    USE_MINI_MASK = True 
    MAX_GT_INSTANCES = 500
"""
    

#1st try, all layers, 100 epochs, imagenet  score: 0.337
"""
class BowlConfig(Config):
    
    # Give the configuration a recognizable name
    NAME = "bowl"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # background + nuclei

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 600

    STEPS_PER_EPOCH = None

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    USE_MINI_MASK = True

    MAX_GT_INSTANCES = 256

    DETECTION_MAX_INSTANCES = 512

    RESNET_ARCHITECTURE = "resnet50"
"""

bowl_config = BowlConfig()
bowl_config.display()
