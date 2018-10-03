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




#11th
###config
class BowlConfig(Config):    
    # Give the configuration a recognizable name
    NAME = "bowl"
    # NUMBER OF GPUs to use. For CPU training, use 1
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 1

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 598

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 66

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet101"

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4,8,16,32,64]

    # Number of classification classes (including background)
    NUM_CLASSES = 1  # Override in sub-classes

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8,16,32,64,128)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 320

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resizing
    # Generally, use the "square" resizing mode for training and inferencing
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    # up scaling. For example, if set to 2 then images are scaled up to double
    # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    # Howver, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    IMAGE_MIN_SCALE = 2

    # Image mean (RGB)
    MEAN_PIXEL = np.array([0, 0, 0])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 600

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 500

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 500

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.7

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 1e-03
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when inferencing
    TRAIN_BN = False  # Defaulting to False since batch size is often small

    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0


"""
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
