from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
import pandas as pd
import cv2
import os
import numpy as np
from tqdm import tqdm
from inference_config import inference_config
from bowl_dataset import BowlDataset
from mrcnn.utils import rle_encode, rle_decode, rle_to_string
import functions as f
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
model_path = os.path.join(MODEL_DIR, "9", "mask_rcnn_bowl_0050.h5")
# model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
#assert model_path != "./logs/9/mask_rcnn_bowl_0100.h5", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

dataset_test = BowlDataset()
dataset_test.load_bowl('stage2_test_final')
dataset_test.prepare()

output = []
sample_submission = pd.read_csv('stage2_sample_submission_final.csv')
ImageId = []
EncodedPixels = []
for image_id in tqdm(sample_submission.ImageId):
    image_path = os.path.join('stage2_test_final', image_id, 'images', image_id + '.png')
    
    original_image = cv2.imread(image_path)
    results = model.detect([original_image], verbose=0)
    r = results[0]
    
    masks = r['masks']
    ImageId_batch, EncodedPixels_batch = f.numpy2encoding_no_overlap2(masks, image_id, r['scores'])
    ImageId += ImageId_batch
    EncodedPixels += EncodedPixels_batch



f.write2csv('submission_v2_final_9_50_correct.csv', ImageId, EncodedPixels)
