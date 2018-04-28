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



f.write2csv('submission_v2_final_test123.csv', ImageId, EncodedPixels)



def rle_encode(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[1]*shape[0], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((shape[1], shape[0])).T