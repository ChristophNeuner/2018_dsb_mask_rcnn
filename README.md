Install pycocotools and COCO pretrained weights (mask_rcnn_coco.h5). General idea is described here (https://github.com/matterport/Mask_RCNN#installation). Keep in mind, to install pycocotools properly, it's better to run make install instead of make.

    For a single GPU training, run:

CUDA_VISIBLE_DEVICES="0" python train.py

    To generate a submission, run:

CUDA_VISIBLE_DEVICES="0" python inference.py

This will create submission.csv in the repo and overwrite the old one (you're welcome to fix this with a PR).