"""
SHerk's version of demo
"""
import argparse
import os
import sys
from glob import glob
import random

sys.path.insert(0, 'lib')

import cv2


from config import cfg, update_config
from detectors.detector_factory import detector_factory

# load arguments
parser = argparse.ArgumentParser()
parser.add_argument('--cfg')
parser.add_argument('--MODEL_PATH')
parser.add_argument('--DEBUG', type=int)

args = parser.parse_args([
  '--cfg', 'experiments/res_50_512x512.yaml',
  '--MODEL_PATH', 'models/res50_cloud_44.pth',
  '--DEBUG', '1'
])

update_config(cfg, args.cfg)
cfg.defrost()
cfg.TEST.MODEL_PATH = args.MODEL_PATH
cfg.DEBUG = args.DEBUG
cfg.freeze()

# load the detector
Detector = detector_factory['multi_pose']
detector = Detector(cfg)



# load images
# image_folder = '/home/ubuntu/Pictures/person/dense/*'
image_folder = 'images/testpng/*.png'
images = glob(image_folder)
# random.shuffle(images)  # shuffle images list



for image in images:
  ret = detector.run(image)
  

  



