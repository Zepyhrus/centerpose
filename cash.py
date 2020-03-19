"""
This script is used to modify the output size of the network
"""

import argparse as ap
import os
from os.path import join, split
import time
import sys


import numpy as np
import pycocotools.coco as coco
import torch.utils.data as data


import cv2


import torch
import torchvision
import torch.utils.data as tdata

sys.path.insert(0, 'lib')

from config import cfg, update_config
from datasets.coco_hp import COCOHP
from datasets.multi_pose import MultiPoseDataset

# 
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)

# load arguments
parser = ap.ArgumentParser()
parser.add_argument('--cfg', required=True)
parser.add_argument('--model')
parser.add_argument('--debug', type=int)

args = parser.parse_args([
  '--cfg', 'experiments/res_50_512x512.yaml',
  '--model', 'models/res50_cloud_87.pth',
  '--debug', '1'
])

update_config(cfg, args.cfg)
cfg.defrost()
cfg.TEST.MODEL_PATH = args.model
cfg.DEBUG = args.debug
cfg.freeze()


# load dataset
class Datasets(MultiPoseDataset, COCOHP):
  pass


dataset = Datasets(cfg, 'train')
data_loader = tdata.DataLoader(
  dataset,
  batch_size=1, # cfg.TRAIN.BATCH_SIZE,
  shuffle=True,
  pin_memory=True,
  drop_last=True,
  sampler=None
)

num_iters = data_loader.__len__()

# initialized from COCO_HP
# points to be removed: 1, 2, 3, 4
num_classes = 1
num_joints = 13
limb = [
  (0, 1), (0, 2), # head
  (1, 3), (3, 5), (2, 4), (4, 6),        # arms
  (1, 2), (2, 8), (1, 7), (7, 8),     # body
  (7, 9), (9, 11), (8, 10), (10, 12)] # legs

data_dir = join(cfg.DATA_DIR, 'coco')
img_dir = join(data_dir, 'train2017')
anno_path = join(data_dir, 'annotations', 'person_keypoints_train2017.json')
max_objs = 32
_valid_ids = [1]
class_name = ['__background__', 'person']
_data_rang = np.random.RandomState(123)
_eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
_eig_vec = np.array([
  [-0.58752847, -0.69563484, 0.41340352],
  [-0.5832747, 0.00994535, -0.81221408],
  [-0.56089297, 0.71832671, 0.41158938]
], dtype=np.float32)

self_coco = coco.COCO(anno_path)
images = self_coco.getImgIds()
catIds = self_coco.getCatIds(class_name[-1])
num_samples = len(images)


# initialized from MultiPoseDataset
for _ in range(100):
  index = 25770 # np.random.randint(num_samples)
  print(index)

  img_id = images[index]
  file_name = self_coco.loadImgs(ids=[img_id])[0]['file_name']
  img_path = join(img_dir, file_name)
  ann_ids = self_coco.getAnnIds(imgIds=[img_id])
  anns = self_coco.loadAnns(ids=ann_ids)
  anns = list(filter(lambda x:x['category_id'] in _valid_ids and x['iscrowd']!= 1, anns))

  for ann in anns:
    ann['numkeypoints'] = 13
    del ann['keypoints'][3:15]

  img = cv2.imread(img_path)


  for ann in anns:
    bbox = ann['bbox']
    kps = np.array(ann['keypoints']).astype(np.int).reshape(-1, 3)

    # cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), GREEN, 1)
    for i, kp in enumerate(kps):
      cv2.circle(img, (kp[0], kp[1]), 0, GREEN, 2)
      cv2.putText(img, str(i), (kp[0], kp[1]), 0, 0.5, GREEN, 2)
  
    for i, j in limb:
      if kps[i][2]*kps[j][2] == 0: continue
      cv2.line(img, tuple(kps[i][:2]), tuple(kps[j][:2]), GREEN, 2)

  cv2.imshow('_', img)
  if cv2.waitKey(0) == 27: break

