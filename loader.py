"""
Used to combine COCO and OCHuman data together,
and create a uniform dataloader.
"""
import os
from os.path import join, split
import json

import numpy as np

import cv2
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 12)
plt.rcParams['figure.autolayout'] = True

import pycocotools.coco as coco

# donot use ochuman api, load data manually

with open('./data/ochuman/labels/ochuman_coco_format_test_range_0.00_1.00.json') as f:
  data_test = json.load(f)

with open('./data/ochuman/labels/ochuman_coco_format_val_range_0.00_1.00.json') as f:
  data_val = json.load(f)

with open('./data/ochuman/labels/ochuman.json') as f:
  data_train = json.load(f)



"""
# the test dataset and val dataset has no intersection
val = [x['file_name'] for x in data_val['images']]
test = [x['file_name'] for x in data_test['images']]
print(len([x for x in val if x in test]))
"""
# combine test/validation data together
def convert(data):
  # convert validation/test data to train data format
  new = {}
  new['images'] = data['images'].copy()

  # add annotations to image
  for image in new['images']:
    image['annotations'] = []

  for anno in data['annotations']:
    annotation = {} # creat an annotation to be registed to image
    bbox = anno['bbox']
    annotation['bbox'] = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
    annotation['keypoints'] = anno['keypoints']

    new['images'][anno['image_id']-1]['annotations'].append(annotation)

  return new

new_val = convert(data_val)
new_test = convert(data_test)

new = {}
new['images'] = new_val['images'] + new_test['images']

print(len(new['images']))

def strip(data_train):
  merge = {}

  for image in data_train['images']:
    im = {} # creat a new image object
    im['file_name'] = image['file_name']
    im['annotations'] = []

    for annotation in image['annotations']:
      anno = {}
      anno['bbox'] = annotation['bbox']
      anno['keypoints'] = annotation['keypoints']
      im['annotations'].append(anno)

    merge[im['file_name']] = im['annotations']

  return merge

new = strip(new)
data_train = strip(data_train)

cnt = 0
for item in new:
  cnt += len(new[item])
print('test/val people: ', cnt)

cnt = 0
for item in data_train:
  cnt += len(data_train[item])
print('train people: ', cnt)

for image in data_train:
  try:
    data_train[image] += new[image]
  except:
    pass

cnt = 0
for item in data_train:
  cnt += len(data_train[item])
print('total people: ', cnt)

import pickle

with open('ochuman_partial.pickle', 'wb') as f:
  pickle.dump(data_train, f)

# visualization
empty = 0
for item in data_train:
  img = cv2.imread(join('./data/ochuman/images', item))
  for anno in data_train[item]:
    bbox = anno['bbox']
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    if anno['keypoints'] is None:
      empty += 1
      continue
    kps = np.array(anno['keypoints']).reshape(-1, 3).astype(np.int)
    for kp in kps:
      if kp[2] != 0:
        cv2.circle(img, (kp[0], kp[1]), 0, (0, 255, 0), 3)

  cv2.imshow('_', img)
  if cv2.waitKey(0) == 27: break

print('empty bbox: ', empty)










"""
# visualization for train dataset
for image in new['images']:
  img = cv2.imread(join('./data/ochuman/images', image['file_name']))

  annos = image['annotations']

  for anno in annos:
    bbox = anno['bbox']
    kps = anno['keypoints']

    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
    if kps is None: continue
    
    for i in range(len(kps) // 3):
      cv2.circle(img, (int(kps[3*i]), int(kps[3*i+1])), 0, (0, 0, 255), 3)

  cv2.imshow('_', img)
  if cv2.waitKey(0) == 27: break
"""


"""
# visualization for validation/test dataset

for anno in data['annotations']:
  image = join('./data/ochuman/images', data['images'][anno['image_id']-1]['file_name'])
  bbox = anno['bbox']
  kps = anno['keypoints']
  kps_num = anno['num_keypoints']


  img = cv2.imread(image)

  cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 3)

  for i in range(kps_num):
    cv2.circle(img, (int(kps[3*i]), int(kps[3*i+1])), 0, (0, 0, 255), 3)

  cv2.imshow('_', img)
  if cv2.waitKey(0) == 27: break
"""


"""
# using ochuman api
from ochumanApi.ochuman import OCHuman, Poly2Mask
from ochumanApi import vis

ochuman = OCHuman(AnnoFile='./data/ochuman/labels/ochuman.json', Filter='kpt')

image_ids = ochuman.getImgIds()
print('Total images: %d' % len(image_ids))

ImgDir = './data/ochuman/images/'

for _ in range(5):
  idx = np.random.randint(len(image_ids))
  data = ochuman.loadImgs(imgIds=[image_ids[idx]])[0]

  img = cv2.imread(os.path.join(ImgDir, data['file_name']))
  height, width = data['height'], data['width']

  colors = [
    [255, 0,    0   ],
    [255, 255,  0   ],
    [0,   255,  0   ],
    [0,   255,  255 ],
    [0,   0,    255 ],
    [255, 0,    255 ]
  ]

  for i, anno in enumerate(data['annotations']):
    bbox = anno['bbox']
    kpt = anno['keypoints']
    segm = anno['segms']
    max_iou = anno['max_iou']

    img = vis.draw_bbox(img, bbox, thickness=3, color=colors[i%len(colors)])
    if segm is not None:
      mask = Poly2Mask(segm)
      img = vis.draw_mask(img, mask, thickness=3, color=colors[i%len(colors)])
    if kpt is not None:
      img = vis.draw_skeleton(img, kpt, connection=None, colors=colors[i%len(colors)], bbox=bbox)

  cv2.imshow('_', img)
  if cv2.waitKey(0) == 27: break
"""

