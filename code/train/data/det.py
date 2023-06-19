# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PART OF THIS FILE AT ALL TIMES.


# This file has been modified by Maciej Baczmański for usage in https://github.com/maciekbaczmanski/mt_kria repository, under Apache 2.0 license.
# All files were modified to update the repository to work with newest versions of libraries, and to train and evaluate our own MultiTask V3 model.

import os.path as osp

import cv2
import numpy as np
import torch
import torch.utils.data as data

DET_CLASSES = (  # always index 0
    'car', 'sign', 'person')
'''
DET_ROOT = "/scratch/workspace/data/multi_task_det5_seg16/detection/Waymo_bdd_txt"
'''

CAR_INCLUDE = ('car', 'truck', 'bus', 'vehicle')
MOTOR_INCLUDE = ('motor', 'bike')
PERSON_INCLUDE = ('person', 'rider', 'cyclist', *MOTOR_INCLUDE)
SIGN_INCLUDE = ('Sign',)


class DetAnnotationTransform(object):

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = dict(zip(DET_CLASSES, range(len(DET_CLASSES))))

    def __call__(self, target, width, height):
        res = []
        for line in target:
            component = line.rstrip().split(' ')
            name = component[0].lower()
            bndbox = []

            center_x = float(component[1]) 
            center_y = float(component[2]) 
            w_x = float(component[3])
            w_y = float(component[4]) 

            x1 = (center_x - w_x/2)
            x2 = (center_x + w_x/2)
            y1 = (center_y - w_y/2)
            y2 = (center_y + w_y/2)

            bndbox.append(x1)
            bndbox.append(y1)
            bndbox.append(x2)
            bndbox.append(y2)

            label_idx = int(name)
            bndbox.append(label_idx)
            res += [bndbox]
        target.close()

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class Detection(data.Dataset):
    # def __init__(self, root='/home/caiyi/data/bdd100k/',
    def __init__(self, DET_ROOT, image_sets='train', transform=None, target_transform=DetAnnotationTransform(),
                 debug=False):
        self.debug = debug
        self.root = DET_ROOT
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self._annopath = osp.join('%s', 'detection', '%s.txt')
        self._imgpath = osp.join('%s', 'images', '%s.jpg')
        self.ids = list()
        rootpath = osp.join(self.root, self.image_set)
        for line in open(osp.join(rootpath, self.image_set + '.txt')):
            self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = open(self._annopath % img_id)
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            if len(target.shape) != 2:
                print("img_id: {}, target.shape: {}".format(self._annopath % img_id, target.shape))
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            boxes[:, 0] *= 512.
            boxes[:, 2] *= 512.
            boxes[:, 1] *= 320.
            boxes[:, 3] *= 320.
            if self.debug:
                from pathlib import Path
                folder = Path("debug/det")
                folder.mkdir(parents=True, exist_ok=True)
                cv2.imwrite((folder / f"{self.__class__.__name__}_{index}.jpg").as_posix(),
                            (img * 255. + np.array([104, 117, 123])[None, None, :]).astype('uint8'))
            # to rgb
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        img_id = self.ids[index]
        anno = open(self._annopath % img_id)
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
