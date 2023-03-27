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

import os.path as osp

import cv2
import numpy as np
import torch
import torch.utils.data as data

SEG_CLASSES = ('background', 'car', 'sign', 'road')
'''
SEG_ROOT = osp.join("/scratch/workspace/data/multi_task_det5_seg16/segmentation")
'''


class DrivableSegmentation(data.Dataset):
    def __init__(self, SEG_ROOT, image_sets='train', transform=None, debug=False):
        self.debug = debug
        self.root = SEG_ROOT
        self.image_set = image_sets
        self.transform = transform
        self._imgpath = osp.join('%s', 'images', '%s.jpg')
        self._segpath = osp.join('%s', 'seg', '%s_drivable_id.png')
        self.ids = list()
        rootpath = osp.join(self.root, self.image_set)
        for line in open(osp.join(rootpath, self.image_set + '.txt')):
            self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, h, w, seg = self.pull_item(index)
        return im, seg

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id)
        seg = cv2.imread(self._segpath % img_id, cv2.IMREAD_GRAYSCALE)
        height, width, channels = img.shape

        if self.transform is not None:
            img, seg = self.transform(img, seg)
            # to rgb
            if self.debug:
                from pathlib import Path
                folder = Path("debug/drive")
                folder.mkdir(parents=True, exist_ok=True)
                cv2.imwrite((folder / f"{self.__class__.__name__}_{index}.jpg").as_posix(),
                            (img * 255. + np.array([104, 117, 123])[None, None, :]).astype('uint8'))
            img = img[:, :, (2, 1, 0)]
        return torch.from_numpy(img).permute(2, 0, 1), height, width, torch.from_numpy(seg)

    def pull_image(self, index):
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_tensor(self, index):
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
