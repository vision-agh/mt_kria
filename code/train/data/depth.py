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

import itertools
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
# PART OF THIS FILE AT ALL TIMES.
from PIL import Image


def rgba(r):
    """Generates a color based on range.

    Args:
      r: the range value of a given point.
    Returns:
      The color for a given range
    """
    c = plt.get_cmap('jet')((r % 20.0) / 20.0)
    c = list(c)
    c[-1] = 0.5  # alpha
    return c


class KittiDepth(data.Dataset):
    def __init__(self, DEPTH_ROOT, image_sets='train', transform=None, debug=False):
        self.debug = debug
        self.root = DEPTH_ROOT
        self.image_set = image_sets
        self.transform = transform
        self.ids = list()
        self._init()

    def _init(self):
        for line in open(osp.join(self.root, self.image_set + '.txt')):
            img_path, label_path, focal_length, baseline = line.split(' ')
            self.ids.append((img_path, label_path, float(focal_length), float(baseline)))

    def __getitem__(self, index):
        im, h, w, seg, focal_length, baseline = self.pull_item(index)
        return im, seg, focal_length, baseline

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def depth_map_to_rgbimg(depth_map):
        depth_map = np.asarray(np.squeeze((255 - torch.clamp_max(depth_map * 4, 250)).byte().numpy()), np.uint8)
        depth_map = np.asarray(cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB), np.uint8)
        return depth_map

    def pull_item(self, index):
        img_path, label_path, focal_length, baseline = self.ids[index]
        # print(osp.join(self.root, img_path))
        img = cv2.imread(osp.join(self.root, img_path))
        depth_map = np.asarray(Image.open(osp.join(self.root, label_path)), np.float32)
        depth_map = np.expand_dims(depth_map, axis=2) / 256.0
        height, width, channels = img.shape

        if self.transform is not None:
            img, depth_map, info = self.transform(img, depth_map)
            scale_h = info['scale_h']
            focal_length *= scale_h
            if self.debug:
                xs = []
                ys = []
                colors = []
                depmap_debug = np.copy(depth_map)
                depmap_debug = depmap_debug[:, :, 0]
                depth_map_h, depth_map_w = depmap_debug.shape
                for i, j in itertools.product(range(depth_map_h), range(depth_map_w)):
                    if depmap_debug[i, j] != 0:
                        xs.append(j)
                        ys.append(i)
                        colors.append(rgba(depmap_debug[i, j]))
                from pathlib import Path
                im_show = (img * 255. + np.array([104, 117, 123])[None, None, :]).astype('uint8')
                im_show = cv2.resize(im_show, (depth_map_w, depth_map_h))
                folder = Path("debug/depth")
                folder.mkdir(parents=True, exist_ok=True)
                plt.figure(figsize=(20, 12))
                plt.imshow(im_show[:, :, ::-1])
                plt.grid("off")
                plt.scatter(xs, ys, c=colors, s=5.0, edgecolors="none")
                plt.savefig((folder / f"{self.__class__.__name__}_{index}.jpg").as_posix())
                plt.close()
            # to rgb
            img = img[:, :, (2, 1, 0)]
        focal_length = torch.from_numpy(np.array([focal_length], dtype=np.float32))
        baseline = torch.from_numpy(np.array([baseline], dtype=np.float32))
        return torch.from_numpy(img).permute(2, 0, 1), \
               height, \
               width, \
               torch.from_numpy(depth_map.copy()).permute(2, 0, 1), \
               focal_length, \
               baseline
