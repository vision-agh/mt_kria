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


from __future__ import division

from itertools import product as product
from math import sqrt as sqrt

import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self, cfg, return_lv_info=False):
        super(PriorBox, self).__init__()
        self.return_lv_info = return_lv_info
        self.image_size = cfg['resize']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            lv_anchor = []
            for i, j in product(range(f[0]), range(f[1])):
                f_kx = self.image_size[1] / f[1]
                f_ky = self.image_size[0] / f[0]
                # unit center x,y
                cx = (j + 0.5) * f_kx
                cy = (i + 0.5) * f_ky

                # aspect_ratio: 1
                # rel size: min_size
                s_kx = self.min_sizes[k]
                s_ky = self.min_sizes[k]
                lv_anchor += [cx, cy, s_kx, s_ky]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_primex = sqrt(s_kx * self.max_sizes[k])
                s_k_primey = sqrt(s_ky * self.max_sizes[k])
                lv_anchor += [cx, cy, s_k_primex, s_k_primey]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    lv_anchor += [cx, cy, s_kx * sqrt(ar), s_ky / sqrt(ar)]
                    lv_anchor += [cx, cy, s_kx / sqrt(ar), s_ky * sqrt(ar)]
            lv_anchor = torch.Tensor(lv_anchor).view(-1, 4)
            lv_anchor = torch.cat([
                lv_anchor[:, :2] - lv_anchor[:, 2:] / 2.,
                lv_anchor[:, :2] + lv_anchor[:, 2:] / 2.], 1)
            mean.append(lv_anchor)
        # back to torch land
        num_anchors_per_lv = [a.size(0) for a in mean]
        output = torch.cat(mean, 0)
        if self.clip:
            output.clamp_(max=1, min=0)
        if self.return_lv_info:
            return output, num_anchors_per_lv
        return output


if __name__ == '__main__':
    solver = {
        'k1': 8,
        'k2': 8,
        'act_clip_val': 8,
        'warmup': False,
        'det_classes': 5,
        'seg_classes': 6,
        'drivable_classes': 3,
        'reg_depth': 80,
        'seg_lane': 2,
        'lr_steps': (12000, 18000),
        'max_iter': 20010,
        'feature_maps': [(80, 128), (40, 64), (20, 32), (10, 16), (5, 8), (3, 6), (1, 4)],
        'resize': (320, 512),
        'steps': [4, 8, 16, 32, 64, 128, 256],
        'min_sizes': [10, 30, 60, 100, 160, 220, 280],
        'max_sizes': [30, 60, 100, 160, 220, 280, 340],
        'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2]],
        'variance': [0.1, 0.2],
        'clip': False,
    }
    pb = PriorBox(solver,return_lv_info=True)
    out = pb.forward()
    print()