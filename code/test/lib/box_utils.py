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

# -*- coding: utf-8 -*-

import numpy as np



# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    count = 0
    # keep = scores.new(scores.size(0)).zero_().long()
    keep = np.zeros(scores.shape[0], dtype=np.int_)
    if boxes.size == 0:
        return keep, count
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = np.multiply(x2 - x1, y2 - y1)
    idx = np.argsort(scores, axis=0)
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = np.empty_like(boxes)
    yy1 = np.empty_like(boxes)
    xx2 = np.empty_like(boxes)
    yy2 = np.empty_like(boxes)
    w = np.empty_like(boxes)
    h = np.empty_like(boxes)

    # keep = torch.Tensor()
    # print(idx)
    while idx.size > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.shape == 1: #TODO na pewno???
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        xx1 = np.take(x1,  idx, axis=0)
        yy1 = np.take(y1, idx, axis=0)
        xx2 = np.take(x2, idx, axis=0)
        yy2 = np.take(y2, idx, axis=0)
        
        # store element-wise max with next highest score

        xx1 = np.clip(xx1, x1[i], None)
        yy1 = np.clip(yy1, y1[i], None)
        xx2 = np.clip(xx2, None, x2[i])
        yy2 = np.clip(yy2, None, y2[i])

        w = np.resize(w,xx2.shape)
        h = np.resize(h,yy2.shape)
        
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration

        w = np.clip(w, 0.0, None)
        h = np.clip(h, 0.0, None)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = np.take(area, idx, axis=0)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.__le__(overlap)]
    return keep, count
