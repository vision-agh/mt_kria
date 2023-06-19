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

import numpy as np
import torch

from ..box_utils import nms

cfg = {
    'variance': [0.1, 0.2]
}


def delta2bbox(rois,
               deltas,
               means=(0., 0., 0., 0.),
               stds=(1., 1., 1., 1.),
               max_shape=None,
               wh_ratio_clip=16 / 1000):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 4)
        deltas (Tensor): Encoded offsets with respect to each roi.
            Has shape (N, 4 * num_classes). Note N = num_anchors * W * H when
            rois is a grid of anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates
        max_shape (tuple[int, int]): Maximum bounds for boxes. specifies (H, W)
        wh_ratio_clip (float): Maximum aspect ratio for boxes.

    Returns:
        Tensor: Boxes with shape (N, 4), where columns represent
            tl_x, tl_y, br_x, br_y.

    References:
        .. [1] https://arxiv.org/abs/1311.2524

    Example:
        >>> rois = torch.Tensor([[ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 5.,  5.,  5.,  5.]])
        >>> deltas = torch.Tensor([[  0.,   0.,   0.,   0.],
        >>>                        [  1.,   1.,   1.,   1.],
        >>>                        [  0.,   0.,   2.,  -1.],
        >>>                        [ 0.7, -1.9, -0.5,  0.3]])
        >>> delta2bbox(rois, deltas, max_shape=(32, 32))
        tensor([[0.0000, 0.0000, 1.0000, 1.0000],
                [0.1409, 0.1409, 2.8591, 2.8591],
                [0.0000, 0.3161, 4.1945, 0.6839],
                [5.0000, 5.0000, 5.0000, 5.0000]])
    """

    means = np.asarray(means).reshape(1, -1).repeat(1, deltas.shape[1] // 4)
    stds = np.asarray(stds).reshape(1, -1).repeat(1, deltas.shape[1] // 4)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = np.clip(dw, -max_ratio, max_ratio)
    dh = np.clip(dh, -max_ratio, max_ratio)

    # Compute center of each roi

    px = np.broadcast_to(np.expand_dims(((rois[:, 0] + rois[:, 2]) * 0.5),axis=1),dx.shape)
    py = np.broadcast_to(np.expand_dims(((rois[:, 1] + rois[:, 3]) * 0.5),axis=1),dy.shape)
    # Compute width/height of each roi
    pw = np.broadcast_to(np.expand_dims((rois[:, 2] - rois[:, 0]),axis=1),dw.shape)
    ph = np.broadcast_to(np.expand_dims((rois[:, 3] - rois[:, 1]),axis=1),dh.shape)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * np.exp(dw)
    gh = ph * np.exp(dh)
    # Use network energy to shift the center of each roi
    gx = px + pw * dx
    gy = py + ph * dy
    # Convert center-xy/width/height to top-left, bottom-right
    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    bboxes = np.stack((x1, y1, x2, y2), axis=-1).reshape(deltas.shape)
    return bboxes


class Detect():
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def forward(self, loc_ori, conf_data, prior_data, bseg_prob=None):
        loc_data = loc_ori[:, :, 0:4]
        dim = loc_ori.shape[2]
        if dim == 6:
            ori_data = loc_ori[:, :, 4:6]
        num = loc_data.shape[0]  # batch size
        num_priors = prior_data.shape[0]
        output = np.zeros((num, self.num_classes, self.top_k, dim + 1))

        conf_preds = np.transpose(conf_data.reshape(num, num_priors,
                                    self.num_classes),axes=(0,2,1))

        for i in range(num):
            stds = [self.variance[0], self.variance[0], self.variance[1], self.variance[1]]
            decoded_boxes = delta2bbox(prior_data, loc_data[i], stds=stds, max_shape=(320, 512))
            # For each class, perform nms
            conf_scores = np.copy(conf_preds[i])
            if dim == 6:
                ori_values = np.copy(ori_data[i])

            # seg_prob = bseg_prob[i]
            for cl in range(self.num_classes):
                c_mask = conf_scores[cl].__gt__(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.ndim == 0:
                    continue

                l_mask = np.broadcast_to(np.expand_dims(c_mask,axis=1),decoded_boxes.shape)
                boxes = decoded_boxes[l_mask].reshape(-1, 4)
                if dim == 6:

                    o_mask = np.broadcast_to(np.expand_dims(c_mask,axis=1),ori_values.shape)
                    oris = ori_values[o_mask].clamp(min=-1, max=1).reshape(-1, 2)
                # idx of highest scoring and non-overlapping boxes per class

                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)

                if dim == 4:
                    output[i, cl, :count] = \
                        np.concatenate((np.expand_dims(scores[ids[:count]], 1),
                                   boxes[ids[:count]]), 1)
                elif dim == 6:
                    output[i, cl, :count] = \
                        np.concatenate((np.expand_dims(scores[ids[:count]], 1),
                                   boxes[ids[:count]],
                                   oris[ids[:count]]), 1)
        # flt = output.contiguous().reshape(num, -1, dim + 1)
        # _, idx = flt[:, :, 0].sort(1, descending=True)
        # _, rank = idx.sort(1)
        # flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output
