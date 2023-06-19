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

import math
import os
import time
from collections import OrderedDict
# from ptflops import get_model_complexity_info #TODO UNCOMMENT COMPLEXITY INFO
import cv2
import imgaug.augmenters as iaa
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from matplotlib.colors import Normalize
from mmcv.runner import load_state_dict
from torch.autograd import Variable

import model_res18v2 as model_res18
from config import parse_args, solver, MEANS, BBOX_NAMES
from layers import *

labelmap_det = (  # always index 0
    '0', '1', '2', '3', '4')

args = parse_args()


class CropKitti(object):
    def crop_img(self, img):
        height, width, channels = img.shape
        top, left = int(height - 352), int((width - 1216) / 2)
        return img[top:top + 352, left:left + 1216]
        # return img[:, left:left + 1216]

    def __call__(self, img, seg=None):
        if seg is None:
            return self.crop_img(img)
        else:
            return self.crop_img(img), self.crop_img(seg)


class KittiCrop(object):
    def __init__(self, aspect_ratios=(1280 / 720, 1920 / 1280, 1920 / 886, 2048 / 1024)):
        self.aspect_ratios = aspect_ratios

    def __call__(self, image, seg=None):
        aspect_ratio = self.aspect_ratios[0]
        crop = iaa.CropToAspectRatio(aspect_ratio, position='center-top').to_deterministic()
        if seg is not None:
            return crop(image=image), crop(image=seg)
        else:
            return crop(image=image)


def base_transform(image, size, mean):
    h, w, c = image.shape
    scale_h = size[0] / h
    scale_w = size[1] / w
    x = cv2.resize(image, (size[1], size[0])).astype(np.float32)
    x = x.astype(np.float32)
    x -= mean
    x = x / 255.0
    x = x[:, :, ::-1].copy()
    info = dict(
        scale_h=scale_h,
        scale_w=scale_w
    )
    return x, info


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image):
        return base_transform(image, self.size, self.mean)


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def get_voc_results_file_template(cls):
    filename = 'det_test' + '_%s.txt' % (cls)
    filedir = os.path.join(args.save_folder, 'det')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, ids):
    for cls_ind, cls in enumerate(labelmap_det):
        print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(cls)
        with open(filename, 'w') as f:
            for im_ind, index in enumerate(ids):
                dets = all_boxes[cls_ind + 1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    if dets.shape[1] == 5:
                        f.write('{:s} {:s} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\n'.
                                format(index[1], cls, dets[k, -1],
                                       dets[k, 0] , dets[k, 1] ,
                                       dets[k, 2] , dets[k, 3] ))
                    elif dets.shape[1] == 7:
                        angle_sin = math.asin(dets[k, 4]) / math.pi * 180
                        angle_cos = math.acos(dets[k, 5]) / math.pi * 180
                        if dets[k, 4] >= 0 and dets[k, 5] >= 0:
                            angle = (angle_sin + angle_cos) / 2
                        elif dets[k, 4] < 0 and dets[k, 5] >= 0:
                            angle = (angle_sin - angle_cos) / 2
                        elif dets[k, 4] < 0 and dets[k, 5] < 0:
                            angle = (-180 - angle_sin - angle_cos) / 2
                        elif dets[k, 4] >= 0 and dets[k, 5] < 0:
                            angle = (angle_cos + 180 - angle_sin) / 2
                        f.write('{:s} {:s} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\n'.
                                format(index[1], cls, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
                        # angle))


def test_net(save_folder, net, device, ids, detect, transform, thresh=0.01, has_ori=False):
    num_images = len(ids)
    softmax = nn.Softmax(dim=-1)
    if args.i_depth:
        img_path = os.path.join('%s', '%s')
    else:
        if args.img_mode == 2:
            img_path = os.path.join('%s', 'images', '%s.jpg')
        elif args.img_mode == 1:
            img_path = os.path.join('%s', '%s.jpg')
        else:
            raise ValueError("img_mode: {} does not implemented".format(str(args.img_mode)))

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap_det) + 1)]
    save_seg_root = os.path.join(save_folder, 'seg')
    save_drivable_root = os.path.join(save_folder, 'drivable')
    save_lane_root = os.path.join(save_folder, 'lane')
    save_depth_root = os.path.join(save_folder, 'depth')
    os.makedirs(save_drivable_root, exist_ok=True)
    os.makedirs(save_depth_root, exist_ok=True)
    if not os.path.exists(save_seg_root):
        os.mkdir(save_seg_root)
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    full_detect_time = []
    for i in range(num_images):
        img_id = ids[i]
        file_path = img_path % img_id
        im = cv2.imread(file_path)
        if im is None:
            print(file_path)
        if args.i_depth:
            im = CropKitti()(im)
            im = KittiCrop()(im)
        im_size = (im.shape[1], im.shape[0])
        im_tensor, info = transform(im)
        scale_h = info['scale_h']
        x = torch.from_numpy(im_tensor).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        x = x.to(device)

        _t['im_detect'].tic()
        if args.quant_mode == 'float':
            loc_or, conf_dat, seg_data, drivable_data, depth_data, lane_data = net(x)
            conf_dat, centerness = conf_dat

        else:
            loc_0, loc_1, loc_2, loc_3, loc_4, loc_5, loc_6, \
            conf_0, conf_1, conf_2, conf_3, conf_4, conf_5, conf_6, \
            centerness_0, centerness_1, centerness_2, centerness_3, centerness_4, centerness_5, centerness_6, \
            seg_data, drivable_data, depth_data, lane_data = net(x)
            loc_or = (loc_0, loc_1, loc_2, loc_3, loc_4, loc_5, loc_6)
            conf_dat = (conf_0, conf_1, conf_2, conf_3, conf_4, conf_5, conf_6)
            centerness = (
                centerness_0, centerness_1, centerness_2, centerness_3, centerness_4, centerness_5, centerness_6)
        if not args.i_det:
            loc_or = None
            conf_dat = None
        if not args.i_seg:
            seg_data = None
        if not args.i_drive:
            drivable_data = None
        if not args.i_depth:
            depth_data = None
        if not args.i_lane:
            lane_data = None

        detect_time = _t['im_detect'].toc(average=False)
        full_detect_time.append(detect_time)

        if seg_data is not None:
            seg_prob = torch.nn.functional.softmax(seg_data, 1)[0].permute(1, 2, 0).cpu().numpy()
            seg_prob = cv2.resize(seg_prob, im_size, interpolation=cv2.INTER_LINEAR)
            seg_data = np.squeeze(seg_data.data.max(1)[1].cpu().numpy(), axis=0)
            seg_data = cv2.resize(seg_data, im_size, interpolation=cv2.INTER_NEAREST)
            save_seg_file = os.path.join(save_seg_root, img_id[1] + '.png')
            cv2.imwrite(save_seg_file, seg_data)

        if loc_or is not None and conf_dat is not None:
            priorbox = PriorBox(solver)
            priors = Variable(priorbox.forward(), volatile=True).to(device)
            loc_ori = list()
            conf_data = list()
            centerness_data = list()
            for loc in loc_or:
                loc_ori.append(loc.permute(0, 2, 3, 1).contiguous().to(device))
            loc_ori = torch.cat([o.view(o.size(0), -1) for o in loc_ori], 1).to(device)
            loc_ori = loc_ori.view(loc_ori.size(0), -1, 4).to(device)
            for conf, cterness in zip(conf_dat, centerness):
                conf_data.append(conf.permute(0, 2, 3, 1).contiguous().to(device))
                centerness_data.append(cterness.permute(0, 2, 3, 1).contiguous().to(device))
            conf_data = torch.cat([o.view(o.size(0), -1) for o in conf_data], 1).to(device)
            centerness_data = torch.cat([o.view(o.size(0), -1) for o in centerness_data], 1).to(device)
            conf_data = conf_data.view(conf_data.size(0), -1, solver['det_classes']).to(device)
            centerness_data = centerness_data.view(centerness_data.size(0), -1, 1).to(device)
            pred = detect(loc_ori, torch.sigmoid(conf_data) * torch.sigmoid(centerness_data), priors)
            detections = pred.data

            for j in range(detections.size(1)):
                dets = detections[0, j, :]
                feature_dim = dets.size(1)
                mask = dets[:, 0].gt(0.).expand(feature_dim, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, feature_dim)
                if dets.dim() == 0:
                    continue
                boxes = dets[:, 1:5]
                boxes[:, 0] /= 512
                boxes[:, 2] /= 512
                boxes[:, 1] /= 320
                boxes[:, 3] /= 320
                boxes[:, 0] *= im_size[0]
                boxes[:, 2] *= im_size[0]
                boxes[:, 1] *= im_size[1]
                boxes[:, 3] *= im_size[1]
                scores = dets[:, 0].cpu().numpy()

                if has_ori:
                    cls_dets = np.hstack((dets[:, 1:feature_dim].cpu().numpy(),
                                          scores[:, np.newaxis])).astype(np.float32,
                                                                         copy=False)
                else:
                    cls_dets = np.hstack((boxes.cpu().numpy(),
                                          scores[:, np.newaxis])).astype(np.float32,
                                                                         copy=False)
                all_boxes[j + 1][i] = cls_dets

        if drivable_data is not None:
            drivable_data = np.squeeze(drivable_data.data.max(1)[1].cpu().numpy(), axis=0)
            drivable_data = cv2.resize(drivable_data, im_size, interpolation=cv2.INTER_NEAREST)
            save_drivable_file = os.path.join(save_drivable_root, img_id[1] + '.png')
            cv2.imwrite(save_drivable_file, drivable_data)
        if lane_data is not None:
            lane_data = np.squeeze(lane_data.data.max(1)[1].cpu().numpy(), axis=0)
            lane_data = cv2.resize(lane_data, im_size, interpolation=cv2.INTER_NEAREST)
            save_lane_file = os.path.join(save_lane_root, img_id[1] + '.png')
            cv2.imwrite(save_lane_file, lane_data)
        if depth_data is not None:
            pred_depth = depth_data.data.cpu().numpy().squeeze()
            pred_depth = cv2.resize(pred_depth, im_size, interpolation=cv2.INTER_LINEAR)
            filename_pred_png = os.path.join(save_depth_root, img_id[1].replace('jpg', 'png'))
            if 'png' not in filename_pred_png:
                filename_pred_png = filename_pred_png + '.png'
            filename_pred_dir = os.path.dirname(filename_pred_png)
            os.makedirs(filename_pred_dir, exist_ok=True)
            pred_depth_scaled = pred_depth * 256.0 * scale_h
            pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
            cv2.imwrite(filename_pred_png, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        print('im_detect: {:d}/{:d} {:.3f}s\r'.format(i + 1,
                                                      num_images, detect_time), end='')

    print(f"DETECTION TIMES: {np.mean(full_detect_time[1:])}")
    print(f"FPS: {1/np.mean(full_detect_time[1:])}")

    if args.i_det:
        print('Saving the detection results...')
        write_voc_results_file(all_boxes, ids)


def get_palette(dataset):
    if dataset == 'bdd' or dataset == 'bdd_toy':
        return np.array([[0, 0, 0], [0, 255, 127], [0, 127, 255], [255, 0, 0]], dtype=np.uint8)
    if dataset == 'pascal' or dataset == 'pascal_toy':
        n = 21
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while (lab > 0):
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i = i + 1
                lab >>= 3
        res = np.array(palette)
        res = np.array([[res[i], res[i + 1], res[i + 2]] for i in range(0, len(res), 3)], dtype=np.uint8)
        return res


def demo(save_folder, net, device, ids, detect, transform, thresh=0.01, has_ori=False,
         save_image=True, save_video=None):
    num_images = len(ids)
    full_detect_time = []
    if args.img_mode == 2:
        img_path = os.path.join('%s', 'images', '%s.jpg')
    elif args.img_mode == 1:
        img_path = os.path.join('%s', '%s.jpg')
    else:
        raise ValueError("img_mode: {} does not implemented".format(str(args.img_mode)))
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap_det) + 1)]
    os.makedirs(save_folder, exist_ok=True)
    save_seg_root = os.path.join(save_folder, 'det_and_seg')
    save_drivable_root = os.path.join(save_folder, 'drivable')
    save_depth_root = os.path.join(save_folder, 'depth')
    os.makedirs(save_drivable_root, exist_ok=True)
    os.makedirs(save_depth_root, exist_ok=True)
    if not os.path.exists(save_seg_root):
        os.mkdir(save_seg_root)
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer(), 'postprocess': Timer()}
    label_colours = cv2.imread('cityscapes19.png', 1).astype(np.uint8)
    figsize = [19.2 / 3 * 2, 10.8]
    dpi = 100
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        vout = cv2.VideoWriter(save_video, fourcc, 30, (int(figsize[0] * 100), int(figsize[1] * 100)))

    for i in range(num_images):
        img_id = ids[i]
        file_path = img_path % img_id
        im = cv2.imread(file_path)
        # im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        raw_im = np.copy(im)
        im_size = (im.shape[1], im.shape[0])
        im_tensor, info = transform(im)
        scale_h = info['scale_h']
        x = torch.from_numpy(im_tensor).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        if args.device == 'gpu':
            x = x.to(device)
        _t['im_detect'].tic()

        # ONNX GENERATION
        # torch.onnx.export(net, x, "model.onnx",output_names=["detection","segmentation", "drivable", "depth", "lane"])
        # return 0
        if args.quant_mode == 'float':
            loc_or, conf_dat, seg_data, drivable_data, depth_data, lane_data = net(x)
            conf_dat, centerness = conf_dat
            # loc_0, loc_1, loc_2, loc_3, loc_4, loc_5, loc_6 = loc_or
            # conf_0, conf_1, conf_2, conf_3, conf_4, conf_5, conf_6 = conf_dat
            # centerness_0, centerness_1, centerness_2, centerness_3, centerness_4, centerness_5, centerness_6 = centerness
        else:
            loc_0, loc_1, loc_2, loc_3, loc_4, loc_5, loc_6, \
            conf_0, conf_1, conf_2, conf_3, conf_4, conf_5, conf_6, \
            centerness_0, centerness_1, centerness_2, centerness_3, centerness_4, centerness_5, centerness_6, \
            seg_data, drivable_data, depth_data, lane_data = net(x)
            loc_or = (loc_0, loc_1, loc_2, loc_3, loc_4, loc_5, loc_6)
            conf_dat = (conf_0, conf_1, conf_2, conf_3, conf_4, conf_5, conf_6)
            centerness = (
                centerness_0, centerness_1, centerness_2, centerness_3, centerness_4, centerness_5, centerness_6)
        detect_time = _t['im_detect'].toc(average=False)
        full_detect_time.append(detect_time)

        # COMPLEXITY INFO GENERATION
        # macs, params = get_model_complexity_info(net, (3, 320, 512), as_strings=True,
        #                                    print_per_layer_stat=True, verbose=False)
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        priorbox = PriorBox(solver)
        priors = Variable(priorbox.forward(), volatile=True).to(device)
        loc_ori = list()
        conf_data = list()
        centerness_data = list()
        for loc in loc_or:
            loc_ori.append(loc.permute(0, 2, 3, 1).contiguous().to(device))
        loc_ori = torch.cat([o.view(o.size(0), -1) for o in loc_ori], 1).to(device)
        loc_ori = loc_ori.view(loc_ori.size(0), -1, 4).to(device)
        for conf, cterness in zip(conf_dat, centerness):
            conf_data.append(conf.permute(0, 2, 3, 1).contiguous().to(device))
            centerness_data.append(cterness.permute(0, 2, 3, 1).contiguous().to(device))
        conf_data = torch.cat([o.view(o.size(0), -1) for o in conf_data], 1).to(device)
        centerness_data = torch.cat([o.view(o.size(0), -1) for o in centerness_data], 1).to(device)
        conf_data = conf_data.view(conf_data.size(0), -1, solver['det_classes']).to(device)
        centerness_data = centerness_data.view(centerness_data.size(0), -1, 1).to(device)
        pred = detect(loc_ori, torch.sigmoid(conf_data) * torch.sigmoid(centerness_data), priors)
        detections = pred.data
        seg_data = np.squeeze(seg_data.data.max(1)[1].cpu().numpy(), axis=0)
        seg_data = cv2.resize(seg_data, im_size, interpolation=cv2.INTER_NEAREST)
        seg_data_color = np.dstack([seg_data, seg_data, seg_data]).astype(np.uint8)
        seg_data_color_img = cv2.LUT(seg_data_color, label_colours)

        if drivable_data is not None:
            drivable_data_logits = drivable_data.data.cpu().numpy()
            drivable_data = np.squeeze(drivable_data.data.max(1)[1].cpu().numpy(), axis=0)
            drivable_data = cv2.resize(drivable_data, im_size, interpolation=cv2.INTER_NEAREST)
            drivable_data_color_img = get_palette('bdd')[:, ::-1][drivable_data.squeeze()]
        if lane_data is not None:
            if args.lane_mode == 'lnpy':
                lane_data_logits = lane_data.data.cpu().numpy()
                lane_data = np.squeeze(torch.sigmoid(lane_data).data.cpu().numpy()[:, 0, :, :], axis=0)
                lane_data = cv2.resize(lane_data, im_size, interpolation=cv2.INTER_NEAREST)
                lane_data_color_img = \
                    np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)[:, ::-1][
                        (lane_data > 0.8).astype('int32').squeeze()]
            else:
                lane_data_logits = lane_data.data.cpu().numpy()
                lane_data = np.squeeze(lane_data.data.max(1)[1].cpu().numpy(), axis=0)
                lane_data = cv2.resize(lane_data, im_size, interpolation=cv2.INTER_NEAREST)
                lane_data_color_img = \
                    np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)[:, ::-1][
                        lane_data.squeeze()]
        if depth_data is not None:
            pred_depth = depth_data.data.cpu().numpy().squeeze()
            # pred_depth = cv2.resize(pred_depth, im_size, interpolation=cv2.INTER_LINEAR)
            pred_depth = cv2.resize(pred_depth, im_size, interpolation=cv2.INTER_NEAREST)
            pred_depth[pred_depth > 80] = 80
            pred_depth[pred_depth < 1] = 1
            pred_depth *= scale_h
            inverse_depth = 1 / pred_depth
            # inverse_depth = 1 / (np.log(pred_depth) + 1e-9) - 1
            # inverse_depth = 90 - pred_depth
            filename_pred_png = os.path.join(save_depth_root, img_id[1])
            filename_pred_dir = os.path.dirname(filename_pred_png)
            os.makedirs(filename_pred_dir, exist_ok=True)
            vmax = np.percentile(inverse_depth, 85)
            normalizer = Normalize(vmin=inverse_depth.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(inverse_depth)[:, :, :3] * 255).astype(np.uint8)
        
        # skip j = 0, because it's the background class
        count = 0
        final_dets = []
        for j in range(detections.size(1)):
            dets = detections[0, j, :]
            feature_dim = dets.size(1)
            mask = dets[:, 0].gt(0.).expand(feature_dim, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, feature_dim)
            if dets.dim() == 0:
                continue
            boxes = dets[:, 1:5]
            scores = dets[:, 0].cpu().numpy()
            final_dets.append((boxes.cpu().numpy(), scores))
            if has_ori:
                cls_dets = np.hstack((dets[:, 1:feature_dim].cpu().numpy(),
                                      scores[:, np.newaxis])).astype(np.float32,
                                                                     copy=False)
            else:
                cls_dets = np.hstack((boxes.cpu().numpy(),
                                      scores[:, np.newaxis])).astype(np.float32,
                                                                     copy=False)
            all_boxes[j+1][i] = cls_dets
            count = count + 1

        plt.axis('off')
        plt.figure(dpi=dpi, figsize=figsize)
        im_seg = np.array(np.zeros(raw_im.shape))
        im_seg[seg_data_color == 16] = raw_im[seg_data_color == 16]
        im_seg[seg_data_color != 16] = raw_im[seg_data_color != 16] / 2 + seg_data_color_img[seg_data_color != 16] / 2
        plt.subplot(2, 2, 2)
        plt.imshow(im_seg[:, :, ::-1].astype('uint8'))
        plt.axis('off')
        if drivable_data is not None:
            im_drivable = np.array(np.zeros(raw_im.shape))
            im_drivable[drivable_data == 0] = raw_im[drivable_data == 0]
            im_drivable[drivable_data != 0] = raw_im[drivable_data != 0] / 2 + drivable_data_color_img[
                drivable_data != 0] / 2
            plt.subplot(2, 2, 3)
            plt.imshow(im_drivable[:, :, ::-1].astype('uint8'))
            plt.axis('off')
        
        im_fin = np.array(np.zeros(raw_im.shape))
        x1 = np.bitwise_and(drivable_data == 0 , lane_data == 0)

        im_fin[x1 == True] = raw_im[x1 == True] 
        
        im_fin[drivable_data != 0] = raw_im[drivable_data != 0] / 2 + drivable_data_color_img[drivable_data != 0] / 3
        im_fin[lane_data != 0] = raw_im[lane_data != 0] / 2 + lane_data_color_img[lane_data != 0] / 2

        if lane_data is not None:
            im_lane = im
            im_lane[lane_data == 0] = im[lane_data == 0]
            im_lane[lane_data != 0] = im[lane_data != 0] / 2 + lane_data_color_img[lane_data != 0] / 2
        count = 0       
        for boxes, scores in final_dets:
            boxes[:, 0] /= 512
            boxes[:, 2] /= 512
            boxes[:, 1] /= 320
            boxes[:, 3] /= 320
            boxes[:, 0] *= raw_im.shape[1]
            boxes[:, 2] *= raw_im.shape[1]
            boxes[:, 1] *= raw_im.shape[0]
            boxes[:, 3] *= raw_im.shape[0]
            for num in range(len(boxes[:, 0])):
                if scores[num] > 0.35:
                    p1 = (int(boxes[num, 0]), int(boxes[num, 1]))
                    p2 = (int(boxes[num, 2]), int(boxes[num, 3]))
                    cv2.rectangle(im, p1, p2, (0, 0, 255), 2)
                    cv2.rectangle(im_fin, p1, p2, (0, 0, 255), 2)
                    p3 = (max(p1[0], 20), max(p1[1], 20))
                    title = "%s" % (BBOX_NAMES[count])
                    cv2.putText(im, title, p3, cv2.FONT_ITALIC, 0.9, (0, 0, 255), 2)
                    cv2.putText(im_fin, title, p3, cv2.FONT_ITALIC, 0.9, (0, 0, 255), 2)
            count += 1
        plt.subplot(2, 2, 1)
        plt.imshow(im[:, :, ::-1].astype('uint8'))
        plt.axis('off')
        if colormapped_im is not None:
            plt.subplot(2, 2, 4)
            plt.imshow(im_fin[:, :, ::-1].astype('uint8'))
            plt.axis('off')
        print('im_detect: {:d}/{:d} {:.3f}s\r'.format(i + 1,
                                                      num_images, detect_time), end="")
        save_png = os.path.join(save_folder, img_id[1] + '.png')
        save_fin = save_png.replace(".png","_fin.png")
        plt.subplots_adjust(wspace=0.05, hspace=0.01)
        if save_image:
            plt.savefig(save_png)
            cv2.imwrite(save_fin, im_fin)
        if save_video:
            fig = plt.gcf()
            fig.canvas.draw()
            # convert canvas to image
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                                sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # print(img.shape)
            # img is rgb, convert to opencv's default bgr
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            vout.write(img)
        plt.close()
    if save_video:
        vout.release()
    print(f"DETECTION TIMES: {np.mean(full_detect_time[1:])}")
    print(f"FPS: {1/np.mean(full_detect_time[1:])}")
    print('Saving the detection results...')
    write_voc_results_file(all_boxes, ids)


if __name__ == '__main__':
    # load net
    det_classes = solver['det_classes']  # +1 for background
    seg_classes = solver['seg_classes']
    drivable_classes = solver['drivable_classes']
    reg_depth = solver['reg_depth']
    seg_lane = solver['seg_lane']

    if args.dump_xmodel:
        args.device = 'cpu'
        args.val_batch_size = 1

    if args.device == 'cpu':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        cudnn.benchmark = True

    net = model_res18.build_model(det_classes, seg_classes, drivable_classes, reg_depth, seg_lane, dev=args.device).to(device)
    state_dict = torch.load(args.trained_model, map_location=device)
    if 'model' in state_dict:
        state_dict = state_dict['model']

    if 'model' in state_dict:
        state_dict = state_dict['model']

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[0:7] == 'module.':
            name = k[7:]
            new_state_dict[name] = v
        else:
            name = k
            new_state_dict[name] = v
    load_state_dict(net, new_state_dict, strict=False)
    net.eval()
    torch.set_grad_enabled(False)
    print('Finished loading model!')

    if args.quant_mode == 'float':
        quant_model = net
    else:
        from pytorch_nndct.apis import torch_quantizer, dump_xmodel

        example = torch.rand(1, 3, 320, 512)
        quantizer = torch_quantizer(args.quant_mode, net, (example), output_dir=args.quant_dir, device=device)
        quant_model = quantizer.quant_model

    if args.eval:
        ids = list()
        if args.i_depth_eigen:
            for i, line in enumerate(open('evaluation/eigen_test_files_with_gt.txt')):
                ids.append((args.image_root, os.path.join('inputs', line.strip().split(' ')[0])))
                if args.max_test and i > args.max_test - 1:
                    break
        else:
            for i, line in enumerate(open(os.path.join(args.image_root, args.image_list))):
                if args.i_depth and not args.i_depth_eigen:
                    # if 'camera1' not in line:
                    #    continue
                    line = line.strip().split(' ')[0]
                else:
                    line = line.strip()
                ids.append((args.image_root, line))
                if args.max_test and i > args.max_test - 1:
                    break
        detect = Detect(det_classes, 0, 200, args.confidence_threshold,
                        0.45)  # num_classes, backgroung_label, top_k, conf_threshold, nms_threshold

        if args.quant_mode == 'test' and args.dump_xmodel:
            # deploy_check= True if args.dump_golden_data else False
            test_net(args.save_folder, quant_model, device, ids[:1], detect,
                     BaseTransform(solver['resize'], MEANS),
                     thresh=args.confidence_threshold, has_ori=False)
            dump_xmodel(args.quant_dir, deploy_check=True)
        else:
            test_net(args.save_folder, quant_model, device, ids, detect,
                     BaseTransform(solver['resize'], MEANS),
                     thresh=args.confidence_threshold, has_ori=False)

        if args.quant_mode == 'calib':
            quantizer.export_quant_config()

    else:
        ids = list()
        for line in open(args.demo_image_list):
            ids.append((args.image_root, line.strip()))
        detect = Detect(det_classes, 0, 200, args.confidence_threshold,
                        0.45)  # num_classes, backgroung_label, top_k, conf_threshold, nms_threshold
        demo(args.demo_save_folder, quant_model, device, ids, detect,
             BaseTransform(solver['resize'], MEANS),
             thresh=args.confidence_threshold, has_ori=False, save_image=args.save_image, save_video=args.save_video)
