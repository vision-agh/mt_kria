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

from collections import OrderedDict
# from ptflops import get_model_complexity_info #TODO UNCOMMENT COMPLEXITY INFO
import cv2

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from mmcv.runner import load_state_dict
from torch.autograd import Variable

import model_res18v2 as model_res18
from config import parse_args, solver, MEANS, BBOX_NAMES
from layers import *

import time
import numpy as np
import cv2

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import math

import pickle

import time

from scipy.special import expit
import serial



def letterbox_image(image, size):
    ih, iw, _ = image.shape
    w, h = size
    scale = min(w/iw, h/ih)
    #print(scale)
    
    nw = int(iw*scale)
    nh = int(ih*scale)
    #print(nw)
    #print(nh)

#     image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_LINEAR)
    
    new_image = np.ones((h,w,3), np.uint8) * 128
    h_start = (h-nh)//2
    w_start = (w-nw)//2
    new_image[h_start:h_start+nh, w_start:w_start+nw, :] = image
    return new_image

def pre_process(image, model_image_size):
    image = image[...,::-1]
    image_h, image_w, _ = image.shape
 
    if model_image_size != (None, None):
        assert model_image_size[0]%32 == 0, 'Multiples of 32 required'
        assert model_image_size[1]%32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
    else:
        new_image_size = (image_w - (image_w % 32), image_h - (image_h % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0) 	
    return image_data

with open("priors.pckl","rb") as f:
    priors = pickle.load(f)
detect = Detect(5, 0, 200, 0.05, 0.45)
label_colours = cv2.imread('cityscapes19.png', 1).astype(np.uint8)



def process_raw(im, device, net, detect):

    x = im.astype(np.float32)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    x = x.to(device)



    start_job = time.time()
    loc_or, conf_dat, seg_data, drivable_data, depth_data, lane_data = net(x)

    end_job = time.time()
    print(f"job time: {end_job - start_job}, FPS: {1/(end_job - start_job)}")

    conf_dat, centerness = conf_dat

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
    seg_raw = np.squeeze(seg_data.data.max(1)[1].cpu().numpy(), axis=0)
 
    if drivable_data is not None:
        drivable_data_logits = drivable_data.data.cpu().numpy()
        driv = np.squeeze(drivable_data.data.max(1)[1].cpu().numpy(), axis=0)
        driv = cv2.resize(driv, (512,320), interpolation=cv2.INTER_NEAREST)

    if lane_data is not None:
        if args.lane_mode == 'lnpy':
            lane_raw = np.squeeze(torch.sigmoid(lane_data).data.cpu().numpy()[:, 0, :, :], axis=0)
            lane_raw = cv2.resize(lane_raw, (512,320), interpolation=cv2.INTER_NEAREST)
        else:
            lane_raw = np.squeeze(lane_data.data.max(1)[1].cpu().numpy(), axis=0)
            lane_raw = cv2.resize(lane_raw, (512,320), interpolation=cv2.INTER_NEAREST)
    
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

        count = count + 1



    return final_dets, lane_raw, seg_raw, driv





def process(input_image, device, net, detect, debug = False):
    start = time.time()
    # final_dets, lane_raw, seg_raw, driv = process_raw(input_image)
    final_dets, lane_raw, seg_raw, driv = process_raw(input_image, device, net, detect)
    end = time.time()
    print(f"raw process with preprocessing: {end - start}, FPS: {1/(end - start)}")
    
    BBOX_NAMES = ['human', 'light_yellow', 'light_green', 'light_red', 'obstacle']
    input_image = cv2.resize(input_image, (512,320))
    # input_image = cv2.cvtColor(cv2.resize(input_image, (512,320)),cv2.COLOR_RGB2BGR)
    count = 0
    for boxes, scores in final_dets:
        for num in range(len(boxes[:, 0])):
            if scores[num] > 0.55:
                p1 = (int(boxes[num, 0]), int(boxes[num, 1]))
                p2 = (int(boxes[num, 2]), int(boxes[num, 3]))
                
                if debug:
                    cv2.rectangle(input_image, p1, p2, (0, 0, 255), 2)
                    p3 = (max(p1[0], 20), max(p1[1], 20))
                    title = "%s" % (BBOX_NAMES[count])
                    cv2.putText(input_image, title, p3, cv2.FONT_ITALIC, 0.9, (0, 0, 255), 2)
        count += 1


    # print(driv[1].shape)
    # driv = driv[:,:,1]
    
    # driv = driv.argmax(2).astype('float32')
#     if debug:
    driv = cv2.resize(driv, (512,320))


    # seg_data = seg_raw.argmax(2)

    seg_stack =  np.dstack([seg_raw, seg_raw, seg_raw]).astype(np.uint8)
    if debug:
        seg = cv2.LUT(seg_stack,label_colours)
    else:
        seg = seg_stack
    lane_data = lane_raw
    lane = expit(lane_data)

    return input_image, seg, driv, lane

def find_lines(lane,lane2, should_plot=False):
    lane_bin = np.zeros(lane.shape)
    lane_bin[lane>0.25]=255

    lane_bin[280:,:]=0
    lane2[280:,:,:]=0


    edges = cv2.Canny(np.uint8(lane_bin), 75, 100)
    

    # lines = cv2.HoughLinesP(edges, 2, np.pi / 180, 50, None, 50, 20)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 45, None, 50, 20)

    if lines is not None:

            leftlinedict = {
                 "ln": [],
                 "ang": [],
                 "x": [],
                 "y": []
            }
            for i in range(0, len(lines)):
                l = lines[i][0]
                x1, y1, x2, y2 = lines[i][0]
                
                # print(abs(y1 - y2), lngth)
                # stop_roi = max(100,max(x1,x2))-min(400,min(x1,x2))
                stop_roi = np.abs(min(400,max(100,x1))-min(400,max(100,x2)))/np.abs(x1-x2)
                if abs(y1 - y2) < 10 and min(y1,y2) > 175 and stop_roi>0.5:
                    # print(f"line(x1,y1,x2,y2): {l}abs:{abs(x1 - x2)}, len: {lngth},stop roi: {stop_roi}")
                    if should_plot:
                        cv2.line(lane2, (x1, y1), (x2, y2), (255, 0, 0), 3, cv2.LINE_AA)
                    # if(min(y1,y2) > 220):
                    #      print("STOP")
                elif max(y1,y2)>200 and min(x1,x2)<125:
                    lngth = np.sqrt(np.square(x1-x2)+np.square(y1-y2))
                    angle = np.abs(y1-y2) / np.abs(x1-x2)

                    leftlinedict["ln"].append(lngth)
                    leftlinedict["ang"].append(angle)

                    if should_plot:
                        leftlinedict["x"].append(min(x1,x2))
                        leftlinedict["y"].append(max(y1,y2))

                elif should_plot:
                    cv2.line(lane2, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
            if len(leftlinedict["ln"]):
                lngth = np.max(leftlinedict["ln"])
                angle = np.mean(leftlinedict["ang"])
                if should_plot:
                    startx = int(np.mean(leftlinedict["x"]))
                    starty = int(np.mean(leftlinedict["y"]))
                    X = lngth/np.sqrt(1+np.square(angle))
                    Y = angle * X
                    endx = int(startx + X)
                    endy = int(starty - Y)
                    cv2.line(lane2, (startx, starty), (endx, endy), (0, 255, 0), 3, cv2.LINE_AA)
    if should_plot:
        plt.imshow(lane2)
        plt.gray()
        plt.show()


# frame_in_w = 512
# frame_in_h = 320
# videoIn = cv2.VideoCapture(0 + cv2.CAP_V4L2)
# videoIn.set(cv2.CAP_PROP_FRAME_WIDTH, frame_in_w);
# videoIn.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_in_h);
# print("capture device is open: " + str(videoIn.isOpened()))

args = parse_args()



if __name__ == '__main__':
    # load net
    det_classes = solver['det_classes']  # +1 for background
    seg_classes = solver['seg_classes']
    drivable_classes = solver['drivable_classes']
    reg_depth = solver['reg_depth']
    seg_lane = solver['seg_lane']

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

    # MAIN -------------------------------------------------------------
    # pics = [        'demo/img_00254.jpg','img_00301.jpg','img_01250.png','img_02521.png','img_02539.png',
    #     'img_02575.png','img_00793.png','img_00795.png','img_00796.png']

    path = 'demo_data/images/img_00254.jpg'
    try:
        # for path in pics:
        while True:
            debug = False
        #     img = cv2.resize(cv2.imread(path),(512,320))
            start = time.time()
            # ret, frame_vga = videoIn.read()
            # img = cv2.resize(frame_vga,(512,320))
            img = cv2.resize(cv2.imread(path),(512,320))

        #     print(ret, frame_vga.shape)

            input_image, seg, driv,  lane = process(img, device, net, detect,debug = debug)
            # input_image, seg, driv,  lane = process(img, debug = debug)



            driv2 = np.zeros_like(input_image)
            driv2[driv>0.5]=(128,128,128)


            lane2 = np.zeros_like(input_image)
            lane2[driv>0.5]=(128,128,128)
            lane2[lane>0.15]=(255,255,255)
            if debug:
                row1 = np.hstack((input_image, seg))
                row2 = np.hstack((driv2, lane2))

                final = np.vstack((row1,row2))

                # plt.figure(figsize = (15,15)) 
                # plt.imshow(final)
                # plt.axis('off')
                # plt.show()
                input_image = cv2.cvtColor(input_image,cv2.COLOR_RGB2BGR)
                plt.imshow(input_image)
                plt.show()
            # print(lane.shape)

            find_lines(lane,lane2,should_plot=debug)
            # ser.write("0,0,0\n".encode())
            end = time.time()
            print(f"full proces: {end - start}, FPS: {1/(end - start)}\n\n")
    except KeyboardInterrupt:
        pass

 