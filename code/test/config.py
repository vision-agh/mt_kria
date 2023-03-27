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

import argparse
import os

import torch

solver = {
    'k1': 8,
    'k2': 8,
    'act_clip_val': 8,
    ' warmup': False,
    'det_classes': 5,
    'seg_classes': 6,
    'drivable_classes': 3, #TODO PUBL -> 2
    'reg_depth': 80,
    'seg_lane': 2,
    'lr_steps': (12000, 18000),
    # 'lr_steps': (5, 10),
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
MEANS = (104, 117, 123)
BBOX_NAMES = ['human', 'light_yellow', 'light_green', 'light_red', 'obstacle']


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Evaluation')
    parser.add_argument('--trained_model',
                        default='./weights/iter_v3_finetune_4000.pth', type=str,
                        help='Trained state_dict file path to open')
    parser.add_argument('--save_folder', default='result/', type=str,
                        help='File path to save results')
    parser.add_argument('--demo_save_folder', default='demo/', type=str,
                        help='File path to save results')
    parser.add_argument('--confidence_threshold', default=0.01, type=float,
                        help='Detection confidence threshold')
    parser.add_argument('--image_root', default=None, type=str,
                        help='image_root_path')
    parser.add_argument('--image_list', default=None, type=str,
                        help='image_list')
    parser.add_argument('--demo_image_list', default=None, type=str,
                        help='demo_image_list')
    parser.add_argument('--eval', action='store_true',
                        help='evaluation mode')
    parser.add_argument('--i_det', action='store_true', default=False, help='Detection test mode')
    parser.add_argument('--i_seg', action='store_true', default=False, help='Segmentation test mode')
    parser.add_argument('--i_drive', action='store_true', default=False, help='Drivable area test mode')
    parser.add_argument('--i_depth', action='store_true', default=False, help='Depth estimation test mode')
    parser.add_argument('--i_lane', action='store_true', default=False, help='Lane segmentation mode')
    parser.add_argument('--i_depth_eigen', action='store_true', default=False,
                        help='Depth estimation kitti eigen dataset test mode')
    parser.add_argument('--save_video', default=None, help='Save video')
    parser.add_argument('--save_image', default=True, type=str2bool, help='Save video')

    parser.add_argument('--img_mode', type=int, default=1)
    parser.add_argument('--quant_dir', type=str, default='quantize_result',
                        help='Path to save quant info')
    parser.add_argument('--quant_mode', default='calib', choices=['float', 'calib', 'test'],
                        help='Quantization mode. 0: no quantization, evaluate float model, '
                             'calib: quantize, '
                             'test: evaluate quantized model')
    parser.add_argument('--dump_xmodel', dest='dump_xmodel', action='store_true',
                        help='Dump xmodel after test')
    parser.add_argument('--device', default='gpu', choices=['gpu', 'cpu'],
                        help='Assign runtime device')
    parser.add_argument('--lane_mode', default='png', choices=['png', 'npy'],
                        help='Lane data format')
    parser.add_argument('--model', default='v2', help='Model')
    parser.add_argument('--max_test', default=None, type=int, help='Max test num')
    parser.add_argument('--no_aspectratio', default=True, type=str2bool,
                        help='Detection without anchor of various aspect ratio')
    parser.add_argument('--onnx', default=None, type=int, help='Path to export onnx')

    args = parser.parse_args()

    if args.no_aspectratio:
        solver['aspect_ratios'] = [[], [], [], [], [], [], []]

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    if torch.cuda.is_available() and not args.dump_xmodel:
        if args.device == 'gpu':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            print("WARNING: It looks like you have a CUDA device, but aren't using \
                  CUDA.  Run with --cuda for optimal eval speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    return args
