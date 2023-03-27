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
import argparse
import os.path

# gets home dir cross platform
import torch

HOME = os.path.expanduser("~")

MEANS = (104, 117, 123)

# CONFIGS
solver = {
    'k1': 8,
    'k2': 8,
    'act_clip_val': 8,
    'warmup': False,
    'det_classes': 5,
    'seg_classes': 6,
    'seg_drivable': 3, #TODO PUBL -> 2
    'seg_lane': 2,
    'reg_depth': 80,
    'lr_steps': (25000, 35000, 45000),
    'max_iter': 50000,
    'feature_maps': [(80, 128), (40, 64), (20, 32), (10, 16), (5, 8), (3, 6), (1, 4)],
    'resize': (320, 512),
    'steps': [4, 8, 16, 32, 64, 128, 256],
    'min_sizes': [10, 30, 60, 100, 160, 220, 280],
    'max_sizes': [30, 60, 100, 160, 220, 280, 340],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2]],
    'variance': [0.1, 0.2],
    'clip': False,
}


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Training With Pytorch')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size for training')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--load_from', default=None, type=str,
                        help='Checkpoint state_dict file to training from')
    parser.add_argument('--start_iter', default=1, type=int,
                        help='Resume training at this iter')
    parser.add_argument('--max_iter', default=None, type=int,
                        help='Max training iter')
    parser.add_argument('--lr_steps', default=None, type=int,
                        help='Learning rate steps for SGD')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use CUDA to train model')
    parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                        help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for SGD')
    parser.add_argument('--topk', default=9, type=int, help='atss topk')
    parser.add_argument('--lossw_seg', default=1.0, type=float, help='Loss weight for segmentation')
    parser.add_argument('--lossw_det', default=1.0, type=float, help='Loss weight for detection')
    parser.add_argument('--lossw_drive', default=1.0, type=float, help='Loss weight for drivable area')
    parser.add_argument('--lossw_lane', default=1.0, type=float, help='Loss weight for lane segmentation')
    parser.add_argument('--lossw_depth', default=1.0, type=float, help='Loss weight for depth estimation')
    parser.add_argument('--seed', default=14, type=int, help='Seed')
    parser.add_argument('--weight_decay', default=0., type=float,
                        help='Weight decay')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--save_folder', default='./float/train',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--DET_ROOT', default='../../data/multi_task_det5_seg16/detection/Waymo_bdd_txt',
                        help='Directory for detection data')
    parser.add_argument('--DEPTH_ROOT', default='../../data/multi_task_det5_seg16/depth',
                        help='Directory for depth data')
    parser.add_argument('--SEG_ROOT', default='../../data/multi_task_det5_seg16/segmentation',
                        help='Directory for segmentation data')
    parser.add_argument('--DRIVABLE_ROOT', default='../../data/multi_task_det5_seg16/drivable',
                        help='Directory for drivable segmentation data')
    parser.add_argument('--LANE_ROOT', default='../../data/multi_task_det5_seg16/lane',
                        help='Directory for lane segmentation data')
    parser.add_argument('--warmup', default='linear', help='Warmup method')
    parser.add_argument('--warmup_iters', default=1000, type=int, help='Warmup iters')
    parser.add_argument('--finetune', default=False, action='store_true', help='Freeze backbone')
    parser.add_argument('--finetune_bb', default=False, action='store_true', help='Freeze all task head')
    parser.add_argument('--finetune_task', default=None, choices=['det', 'seg', 'lane', 'drive', 'depth'], type=str,
                        help='Finetune single task head')
    parser.add_argument('--optm', default='sgd', choices=['sgd'], type=str,
                        help='Optimizer')
    parser.add_argument('--debug', default=False, action='store_true', help='Debug mode')
    parser.add_argument('--lane_mode', default='png', choices=['png', 'npy'], help='Lane data format')
    parser.add_argument('--train_weight', default=False, action='store_true', help='Balance loss weight')
    parser.add_argument('--no_aspectratio', default=True, type=str2bool,
                        help='Detection without anchor of various aspect ratio')
    args = parser.parse_args()

    if args.no_aspectratio:
        solver['aspect_ratios'] = [[], [], [], [], [], [], []]

    if args.max_iter:
        max_iter = int(args.max_iter)
        solver['max_iter'] = max_iter

    if args.lr_steps:
        lr_steps = tuple(map(int, args.lr_steps.split(',')))
        solver['lr_steps'] = lr_steps

    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print("WARNING: It looks like you have a CUDA device, but aren't " +
                  "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    return args
