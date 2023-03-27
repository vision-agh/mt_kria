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

import os
import random
import time
from collections import OrderedDict
from math import log

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from mmcv.runner import load_state_dict
from torch.autograd import Variable

import model_res18v2 as model_res18
from data import *
from data.depth import KittiDepth
from data.det import *
from data.drivable_area import DrivableSegmentation
from data.lane import LaneSegmentation
from data.seg import *
from layers.modules import AtssLoss
from loss import *
from utils.depth_augmentations import KittiDepthAugmentation
from utils.det_augmentations import DetAugmentation
from utils.seg_augmentations import SegAugmentation, LaneNpyAugmentation

EPS = 1e-6


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
        rank_shift (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train():
    args = parse_args()

    if not args.cuda:
        print("TRAINING ON CPU IS NOT SUPPORTED")
        return -1

    if args.finetune and args.finetune_task:
        train_det = False if args.finetune_task != 'det' else True
        train_seg = False if args.finetune_task != 'seg' else True
        train_drivable = False if args.finetune_task != 'drive' else True
        train_depth = False if args.finetune_task != 'depth' else True
        train_lane = False if args.finetune_task != 'lane' else True
    else:
        train_det = True
        train_seg = True
        train_drivable = True
        train_depth = False
        train_lane = True

    print('Start training')
    set_random_seed(args.seed, deterministic=True)

    if train_seg:
        dataset_seg = Segmentation(SEG_ROOT=args.SEG_ROOT,
                                   transform=SegAugmentation(solver['resize'],
                                                             MEANS), debug=args.debug)
    if train_drivable:
        dataset_drivable = DrivableSegmentation(SEG_ROOT=args.DRIVABLE_ROOT,
                                                transform=SegAugmentation(solver['resize'],
                                                                          MEANS), debug=args.debug)
    if train_det:
        dataset_det = Detection(DET_ROOT=args.DET_ROOT,
                                transform=DetAugmentation(solver['resize'],
                                                          MEANS), debug=args.debug)
    if train_depth:
        dataset_depth = KittiDepth(DEPTH_ROOT=os.path.join(args.DEPTH_ROOT, 'kitti'),
                                   transform=KittiDepthAugmentation(solver['resize'],
                                                                    MEANS), debug=args.debug)
    if train_lane:
        if args.lane_mode == 'npy':
            ltf = LaneNpyAugmentation(solver['resize'], MEANS)
        else:
            ltf = SegAugmentation(solver['resize'], MEANS)
        dataset_lane = LaneSegmentation(SEG_ROOT=args.LANE_ROOT, mode=args.lane_mode,
                                        transform=ltf, debug=args.debug)

    print('dataset create success')

    net = model_res18.build_model(solver['det_classes'], solver['seg_classes'], solver['seg_drivable'],
                                  solver['reg_depth'], solver['seg_lane'])

    print('build model success')

    step_index = sum([args.start_iter > lr_step for lr_step in solver['lr_steps']])
    start_lr = args.lr * (args.gamma ** step_index)

    backbone_params = {}
    depth_params = {}
    drivable_params = {}
    seg_params = {}
    lane_params = {}
    det_params = {}
    neck_params = {}
    for name, param in net.named_parameters():
        if 'resnet18_32s' in name:
            backbone_params[name] = param
            if args.finetune:
                param.requires_grad_(False)
        elif 'depth' in name:
            depth_params[name] = param
            if not train_depth:
                param.requires_grad_(False)
            if args.finetune_bb:
                param.requires_grad_(False)
        elif 'drivable' in name:
            drivable_params[name] = param
            if not train_drivable:
                param.requires_grad_(False)
            if args.finetune_bb:
                param.requires_grad_(False)
        elif 'seg' in name:
            seg_params[name] = param
            if not train_seg:
                param.requires_grad_(False)
            if args.finetune_bb:
                param.requires_grad_(False)
        elif 'det' in name or 'loc_' in name or 'conf_' in name or 'centerness' in name:
            det_params[name] = param
            if not train_det:
                param.requires_grad_(False)
            if args.finetune_bb:
                param.requires_grad_(False)
        elif 'lane' in name:
            lane_params[name] = param
            if not train_lane:
                param.requires_grad_(False)
            if args.finetune_bb:
                param.requires_grad_(False)
        else:
            neck_params[name] = param
            if args.finetune:
                param.requires_grad_(False)
    print("==========================================================")
    print("Depth params: ")
    print(depth_params.keys())
    print("==========================================================")
    print("Drive params: ")
    print(drivable_params.keys())
    print("==========================================================")
    print("Seg params: ")
    print(seg_params.keys())
    print("==========================================================")
    print("Det params: ")
    print(det_params.keys())
    print("==========================================================")
    print("Lane params: ")
    print(lane_params.keys())
    print("==========================================================")
    print("Neck params: ")
    print(neck_params.keys())

    weight_det = nn.Parameter(-log(args.lossw_det) * torch.ones(1))
    weight_lane = nn.Parameter(-log(args.lossw_lane) * torch.ones(1))
    weight_seg = nn.Parameter(-log(args.lossw_seg) * torch.ones(1))
    weight_drivable = nn.Parameter(-log(args.lossw_drive) * torch.ones(1))
    weight_depth = nn.Parameter(-log(args.lossw_depth) * torch.ones(1))
    loss_weights_params = dict(
        weight_depth=weight_depth,
        weight_det=weight_det,
        weight_lane=weight_lane,
        weight_seg=weight_seg,
        weight_drivable=weight_drivable
    )

    if args.optm == 'sgd':
        optimizer = optim.SGD([
            {"params": {**backbone_params, **drivable_params, **seg_params, **det_params, **lane_params,
                        **neck_params}.values(), "weight_decay": args.weight_decay},
            {"params": {**depth_params, **loss_weights_params}.values(), "weight_decay": 0.}], lr=start_lr,
            momentum=args.momentum)
        base_lr = [group['lr'] for group in optimizer.param_groups]
    else:
        raise ValueError(f"optm {args.optm} not implemented")

    if args.load_from:
        print('Loading pretraining {}...'.format(args.load_from))
        state_dict = torch.load(args.load_from)
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

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        state_dict = torch.load(args.resume)
        if 'model' in state_dict:
            with torch.no_grad():
                loss_weights = state_dict['loss_weights']
                weight_det.set_(loss_weights['weight_det'].data)
                weight_lane.set_(loss_weights['weight_lane'].data)
                weight_seg.set_(loss_weights['weight_seg'].data)
                weight_drivable.set_(loss_weights['weight_drivable'].data)
                weight_depth.set_(loss_weights['weight_depth'].data)
            optimizer_dict = state_dict['optimizers']
            optimizer.load_state_dict(optimizer_dict['optimizer'])
            state_dict = state_dict['model']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k[0:7] == 'module.':
                name = k[7:]
                new_state_dict[name] = v
            else:
                name = k
                new_state_dict[name] = v
        net.load_state_dict(new_state_dict)

    if args.cuda:
        net = torch.nn.DataParallel(net)

    if args.cuda:
        print("adas")
        net = net.cuda()
        weight_det = weight_det.cuda()
        weight_lane = weight_lane.cuda()
        weight_seg = weight_seg.cuda()
        weight_drivable = weight_drivable.cuda()
        weight_depth = weight_depth.cuda()
    if not args.train_weight:
        weight_det.requires_grad = False
        weight_lane.requires_grad = False
        weight_seg.requires_grad = False
        weight_drivable.requires_grad = False
        weight_depth.requires_grad = False

    criterion_det = AtssLoss(solver['det_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, net.module.num_anchors_per_lv, topk=args.topk,
                             use_gpu=args.cuda)
    criterion_depth = SilogLoss()
    lovasz_softmax_loss_seg = LovaszSoftmaxLoss(weight=[1.] * solver['seg_classes'])
    lovasz_softmax_loss_lane = LovaszSoftmaxLoss(weight=[1.] * solver['seg_lane'])
    lovasz_softmax_loss_drivable = LovaszSoftmaxLoss(weight=[1.] * solver['seg_drivable'])

    net.train()

    def _freeze_norm_stats(model):
        try:
            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
                    print(f"freeze norm: {m}")
        except ValueError:
            print("errrrrrrrrrrrrrroooooooorrrrrrrrrrrr with instancenorm")
            return

    if args.finetune or args.finetune_bb:
        _freeze_norm_stats(net)

    if train_det:
        epoch_size_det = len(dataset_det) // args.batch_size
    if train_seg:
        epoch_size_seg = len(dataset_seg) // args.batch_size
    if train_drivable:
        epoch_size_drivable = len(dataset_drivable) // args.batch_size
    if train_depth:
        epoch_size_depth = len(dataset_depth) // args.batch_size
    if train_lane:
        epoch_size_lane = len(dataset_lane) // args.batch_size

    print('Using the specified args:')
    print(args)

    generator=torch.Generator(device='cuda')

    if train_det:
        data_loader_det = data.DataLoader(dataset_det, int(args.batch_size),
                                          num_workers=args.num_workers,
                                          shuffle=True, collate_fn=detection_collate,
                                          pin_memory=True,
                                          generator=generator)

    if train_seg:
        data_loader_seg = data.DataLoader(dataset_seg, int(args.batch_size),
                                          num_workers=args.num_workers,
                                          shuffle=True, collate_fn=segmentation_collate,
                                          pin_memory=True,
                                          generator=generator)

    if train_drivable:
        data_loader_drivable = data.DataLoader(dataset_drivable, int(args.batch_size),
                                               num_workers=args.num_workers,
                                               shuffle=True, collate_fn=segmentation_collate,
                                               pin_memory=True,
                                               generator=generator)
    if train_depth:
        data_loader_depth = data.DataLoader(dataset_depth, int(args.batch_size),
                                            num_workers=args.num_workers,
                                            shuffle=True, collate_fn=depth_collate,
                                            pin_memory=True,
                                            generator=generator)
    if train_lane:
        data_loader_lane = data.DataLoader(dataset_lane, int(args.batch_size),
                                           num_workers=args.num_workers,
                                           shuffle=True, collate_fn=segmentation_collate,
                                           pin_memory=True,
                                           generator=generator)

    # create batch iterator
    if train_det:
        batch_iterator_det = iter(data_loader_det)
    if train_seg:
        batch_iterator_seg = iter(data_loader_seg)
    if train_drivable:
        batch_iterator_drivable = iter(data_loader_drivable)
    if train_depth:
        batch_iterator_depth = iter(data_loader_depth)
    if train_lane:
        batch_iterator_lane = iter(data_loader_lane)

    for iteration in range(args.start_iter, solver['max_iter']):
        if train_det:
            if iteration != 0 and (iteration % epoch_size_det == 0):
                batch_iterator_det = iter(data_loader_det)
        if train_seg:
            if iteration != 0 and (iteration % epoch_size_seg == 0):
                batch_iterator_seg = iter(data_loader_seg)
        if train_drivable:
            if iteration != 0 and (iteration % epoch_size_drivable == 0):
                batch_iterator_drivable = iter(data_loader_drivable)
        if train_depth:
            if iteration != 0 and (iteration % epoch_size_depth == 0):
                batch_iterator_depth = iter(data_loader_depth)
        if train_lane:
            if iteration != 0 and (iteration % epoch_size_lane == 0):
                batch_iterator_lane = iter(data_loader_lane)

        if args.warmup == 'linear' and iteration < args.warmup_iters:
            rate = iteration / args.warmup_iters
            for group, lr in zip(optimizer.param_groups, base_lr):
                group['lr'] = lr * rate
        else:
            if iteration in solver['lr_steps']:
                step_index += 1
                adjust_learning_rate(args.lr, optimizer, args.gamma, step_index)

        optimizer.zero_grad()
        for iter_round in range(1):
            log_var = {}
            # load train data
            # forward
            t0 = time.time()
            if train_seg:
                images_seg, seg = next(batch_iterator_seg)
                if args.cuda:
                    images_seg = Variable(images_seg.cuda())
                    with torch.no_grad():
                        seg = Variable(seg.cuda())
                else:
                    images_seg = Variable(images_seg)
                    with torch.no_grad():
                        seg = Variable(seg)
                out_seg = net(images_seg)
                _, _, seg_data, _, _, depth_data, _ = out_seg
                loss_m = lovasz_softmax_loss_seg(seg_data, seg)
                loss_m = loss_m * torch.exp(-weight_seg)
                if args.train_weight:
                    loss_m = loss_m + weight_seg
                log_var['loss_seg'] = loss_m.detach().cpu().numpy()
                loss_m.backward()
                del images_seg
                del seg
                del out_seg
                del seg_data
                del depth_data

            if train_drivable:
                images_drivable, drivable = next(batch_iterator_drivable)
                if args.cuda:
                    images_drivable = Variable(images_drivable.cuda())
                    with torch.no_grad():
                        drivable = Variable(drivable.cuda())
                else:
                    images_drivable = Variable(images_drivable)
                    with torch.no_grad():
                        drivable = Variable(drivable)
                out_drivable = net(images_drivable)
                _, _, _, _, drivable_data, _, _ = out_drivable
                # loss_drivable = softmax_focal_loss2d(drivable_data, drivable)
                if drivable_data.size() != drivable.size():
                    drivable_data = F.interpolate(drivable_data, size=drivable.size()[-2:], mode='bilinear')

                loss_drivable = lovasz_softmax_loss_drivable(drivable_data, drivable)
                loss_drivable = loss_drivable * torch.exp(-weight_drivable)
                if args.train_weight:
                    loss_drivable = loss_drivable + weight_drivable
                log_var['loss_drivable'] = loss_drivable.detach().cpu().numpy()
                loss_drivable.backward()
                del images_drivable
                del drivable
                del out_drivable
                del drivable_data

            if train_lane:
                images_lane, lane = next(batch_iterator_lane)
                if args.cuda:
                    images_lane = Variable(images_lane.cuda())
                    with torch.no_grad():
                        lane = Variable(lane.cuda())
                else:
                    images_lane = Variable(images_lane)
                    with torch.no_grad():
                        lane = Variable(lane)
                out_lane = net(images_lane)
                _, _, _, _, _, _, lane_data = out_lane
                if lane_data.size() != lane.size():
                    lane_data = F.interpolate(lane_data, size=lane.size()[-2:], mode='bilinear')
                if args.lane_mode == 'npy':
                    lane_target = lane / 255.
                    loss_lane = torch.nn.BCEWithLogitsLoss(
                        pos_weight=(lane_data < 0.2).sum() / (lane_data > 0.5).sum())(lane_data[:, 0, :, :],
                                                                                      lane_target)
                else:
                    loss_lane = lovasz_softmax_loss_lane(lane_data, lane)
                loss_lane = loss_lane * torch.exp(-weight_lane)
                if args.train_weight:
                    loss_lane = loss_lane + weight_lane
                log_var['loss_lane'] = loss_lane.detach().cpu().numpy()
                loss_lane.backward()
                del images_lane
                del lane

            if train_depth:
                images_depth, depthmap, focal_lenghts, baseline = next(batch_iterator_depth)

                if args.cuda:
                    images_depth = Variable(images_depth.cuda())
                    with torch.no_grad():
                        # cuda
                        depthmap = [Variable(d) for d in depthmap]
                        focal_lenghts = Variable(focal_lenghts.cuda())
                        baseline = Variable(baseline.cuda())
                else:
                    images_depth = Variable(images_depth)
                    with torch.no_grad():
                        depthmap = [Variable(d) for d in depthmap]
                        focal_lenghts = Variable(focal_lenghts)
                        baseline = Variable(baseline)
                out_depth = net(images_depth)
                _, _, seg_data, _, _, depth_data, _ = out_depth
                valid_depthmap = []
                valid_depthdata = []
                for di, depthmap_i in enumerate(depthmap):
                    if args.cuda:
                        depthmap_i = depthmap_i.cuda()
                    depthmap_i = depthmap_i[None]
                    mask_i = depthmap_i > 1.0
                    depth_data_i = depth_data[di][None]
                    depth_data_i = depth_data_i * focal_lenghts[di][None] / 637.5751
                    if depthmap_i.size() != depth_data_i.size():
                        depth_data_i = F.interpolate(depth_data_i, size=depthmap_i.size()[-2:], mode='nearest')
                    valid_depthmap_i = depthmap_i[mask_i]
                    valid_depthdata_i = depth_data_i[mask_i]
                    valid_depthmap.append(valid_depthmap_i)
                    valid_depthdata.append(valid_depthdata_i)
                valid_depthmap = torch.cat(valid_depthmap)
                valid_depthdata = torch.cat(valid_depthdata)
                loss_depth = criterion_depth(valid_depthdata, valid_depthmap,
                                             torch.ones_like(valid_depthdata).to(torch.bool))

                loss_depth = loss_depth * torch.exp(-weight_depth)
                if args.train_weight:
                    loss_depth = loss_depth + weight_depth
                log_var['loss_depth'] = loss_depth.detach().cpu().numpy()
                loss_depth.backward()
                del images_depth
                del depthmap
                del focal_lenghts
                del baseline
                del out_depth
                del depth_data
                del seg_data

            if train_det:
                images_det, targets = next(batch_iterator_det)
                if args.cuda:
                    images_det = Variable(images_det.cuda())
                    with torch.no_grad():
                        targets = [Variable(ann.cuda()) for ann in targets]
                else:
                    images_det = Variable(images_det)
                    with torch.no_grad():
                        targets = [Variable(ann) for ann in targets]
                out_det = net(images_det)
                loss_l, loss_c, loss_centerness = criterion_det(out_det, targets)
                loss_det = loss_l + loss_c + loss_centerness
                log_var['loss_det_loc'] = loss_l.detach().cpu().numpy()
                log_var['loss_det_cls'] = loss_c.detach().cpu().numpy()
                log_var['loss_det_cen'] = loss_centerness.detach().cpu().numpy()
                loss_det = loss_det * torch.exp(-weight_det)
                if args.train_weight:
                    loss_det = loss_det + weight_det
                log_var['loss_det'] = loss_det.detach().cpu().numpy()
                loss_det.backward()
                del images_det
                del targets

        optimizer.step()
        t1 = time.time()

        if iteration % 50 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration), end=' ')
            print('Learning_Rate: %f ||' % (optimizer.param_groups[0]['lr']), end=' ')
            for k in sorted(log_var.keys()):
                v = log_var[k]
                print("%s : %.4f" % (k.capitalize(), v), end=' ')

        if iteration % 500 == 0:
            print('Saving state, iter:', iteration)
            state_dict = {'model': net.state_dict(), 'optimizers': {}, 'loss_weights':
                dict(weight_depth=weight_depth, weight_det=weight_det, weight_lane=weight_lane, weight_seg=weight_seg,
                     weight_drivable=weight_drivable)}
            state_dict['optimizers']['optimizer'] = optimizer.state_dict()
            torch.save(state_dict,
                       os.path.join(args.save_folder, 'iter_' +
                                    repr(iteration) + '.pth'))
    state_dict = {'model': net.state_dict(), 'optimizers': {}, 'loss_weights':
        dict(weight_depth=weight_depth, weight_det=weight_det, weight_lane=weight_lane, weight_seg=weight_seg,
             weight_drivable=weight_drivable)}
    state_dict['optimizers']['optimizer'] = optimizer.state_dict()
    torch.save(state_dict,
               os.path.join(args.save_folder, 'final' + '.pth'))


def adjust_learning_rate(initial_lr, optimizer, gamma, step):
    lr = initial_lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
