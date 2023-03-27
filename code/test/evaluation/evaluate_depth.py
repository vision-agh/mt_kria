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
import pathlib

import imgaug.augmenters as iaa
import numpy as np
import torch
from PIL import Image


class CropKitti(object):
    def crop_img(self, img):
        height, width, channels = img.shape
        top, left = int(height - 352), int((width - 1216) / 2)
        return img[top:top + 352, left:left + 1216]

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
            image, seg = crop(image=image), crop(image=seg)
            return image, seg
        else:
            image = crop(image=image)
            return image


class SilogLoss(torch.nn.Module):
    def __init__(self):
        super(SilogLoss, self).__init__()

    def forward(self, ip, target, ratio=10, ratio2=0.85):
        ip = ip.reshape(-1)
        target = target.reshape(-1)

        mask = (target > 1) & (target < 81)
        masked_ip = torch.masked_select(ip.float(), mask)
        masked_op = torch.masked_select(target, mask)

        log_diff = torch.log(masked_ip * ratio) - torch.log(masked_op * ratio)
        log_diff_masked = log_diff

        silog1 = torch.mean(log_diff_masked ** 2)
        silog2 = ratio2 * (torch.mean(log_diff_masked) ** 2)
        silog_loss = torch.sqrt(silog1 - silog2) * ratio
        return silog_loss


loss_names = ["Silog", "rmse", "rmse_log", "abs_rel", "sq_rel", "d1", "d2", "d3"]  # This is used for printing


def CalculateLosses(pred, gt):
    pred = torch.from_numpy(pred).cuda()
    gt = torch.from_numpy(gt).cuda()
    with torch.no_grad():
        Silog = SilogLoss()
        silog_loss = Silog(pred, gt, 100, 1)

        pred = pred.reshape(-1)
        gt = gt.reshape(-1)

        # Filtering, similar to the one used in official implementation
        mask = torch.tensor((gt > 1e-3) & (gt < 80), dtype=torch.bool).cuda()
        masked_pred = torch.masked_select(pred, mask)
        masked_gt = torch.masked_select(gt, mask)

        masked_pred[masked_pred < 1e-3] = 1e-3
        masked_pred[masked_pred > 80] = 80

        silog_loss = silog_loss.item()
        rmse = torch.sqrt(torch.mean((masked_gt - masked_pred) ** 2)).item()
        rmse_log = torch.sqrt(((torch.log(masked_gt) - torch.log(masked_pred)) ** 2).mean()).item()
        abs_rel = torch.mean(torch.abs(masked_gt - masked_pred) / masked_gt).item()
        sq_rel = torch.mean((masked_gt - masked_pred) ** 2 / masked_gt).item()
        thresh = torch.max((masked_gt / masked_pred), (masked_pred / masked_gt))
        d1 = (thresh < 1.25).float().mean().item()
        d2 = (thresh < 1.25 ** 2).float().mean().item()
        d3 = (thresh < 1.25 ** 3).float().mean().item()

        return [silog_loss, rmse, rmse_log, abs_rel, sq_rel, d1, d2, d3]


def parse_args():
    """Use argparse to get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('gt', help='path to ground truth')
    parser.add_argument('result', help='path to results to be evaluated')
    parser.add_argument('--eigen', action='store_true', default=False, help='')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    predict_root = pathlib.Path(args.result).absolute()
    gt_root = pathlib.Path(args.gt).absolute()
    losses = []
    missing = 0
    if args.eigen:
        drives_path = predict_root / 'inputs'
        val_drives_path = gt_root / 'data_depth_annotated' / 'val'
        drive_focal_lengths = {}
        subdir = "image_02"
        for drive in drives_path.iterdir():
            calib_path = gt_root / 'inputs' / drive.name / 'calib_cam_to_cam.txt'
            calib = {}
            with open(calib_path, encoding="utf8") as f:
                for line in f:
                    line_name, line_data = line.split(":")[:2]
                    calib[line_name] = line_data.split(" ")
            drive_focal_lengths[drive.name] = float(calib["P_rect_02"][1])
            for drive_i in (drives_path / drive).iterdir():
                if not drive_i.is_dir():
                    continue
                for drive_i_predict in (drive_i / subdir / "data").iterdir():
                    drive_i_label = val_drives_path / drive_i.name / "proj_depth" / "groundtruth" / subdir / drive_i_predict.name
                    if not drive_i_label.is_file():
                        missing += 1
                        continue
                    depth_map = np.asarray(Image.open(drive_i_label.as_posix()), np.float32)
                    depth_map = np.expand_dims(depth_map, axis=2) / 256.0
                    depth_map = CropKitti()(depth_map)
                    depth_map = KittiCrop()(depth_map)
                    label_focal = drive_focal_lengths[drive.name]
                    predict_map = np.asarray(Image.open(drive_i_predict.as_posix()), np.float32)
                    predict_map = np.expand_dims(predict_map, axis=2) / 256.0
                    predict_map = predict_map * label_focal / 637.5751
                    loss = CalculateLosses(predict_map, depth_map)
                    losses += [loss]
    else:
        with open(gt_root / 'val.txt', 'r') as f:
            for line in f:
                pname, aname, label_focal = line.strip().split(' ')
                label_focal = float(label_focal)
                drive_i_label = gt_root / aname
                drive_i_predict = (predict_root / pname).with_suffix('.png')
                if not drive_i_predict.is_file():
                    missing += 1
                    continue
                depth_map = np.asarray(Image.open(drive_i_label.as_posix()), np.float32)
                depth_map = np.expand_dims(depth_map, axis=2) / 256.0
                depth_map = CropKitti()(depth_map)
                depth_map = KittiCrop()(depth_map)
                predict_map = np.asarray(Image.open(drive_i_predict.as_posix()), np.float32)
                predict_map = np.expand_dims(predict_map, axis=2) / 256.0
                predict_map = predict_map * label_focal / 637.5751
                loss = CalculateLosses(predict_map, depth_map)
                losses += [loss]
    losses = np.array(losses)
    if missing:
        print(f"Totel missing: {missing}")
    for i in range(len(loss_names)):
        print(loss_names[i], ":", np.mean(losses[:, i]))
