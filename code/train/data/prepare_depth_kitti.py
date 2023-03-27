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

# Depth dataset
# + /multi_task_det5_seg16/depth
#   + inputs
#       + 2011_09_26/2011_09_26_drive_0009_sync/image_03
#           + /data/0000000418.png
#           + ...
#       ...
#   + data_depth_annotated
#       + train
#           + 2011_09_26_drive_0009_sync/proj_depth/groundtruth/image_03
#               + 0000000418.png
#               ...
#       + val
#           ...
import pathlib
import sys


def prepare_depth(kitti_root):
    kitti_root = pathlib.Path(kitti_root).absolute()
    drives_path = kitti_root / 'inputs'
    train_drives_path = kitti_root / 'data_depth_annotated' / 'train'
    val_drives_path = kitti_root / 'data_depth_annotated' / 'val'

    train_drives = []
    val_drives = []
    drive_focal_lengths = {}
    drive_baseline = {}
    for drive in drives_path.iterdir():
        calib_path = drive / 'calib_cam_to_cam.txt'
        calib = {}
        with open(calib_path, encoding="utf8") as f:
            for line in f:
                line_name, line_data = line.split(":")[:2]
                calib[line_name] = line_data.split(" ")
        baseline = - float(calib["P_rect_03"][4]) / float(calib["P_rect_03"][1])
        drive_focal_lengths[drive.name] = float(calib["P_rect_03"][1])
        drive_baseline[drive.name] = baseline
        for drive_i in (drives_path / drive).iterdir():
            if not drive_i.is_dir():
                continue
            train_drive_images_path = train_drives_path / drive_i.name / "proj_depth" / "groundtruth" / "image_03"
            val_drive_images_path = val_drives_path / drive_i.name / "proj_depth" / "groundtruth" / "image_03"
            if train_drive_images_path.is_dir():
                for train_drive_label in train_drive_images_path.iterdir():
                    train_drive_image = drive_i / "image_03" / "data" / train_drive_label.name
                    train_drives.append(
                        (train_drive_image.relative_to(kitti_root).as_posix(),
                         train_drive_label.relative_to(kitti_root).as_posix(), drive_focal_lengths[drive.name],
                         drive_baseline[drive.name]))
            elif val_drive_images_path.is_dir():
                for val_drive_label in val_drive_images_path.iterdir():
                    val_drive_image = drive_i / "image_03" / "data" / val_drive_label.name
                    val_drives.append(
                        (val_drive_image.relative_to(kitti_root).as_posix(),
                         val_drive_label.relative_to(kitti_root).as_posix(), drive_focal_lengths[drive.name],
                         drive_baseline[drive.name]))
            else:
                print("Drive {} does not exist, continue".format(drive_i.name))
    with (kitti_root / "train.txt").open('w') as f:
        for train_drive in train_drives:
            f.write("{} {} {} {}\n".format(*train_drive))
    with (kitti_root / "val.txt").open('w') as f:
        for val_drive in val_drives:
            f.write("{} {} {}\n".format(*val_drive[:-1]))


if __name__ == '__main__':
    prepare_depth(sys.argv[1])
