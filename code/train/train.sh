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

DET_ROOT=../../data/multi_task_det5_seg16/detection/Waymo_bdd_txt
SEG_ROOT=../../data/multi_task_det5_seg16/segmentation
DEPTH_ROOT=../../data/multi_task_det5_seg16/depth
SAVE_FOLDER=${1}

shift

echo "Conducting train..."
python train.py --save_folder "${SAVE_FOLDER}" --DET_ROOT ${DET_ROOT} --SEG_ROOT ${SEG_ROOT} --DEPTH_ROOT ${DEPTH_ROOT} "$@"
