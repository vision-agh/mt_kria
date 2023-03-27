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

echo "Preparing dataset..."
CAL_DATASET=../../data/multi_task_det5_seg16/lane/train/images
WEIGHTS=../../float6/final.pth
# WEIGHTS=../../float/pt_MTv3-resnet18_mixed_320_512_25.44G_1.4.pth


echo "Conducting calibration test..."
IMG_LIST=train.txt

shift

python -W ignore test.py --img_mode 1 --trained_model "${WEIGHTS}" --image_root ${CAL_DATASET} --image_list ${IMG_LIST} --eval --quant_mode calib "$@"

echo "Conducting quantized Detection test..."
bash eval_det.sh "$WEIGHTS" --quant_mode test "$@"

echo "Conducting quantized Segmentation test..."
bash eval_seg.sh "$WEIGHTS" --quant_mode test "$@"

echo "Conducting quantized Drivable area test..."
bash eval_drivable.sh "$WEIGHTS" --quant_mode test "$@"

echo "Conducting quantized Lane segmentation test..."
bash eval_lane.sh "$WEIGHTS" --quant_mode test "$@"

# echo "Conducting quantized Depth estimation test..."
# bash eval_depth_eigen.sh "$WEIGHTS" --quant_mode test "$@"

#python -W ignore test.py --img_mode 1 --trained_model "${WEIGHTS}" --image_root ${CAL_DATASET} --image_list ${IMG_LIST} --eval --quant_mode test --dump_xmodel "$@"
