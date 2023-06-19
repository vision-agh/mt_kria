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

DATASET=demo_data/
WEIGHTS=../../float6/final.pth
IMG_LIST=demo_data/demo_list.txt
CONF_THRESH=0.05

shift

rm -rf demo/*

python -W ignore ./test.py --confidence_threshold ${CONF_THRESH} --trained_model ${WEIGHTS} --image_root ${DATASET} --demo_image_list ${IMG_LIST} --img_mode 2 --quant_mode float "$@"
