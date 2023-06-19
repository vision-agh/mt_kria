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

DATASET=../../data/multi_task_det5_seg16/detection/val/
WEIGHTS=../../float6/final.pth
IMG_LIST=det_val.txt
GT_FILE=${DATASET}/det_gt.txt
SAVE_FOLDER=../../results
rm -rf "$SAVE_FOLDER"/det/
mkdir -p "$SAVE_FOLDER"/det/
DT_FILE=${SAVE_FOLDER}/det_test_all.txt
TEST_LOG=${SAVE_FOLDER}/det_log.txt

shift

echo "python -W ignore test.py --i_det --save_folder ${SAVE_FOLDER} --trained_model ${WEIGHTS}  --image_root ${DATASET} --image_list ${IMG_LIST} --img_mode 2 --eval --quant_mode float "$@"" >> ${TEST_LOG}
python -W ignore test.py --i_det --save_folder ${SAVE_FOLDER} --trained_model ${WEIGHTS}  --image_root ${DATASET} --image_list ${IMG_LIST} --img_mode 2 --eval --quant_mode float "$@"
cat ${SAVE_FOLDER}/det/* > ${DT_FILE}
python ./evaluation/evaluate_det.py -gt_file ${GT_FILE} -detection_metric map -result_file ${DT_FILE} | tee ${TEST_LOG}
echo "Test report is saved to ${TEST_LOG}"

rm -rf "$SAVE_FOLDER"/det/
