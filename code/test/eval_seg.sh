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

DATASET=../../data/multi_task_det5_seg16/segmentation/val/
WEIGHTS=${1}
IMG_LIST=seg_val.txt
SAVE_FOLDER=../../results/
mkdir -p "$SAVE_FOLDER"
GT_FILE=${DATASET}/seg_label/
DT_FILE=${SAVE_FOLDER}/seg/
TEST_LOG=${SAVE_FOLDER}/seg_log.txt
rm -rf "$DT_FILE"
mkdir -p "$DT_FILE"

shift

echo "python -W ignore  test.py --i_seg --save_folder ${SAVE_FOLDER} --trained_model ${WEIGHTS}  --image_root ${DATASET} --image_list ${IMG_LIST} --img_mode 2 --eval --quant_mode float "$@"" >> ${TEST_LOG}
python -W ignore  test.py --i_seg --save_folder ${SAVE_FOLDER} --trained_model ${WEIGHTS}  --image_root ${DATASET} --image_list ${IMG_LIST} --img_mode 2 --eval --quant_mode float "$@"
python ./evaluation/evaluate_seg.py seg ${GT_FILE} ${DT_FILE} | tee -a ${TEST_LOG}
echo "Test report is saved to ${TEST_LOG}"

rm -rf "$DT_FILE"
