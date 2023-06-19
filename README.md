### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Eval](#eval)
4. [Performance](#performance)
5. [Model_info](#model_info)

### Installation
1. Environment requirement
    - pytorch, opencv, ...
    - vai_q_pytorch(Optional, required by quantization)
    - XIR Python frontend (Optional, required by quantization)

2. Installation with Docker


   a. Please refer to [vitis-ai](https://github.com/Xilinx/Vitis-AI/tree/master/) for how to obtain the docker image.


   b. Activate pytorch virtual envrionment in docker:
   ```shell
   conda activate vitis-ai-pytorch
   ```
   c. Install python dependencies using conda:
   ```shell
   pip install opencv-python imgaug==0.4.0 tqdm==4.60.0 --user
   # install mmcv-full 1.2.0 follow instruction from https://github.com/open-mmlab/mmcv
   sudo apt-get update && sudo apt-get install cuda-toolkit-11-0
   export CUDA_HOME=/usr/local/cuda
   # the default gcc version is gcc-9, we should config to gcc-8 to make it compatible with cudatoolkit=11.0
   sudo update-alternatives --config gcc 
   pip install pip install mmcv-full==1.2.0 --user
   ```

3. Install with conda
   ```bash
   conda create -n multi_task_v3 python=3.6
   conda activate multi_task_v3
   conda install pytorch=1.4.0 torchvision cudatoolkit=10.0.130
   python -m pip install opencv-python imgaug==0.4.0
   # install mmcv-full 1.2.0 follow instruction from https://github.com/open-mmlab/mmcv
   python -m pip install pip install mmcv-full==1.2.0
   ```   
   
### Preparation

1. Dataset description

   Object Detection: BDD100K

   Segmentation: CityScapes + BDD100K
   
   Drivable Area: BDD100K
   
   Lane Segmentation: BDD100K with lane edge label transformed to lane segmentation label
   
   Depth Estimation: KITTI

2. Dataset Directory Structure like:
   ```markdown
   + data/multi_task_det5_seg16
     + detection
       + bdd_txt
         + train
           + train.txt
           + detection
             + images_id1.txt
             + images_id2.txt
           + images
             + images_id1.jpg
             + images_id2.jpg
         + val
           + images
             + images_id1.jpg
             + images_id2.jpg
           + det_gt.txt
           + det_val.txt
     + segmentation
       + train
         + train.txt
         + seg
           + images_id1.png
           + images_id2.png
         + images
           + images_id1.jpg
           + images_id2.jpg
       + val
         + images
           + images_id1.jpg
           + images_id2.jpg
          + seg_label
           + images_id1.png
           + images_id2.png      
         + seg_val.txt
     + lane
       + train
         train.txt
         + images
           + images_id1.jpg
           + images_id2.jpg
         + seg  
           + images_id1.png
           + images_id2.png      
       val
         + val.txt
         + images
           + images_id1.jpg
           + images_id2.jpg
         + seg  
           + images_id1.png
           + images_id2.png              
     + drivable
       + train
         train.txt
         + images
           + images_id1.jpg
           + images_id2.jpg
         + seg  
           + images_id1.png
           + images_id2.png      
       val
         + val.txt
         + images
           + images_id1.jpg
           + images_id2.jpg
         + seg  
           + images_id1.png
           + images_id2.png              
     + depth
       + kitti
         + train.txt
         + val.txt
         + data_depth_annotated
           + train
             + 2011_09_26_drive_0001_sync/proj_depth/groundtruth/
               + image_02
                 + image_id1.png
                 + image_id2.png
               + image_03
                 + image_id1.png
                 + image_id2.png
             + 2011_09_26_drive_0009_sync/proj_depth/groundtruth/
               + image_02
                 + image_id1.png
                 + image_id2.png
               + image_03
                 + image_id1.png
                 + image_id2.png
           + val
              + 2011_09_26_drive_0002_sync/proj_depth/groundtruth/
               + image_02
                 + image_id1.png
                 + image_id2.png
               + image_03
                 + image_id1.png
                 + image_id2.png
             + 2011_09_26_drive_0005_sync/proj_depth/groundtruth/
               + image_02
                 + image_id1.png
                 + image_id2.png
               + image_03
                 + image_id1.png
                 + image_id2.png    
         + inputs
           + 2011_09_26    
             + 2011_09_26_drive_0001_sync
               + images_02/data
                 + image_id1.png
                 + image_id2.png 
               + images_03/data
                 + image_id1.png
                 + image_id2.png 
             + calib_cam_to_cam.txt
             + calib_imu_to_velo.txt
             + calib_velo_to_cam.txt
   
     images: original images
     seg_label: segmentation ground truth
     det_gt.txt: detectioin ground truth
        image_name label_1 xmin1 ymin1 xmax1 ymax1
        image_name label_2 xmin2 ymin2 xmax2 ymax2
     det_val.txt:
        images id for detection evaluation
     seg_val.txt:
        images id for segmentation evaluation
   ```
   
3. Training Dataset preparation
    ```    
    1. Detection data: BDD100K
       Download from http://bdd-data.berkeley.edu
       use bdd_to_yolo.py convert .json labels to .txt.
       Formate:
            image_name label_1 xmin1 ymin1 xmax1 ymax1
    2. Segmentation data: Cityscapes
        Download from www.cityscapes-dataset.net
        We modify 19 calses to 16 classes which needs preprocessing
        Download codes from https://github.com/mcordts/cityscapesScripts
        replace downloaded /cityscapesScripts/cityscapesscripts/helpers/labels.py with ./data/labels.py
        Then process original datasets to our setting
    3. Lane Segmentation
       Download from http://bdd-data.berkeley.edu
       The lanes of BDD100K are labelled by one or two lines. 
       To get better segmentation  results, 
       preprocess BDD100K lane data by finding those two lines labeled edge and use their inter area as segmentation label.
       For single line, we dialate the line to 8 pixel width segmentation label.
    4. Drivable area data: BDD100K
       Download from http://bdd-data.berkeley.edu
    5. Depth data: KITTI
    ```   
   
### Eval

1. Demo

   ```shell
   # Download cityscapes19.png from https://raw.githubusercontent.com/695kede/xilinx-edge-ai/230d89f7891112d60b98db18bbeaa8b511e28ae2/docs/Caffe-Segmentation/Segment/workspace/scripts/cityscapes19.png
   # put cityscapes19.png at ./code/test/
   cd code/test/
   bash ./run_demo.sh WEIGHT_PATH 
   #the demo pics will be saved at /code/test/demo
   ```

2. Evaluate Detection Performance

   ```shell
   cd code/test/
   bash ./eval_det.sh WEIGHT_PATH 
   #the results will be saved at WEIGHT_FOLDER/det_log.txt
   ```

3. Evaluate Segmentation Performance

   ```shell
   cd code/test/
   bash ./eval_seg.sh WEIGHT_PATH 
   # the results will be saved at WEIGHT_FOLDER/seg_log.txt
   ```

4. Evaluate Drivable Area Performance

   ```shell
   cd code/test/
   bash ./eval_drivable.sh WEIGHT_PATH 
   # the results will be saved at WEIGHT_FOLDER/drivable_log.txt
   ```
   
5. Evaluate Lane Segmentation Performance

   ```shell
   cd code/test/
   bash ./eval_lane.sh WEIGHT_PATH 
   # the results will be saved at WEIGHT_FOLDER/lane_log.txt
   ```
   
6. Evaluate Depth Performance

   ```shell
   cd code/test/
   bash ./eval_depth_eigen.sh WEIGHT_PATH 
   # the results will be saved at WEIGHT_FOLDER/depth_log.txt
   ```

7. Quantize and quantized model evaluation
   ```shell
   cd code/test/
   # export CUDA_HOME
   bash ./run_quant.sh WEIGHT_PATH 
   ```

8. Training
   ```shell
   cd code/train/
   # modify configure if you need, includes data root, weight path,...
   bash ./train.sh WEIGHT_SAVE_FOLDER 
   ```

### Performance

   ```markdown
Detection test images: bdd100+Waymo val 10000
Segmentation test images: bdd100+CityScapes val 1500
Drivable area test images: bdd100 val 10000
Lane segmentation test images: bdd100 val 10000
Depth estimation test images: kitti eigen split
Classes-detection: 4
Classes-segmentation: 16
Lane-segmentation: 2
Drivable-area: 3
Depth-estimation: 1

Input size: 320x512
Flops: 25.44G
   ```

| model  | Det mAP(%)  | Seg mIOU(%)  | Lane IOU(%)  | Drivable mIOU(%)  | Depth SILog  |
|---|---|---|---|---|---|
| float  | 51.2  | 58.14  | 43.71  | 82.57  | 8.78 |
| quant  | 50.9  | 57.52  | 44.01  | 82.30 | 9.32 |

Depth estimation validation image is center-top cropped to aspect ratio 1.78.

### Model_info 

1. Data preprocess 

   ```markdown
   data channel order: RGB(0~255)                  
   resize: h * w = 320 * 512 (cv2.resize(image, (new_w, new_h)).astype(np.float32))
   mean: (104, 117, 123), input = input - mean
   ```

   

### LICENSE NOTICE

Original repository was downloaded using link provided here:
https://github.com/Xilinx/Vitis-AI/blob/v2.5/model_zoo/model-list/pt_multitaskv3_mixed_320_512_25.44G_2.5/model.yaml

Original copyright belongs to Xilinx Inc.
Below files were modified/ added in compliance with Apache 2.0 license: 

```
code/test/config.py
code/test/demo_data/demo_list.txt
code/test/demo_data/images/FRONT_41_157.jpg
code/test/demo_data/images/frame_0.jpg
code/test/demo_data/images/frame_289.jpg
code/test/demo_data/images/frame_4512.jpg
code/test/demo_data/images/frame_5873.jpg
code/test/demo_data/images/img_00254.jpg
code/test/demo_data/images/img_00266.jpg
code/test/demo_data/images/img_00339.jpg
code/test/demo_data/images/img_00374.jpg
code/test/demo_data/images/img_00406.jpg
code/test/demo_data/images/img_00465.jpg
code/test/demo_data/images/img_00468.jpg
code/test/demo_data/images/yolop/frame_0.jpg
code/test/demo_data/images/yolop/frame_289.jpg
code/test/demo_data/images/yolop/frame_4512.jpg
code/test/demo_data/images/yolop/frame_5873.jpg
code/test/demo_data/images/yolop2/img_00266.jpg
code/test/demo_data/images/yolop2/img_00339.jpg
code/test/demo_data/images/yolop2/img_00406.jpg
code/test/demo_data/images/yolop2/img_00465.jpg
code/test/eval_depth.sh
code/test/eval_depth_eigen.sh
code/test/eval_det.sh
code/test/eval_drivable.sh
code/test/eval_lane.sh
code/test/eval_seg.sh
code/test/evaluation/evaluate_det.py
code/test/evaluation/evaluate_seg.py
code/test/layers/functions/prior_box.py
code/test/model_res18.py
code/test/model_res18v2.py
code/test/resnet.py
code/test/run_demo.sh
code/test/run_deploy.sh
code/test/run_quant.sh
code/test/test.py
code/train/data/config.py
code/train/data/det.py
code/train/data/drivable_area.py
code/train/data/lane.py
code/train/loss.py
code/train/model.py
code/train/model_res18v2.py
code/train/resnet.py
code/train/train.py
code/train/train.sh
code/train/utils/det_augmentations.py
data/.gitignore
environment.yml
code/test/layers/box_utils.py
code/test/layers/functions/detection.py
code/test/test.py
float6/final.pth
```

Files were modified to update the repository to work with newest versions of libraries, and to train and evaluate our own MultiTask V3 model.
All modifications can be seen in commit history in this repository.

License is available in `LICENSE.txt` file.


