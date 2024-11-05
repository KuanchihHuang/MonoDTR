# MonoDTR

**MonoDTR: Monocular 3D Object Detection with Depth-Aware Transformer** (CVPR 2022) [[paper](https://arxiv.org/abs/2203.10981)]\
Kuan-Chih Huang, Tsung-Han Wu, Hung-Ting Su, Winston H. Hsu.

<img src="resources/example.jpg" alt="vis" style="zoom:50%;" />

This branch is used for KITTI-360 dataset. The results on KITTI-360 test set can be found [here](https://www.cvlibs.net/datasets/kitti-360/eval_bbox_detect_detail.php?benchmark=bbox3d&result=f5508c2c6753b33341f66f1d965f9df51d8671a1).

## Setup

- **Requirements**

  1. Python 3.8
  2. [PyTorch](http://pytorch.org) 1.11
  3. Torchvision 0.12
  4. Cuda 11.3
  5. Ubuntu 20.04

  This branch is tested with NVIDIA RTX6000 (48 GB) GPU.
  ```bash
  git clone https://github.com/KuanchihHuang/MonoDTR.git
  cd MonoDTR
  git checkout kitti360
  ```

- **Cuda & Python**

  - Create a python conda environment and activate it.
    ```bash
    conda create -n monodtr python=3.8 -y
    conda activate monodtr
    ```

  - Install the pytorch and python dependencies using the `requirements.txt` file. 
    ```bash
    pip install -r requirements.txt
    ```
    Git clone kitti360 scripts.
    ```bash
    git clone https://github.com/autonomousvision/kitti360Scripts
    cd kitti360Scripts
    pip install .
    cd ..
    #if you encounter the error about CSUPPORT, set CSUPPORT = False at Line 26 in kitti360scripts/viewer/kitti360Viewer3DRaw.py 
    ```

- **KITTI-360 Data**
  - Download the [KITTI-360](http://www.cvlibs.net/datasets/kitti-360/).
  - Download the processed KITTI-360 `train_val` and dummy `testing` from [labels](https://drive.google.com/file/d/1h1VmHNdoIKRecJKANt1Wj_-nDNX_HCQG/view?usp=sharing) generated by [SeaBird](https://github.com/abhi1kumar/SeaBird/tree/main/PanopticBEV).

  - Arrange datasets as

```bash
MonoDTR
├── data
│      └── kitti_360
│             ├── ImageSets
│             ├── KITTI-360
│             │      ├── calibration
│             │      ├── data_2d_raw
│             │      ├── data_3d_raw
│             │      ├── data_3d_boxes
│             │      └── data_poses
│             ├── train_val
│             │      ├── calib
│             │      ├── label
│             └── testing
│                    ├── calib
│                    ├── label
│ ...
```

Convert LiDAR to depth map.
```bash
python data/kitti_360/save_depth.py
```

Next, link the corresponding images and depths.

```bash
python data/kitti_360/setup_split.py
```

You should see the following structure with `61056` samples in each sub-folder of `train_val` split, and `910` samples in each 
sub-folder of `testing` split.

```bash
MonoDTR
├── data
│      └── kitti_360
│             ├── ImageSets
│             ├── train_val
│             │      ├── calib
│             │      ├── image
│             │      ├── label
│             │      ├── depth
│             ├── testing
│             │      ├── calib
│             │      ├── image
│             │      ├── label
│             │      ├── depth
```

Next, to create the datababe for train/val split:

```bash
./launchers/det_precompute.sh config/config_kitti360.py train 
```

To create the datababe for trainval/test split:
```bash
./launchers/det_precompute.sh config/config_kitti360_test.py train 
./launchers/det_precompute.sh config/config_kitti360_test.py test
```

## Training

Train the model for train/val split:
```bash
 ./launcher/train.sh config/config_kitti360.py 0 $EXP_NAME
```
Train the model for trainval/test split:
```bash
./launcher/train.sh config/config_kitti360_test.py 0 $EXP_NAME
```

## Testing

### Model Zoo

We provide models/predictions for the main experiments on KITTI-360 Val data splits available to download here.

| Data_Splits | Method  | Config<br/>(Run)                                          | Weight<br>/Pred  | Metrics | Lrg<br/>(50) | Car<br/>(50) | Mean<br/>(50) | Lrg<br/>(25) | Car<br/>(25) | Mean<br/>(25) 
|------------|---------|------------------------------------------------------------------|----------|--------|----------|-----------|----------|-----------|----------------|----
| KITTI-360 Val  | MonoDTR | [config_kitti360](config/config_kitti360.py)          | [gdrive](https://drive.google.com/file/d/145_kWshU3in3Z5ElFnEczeja8uHSSola/view?usp=sharing) | AP   | 6.93 | 50.20 | 28.57 | 22.93 | 58.12 | 40.53 
| KITTI-360 Test | MonoDTR | [config_kitti360_test](config/config_kitti360_test.py)        | [gdrive]() | AP   |   -   |   -   | 3.02 |   -   |   -   | 39.76 

### Testing Pre-trained Models

Place models in the root folder as follows:

```bash
MonoDTR/workdirs/MonoDTR
├── checkpoint
│      ├── MonoDTR_19.pth
├── output
│      ├── training
│      ├── validation
│      │       └── data
│      │       └── data_mapped
│      │       └── data_kitti_360_format_mapped/
```

To test, execute the following command:
```bash
 ./launcher/eval.sh config/config.py 0 workdirs/MonoDTR/checkpoint/MonoDTR_19.pth validation
```
You will get the output at `MonoDTR/workdirs/MonoDTR/output/validation/data`.

Then, map the prediction file from VisualDet3D to KITTI-360. Please modify `pred_dir` and `split` in `convert_pred_kitti360.py` and run:
```bash
python data/kitti_360/convert_pred_kitti360.py
```
To get the evaluation results:
```bash
python eval_kitti360.py
```
We use the same evaluation protocol of [SeaBird](https://github.com/abhi1kumar/SeaBird/tree/main/PanopticBEV).

## Citation

If you find our work useful in your research, please consider citing:

```
@inproceedings{huang2022monodtr,
    author = {Kuan-Chih Huang and Tsung-Han Wu and Hung-Ting Su and Winston H. Hsu},
    title = {MonoDTR: Monocular 3D Object Detection with Depth-Aware Transformer},
    booktitle = {CVPR},
    year = {2022}    
}
```

## Acknowledgements
Our codes are mainly based on [visualDet3D](https://github.com/Owen-Liuyuxuan/visualDet3D), and also benefits from [CaDDN](https://github.com/TRAILab/CaDDN), [MonoDLE](https://github.com/xinzhuma/monodle), and [LoFTR](https://github.com/zju3dv/LoFTR). The data preprocessing and evaluation on [KITTI360 data](https://github.com/autonomousvision/kitti360Scripts) are from [SeaBird](https://github.com/abhi1kumar/SeaBird). Thanks for their contributions!
