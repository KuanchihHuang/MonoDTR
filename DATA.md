## Data Preparation

Please download [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize the data as follows:


```
#MonoDTR_ROOT
  |data/
    |KITTI/
      |object/			
        |training/
          |calib/
          |image_2/
          |label_2/
          |velodyne/
        |testing/
          |calib/
          |image_2/
```

You can modify the path in config/config.py (for train / val split), and then run the preparation script:


```sh
cd #MonoDTR_ROOT
./launchers/det_precompute.sh config/config.py train # precompute image database and anchors mean/std
python scripts/depth_gt_compute.py --config=config/config.py # precompute depth gt for training
```
