from tqdm import tqdm
import numpy as np
import os
import pickle
import time
import cv2
from fire import Fire
from typing import List, Dict, Tuple
from copy import deepcopy
import skimage.measure
import torch

from _path_init import *
from visualDet3D.networks.heads.anchors import Anchors
from visualDet3D.networks.utils.utils import calc_iou, BBox3dProjector
from visualDet3D.data.pipeline import build_augmentator
from visualDet3D.data.kitti.kittidata import KittiData
from visualDet3D.data.kitti.utils import generate_depth_from_velo
from visualDet3D.utils.timer import Timer
from visualDet3D.utils.utils import cfg_from_file
def denorm(image:np.ndarray, rgb_mean:np.ndarray, rgb_std:np.ndarray)->np.ndarray:
    """
        Denormalize a image.
        Args:
            image: np.ndarray normalized [H, W, 3]
            rgb_mean: np.ndarray [3] among [0, 1] image
            rgb_std : np.ndarray [3] among [0, 1] image
        Returns:
            unnormalized image: np.ndarray (H, W, 3) [0-255] dtype=np.uint8
    """
    image = image * rgb_std + rgb_mean #
    image[image > 1] = 1
    image[image < 0] = 0
    image *= 255
    return np.array(image, dtype=np.uint8)

def process_train_val_file(cfg)-> Tuple[List[str], List[str]]:
    train_file = cfg.data.train_split_file
    val_file   = cfg.data.val_split_file

    with open(train_file) as f:
        train_lines = f.readlines()
        for i  in range(len(train_lines)):
            train_lines[i] = train_lines[i].strip()

    with open(val_file) as f:
        val_lines = f.readlines()
        for i  in range(len(val_lines)):
            val_lines[i] = val_lines[i].strip()

    return train_lines, val_lines

def compute_depth_for_split(cfg,
                                 index_names:List[str], 
                                 data_root_dir:str, 
                                 output_dict:Dict, 
                                 data_split:str='training', 
                                 time_display_inter:int=100):
    save_dir = os.path.join(cfg.path.preprocessed_path, data_split)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    depth_dir = os.path.join(save_dir, 'depth')
    if not os.path.isdir(depth_dir):
        os.mkdir(depth_dir)

    N = len(index_names)
    frames = [None] * N
    print("start reading {} data".format(data_split))
    timer = Timer()
    preprocess = build_augmentator(cfg.data.test_augmentation)
    use_right_img = output_dict["image_3"] 

    for i, index_name in tqdm(enumerate(index_names)):

        # read data with dataloader api
        data_frame = KittiData(data_root_dir, index_name, output_dict)
        
        if use_right_img:
            calib, image, right_image, label, velo = data_frame.read_data()
        else:
            calib, image, label, velo = data_frame.read_data()

        original_image = image.copy()
        baseline = (calib.P2[0, 3] - calib.P3[0, 3]) / calib.P2[0, 0]
        
        if use_right_img:
            image, image_3, P2, P3 = preprocess(original_image, right_image.copy(), p2=deepcopy(calib.P2), p3=deepcopy(calib.P3))
        else:
            image, P2 = preprocess(original_image, p2=deepcopy(calib.P2))
        
        ## gathering depth with point cloud back projection
        depth_left = generate_depth_from_velo(velo[:, 0:3], image.shape[0], image.shape[1], calib.Tr_velo_to_cam, calib.R0_rect, P2)
        depth_left = skimage.measure.block_reduce(depth_left, (4,4), np.max)
        file_name = os.path.join(depth_dir, "P2%06d.png" % i)
        cv2.imwrite(file_name, depth_left)

        if use_right_img:
            depth_right = generate_depth_from_velo(velo[:, 0:3], image.shape[0], image.shape[1], calib.Tr_velo_to_cam, calib.R0_rect, P3)
            depth_right = skimage.measure.block_reduce(depth_right, (4,4), np.max)
            file_name = os.path.join(depth_dir, "P3%06d.png" % i)
            cv2.imwrite(file_name, depth_left)

    print("{} split finished precomputing depth".format(data_split))

def main(config:str="config/config.py"):
    """Main entry point for depth precompute
    config_file(str): path to the config file.
    """
    cfg = cfg_from_file(config)
    torch.cuda.set_device(cfg.trainer.gpu)
    time_display_inter = 100 # define the inverval displaying time consumed in loop
    data_root_dir = cfg.path.data_path # the base directory of training dataset
    calib_path = os.path.join(data_root_dir, 'calib') 
    list_calib = os.listdir(calib_path)
    N = len(list_calib)
    # no need for image, could be modified for extended use
    
    use_right_img = False
    
    output_dict = {
                "calib": True,
                "image": True,
                "image_3" : use_right_img, 
                "label": False,
                "velodyne": True,
            }
    
    train_names, val_names = process_train_val_file(cfg)
    compute_depth_for_split(cfg, train_names, data_root_dir, output_dict, 'training', time_display_inter)

    print("Preprocessing finished")

if __name__ == '__main__':
    Fire(main)
