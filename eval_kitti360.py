# KITTI-360 evaluation on KITTI style outputs
import os
from kitti360_helpers.kitti_360_evalDetection import evaluate_kitti_360_verbose
from kitti360_helpers.kitti_360_evalDetection_windows import evaluate_kitti_360_windows_verbose
from kitti360_helpers.kitti_360_util import convert_kitti_image_text_to_kitti_360_window_npy

#we follow the evaluation protocal from SeaBird (CVPR'24)
#kitti360_helpers are directly adapted from SeaBird/PanopticBEV/panoptic_bev/helpers

if __name__ == "__main__":

    split = "train_val"
    label_folder  = os.path.join("data", split, "label")
    label_folder = "data/kitti_360/train_val/label"
    pred_folder = "/ssd2/kuanchih/MonoDTR/workdirs/MonoDTR/output/validation/data_mapped"
    evaluate_kitti_360_verbose(pred_folder= pred_folder, gt_folder= label_folder)

    max_dist_th           = 4
    replace_low_score_box = True
    convert_kitti_image_text_to_kitti_360_window_npy(pred_folder, split= split, max_dist_th= max_dist_th,
                                                     replace_low_score_box= replace_low_score_box, logger=None, verbose= False)
    
    if split != "testing":
        pred_window_folder    = pred_folder.replace("data", "data_kitti_360_format")
        evaluate_kitti_360_windows_verbose(pred_folder= pred_window_folder, gt_folder= "./data/kitti_360/ImageSets/windows/train_og")
