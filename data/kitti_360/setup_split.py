"""
    Sample Run:
    python data/kitti_360/setup_split.py
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np

def read_lines(path, strip= True):
    with open(path) as f:
        lines = f.readlines()

    if strip:
        # you may also want to remove whitespace characters like `\n` at the end of each line
        lines = [x.strip() for x in lines]

    return lines

BASE_PATH = "data/kitti_360"
IMAGE_EXT = ".png"
DEPTH_EXT = ".npy"

def make_symlink_or_copy(src_path, intended_path, MAKE_SYMLINK = True):
    if not os.path.exists(intended_path):
        if MAKE_SYMLINK:
            os.symlink(src_path, intended_path)
        else:
            command = "cp " + src_path + " " + intended_path
            os.system(command)

def link_original_data(inp_id_list_path, out_id_list_path, split= "trainval"):
    CWD = os.getcwd()
    # Read id lists
    inp_id_list = read_lines(inp_id_list_path)
    out_id_list = read_lines(out_id_list_path)

    assert len(inp_id_list) == len(out_id_list)

    split_folder     = "train_val" if "trainval" in split else "testing"
    out_image_folder = os.path.join(CWD, BASE_PATH, split_folder, "image")
    out_depth_folder = os.path.join(CWD, BASE_PATH, split_folder, "depth")
    os.makedirs(out_image_folder, exist_ok= True)
    os.makedirs(out_depth_folder, exist_ok= True)

    cnt = 0
    for inp_id, out_id in zip(inp_id_list, out_id_list):
        out_image_path = os.path.join(out_image_folder, out_id + IMAGE_EXT)
        out_depth_path = os.path.join(out_depth_folder, out_id + DEPTH_EXT)

        drive, diid    = inp_id.split(";")
        if "testing" in split:
            # Map to one of the trainval ids
            inp_id = "2013_05_28_drive_0004_sync;0000009808"

        inp_image_path = os.path.join(CWD, BASE_PATH, "KITTI-360/data_2d_raw", drive, "image_00/data_rect", diid + IMAGE_EXT)
        inp_depth_path = os.path.join(CWD, BASE_PATH, "KITTI-360/depth", drive, diid + DEPTH_EXT)
        make_symlink_or_copy(src_path= inp_image_path, intended_path= out_image_path)
        if "testing" not in split:
            make_symlink_or_copy(src_path= inp_depth_path, intended_path= out_depth_path)


        cnt += 1
        if cnt % 5000 == 0 or out_id == out_id_list[-1]:
            print("{} images done...".format(cnt))

#===================================================================================================
# Main starts here
#===================================================================================================
# Link train
print('=============== Linking trainval =======================')
inp_id_list_path = "visualDet3D/data/kitti/kitti360_split/org_trainval_det_clean.txt"
out_id_list_path = "visualDet3D/data/kitti/kitti360_split/trainval_det.txt"
link_original_data(inp_id_list_path, out_id_list_path, split= "trainval")

# Link test
print('=============== Linking test =======================')
inp_id_list_path = "visualDet3D/data/kitti/kitti360_split/org_test_det_samp.txt"
out_id_list_path = "visualDet3D/data/kitti/kitti360_split/test_det.txt"
link_original_data(inp_id_list_path, out_id_list_path, split= "testing")



