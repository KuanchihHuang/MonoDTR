import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Mapping predictions from visualDet3D to KITTI360 ')
    
    parser.add_argument(
        '--pred_dir', type=str, 
        default = "/ssd2/kuanchih/MonoDTR/workdirs/MonoDTR/output/validation/data",
        help='Path to the directory containing the prediction file')
    
    # Argument for the split, with validation options
    parser.add_argument(
        '--split', type=str, choices=['validation', 'test'],
        default = "validation",
        help='Specify the split, either "validation" or "test"'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Prediction Directory: {args.pred_dir}")
    print(f"Split: {args.split}")

    if args.split == "validation":
        mapped_file = "./visualDet3D/data/kitti/kitti360_split/val_det_samp.txt"
    else: #for test
        mapped_file = "./visualDet3D/data/kitti/kitti360_split/test_det.txt"
    new_pred_dir = args.pred_dir.replace("data", "data_mapped")
    os.makedirs(new_pred_dir, exist_ok = True) 
    
    pred_files = sorted(os.listdir(args.pred_dir))

    with open(mapped_file) as f:
        train_lines = f.readlines()
        for i  in range(len(train_lines)):
            os.system(f"cp {os.path.join(args.pred_dir,pred_files[i])} {os.path.join(new_pred_dir,train_lines[i].strip())}.txt")
    """
    #NOTE: please modify the 
    if split == "validation": #for val
        val_file = "./visualDet3D/data/kitti/kitti360_split/val_det_samp.txt"
        old_out = "/ssd2/kuanchih/MonoDTR/workdirs/MonoDTR/output/validation/data"
    else: #for test
        val_file = "./visualDet3D/data/kitti/kitti360_split/test_det.txt"
        old_out = "/ssd2/kuanchih/MonoDTR/workdirs_test/MonoDTR/output/test/data"
    
    new_out = old_out.replace("data", "data_mapped")
    
    os.makedirs(new_out, exist_ok = True) 


    ori_out = sorted(os.listdir(old_out))

    with open(val_file) as f:
        train_lines = f.readlines()
        for i  in range(len(train_lines)):
            os.system(f"cp {os.path.join(old_out,ori_out[i])} {os.path.join(new_out,train_lines[i].strip())}.txt")
    """
