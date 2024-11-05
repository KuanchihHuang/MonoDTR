import os




if __name__ == "__main__":

    val_file = "./visualDet3D/data/kitti/kitti360_split/val_det_samp.txt"
    #val_file = "./visualDet3D/data/kitti/kitti360_split/test_det.txt"
    old_out = "/ssd2/kuanchih/MonoDTR/workdirs/MonoDTR/output/validation/data"
    #old_out = "/ssd2/kuanchih/MonoDTR/workdirs_test/MonoDTR/output/test/data"
    new_out = old_out.replace("data", "data_mapped")
    
    os.makedirs(new_out, exist_ok = True) 


    ori_out = sorted(os.listdir(old_out))

    with open(val_file) as f:
        train_lines = f.readlines()
        for i  in range(len(train_lines)):
            os.system(f"cp {os.path.join(old_out,ori_out[i])} {os.path.join(new_out,train_lines[i].strip())}.txt")
