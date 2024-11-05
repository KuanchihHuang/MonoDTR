import os, sys
import numpy as np
from collections import namedtuple
import copy
import glob

from panoptic_bev.helpers.more_util import project_3d_points_in_4D_format, convertRot2Alpha
from panoptic_bev.helpers.file_io import read_csv,read_lines, save_numpy
from panoptic_bev.helpers.kitti_utils import get_calib_from_file
from panoptic_bev.helpers.more_util import custom_print

def get_intrinsics(intrinsic_file_path, cam_id= 0):
    ''' load perspective intrinsics '''
    # Reference:
    # https://github.com/autonomousvision/kitti360Scripts/blob/7c144cb069234bbe75b83e75c9ff2a120eab8b4b/kitti360scripts/helpers/project.py#L111-L136
    intrinsic_loaded = False
    width = -1
    height = -1
    with open(intrinsic_file_path) as f:
        intrinsics = f.read().splitlines()
    for line in intrinsics:
        line = line.split(' ')
        if line[0] == 'P_rect_%02d:' % cam_id:
            temp = [float(x) for x in line[1:]]
            temp = np.reshape(temp, [3,4])
            K    = np.eye(4).astype(np.float32)
            K[:3, :]= temp
            intrinsic_loaded = True
        elif line[0] == 'R_rect_%02d:' % cam_id:
            R_rect = np.eye(4)
            R_rect[:3,:3] = np.array([float(x) for x in line[1:]]).reshape(3,3)
        elif line[0] == "S_rect_%02d:" % cam_id:
            width = int(float(line[1]))
            height = int(float(line[2]))
    assert(intrinsic_loaded==True)
    assert(width>0 and height>0)

    # R_rect is with Y up. Multiply the first row of R_rect by -1 to make Y down.
    # Adjust the intrinsics as well.
    R_rect[1] *= -1.0
    K[:, 1]   *= -1.0

    return K, R_rect

def readVariable(fid,name,M,N):
    # Reference:
    # https://github.com/autonomousvision/kitti360Scripts/blob/7c144cb069234bbe75b83e75c9ff2a120eab8b4b/kitti360scripts/devkits/commons/loadCalibration.py#L9-L33
    # rewind
    fid.seek(0,0)

    # search for variable identifier
    line = 1
    success = 0
    while line:
        line = fid.readline()
        if line.startswith(name):
            success = 1
            break

    # return if variable identifier not found
    if success==0:
      return None

    # fill matrix
    line = line.replace('%s:' % name, '')
    line = line.split()
    assert(len(line) == M*N)
    line = [float(x) for x in line]
    mat = np.array(line).reshape(M, N)

    return mat

def get_ego_to_camera(camera_to_ego_file_path, cam_id= 0):
    # Reference:
    # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/devkits/commons/loadCalibration.py#L35-L51
    # open file
    fid = open(camera_to_ego_file_path, 'r')

    # read variables
    cameras = ['image_%02d' % cam_id]
    lastrow = np.array([0,0,0,1]).reshape(1,4)
    for camera in cameras:
        camera_to_ego = np.concatenate((readVariable(fid, camera, 3, 4), lastrow))

    # close file
    fid.close()
    return np.linalg.inv(camera_to_ego)

def get_world_to_ego(world_to_ego_file_name, frame_id_int):
    # Reference:
    # https://github.com/autonomousvision/kitti360Scripts/blob/7c144cb069234bbe75b83e75c9ff2a120eab8b4b/kitti360scripts/helpers/project.py#L27-L34
    poses  = np.loadtxt(world_to_ego_file_name)
    frames = poses[:,0].astype(np.uint64)
    poses  = np.reshape(poses[:,1:],[-1,3,4])

    pose   =  poses[frames == frame_id_int]
    if pose.shape[0] == 1:
        pose = pose[0]
        world_to_ego = np.linalg.inv(np.concatenate((pose, np.array([0.,0.,0.,1.]).reshape(1,4))))
    else:
        world_to_ego = None

    # world_to_ego_2 = {}
    # for frame, pose in zip(frames, poses):
    #     # pose is ego_to_camera_2. convert to (4,4) and then invert.
    #     world_to_ego_2 = np.linalg.inv(np.concatenate((pose, np.array([0.,0.,0.,1.]).reshape(1,4))))

    return world_to_ego

def get_bounds_of_binary_array(img):
    """
    Bounds of binary 2D array
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    if np.sum(rows) <= 0 or np.sum(cols) <= 0:
        return -1, -1, -1, -1
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return cmin, rmin, cmax, rmax

def get_window_drive_start_end(file_path):
    # eg: data_3d_semantics/train/2013_05_28_drive_0000_sync/static/0000000372_0000000610.ply
    drive_id = int(file_path.split("/")[2].split("_")[4])
    basename = os.path.basename(file_path).split(".")[0]
    start    = int(basename.split("_")[0])
    end      = int(basename.split("_")[1])

    return drive_id, start, end

def get_kitti_style_ground_truth(obj, camera_calib_final, world_to_rect, detect_cat_list, detect_cat_map_dict, w, h, global_seg, thresh= 2.0):
    '''Converts a KITTI-360 object annotation GT to KITTI style GT'''
    # Thresholds for occlusion
    # See https://github.com/autonomousvision/kitti360Scripts/issues/78#issuecomment-1423623920
    VISIBLE_FRAC_THRESHOLD_1 = 0.8
    VISIBLE_FRAC_THRESHOLD_2 = 0.2
    cat_name  = obj.name
    vertices  = obj.vertices
    pose      = obj.R
    globalId  = local2global(obj.semanticId, obj.instanceId)

    if cat_name not in detect_cat_list:
        return None, None, None, None, None, None, None, None, None, None

    center_3d_world = np.mean(vertices, axis= 0)
    proj3d_2d = project_3d_points_in_4D_format(camera_calib_final, points_4d= center_3d_world.reshape((3, 1)), pad_ones= True).transpose()[0]
    # Object depth is too near or far
    if proj3d_2d[2] < -2 or proj3d_2d[2] > 80:
        return None, None, None, None, None, None, None, None, None, None

    # 2D information
    # Get visible bounds of the object in an image
    u_min, v_min, u_max, v_max = get_bounds_of_binary_array(global_seg == globalId)

    # Any of the bounds is negative (i.e. object does not exist)
    if u_min < 0 or v_min < 0 or u_max < 0 or v_max < 0:
        return None, None, None, None, None, None, None, None, None, None

    # first bring vertices in pixel space
    uv_vertices = project_3d_points_in_4D_format(camera_calib_final, points_4d= vertices.transpose(), pad_ones= True).transpose()
    u_min_temp, v_min_temp, _, _ = np.min(uv_vertices, axis= 0)
    u_max_temp, v_max_temp, _, _ = np.max(uv_vertices, axis= 0)

    # Any of the projected pixels is too off
    if u_min_temp <= -thresh*w or u_max_temp >= (1+thresh)*w or v_min_temp <= -thresh*h or v_max_temp >= (1+thresh)*h:
        return None, None, None, None, None, None, None, None, None, None

    # Update projected uv_min if they are larger than min bounds from globalId
    # projected uv_max if they are smaller than max bounds from globalId
    u_min_final = u_min if u_min < u_min_temp else u_min_temp
    v_min_final = v_min if v_min < v_min_temp else v_min_temp
    u_max_final = u_max if u_max > u_max_temp else u_max_temp
    v_max_final = v_max if v_max > v_max_temp else v_max_temp

    # https://github.com/abhi1kumar/groomed_nms/blob/main/data/kitti_split1/devkit/readme.txt#L55-L72
    truncation = 1.0
    if u_min < u_max and v_min < v_max and u_min_final < u_max_final and v_min_final < v_max_final:
        truncation  = 1.0 - ((u_max - u_min)*(v_max - v_min))/((u_max_final - u_min_final)*(v_max_final - v_min_final))

    # Remove the invisible boxes
    # Find out the instance ID of the visible bounding boxes based on our 2D instance segmentation maps,
    # and then retrieve the corresponding 3D bounding boxes.
    # Refernce:
    # https://github.com/autonomousvision/kitti360Scripts/issues/58#issuecomment-1124445995
    box2d_area     = (int(u_max) - int(u_min))*(int(v_max) - int(v_min))
    globalId_cnt   = np.sum(global_seg == globalId)
    visible_frac   = globalId_cnt/box2d_area if box2d_area > 0 else 0.0

    # Object is very less visible
    if visible_frac  <= VISIBLE_FRAC_THRESHOLD_2:
        return None, None, None, None, None, None, None, None, None, None

    if visible_frac > VISIBLE_FRAC_THRESHOLD_1:
        occlusion = 0
    elif visible_frac > VISIBLE_FRAC_THRESHOLD_2:
        occlusion = 1
    else:
        occlusion = 2
    kitti_cat   = detect_cat_map_dict[cat_name]

    # 3D information.
    # Bring 3D coordinates in rectified camera space
    center_3d_rect = np.matmul(world_to_rect[:3, :3], center_3d_world.reshape((3, 1))) + world_to_rect[:3, 3].reshape((3, 1))
    center_3d_rect = center_3d_rect.flatten()
    if np.abs(center_3d_rect[0]) > 50.0 or np.abs(center_3d_rect[1]) > 10.0:
        return None, None, None, None, None, None, None, None, None, None

    # global pose in X(right), Y(inside), Z(up)
    # pose contains both the scaling information and rotation information
    #         --            --     --             --      --                   --
    #        | cos   -sin   0 |   | l3d    0     0  |    | l3d cos  -w3d sin   0 |
    # pose = | sin    cos   0 |   |  0    w3d    0  | =  | l3d sin   w3d cos   0 |
    #        |  0      0    1 |   |  0     0    h3d |    |   0          0     h3d|
    #        --             --    --              --     --                    --

    # dimension
    h3d = pose[2, 2]
    w3d = np.sqrt(pose[0,1]**2 + pose[1,1]**2)
    l3d = np.sqrt(pose[0,0]**2 + pose[1,0]**2)
    dimension = [h3d, w3d, l3d]

    # pose without dim
    pose_wo_dim = copy.copy(pose)
    pose_wo_dim[:, 0] /= l3d
    pose_wo_dim[:, 1] /= w3d
    pose_wo_dim[:, 2] /= h3d
    pose_rect = np.matmul(world_to_rect[:3, :3], pose_wo_dim)

    # KITTI assumes only yaw angle
    pose_rect[1, :2] = 0
    pose_rect[0, 2]  = 0.0
    pose_rect[1, 2]  = 1.0
    pose_rect[2, 2]  = 0
    # KITTI yaw convention is clockwise. See kitti/training/label_2/000044.txt as an example
    # Convert to anticlockwise otherwise 90 rotations look entirely opposite
    yaw   = - np.arctan2(pose_rect[2, 0], pose_rect[0, 0])
    alpha = convertRot2Alpha(ry3d= yaw, z3d= center_3d_rect[2], x3d= center_3d_rect[0])

    # Update projected boxes to lie within the image bounds
    u_min_temp = np.min([np.max([u_min_temp, 0.0]), w])
    u_max_temp = np.min([np.max([u_max_temp, 0.0]), w])
    v_min_temp = np.min([np.max([v_min_temp, 0.0]), h])
    v_max_temp = np.min([np.max([v_max_temp, 0.0]), h])

    # Use projected 3D box as the 2D box
    # bbox_2d = [u_min_temp, v_min_temp, u_max_temp, v_max_temp]
    # Use bounds of 2D box
    bbox_2d = [u_min, v_min, u_max, v_max]

    #       0  1   2      3   4   5   6    7    8    9   10   11   12   13
    # cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d
    output_str = ("{} {:.2f} {:1d} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}"
            .format(kitti_cat, truncation, occlusion, alpha,
                    bbox_2d[0], bbox_2d[1], bbox_2d[2], bbox_2d[3],
                    dimension[0], dimension[1], dimension[2],
                    center_3d_rect[0], center_3d_rect[1] + h3d/2.0, center_3d_rect[2], yaw))

    return kitti_cat, truncation, occlusion, alpha, bbox_2d, dimension, center_3d_rect, yaw, proj3d_2d, output_str


def convert_kitti_image_text_to_kitti_360_window_npy(result_folder, output_folder= None, split= "validation", max_dist_th= 4, replace_low_score_box= True, logger= None, verbose= False):

    # Ref: https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/labels.py#L80-L95
    labels_to_id_mapping = {'Car': 26, 'Building': 11}
    evaluate_classes     = ['Car', 'Building']

    if split == "testing":
        mapping_file      = "/ssd2/kuanchih/dataset/SeaBird/PanopticBEV/data/kitti_360/kitti_360/test_det_org.txt"
        calib_folder      = "/ssd2/kuanchih/dataset/SeaBird/PanopticBEV/data/kitti_360/testing/calib/"
        window_list_path  = "/ssd2/kuanchih/dataset/SeaBird/PanopticBEV/data/kitti_360/ImageSets/windows/2013_05_18_drive_test.txt"
    else:
        mapping_file      = "data/kitti_360/kitti_360/val_det_org.txt"
        calib_folder      = "data/kitti_360/train_val/calib/"
        window_list_path  = "data/kitti_360/ImageSets/windows/2013_05_28_drive_val.txt"

    oracle_flag = False
    if "data/kitti_360/" in result_folder:
        oracle_flag       = True
        output_folder     = os.path.join("output/oracle_img_to_win", "{}_th_{:.0f}_replace_{}".format(split, max_dist_th, str(replace_low_score_box)))
    elif output_folder is None:
        output_folder     = result_folder.replace("data", "data_kitti_360_format")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # First get the windows
    window_data       = read_lines(window_list_path)
    windows = []
    for i, window_text in enumerate(window_data):
        drive_id, start, end  = get_window_drive_start_end(file_path= window_text)
        windows.append([drive_id, start, end])
    num_windows    = len(windows)
    data_to_write  = [None] * num_windows

    # Then get the image prediction paths
    pred_file_list = sorted(glob.glob(result_folder + "/*.txt"))

    custom_print("=> Writing windows..."                           , logger= logger)
    custom_print("Split         = {}".format(split)                , logger= logger)
    custom_print("Repl low box  = {}".format(replace_low_score_box), logger= logger)
    custom_print("Max dist th   = {}".format(max_dist_th)          , logger= logger)
    custom_print("#Windows=     = {}".format(num_windows)          , logger= logger)
    custom_print("#files        = {}".format(len(pred_file_list))  , logger= logger)
    custom_print("Pred_folder   = {}".format(result_folder)        , logger= logger)
    custom_print("Output_folder = {}".format(output_folder)        , logger= logger)

    if oracle_flag:
        samp_file_id_path = "data/kitti_360/ImageSets/val_det_samp.txt"
        samp_file_id_list = read_lines(samp_file_id_path)
        print("Only using file ids present in {}".format(samp_file_id_path))

    for i, pred_file_path in enumerate(pred_file_list):
        if verbose:
            if (i+1)% 500 == 0 or (i+1) == len(pred_file_list):
                custom_print("{} images processing".format(i+1), logger= logger)

        # For oracle stuff, only process the ones which are in the sampled file
        if oracle_flag and os.path.basename(pred_file_path).split(".")[0] not in samp_file_id_list:
            continue

        calib_file_path = os.path.join(calib_folder, os.path.basename(pred_file_path))
        calib           = get_calib_from_file(calib_file_path)
        world2rect      = calib['World2Rect']
        rect2world      = np.linalg.inv(np.concatenate((world2rect, np.array([0.,0.,0.,1.]).reshape(1,4))))

        #get predictions
        predictions_img = read_csv(pred_file_path, ignore_warnings= True, use_pandas= True)
        if predictions_img is not None:
            # Get object in world frame
            num_boxes  = predictions_img.shape[0]
            class_boxes= predictions_img[:, 0]
            data_boxes = predictions_img[:, 1:].astype(np.float32)

            #       0  1   2      3   4   5   6    7    8    9   10   11   12   13     14
            # cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score
            alpha = data_boxes[:, 2]
            x1  = data_boxes[:, 3]
            y1  = data_boxes[:, 4]
            x2  = data_boxes[:, 5]
            y2  = data_boxes[:, 6]
            h3d = data_boxes[:, 7]
            w3d = data_boxes[:, 8]
            l3d = data_boxes[:, 9]
            x3d = data_boxes[:, 10]
            y3d = data_boxes[:, 11] - h3d/2.0
            z3d = data_boxes[:, 12]
            ry3d = data_boxes[:,13]
            if data_boxes.shape[1] > 14:
                score = data_boxes[:, 14]
            else:
                score = np.ones(num_boxes)

            # Convert to global / world / map coordinate
            # https://github.com/autonomousvision/kitti360Scripts/issues/65#issue-1297060686
            global_pts = np.matmul(rect2world, np.vstack((x3d, y3d, z3d, np.ones(num_boxes))) ).T  # N x 4
            global_pts = global_pts[:, :3] # N x 3

            # The object frame of reference is X right, Z up and Y inside
            # The heading angle (\theta) is positive ANTI-clockwise from positive X-axis about the Z-axis.
            # The rotation matrix rot_z is from
            # https://github.com/autonomousvision/kitti360Scripts/blob/e0e3442991d3cf4c69debb84b48fcab3aabf2516/kitti360scripts/evaluation/semantic_3d/evalDetection.py#L186-L192
            # If \theta = 90, rotation matrix rot_z transforms the point (l/2,0) to (0,l/2).
            #                    +Y
            #                    |
            #                    |
            #                    |
            #                    |
            #                    |        (l/2, 0)
            #                    |**********--------------+ X
            #
            #                    + Y
            #                    |
            #                    * (0,l/2)
            #                    *
            #                    *
            #                    *
            #                    *------------------------+ X
            # This example confirms that heading angle is positive anti-clockwise from positive X-axis.
            # KITTI format is positive clockwise from positive X-axis. So, add a negative sign
            heading_angle    =  - ry3d

            # Get int id based stuff
            class_id  = np.zeros(num_boxes, )
            class_id[class_boxes == "Car"]      = labels_to_id_mapping['Car']
            class_id[class_boxes == "Building"] = labels_to_id_mapping['Building']

            # center_x, center_y, center_z, size_x, size_y, size_z, heading_angle, id, confidence
            # center_x, center_y, center_z, length, width, height,  clockwise    , id, confidence
            image_data = np.hstack((global_pts, l3d[:, np.newaxis], w3d[:, np.newaxis], h3d[:, np.newaxis], heading_angle[:, np.newaxis], class_id[:, np.newaxis], score[:, np.newaxis]))

            # Check out the drive id of the file if it is in the window
            real_path  = os.path.realpath(calib_file_path)
            real_path = os.path.realpath(calib_file_path.replace("calib","image").replace("txt", "png"))
            frame_id   = int(os.path.basename(real_path).split(".")[0])
            #print(real_path)
            #import pdb; pdb.set_trace()
            drive_str  = [s for i, s in enumerate(real_path.split("/")) if "drive" in s][0]
            drive_id   = int(drive_str.split("_")[4])

            for w, (window_data, window) in enumerate(zip(data_to_write, windows)):
                window_drive_id, window_start, window_end = window
                if drive_id == window_drive_id and window_start <= frame_id and frame_id <= window_end:
                    if window_data is None:
                        data_to_write[w] = image_data
                    else:
                        # Do a center based matching
                        image_centers  =  image_data[:, :3]
                        window_centers = window_data[:, :3]
                        from scipy.spatial.distance import cdist
                        dist_mat = cdist(image_centers, window_centers, metric='cityblock') # N x W
                        #boxes which are sufficiently far are the ones we care about
                        dist_min   = np.min(dist_mat, axis= 1)
                        dist_argmin= np.argmin(dist_mat, axis= 1)
                        far_ind    = dist_min >= max_dist_th
                        near_ind   = np.logical_not(far_ind)
                        if replace_low_score_box and near_ind.sum() > 0:
                            # Compare scores of near_ind and window_data
                            win_index             = dist_argmin[near_ind]
                            score_img             = image_data [near_ind, 8]
                            score_window          = window_data[win_index, 8]
                            cat_img               = image_data [near_ind, 7]
                            cat_window            = window_data[win_index, 7]
                            update_win_index_bool = np.logical_and(score_img > score_window, cat_img == cat_window)
                            if update_win_index_bool.sum() > 0:
                                # Update window_data since we have better boxes
                                update_win_index      = win_index[update_win_index_bool]
                                update_img_index      = np.arange(near_ind.shape[0])[near_ind][update_win_index_bool]
                                window_data[update_win_index, :] = image_data[update_img_index, :]
                                data_to_write[w] = window_data
                        if far_ind.sum() > 0:
                            image_data = image_data[far_ind]
                            data_to_write[w] = np.vstack((window_data, image_data))

    # Now write in the desired format
    # https://github.com/autonomousvision/kitti360Scripts/tree/master/kitti360scripts/evaluation/semantic_3d#output-for-3d-bounding-box-detection
    if verbose:
        custom_print("Writing windows", logger= logger)
    for w, (window_data, window) in enumerate(zip(data_to_write, windows)):
        if window_data is None:
            # add a dummy Car entry to the window
            window_data = np.array([[10, 10, 10, 4, 2, 2., 1.57, 26, 0.01]])
        window_drive_id, window_start, window_end = window
        # 0008_0000000002_0000000245.npy
        path = os.path.join(output_folder, "{:04d}_{:010d}_{:010d}.npy".format(window_drive_id, window_start, window_end))
        save_numpy(path, numpy_variable= window_data, show_message= False)
        if verbose and (w+1) % 10 == 0 or w == len(windows)-1:
            custom_print("{:d} windows done".format(w+1), logger= logger)


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'kittiId'     , # An integer ID that is associated with this label for KITTI-360
                    # NOT FOR RELEASING

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'ignoreInInst', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations of instance segmentation or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    kittiId,    trainId   category            catId     hasInstances   ignoreInEval   ignoreInInst   color
    Label(  'unlabeled'            ,  0 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        1 ,         0 , 'flat'            , 1       , False        , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        3 ,         1 , 'flat'            , 1       , False        , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,        2 ,       255 , 'flat'            , 1       , False        , True         , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,        10,       255 , 'flat'            , 1       , False        , True         , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        11,         2 , 'construction'    , 2       , True         , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        7 ,         3 , 'construction'    , 2       , False        , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        8 ,         4 , 'construction'    , 2       , False        , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,        30,       255 , 'construction'    , 2       , False        , True         , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,        31,       255 , 'construction'    , 2       , False        , True         , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,        32,       255 , 'construction'    , 2       , False        , True         , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        21,         5 , 'object'          , 3       , True         , False        , True         , (153,153,153) ),
    Label(  'polegroup'            , 18 ,       -1 ,       255 , 'object'          , 3       , False        , True         , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        23,         6 , 'object'          , 3       , True         , False        , True         , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        24,         7 , 'object'          , 3       , True         , False        , True         , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        5 ,         8 , 'nature'          , 4       , False        , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        4 ,         9 , 'nature'          , 4       , False        , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,        9 ,        10 , 'sky'             , 5       , False        , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,        19,        11 , 'human'           , 6       , True         , False        , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,        20,        12 , 'human'           , 6       , True         , False        , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,        13,        13 , 'vehicle'         , 7       , True         , False        , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,        14,        14 , 'vehicle'         , 7       , True         , False        , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,        34,        15 , 'vehicle'         , 7       , True         , False        , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,        16,       255 , 'vehicle'         , 7       , True         , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,        15,       255 , 'vehicle'         , 7       , True         , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,        33,        16 , 'vehicle'         , 7       , True         , False        , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,        17,        17 , 'vehicle'         , 7       , True         , False        , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,        18,        18 , 'vehicle'         , 7       , True         , False        , False        , (119, 11, 32) ),
    Label(  'garage'               , 34 ,        12,         2 , 'construction'    , 2       , True         , True         , True         , ( 64,128,128) ),
    Label(  'gate'                 , 35 ,        6 ,         4 , 'construction'    , 2       , False        , True         , True         , (190,153,153) ),
    Label(  'stop'                 , 36 ,        29,       255 , 'construction'    , 2       , True         , True         , True         , (150,120, 90) ),
    Label(  'smallpole'            , 37 ,        22,         5 , 'object'          , 3       , True         , True         , True         , (153,153,153) ),
    Label(  'lamp'                 , 38 ,        25,       255 , 'object'          , 3       , True         , True         , True         , (0,   64, 64) ),
    Label(  'trash bin'            , 39 ,        26,       255 , 'object'          , 3       , True         , True         , True         , (0,  128,192) ),
    Label(  'vending machine'      , 40 ,        27,       255 , 'object'          , 3       , True         , True         , True         , (128, 64,  0) ),
    Label(  'box'                  , 41 ,        28,       255 , 'object'          , 3       , True         , True         , True         , (64,  64,128) ),
    Label(  'unknown construction' , 42 ,        35,       255 , 'void'            , 0       , False        , True         , True         , (102,  0,  0) ),
    Label(  'unknown vehicle'      , 43 ,        36,       255 , 'void'            , 0       , False        , True         , True         , ( 51,  0, 51) ),
    Label(  'unknown object'       , 44 ,        37,       255 , 'void'            , 0       , False        , True         , True         , ( 32, 32, 32) ),
    Label(  'license plate'        , -1 ,        -1,        -1 , 'vehicle'         , 7       , False        , True         , True         , (  0,  0,142) ),
]


#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# KITTI-360 ID to cityscapes ID
kittiId2label   = { label.kittiId : label for label in labels           }


MAX_N = 1000
def local2global(semanticId, instanceId):
    globalId = semanticId*MAX_N + instanceId
    if isinstance(globalId, np.ndarray):
        return globalId.astype(np.int)
    else:
        return int(globalId)

def global2local(globalId):
    semanticId = globalId // MAX_N
    instanceId = globalId % MAX_N
    if isinstance(globalId, np.ndarray):
        return semanticId.astype(np.int), instanceId.astype(np.int)
    else:
        return int(semanticId), int(instanceId)

class KITTI360Bbox3D:
    # Constructor
    def __init__(self):
        # KITTI360Object.__init__(self)
        # the polygon as list of points
        self.vertices  = []
        self.faces  = []
        self.lines = [[0,5],[1,4],[2,7],[3,6],
                      [0,1],[1,3],[3,2],[2,0],
                      [4,5],[5,7],[7,6],[6,4]]
        self.dim   = None

        # the ID of the corresponding object
        self.semanticId = -1
        self.instanceId = -1
        self.annotationId = -1

        # the window that contains the bbox
        self.start_frame = -1
        self.end_frame = -1

        # timestamp of the bbox (-1 if statis)
        self.timestamp = -1

        # projected vertices
        self.vertices_proj = None

        # name
        self.name = ''

    def __str__(self):
        return self.name

    def parseOpencvMatrix(self, node):
        rows = int(node.find('rows').text)
        cols = int(node.find('cols').text)
        data = node.find('data').text.split(' ')

        mat = []
        for d in data:
            d = d.replace('\n', '')
            if len(d)<1:
                continue
            mat.append(float(d))
        mat = np.reshape(mat, [rows, cols])
        return mat

    def parseVertices(self, child):
        transform = self.parseOpencvMatrix(child.find('transform'))
        R = transform[:3,:3]
        T = transform[:3,3]
        vertices = self.parseOpencvMatrix(child.find('vertices'))
        faces = self.parseOpencvMatrix(child.find('faces'))

        self.dim = np.max(vertices, axis= 0) - np.min(vertices, axis= 0)

        vertices = np.matmul(R, vertices.transpose()).transpose() + T
        self.vertices = vertices
        self.faces = faces
        self.R = R
        self.T = T

    def parseBbox(self, child):
        semanticIdKITTI = int(child.find('semanticId').text)
        self.semanticId = kittiId2label[semanticIdKITTI].id
        self.instanceId = int(child.find('instanceId').text)
        self.name = kittiId2label[semanticIdKITTI].name

        self.start_frame = int(child.find('start_frame').text)
        self.end_frame = int(child.find('end_frame').text)

        self.timestamp = int(child.find('timestamp').text)

        self.annotationId = int(child.find('index').text) + 1

        self.parseVertices(child)

# Meta class for KITTI360Bbox3D
class Annotation3D:
    # Constructor
    def __init__(self, labelDir='', sequence=''):

        labelPath = glob.glob(os.path.join(labelDir, '*', '%s.xml' % sequence)) # train or test
        if len(labelPath)!=1:
            raise RuntimeError('%s does not exist! Please specify KITTI360_DATASET in your environment path.' % labelPath)
        else:
            labelPath = labelPath[0]
            print('Loading %s...' % labelPath)

        self.init_instance(labelPath)

    def init_instance(self, labelPath):
        # load annotation
        tree = ET.parse(labelPath)
        root = tree.getroot()

        self.objects = defaultdict(dict)

        self.num_bbox = 0

        for child in root:
            if child.find('transform') is None:
                continue
            obj = KITTI360Bbox3D()
            obj.parseBbox(child)
            globalId = local2global(obj.semanticId, obj.instanceId)
            self.objects[globalId][obj.timestamp] = obj
            self.num_bbox+=1

        globalIds = np.asarray(list(self.objects.keys()))
        semanticIds, instanceIds = global2local(globalIds)
        for label in labels:
            if label.hasInstances:
                print(f'{label.name:<30}:\t {(semanticIds==label.id).sum()}')
        print(f'Loaded {len(globalIds)} instances')
        print(f'Loaded {self.num_bbox} boxes')


    def __call__(self, semanticId, instanceId, timestamp=None):
        globalId = local2global(semanticId, instanceId)
        if globalId in self.objects.keys():
            # static object
            if len(self.objects[globalId].keys())==1:
                if -1 in self.objects[globalId].keys():
                    return self.objects[globalId][-1]
                else:
                    return None
            # dynamic object
            else:
                return self.objects[globalId][timestamp]
        else:
            return None
