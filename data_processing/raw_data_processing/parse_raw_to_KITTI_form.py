'''
read the tracklets provided by kitti raw data
write the label file as kitti form
'''
import os
import cv2
import numpy as np
import shutil
from utils.read_dir import ReadDir
import parseTrackletXML as xmlParser

def makedir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)

def obtain_2Dbox(dims, trans, rot, P2, img_xmax, img_ymax):
    '''
    obtain 2D bounding box based on 3D location values
    construct 3D bounding box at first, 2D bounding box is just the minimal and maximal values of 3D bounding box
    '''
    # generate 8 points for bounding box
    h, w, l = dims[0], dims[1], dims[2]
    tx, ty, tz = trans[0], trans[1], trans[2]

    R = np.array([[np.cos(rot), 0, np.sin(rot)],
                  [0, 1, 0],
                  [-np.sin(rot), 0, np.cos(rot)]])

    x_corners = [0, l, l, l, l, 0, 0, 0]  # -l/2
    y_corners = [0, 0, h, h, 0, 0, h, h]  # -h
    z_corners = [0, 0, 0, w, w, w, w, 0]  # -w/2

    x_corners = [i - l / 2 for i in x_corners]
    y_corners = [i - h for i in y_corners]
    z_corners = [i - w / 2 for i in z_corners]

    corners_3D = np.array([x_corners, y_corners, z_corners])
    corners_3D = R.dot(corners_3D)
    corners_3D += np.array([tx, ty, tz]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
    corners_2D = P2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]
    corners_2D = corners_2D[:2]

    for i in range(len(corners_2D[0, :])):
        if corners_2D[0, i] < 0:
            corners_2D[0, i] = 0
        elif corners_2D[0, i] > img_xmax:
            corners_2D[0, i] = img_xmax

    for j in range(len(corners_2D[1, :])):
        if corners_2D[1, j] < 0:
            corners_2D[1, j] = 0
        elif corners_2D[1, j] > img_ymax:
            corners_2D[1, j] = img_ymax

    xmin, xmax = int(min(corners_2D[0,:])), int(max(corners_2D[0,:]))
    ymin, ymax = int(min(corners_2D[1,:])), int(max(corners_2D[1,:]))

    bbox = [xmin, ymin, xmax, ymax]
    
    return bbox

def local_ori(trans, rot):
    '''
    compute local orientation value based on global orientation and translation values
    '''
    local_ori = rot - np.arctan(trans[0]/trans[2])
    return round(local_ori,2)

# Read transformation matrices
def read_transformation_matrix():
    for line in open(os.path.join(tracklet_path, 'calib_velo_to_cam.txt')):
        if 'R:' in line:
            R = line.strip().split(' ')
            R = np.asarray([float(number) for number in R[1:]])
            R = np.reshape(R, (3,3))

        if 'T:' in line:
            T = line.strip().split(' ')
            T = np.asarray([float(number) for number in T[1:]])
            T = np.reshape(T, (3,1))

    for line in open(os.path.join(tracklet_path, 'calib_cam_to_cam.txt')):
        if 'R_rect_00:' in line:
            R0_rect = line.strip().split(' ')
            R0_rect = np.asarray([float(number) for number in R0_rect[1:]])
            R0_rect = np.reshape(R0_rect, (3,3))

    # recifying rotation matrix
    R0_rect = np.append(R0_rect, np.zeros((3,1)), axis=1)
    R0_rect = np.append(R0_rect, np.zeros((1,4)), axis=0)
    R0_rect[-1,-1] = 1

    #The rigid body transformation from Velodyne coordinates to camera coordinates
    Tr_velo_to_cam = np.concatenate([R,T],axis=1)
    Tr_velo_to_cam = np.append(Tr_velo_to_cam, np.zeros((1,4)), axis=0)
    Tr_velo_to_cam[-1,-1] = 1

    transform = np.dot(R0_rect, Tr_velo_to_cam)

    # FIGURE OUT THE CALIBRATION
    for line in open(os.path.join(tracklet_path, 'calib_cam_to_cam.txt')):
        if 'P_rect_02' in line:
            line_P2 = line.replace('P_rect_02', 'P2')
            # print (line_P2)

    P2 = line_P2.split(' ')
    P2 = np.asarray([float(i) for i in P2[1:]])
    P2 = np.reshape(P2, (3,4))

    return transform, line_P2, P2

# Read the tracklets
def write_label(transform, P2):
    for trackletObj in xmlParser.parseXML(os.path.join(tracklet_path, 'tracklet_labels.xml')):
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in trackletObj:
            label_file = label_path + str(absoluteFrameNumber).zfill(10) + '.txt'
            image_file = image_path + str(absoluteFrameNumber).zfill(10) + '.png'
            img = cv2.imread(image_file)
            img_xmax, img_ymax = img.shape[1], img.shape[0]

            translation = np.append(translation, 1)
            translation = np.dot(transform, translation)
            translation = translation[:3]/translation[3]

            rot = -(rotation[2] + np.pi/2)
            if rot > np.pi:
                rot -= 2*np.pi
            elif rot < -np.pi:
                rot += 2*np.pi
            rot = round(rot, 2)

            local_rot = local_ori(translation, rot)


            bbox = obtain_2Dbox(trackletObj.size, translation, rot, P2, img_xmax, img_ymax)

            with open(label_file, 'a') as file_writer:
                line = [trackletObj.objectType] + [int(truncation),int(occlusion[0]),local_rot] + bbox + [round(size, 2) for size in trackletObj.size] \
                + [round(tran, 2) for tran in translation] + [rot]
                line = ' '.join([str(item) for item in line]) + '\n'
                file_writer.write(line)
            
def write_calib(line_P2):
    for image in os.listdir(image_path):
        calib_file = calib_path + image.split('.')[0] + '.txt'

        # Create calib files
        with open(calib_file, 'w') as file_writer:
            file_writer.write(line_P2)

if __name__ == '__main__':
    base_dir = '/media/user/新加卷/kitti_dateset/'
    dir = ReadDir(base_dir=base_dir, subset='tracklet', tracklet_date='2011_09_26',
                  tracklet_file='2011_09_26_drive_0093_sync')
    tracklet_path = dir.tracklet_drive
    label_path = dir.label_dir
    image_path = dir.image_dir
    calib_path = dir.calib_dir
    pred_path = dir.prediction_dir

    makedir(label_path)
    makedir(calib_path)

    transform, line_P2, P2 = read_transformation_matrix()

    write_label(transform, P2)
    write_calib(line_P2)