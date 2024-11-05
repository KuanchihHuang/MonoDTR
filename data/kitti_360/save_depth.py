import os
import sys
from kitti360scripts.viewer.kitti360Viewer3DRaw import Kitti360Viewer3DRaw
from kitti360scripts.helpers.project import CameraPerspective
from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationCameraToPose, loadCalibrationRigid
import numpy as np

import pdb; pdb.set_trace()

kitti360Path = "data/kitti_360/KITTI-360"

seq_list = ["0000", "0002", "0003", "0004", "0005", "0006", "0007", "0009", "0010"]

for seq in seq_list:

    sequence = "2013_05_28_drive_"+seq+"_sync"
    cam_id = 0

    camera = CameraPerspective(kitti360Path, sequence, cam_id)
    velo = Kitti360Viewer3DRaw(mode='velodyne', seq=int(seq))
    fileCameraToVelo = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_velo.txt')
    TrCam0ToVelo = loadCalibrationRigid(fileCameraToVelo)
    fileCameraToPose = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_pose.txt')
    TrCamToPose = loadCalibrationCameraToPose(fileCameraToPose)

    TrVeloToCam = {}
    for k, v in TrCamToPose.items():
        TrCamkToCam0 = np.linalg.inv(TrCamToPose['image_00']) @ TrCamToPose[k]
        TrCamToVelo = TrCam0ToVelo @ TrCamkToCam0
        TrVeloToCam[k] = np.linalg.inv(TrCamToVelo)
    TrVeloToRect = np.matmul(camera.R_rect, TrVeloToCam['image_%02d' % cam_id])


    file_counts = sorted(os.listdir(os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_00/data_rect')))


    for frame in file_counts:
        frame = int(frame.split(".")[0])
        print(frame)

        points = velo.loadVelodyneData(frame)
        points[:, 3] = 1

        pointsCam = np.matmul(TrVeloToRect, points.T).T
        pointsCam = pointsCam[:, :3]

        u, v, depth = camera.cam2image(pointsCam.T)
        u = u.astype(int)
        v = v.astype(int)

        depthMap = np.zeros((camera.height, camera.width))
        mask = np.logical_and(np.logical_and(np.logical_and(u >= 0, u < camera.width), v >= 0), v < camera.height)
        mask = np.logical_and(mask, depth > 0)
        depthMap[v[mask], u[mask]] = depth[mask]
    
        save_dir = os.path.join(kitti360Path, 'depth', sequence)
        os.makedirs(save_dir, exist_ok=True)
        print(save_dir,str(frame).zfill(10)+".npy")
        np.save(os.path.join(save_dir,str(frame).zfill(10)+".npy"),depthMap)
        import pdb; pdb.set_trace()
