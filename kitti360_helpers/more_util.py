"""
    Sample Run:
    python .py
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import math
import torch
import shutil
import cv2
from torch import distributed

def mkdir_if_missing(directory, delete_if_exist=False):
    """
    Recursively make a directory structure even if missing.
    if delete_if_exist=True then we will delete it first
    which can be useful when better control over initialization is needed.
    """
    if delete_if_exist and os.path.exists(directory): shutil.rmtree(directory)

    # check if not exist, then make
    if not os.path.exists(directory):
        os.makedirs(directory)

def project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=False):
    """
    Projects a 3D box into 2D vertices
    Args:
        p2 (nparray): projection matrix of size 4x3
        x3d: x-coordinate of center of object
        y3d: y-coordinate of center of object
        z3d: z-cordinate of center of object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis
    """

    if type(x3d) == np.ndarray:

        p2_batch = np.zeros([x3d.shape[0], 4, 4])
        p2_batch[:, :, :] = p2[np.newaxis, :, :]

        ry3d_cos = np.cos(ry3d)
        ry3d_sin = np.sin(ry3d)

        R = np.zeros([x3d.shape[0], 4, 3])
        R[:, 0, 0] = ry3d_cos
        R[:, 0, 2] = ry3d_sin
        R[:, 1, 1] = 1
        R[:, 2, 0] = -ry3d_sin
        R[:, 2, 2] = ry3d_cos

        corners_3d = np.zeros([x3d.shape[0], 3, 8])

        # setup X
        corners_3d[:, 0, :] = -l3d[:, np.newaxis] / 2
        corners_3d[:, 0, 1:5] = l3d[:, np.newaxis] / 2

        # setup Y
        corners_3d[:, 1, :] = -h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 2:4] = h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 6:8] = h3d[:, np.newaxis] / 2

        # setup Z
        corners_3d[:, 2, :] = -w3d[:, np.newaxis] / 2
        corners_3d[:, 2, 3:7] = w3d[:, np.newaxis] / 2

        # rotate
        corners_3d = R @ corners_3d

        # translate
        corners_3d[:, 0, :] += x3d[:, np.newaxis]
        corners_3d[:, 1, :] += y3d[:, np.newaxis]
        corners_3d[:, 2, :] += z3d[:, np.newaxis]
        corners_3d[:, 3, :] = 1

        # project to 2D
        corners_2d = p2_batch @ corners_3d

        corners_2d[:, :2, :] /= corners_2d[:, 2, :][:, np.newaxis, :]

        verts3d = corners_2d

    elif type(x3d) == torch.Tensor:

        p2_batch = torch.zeros(x3d.shape[0], 4, 4)
        p2_batch[:, :, :] = p2[np.newaxis, :, :]

        ry3d_cos = torch.cos(ry3d)
        ry3d_sin = torch.sin(ry3d)

        R = torch.zeros(x3d.shape[0], 4, 3)
        R[:, 0, 0] = ry3d_cos
        R[:, 0, 2] = ry3d_sin
        R[:, 1, 1] = 1
        R[:, 2, 0] = -ry3d_sin
        R[:, 2, 2] = ry3d_cos

        corners_3d = torch.zeros(x3d.shape[0], 3, 8)

        # setup X
        corners_3d[:, 0, :] = -l3d[:, np.newaxis] / 2
        corners_3d[:, 0, 1:5] = l3d[:, np.newaxis] / 2

        # setup Y
        corners_3d[:, 1, :] = -h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 2:4] = h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 6:8] = h3d[:, np.newaxis] / 2

        # setup Z
        corners_3d[:, 2, :] = -w3d[:, np.newaxis] / 2
        corners_3d[:, 2, 3:7] = w3d[:, np.newaxis] / 2

        # rotate
        corners_3d = torch.bmm(R, corners_3d)

        corners_3d = corners_3d.to(x3d.device)
        p2_batch = p2_batch.to(x3d.device)

        # translate
        corners_3d[:, 0, :] += x3d[:, np.newaxis]
        corners_3d[:, 1, :] += y3d[:, np.newaxis]
        corners_3d[:, 2, :] += z3d[:, np.newaxis]
        corners_3d[:, 3, :] = 1

        # project to 2D
        corners_2d = torch.bmm(p2_batch, corners_3d)

        corners_2d[:, :2, :] /= corners_2d[:, 2, :][:, np.newaxis, :]

        verts3d = corners_2d

    else:

        # compute rotational matrix around yaw axis
        R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)],
                      [0, 1, 0],
                      [-math.sin(ry3d), 0, +math.cos(ry3d)]])

        # 3D bounding box corners
        x_corners = np.array([0, l3d, l3d, l3d, l3d, 0, 0, 0])
        y_corners = np.array([0, 0, h3d, h3d, 0, 0, h3d, h3d])
        z_corners = np.array([0, 0, 0, w3d, w3d, w3d, w3d, 0])

        x_corners += -l3d / 2
        y_corners += -h3d / 2
        z_corners += -w3d / 2

        # bounding box in object co-ordinate
        corners_3d = np.array([x_corners, y_corners, z_corners])

        # rotate
        corners_3d = R.dot(corners_3d)

        # translate
        corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))

        corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
        corners_2D = p2.dot(corners_3D_1)
        corners_2D = corners_2D / corners_2D[2]

        # corners_2D = np.zeros([3, corners_3d.shape[1]])
        # for i in range(corners_3d.shape[1]):
        #    a, b, c, d = argoverse.utils.calibration.proj_cam_to_uv(corners_3d[:, i][np.newaxis, :], p2)
        #    corners_2D[:2, i] = a
        #    corners_2D[2, i] = corners_3d[2, i]

        bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]

        verts3d = corners_2D[:2].T#(corners_2D[:, bb3d_lines_verts_idx][:2]).astype(float).T

    if return_3d:
        return verts3d, corners_3d
    else:
        return verts3d

def backproject_2d_pixels_in_4D_format(p2_inv, points, pad_ones= False):
    """
        Projects 2d points with x and y in pixels and appened with ones to 3D using inverse of projection matrix
        :param p2_inv:   np array 4 x 4
        :param points:   np array 4 x N or 3 x N
        :param pad_ones: whether to pad_ones or not. 3 X N shaped points need to be padded
        :return: coord2d np array 4 x N
        """
    if type(points) == np.ndarray:
        if pad_ones:
            N = points.shape[1]
            points_4d = np.vstack((points, np.ones((1, N))))
        else:
            points_4d = points

        points_4d[0] = np.multiply(points_4d[0], points_4d[2])
        points_4d[1] = np.multiply(points_4d[1], points_4d[2])
        output       = np.matmul(p2_inv, points_4d)

    elif type(points) == torch.Tensor:
        if pad_ones:
            N = points.shape[1]
            points_4d = torch.vstack((points, torch.ones((1, N), dtype= points.dtype, device= points.device)))
        else:
            points_4d = points

        points_4d[0] = points_4d[0] * points_4d[2]
        points_4d[1] = points_4d[1] * points_4d[2]
        output       = torch.matmul(p2_inv, points_4d)

    return output

def project_3d_points_in_4D_format(p2, points_4d, pad_ones= False):
    """
    Projects 3d points appened with ones to 2d using projection matrix
    :param p2:       np array 4 x 4
    :param points:   np array 4 x N
    :return: coord2d np array 4 x N
    """
    N = points_4d.shape[1]
    z_eps = 1e-2

    if type(points_4d) == np.ndarray:
        if pad_ones:
            points_4d = np.vstack((points_4d, np.ones((1, N))))

        coord2d = np.matmul(p2, points_4d)
        ind = np.where(np.abs(coord2d[2]) > z_eps)
    elif type(points_4d) == torch.Tensor:
        if pad_ones:
            points_4d = torch.cat([points_4d, torch.ones((1, N))], dim= 0)

        coord2d = torch.matmul(p2, points_4d)
        ind = torch.abs(coord2d[2]) > z_eps

    coord2d[:2, ind] /= coord2d[2, ind]

    return coord2d

def ex_box_jaccard(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    inter_x1 = np.maximum(np.min(a[:,0]), np.min(b[:,0]))
    inter_x2 = np.minimum(np.max(a[:,0]), np.max(b[:,0]))
    inter_y1 = np.maximum(np.min(a[:,1]), np.min(b[:,1]))
    inter_y2 = np.minimum(np.max(a[:,1]), np.max(b[:,1]))
    if inter_x1>=inter_x2 or inter_y1>=inter_y2:
        return 0.
    x1 = np.minimum(np.min(a[:,0]), np.min(b[:,0]))
    x2 = np.maximum(np.max(a[:,0]), np.max(b[:,0]))
    y1 = np.minimum(np.min(a[:,1]), np.min(b[:,1]))
    y2 = np.maximum(np.max(a[:,1]), np.max(b[:,1]))
    mask_w = np.int(np.ceil(x2-x1))
    mask_h = np.int(np.ceil(y2-y1))
    mask_a = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
    mask_b = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
    a[:,0] -= x1
    a[:,1] -= y1
    b[:,0] -= x1
    b[:,1] -= y1
    mask_a = cv2.fillPoly(mask_a, pts=np.asarray([a], 'int32'), color=1)
    mask_b = cv2.fillPoly(mask_b, pts=np.asarray([b], 'int32'), color=1)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    iou = float(inter)/(float(union)+1e-12)

    return iou

def convertRot2Alpha(ry3d, z3d, x3d):

    if type(z3d) == torch.Tensor:
        alpha = ry3d - torch.atan2(-z3d, x3d) - 0.5 * math.pi
        while torch.any(alpha > math.pi): alpha[alpha > math.pi] -= math.pi * 2
        while torch.any(alpha <= (-math.pi)): alpha[alpha <= (-math.pi)] += math.pi * 2

    elif type(z3d) == np.ndarray:
        alpha = ry3d - np.arctan2(-z3d, x3d) - 0.5 * math.pi
        while np.any(alpha > math.pi): alpha[alpha > math.pi] -= math.pi * 2
        while np.any(alpha <= (-math.pi)): alpha[alpha <= (-math.pi)] += math.pi * 2

    else:
        alpha = ry3d - math.atan2(-z3d, x3d) - 0.5 * math.pi
        while alpha > math.pi: alpha -= math.pi * 2
        while alpha <= (-math.pi): alpha += math.pi * 2

    return alpha

def smax_and_drop(input, do_smax= True):
    b, c, h, w = input.shape
    if do_smax:
        smaxed_map = torch.nn.functional.softmax(input, dim=1)
    else:
        smaxed_map = input
    # Drop the background
    return smaxed_map[:, :(c-1)]

def clip_sigmoid(input):
    output = torch.clamp(torch.sigmoid(input), min= 1e-3, max=1-1e-3).clone()
    return output

def packedSeq_to_tensor(input):
    temp  = []
    for t in range(len(input._tensors)):
        temp.append(input._tensors[t])
    output = torch.stack(temp, dim=0)
    return output

def per_class_to_single_occupancy(per_class_occupancy, num_det_classes):
    bs,_, h, w = per_class_occupancy.shape
    weights    = 1.0 + torch.arange(num_det_classes, dtype= per_class_occupancy.dtype, device= per_class_occupancy.device)
    single     = weights.unsqueeze(0).unsqueeze(2).unsqueeze(2).repeat(bs, 1, h, w) * per_class_occupancy # bs x 11 x 104 x 104
    single     = torch.max(single, dim= 1)[0].unsqueeze(1)                                     # bs x 1  x 104 x 104
    return single

def get_network_inputs(detector= False):
    if detector:
        NETWORK_INPUTS = ["img", "bev_msk", "cat", "iscrowd", "bbx", "calib", "sem_msk", "hm", "reg_mask", "ind", "cat_ind", "wh", "reg", "cls_theta", "h3d", "y3d", "yaw"]
    else:
        # NETWORK_INPUTS = ["img", "bev_msk", "front_msk", "weights_msk", "cat", "iscrowd", "bbx", "calib"]
        NETWORK_INPUTS = ["img", "bev_msk", "cat", "iscrowd", "bbx", "calib", "sem_msk"]
    return NETWORK_INPUTS

def cast_to_cpu_cuda_tensor(input, reference_tensor):
    if reference_tensor.is_cuda and not input.is_cuda:
        input = input.cuda()
    if not reference_tensor.is_cuda and input.is_cuda:
        input = input.cpu()
    return input

def zero_tensor_like(input):
    zeros = torch.zeros((1,))
    return cast_to_cpu_cuda_tensor(zeros, reference_tensor= input)

def get_semantic_GT(img_id, bev_panoptic, metadata):
    bev_semantic = np.zeros(bev_panoptic.shape).astype(np.uint8)
    num_images = len(metadata["images"])
    found_index = -1
    for i in range(num_images):
        if metadata['images'][i]['id'] == img_id:
            found_index = i
            break

    cat_img = []
    calib_img = np.eye(4)
    if found_index >= 0:
        cat_img    = metadata['images'][found_index]['cat']
        calib_temp = np.array(metadata['images'][found_index]['cam_intrinsic'])
        calib_img[:3, :3] = calib_temp

        assert len(np.unique(bev_panoptic)) == len(cat_img)
        for i, cat_curr in enumerate(cat_img):
            bev_semantic[bev_panoptic == i] = cat_curr

    return bev_semantic, cat_img, calib_img

def custom_print(out_string, logger= None, debug= True):
    if logger is not None:
        logger.info(out_string)
    else:
        print(out_string)

# Print an error message and quit
def printError(message):
    print('ERROR: ' + str(message))
    sys.exit(-1)

def draw_line(im, v1, v2, color=(0, 200, 200), thickness=1):

    cv2.line(im, (int(v1[0]), int(v1[1])), (int(v2[0]), int(v2[1])), color, thickness)

def draw_circle(im, pos, radius=5, thickness=1, color=(250, 100, 100), fill=True):

    if fill: thickness = -1

    cv2.circle(im, (int(pos[0]), int(pos[1])), radius, color=color, thickness=thickness)

def draw_transparent_box(im, box, blend=0.5, color=(0, 255, 255)):

    x_s = int(np.clip(min(box[0], box[2]), a_min=0, a_max=im.shape[1]))
    x_e = int(np.clip(max(box[0], box[2]), a_min=0, a_max=im.shape[1]))
    y_s = int(np.clip(min(box[1], box[3]), a_min=0, a_max=im.shape[0]))
    y_e = int(np.clip(max(box[1], box[3]), a_min=0, a_max=im.shape[0]))

    im[y_s:y_e + 1, x_s:x_e + 1, 0] = im[y_s:y_e + 1, x_s:x_e + 1, 0] * blend + color[0] * (1 - blend)
    im[y_s:y_e + 1, x_s:x_e + 1, 1] = im[y_s:y_e + 1, x_s:x_e + 1, 1] * blend + color[1] * (1 - blend)
    if __name__ == '__main__':
        im[y_s:y_e + 1, x_s:x_e + 1, 2] = im[y_s:y_e + 1, x_s:x_e + 1, 2] * blend + color[2] * (1 - blend)

def draw_3d_box(im, verts, color=(0, 200, 200), thickness=1):

    # Make lines
    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
    verts = verts[bb3d_lines_verts_idx]
    for lind in range(0, verts.shape[0] - 1):
        v1 = verts[lind]
        v2 = verts[lind + 1]
        cv2.line(im, (int(v1[0]), int(v1[1])), (int(v2[0]), int(v2[1])), color, thickness)

    draw_transparent_polygon(im, verts[5:9, :], blend=0.5, color=color)

def draw_2d_box(im, verts, color=(0, 200, 200), thickness=1, verts_as_corners= True):
    if verts_as_corners:
        x1, y1 = np.min(verts, axis=0)
        x2, y2 = np.max(verts, axis=0)
    else:
        x1, y1, x2, y2 = verts
    verts =np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]])
    for lind in range(0, verts.shape[0] - 1):
        v1 = verts[lind]
        v2 = verts[lind + 1]
        cv2.line(im, (int(v1[0]), int(v1[1])), (int(v2[0]), int(v2[1])), color, thickness)

def get_polygon_grid(im, poly_verts):

    from matplotlib.path import Path

    nx = im.shape[1]
    ny = im.shape[0]
    #poly_verts = [(1, 1), (5, 1), (5, 9), (3, 2), (1, 1)]

    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T

    path = Path(poly_verts)
    grid = path.contains_points(points)
    grid = grid.reshape((ny, nx))

    return grid

def draw_transparent_polygon(im, verts, blend=0.5, color=(0, 255, 255)):

    mask = get_polygon_grid(im, verts[:4, :])

    im[mask, 0] = im[mask, 0] * blend + (1 - blend) * color[0]
    im[mask, 1] = im[mask, 1] * blend + (1 - blend) * color[1]
    im[mask, 2] = im[mask, 2] * blend + (1 - blend) * color[2]

def interp_color(dist, bounds=[0, 1], color_lo=(0,0, 250), color_hi=(0, 250, 250)):

    percent = (dist - bounds[0]) / (bounds[1] - bounds[0])
    weight  = np.power(percent, 2.0)
    b = color_lo[0] * (1 - weight) + color_hi[0] * weight
    g = color_lo[1] * (1 - weight) + color_hi[1] * weight
    r = color_lo[2] * (1 - weight) + color_hi[2] * weight

    return (b, g, r)

def create_colorbar(height, width, color_lo=(0,0, 250), color_hi=(0, 250, 250)):

    im = np.zeros([height, width, 3])

    for h in range(0, height):

        color = interp_color(h + 0.5, [0, height], color_hi, color_lo)
        im[h, :, 0] = (color[0])
        im[h, :, 1] = (color[1])
        im[h, :, 2] = (color[2])

    return im.astype(np.uint8)

def draw_tick_marks(im, ticks):

    ticks_loc = list(range(0, im.shape[0] + 1, int((im.shape[0]) / (len(ticks) - 1))))

    for tind, tick in enumerate(ticks):
        y = min(max(ticks_loc[tind], 50), im.shape[0] - 10)
        x = im.shape[1] - 200

        draw_text(im, '-{}m'.format(tick), (x, y), lineType=2, scale=2.0, bg_color=None)

def draw_filled_rectangle(im, x_s, x_e, y_s, y_e, bg_color= (0, 255, 255), blend= 0.33, border= 0, border_color= (0, 0, 0)):
    if border > 0:
        im[y_s:y_s+border, x_s:x_e + 1, 0] = im[y_s:y_s+border, x_s:x_e + 1, 0]*blend + border_color[0] * (1 - blend)
        im[y_s:y_s+border, x_s:x_e + 1, 1] = im[y_s:y_s+border, x_s:x_e + 1, 1]*blend + border_color[1] * (1 - blend)
        im[y_s:y_s+border, x_s:x_e + 1, 2] = im[y_s:y_s+border, x_s:x_e + 1, 2]*blend + border_color[2] * (1 - blend)

        im[y_e - border + 1: y_e + 1, x_s:x_e + 1, 0] = im[y_e - border + 1: y_e + 1, x_s:x_e + 1, 0]*blend + border_color[0] * (1 - blend)
        im[y_e - border + 1: y_e + 1, x_s:x_e + 1, 1] = im[y_e - border + 1: y_e + 1, x_s:x_e + 1, 1]*blend + border_color[1] * (1 - blend)
        im[y_e - border + 1: y_e + 1, x_s:x_e + 1, 2] = im[y_e - border + 1: y_e + 1, x_s:x_e + 1, 2]*blend + border_color[2] * (1 - blend)

        im[y_s:y_e + 1, x_s:x_s+border, 0] = im[y_s:y_e + 1, x_s:x_s+border, 0]*blend + border_color[0] * (1 - blend)
        im[y_s:y_e + 1, x_s:x_s+border, 1] = im[y_s:y_e + 1, x_s:x_s+border, 1]*blend + border_color[1] * (1 - blend)
        im[y_s:y_e + 1, x_s:x_s+border, 2] = im[y_s:y_e + 1, x_s:x_s+border, 2]*blend + border_color[2] * (1 - blend)

        im[y_s:y_e + 1, x_e-border + 1:x_e + 1, 0] = im[y_s:y_e + 1, x_e-border + 1:x_e + 1, 0]*blend + border_color[0] * (1 - blend)
        im[y_s:y_e + 1, x_e-border + 1:x_e + 1, 1] = im[y_s:y_e + 1, x_e-border + 1:x_e + 1, 1]*blend + border_color[1] * (1 - blend)
        im[y_s:y_e + 1, x_e-border + 1:x_e + 1, 2] = im[y_s:y_e + 1, x_e-border + 1:x_e + 1, 2]*blend + border_color[2] * (1 - blend)

        # Image
        im[y_s+border:y_e-border + 1, x_s+border:x_e-border + 1, 0] = im[y_s+border:y_e-border + 1, x_s+border:x_e-border + 1, 0]*blend + bg_color[0] * (1 - blend)
        im[y_s+border:y_e-border + 1, x_s+border:x_e-border + 1, 1] = im[y_s+border:y_e-border + 1, x_s+border:x_e-border + 1, 1]*blend + bg_color[1] * (1 - blend)
        im[y_s+border:y_e-border + 1, x_s+border:x_e-border + 1, 2] = im[y_s+border:y_e-border + 1, x_s+border:x_e-border + 1, 2]*blend + bg_color[2] * (1 - blend)

    else:
        im[y_s:y_e + 1, x_s:x_e + 1, 0] = im[y_s:y_e + 1, x_s:x_e + 1, 0]*blend + bg_color[0] * (1 - blend)
        im[y_s:y_e + 1, x_s:x_e + 1, 1] = im[y_s:y_e + 1, x_s:x_e + 1, 1]*blend + bg_color[1] * (1 - blend)
        im[y_s:y_e + 1, x_s:x_e + 1, 2] = im[y_s:y_e + 1, x_s:x_e + 1, 2]*blend + bg_color[2] * (1 - blend)

    return  im

def draw_text(im, text, pos, scale=0.4, color=(0, 0, 0), font=cv2.FONT_HERSHEY_SIMPLEX, bg_color=(0, 255, 255),
              blend=0.33, lineType=1, pad= 0, border= 0, border_color= (0, 0, 0)):

    pos = [int(pos[0]), int(pos[1])]

    if bg_color is not None:

        text_size, _ = cv2.getTextSize(text, font, scale, lineType)
        x_s = int(np.clip(pos[0] - pad, a_min=0, a_max=im.shape[1]))
        x_e = int(np.clip(pos[0] + pad + text_size[0] - 2, a_min=0, a_max=im.shape[1]))
        y_s = int(np.clip(pos[1] - pad - text_size[1] + 2, a_min=0, a_max=im.shape[0]))
        y_e = int(np.clip(pos[1] + pad , a_min=0, a_max=im.shape[0]))

        im = draw_filled_rectangle(im, x_s, x_e, y_s, y_e, bg_color= bg_color, blend= blend, border= border, border_color= border_color)

        pos[0] = int(np.clip(pos[0] + 2, a_min=0, a_max=im.shape[1]))
        pos[1] = int(np.clip(pos[1] - 2, a_min=0, a_max=im.shape[0]))

    cv2.putText(im, text, tuple(pos), font, scale, color, lineType)

def draw_bev(canvas_bev, z3d, l3d, w3d, x3d, ry3d, color=(0, 200, 200), scale=1, thickness=2, text= None):

    w = l3d * scale
    l = w3d * scale
    x = x3d * scale
    z = z3d * scale
    r = ry3d*-1

    corners1 = np.array([
        [-w / 2, -l / 2, 1],
        [+w / 2, -l / 2, 1],
        [+w / 2, +l / 2, 1],
        [-w / 2, +l / 2, 1]
    ])

    ry = np.array([
        [+math.cos(r), -math.sin(r), 0],
        [+math.sin(r), math.cos(r), 0],
        [0, 0, 1],
    ])

    corners2 = ry.dot(corners1.T).T

    corners2[:, 0] += x + canvas_bev.shape[1] / 2
    corners2[:, 1] += z

    draw_line(canvas_bev, corners2[0], corners2[1], color=color, thickness=thickness)
    draw_line(canvas_bev, corners2[1], corners2[2], color=color, thickness=thickness)
    draw_line(canvas_bev, corners2[2], corners2[3], color=color, thickness=thickness)
    draw_line(canvas_bev, corners2[3], corners2[0], color=color, thickness=thickness)

    if text is not None:
        thickness=2
        cv2.putText(canvas_bev, text, (int(corners2[0, 0]), int(corners2[0, 1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), thickness, cv2.LINE_AA)

def draw_border(image, thickness= 4, style= 'left'):
    # image = h x w x 3
    if style in ['top', 'height', 'all']:
        image[:thickness ]          = 0
    if style in ['bottom', 'height', 'all']:
        image[-(thickness+1):-1]    = 0
    if style in ['left', 'width', 'all']:
        image[:, :thickness]        = 0
    if style in ['right', 'width', 'all']:
        image[:, -(thickness+1):-1] = 0

    return image

def imhstack(im1, im2):

    sf = im1.shape[0] / im2.shape[0]

    if sf > 1:
        im2 = cv2.resize(im2, (int(im2.shape[1] / sf), im1.shape[0]))
    elif sf < 1:
        im1 = cv2.resize(im1, (int(im1.shape[1] / sf), im2.shape[0]))


    im_concat = np.hstack((im1, im2))

    return im_concat