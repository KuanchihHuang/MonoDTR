'''
from compute anchors to data_gen

'''
import numpy as np
from config import config as cfg

def compute_anchors(angle):
    """
    compute angle offset and which bin the angle lies in
    input: fixed local orientation [0, 2pi]
    output: [bin number, angle offset]

    For two bins:

    if angle < pi, l = 0, r = 1
        if    angle < 1.65, return [0, angle]
        elif  pi - angle < 1.65, return [1, angle - pi]

    if angle > pi, l = 1, r = 2
        if    angle - pi < 1.65, return [1, angle - pi]
      elif     2pi - angle < 1.65, return [0, angle - 2pi]
    """
    anchors = []

    wedge = 2. * np.pi / cfg().bin  # 2pi / bin = pi
    l_index = int(angle / wedge)  # angle/pi
    r_index = l_index + 1

    # (angle - l_index*pi) < pi/2 * 1.05 = 1.65
    if (angle - l_index * wedge) < wedge / 2 * (1 + cfg().overlap / 2):
        anchors.append([l_index, angle - l_index * wedge])

    # (r*pi + pi - angle) < pi/2 * 1.05 = 1.65
    if (r_index * wedge - angle) < wedge / 2 * (1 + cfg().overlap / 2):
        anchors.append([r_index % cfg().bin, angle - r_index * wedge])

    return anchors


def orientation_confidence_flip(image_data, dims_avg):
    for data in image_data:

        # minus the average dimensions
        data['dims'] = data['dims'] - dims_avg[data['name']]

        # fix orientation and confidence for no flip
        orientation = np.zeros((cfg().bin, 2))
        confidence = np.zeros(cfg().bin)

        anchors = compute_anchors(data['new_alpha'])

        for anchor in anchors:
            # each angle is represented in sin and cos
            orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
            confidence[anchor[0]] = 1

        confidence = confidence / np.sum(confidence)

        data['orient'] = orientation
        data['conf'] = confidence

        # Fix orientation and confidence for random flip
        orientation = np.zeros((cfg().bin, 2))
        confidence = np.zeros(cfg().bin)

        anchors = compute_anchors(2. * np.pi - data['new_alpha'])  # compute orientation and bin
        # for flipped images

        for anchor in anchors:
            orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
            confidence[anchor[0]] = 1

        confidence = confidence / np.sum(confidence)

        data['orient_flipped'] = orientation
        data['conf_flipped'] = confidence

    return image_data

if __name__ == '__main__':
    angle = np.pi/2
    bin = 2
    anchors = compute_anchors(angle)
    orientation = np.zeros((bin, 2))
    confidence = np.zeros(bin)
    for anchor in anchors:
        # each angle is represented in sin and cos
        orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
        confidence[anchor[0]] = 1

    confidence = confidence / np.sum(confidence)
    orientation = np.expand_dims(orientation, axis=0)
    print(anchors)