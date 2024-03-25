import os
import csv
import numpy as np
from config import config as cfg

class KITTILoader():
    def __init__(self, subset='training'):
        super(KITTILoader, self).__init__()

        self.base_dir = cfg().base_dir
        self.KITTI_cat = cfg().KITTI_cat

        label_dir = os.path.join(self.base_dir, subset, 'label_2')
        image_dir = os.path.join(self.base_dir, subset, 'image_2')

        self.image_data = []
        self.images = []

        for i, fn in enumerate(os.listdir(label_dir)):
            label_full_path = os.path.join(label_dir, fn)
            image_full_path = os.path.join(image_dir, fn.replace('.txt', '.png'))

            self.images.append(image_full_path)
            fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw', 'dl',
                          'lx', 'ly', 'lz', 'ry']
            with open(label_full_path, 'r') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)

                for line, row in enumerate(reader):

                    if row['type'] in self.KITTI_cat:
                        if subset == 'training':
                            new_alpha = get_new_alpha(row['alpha'])
                            dimensions = np.array([float(row['dh']), float(row['dw']), float(row['dl'])])
                            annotation = {'name': row['type'], 'image': image_full_path,
                                          'xmin': int(float(row['xmin'])), 'ymin': int(float(row['ymin'])),
                                          'xmax': int(float(row['xmax'])), 'ymax': int(float(row['ymax'])),
                                          'dims': dimensions, 'new_alpha': new_alpha}

                        elif subset == 'eval':
                            dimensions = np.array([float(row['dh']), float(row['dw']), float(row['dl'])])
                            translations = np.array([float(row['lx']), float(row['ly']), float(row['lz'])])
                            annotation = {'name': row['type'], 'image': image_full_path,
                                          'alpha': float(row['alpha']),
                                          'xmin': int(float(row['xmin'])), 'ymin': int(float(row['ymin'])),
                                          'xmax': int(float(row['xmax'])), 'ymax': int(float(row['ymax'])),
                                          'dims': dimensions, 'trans': translations, 'rot_y': float(row['ry'])}


                        self.image_data.append(annotation)

    def get_average_dimension(self):
        dims_avg = {key: np.array([0, 0, 0]) for key in self.KITTI_cat}
        dims_cnt = {key: 0 for key in self.KITTI_cat}

        for i in range(len(self.image_data)):
            current_data = self.image_data[i]
            if current_data['name'] in self.KITTI_cat:
                dims_avg[current_data['name']] = dims_cnt[current_data['name']] * dims_avg[current_data['name']] + \
                                                 current_data['dims']
                dims_cnt[current_data['name']] += 1
                dims_avg[current_data['name']] /= dims_cnt[current_data['name']]
        return dims_avg, dims_cnt

def get_new_alpha(alpha):
    """
    change the range of orientation from [-pi, pi] to [0, 2pi]
    :param alpha: original orientation in KITTI
    :return: new alpha
    """
    new_alpha = float(alpha) + np.pi / 2.
    if new_alpha < 0:
        new_alpha = new_alpha + 2. * np.pi
        # make sure angle lies in [0, 2pi]
    new_alpha = new_alpha - int(new_alpha / (2. * np.pi)) * (2. * np.pi)

    return new_alpha

if __name__ == '__main__':
    base_dir = '/home/user/Deep3DBOX_Keras_Modified/kitti_test'
    KITTI_gen = KITTILoader(subset='training')
    dim_avg, dim_cnt = KITTI_gen.get_average_dimension()
    print(dim_avg, dim_cnt)

    # input = 180 * np.pi / 180
    # a = KITTILoader.get_new_alpha(input)
    # print(a * 180 / np.pi)
