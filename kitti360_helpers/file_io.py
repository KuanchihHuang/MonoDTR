

"""
    File Input Output Operations for different kinds of files.
"""
import os
import numpy as np
import pandas as pd
import cv2
import pickle
import logging
import warnings
import json
import io
import yaml
import umsgpack

def read_image(path, sixteen_bit= False, rgb= False):
    if sixteen_bit:
        return cv2.imread(path, -1)
    else:
        if rgb:
            return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return cv2.imread(path)

def write_image(path, im):
    cv2.imwrite(path, im)

def read_csv(path, delimiter= " ", ignore_warnings= False, use_pandas= False):
    try:
        if ignore_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if use_pandas:
                    data = pd.read_csv(path, delimiter= delimiter, header=None).values
                else:
                    data = np.genfromtxt(path, delimiter= delimiter)
        else:
            if use_pandas:
                data = pd.read_csv(path, delimiter=delimiter, header=None).values
            else:
                data = np.genfromtxt(path, delimiter=delimiter)
    except:
        data = None

    return data

def write_csv(path, numpy_variable, delimiter= " ", save_folder= None, float_format= None):
    if save_folder is not None:
        path = os.path.join(save_folder, path)

    pd.DataFrame(numpy_variable).to_csv(path, sep= delimiter, header=None, index=None, float_format= float_format)

def read_lines(path, strip= True):
    with open(path) as f:
        lines = f.readlines()

    if strip:
        # you may also want to remove whitespace characters like `\n` at the end of each line
        lines = [x.rstrip('\n') for x in lines]

    return lines

def write_lines(path, lines_with_return_character):
    with open(path, 'w') as f:
        f.writelines(lines_with_return_character)

def read_numpy(path, folder= None, show_message= True):
    if folder is not None:
        path = os.path.join(folder, path)

    if show_message:
        logging.info("=> Reading {}".format(path))

    return np.load(path)

def save_numpy(path, numpy_variable, save_folder= None, show_message= True):
    if save_folder is not None:
        path = os.path.join(save_folder, path)

    if show_message:
        logging.info("=> Saving to {}".format(path))
    np.save(path, numpy_variable)

def read_pickle(path):
    """
    De-serialize an object from a provided path
    """
    print("=> Loading pickle {}".format(path))
    with open(path, 'rb') as file:
        return pickle.load(file)

def write_pickle(path, obj):
    """
    Serialize an object to a provided path
    """
    logging.info("=> Saving pickle to {}".format(path))
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

def read_json(path):
    print("=> Loading JSON {}".format(path))
    with open(path, 'rb') as file:
        return json.load(file)

def write_json(path, dict_name, sort_keys= False):
    print("=> Writing JSON {}".format(path))
    if sort_keys:
        with open(path, 'w') as f:
            f.write(json.dumps(dict_name, default=lambda o: o.__dict__, sort_keys= True, indent= 4))
    else:
        with open(path, 'w') as f:
            f.write(json.dumps(dict_name, default=lambda o: o.__dict__))

def read_yaml(path):
    print("=> Loading YAML {}".format(path))
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def write_yaml(path, obj):
    logging.info("=> Saving YAML to {}".format(path))
    with io.open(path, 'w', encoding= 'utf8') as f:
        yaml.dump(obj, f, default_flow_style= False, allow_unicode= True)

def writeDict2JSON(dictName, fileName):
    write_json(path= fileName, dict_name= dictName, sort_keys= True)

def read_panoptic_dataset_binary(bin_path):
    print("=> Loading Panoptic Binary {}".format(bin_path))
    with open(bin_path, "rb") as fid:
        return umsgpack.unpack(fid, encoding="utf-8")

#      0===============================0
#      |    PLY files reader/writer    |
#      0===============================0
#      Hugues THOMAS - 10/02/2017
#      https://github.com/HuguesTHOMAS/KPConv
ply_dtypes = dict([
    (b'int8', 'i1'),
    (b'char', 'i1'),
    (b'uint8', 'u1'),
    (b'uchar', 'u1'),
    (b'int16', 'i2'),
    (b'short', 'i2'),
    (b'uint16', 'u2'),
    (b'ushort', 'u2'),
    (b'int32', 'i4'),
    (b'int', 'i4'),
    (b'uint32', 'u4'),
    (b'uint', 'u4'),
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
])

# Numpy reader format
valid_formats = {'ascii': '', 'binary_big_endian': '>',
                 'binary_little_endian': '<'}

def parse_header(plyfile, ext):
    # Variables
    line = []
    properties = []
    num_points = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        if b'element' in line:
            line = line.split()
            num_points = int(line[2])

        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties


def parse_mesh_header(plyfile, ext):
    # Variables
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None


    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        # Find point element
        if b'element vertex' in line:
            current_element = 'vertex'
            line = line.split()
            num_points = int(line[2])

        elif b'element face' in line:
            current_element = 'face'
            line = line.split()
            num_faces = int(line[2])

        elif b'property' in line:
            if current_element == 'vertex':
                line = line.split()
                vertex_properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
            elif current_element == 'vertex':
                if not line.startswith('property list uchar int'):
                    raise ValueError('Unsupported faces property : ' + line)

    return num_points, num_faces, vertex_properties


def read_ply(filename, triangular_mesh=False):
    """
    Read ".ply" files
    Parameters
    ----------
    filename : string
        the name of the file to read.
    Returns
    -------
    result : array
        data stored in the file
    Examples
    --------
    Store data in file
    >>> points = np.random.rand(5, 3)
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])
    Read the file
    >>> data = read_ply('example.ply')
    >>> values = data['values']
    array([0, 0, 1, 1, 0])

    >>> points = np.vstack((data['x'], data['y'], data['z'])).T
    array([[ 0.466  0.595  0.324]
           [ 0.538  0.407  0.654]
           [ 0.850  0.018  0.988]
           [ 0.395  0.394  0.363]
           [ 0.873  0.996  0.092]])
    """

    with open(filename, 'rb') as plyfile:


        # Check if the file start with ply
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start whith the word ply')

        # get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError('The file is not binary')

        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # PointCloud reader vs mesh reader
        if triangular_mesh:

            # Parse header
            num_points, num_faces, properties = parse_mesh_header(plyfile, ext)

            # Get point data
            vertex_data = np.fromfile(plyfile, dtype=properties, count=num_points)

            # Get face data
            face_properties = [('k', ext + 'u1'),
                               ('v1', ext + 'i4'),
                               ('v2', ext + 'i4'),
                               ('v3', ext + 'i4')]
            faces_data = np.fromfile(plyfile, dtype=face_properties, count=num_faces)

            # Return vertex data and concatenated faces
            faces = np.vstack((faces_data['v1'], faces_data['v2'], faces_data['v3'])).T
            data = [vertex_data, faces]

        else:

            # Parse header
            num_points, properties = parse_header(plyfile, ext)

            # Get data
            data = np.fromfile(plyfile, dtype=properties, count=num_points)

    return data
