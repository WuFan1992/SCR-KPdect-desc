
import numpy as np
import cv2
import os.path as osp

from skimage.io import imread



def load_extrinsic(meta_info):
    if len(meta_info["extrinsic_Tcw"]) == 16:
        Tcw = np.array(meta_info["extrinsic_Tcw"]).reshape(4, 4)
        Tcw = Tcw[:3, :]
    else:
        Tcw = np.array(meta_info["extrinsic_Tcw"]).reshape(3, 4)
    return Tcw


def load_intrinsic(meta_info):
    # if dataset=='default':
    K_param = meta_info["camera_intrinsic"]
    K = np.zeros((3, 3))
    K[0, 0] = K_param[0]
    K[1, 1] = K_param[1]
    K[2, 2] = 1
    K[0, 2] = K_param[2]
    K[1, 2] = K_param[3]
    return K


def load_depth_from_png(tiff_file_path):
    depth = cv2.imread(tiff_file_path, cv2.IMREAD_ANYDEPTH)
    depth[depth==65535]=0
    return depth


def load_one_img(
    base_dir, scene_info, idx, read_img=True):
    
    pose = scene_info["poses"][idx]
    K = scene_info["intrinsics"][idx]
    
    file_name = scene_info["image_paths"][idx]
    depth_file_name = scene_info["depth_paths"][idx]

    img = None
    img_path = osp.join(base_dir, file_name)
    depth_file_name = osp.join(base_dir, depth_file_name)
    if read_img:
        img = imread(img_path)
    depth = load_depth_from_png(depth_file_name)

    depth=depth.astype(np.float32)/1000
    depth[depth < 1e-5] = 0
    return img, depth, pose, K
