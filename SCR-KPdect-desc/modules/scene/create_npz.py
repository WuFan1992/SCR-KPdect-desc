import os
import sys
import numpy as np
from colmap_reader import read_points3D_binary, read_extrinsics_binary, read_intrinsics_binary,qvec2rotmat
from itertools import combinations
from scene_utils import focal2fov, getIntrinsic, getExtrinsic

def get_imagepair_index(path):
    """
     Get the image pair index 
    """
    point3d_path = os.path.join(path, "sparse/0/points3D.bin")
    _, _, _, _, img_ids, _  = read_points3D_binary(point3d_path)
    
    pair_set = set()   # Use the set to remove the duplicate

    for p_id in img_ids:
        imgs = img_ids[p_id]   # image ids observing this 3D point
        # Combination every 2 elements
        for a, b in combinations(imgs, 2):
            # 无序对规范化
            if abs(a-b) < 30:
                continue
            
            if a < b:
                pair_set.add((a, b))
            else:
                pair_set.add((b, a))
    # 转回 list[list]
    imgs_pairs = [list(pair) for pair in pair_set]
    return imgs_pairs


def readColmapCameras(cam_extrinsics, cam_intrinsics):

    max_id = max(cam_extrinsics.keys())
    image_name_list = [""] * (max_id+1)  # ex : img_id=768  image_name = 767.color.png 
    depth_name_list = [""] * (max_id+1)
    intrinsics_list = [np.zeros((3,3)) for _  in range(max_id+1)]
    pose_list = [np.zeros((4,4)) for _  in range(max_id+1)]

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        img_name = extr.name
        image_name_list[key] = img_name
        depth_name_list[key] = img_name.replace(".color.png", ".depth.png")
        
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        ### elif intr.model=="PINHOLE":
        elif intr.model=="PINHOLE" or intr.model=="OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        

        intrinsic = getIntrinsic(FovX, height, width)
        pose = getExtrinsic(R,T)
        intrinsics_list[key] = intrinsic
        pose_list[key] = pose
        
    sys.stdout.write('\n')
    return image_name_list, depth_name_list, intrinsics_list, pose_list

def readSceneInfo(path):
    
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    image_name_list, depth_name_list, intrinsics_list, pose_list = readColmapCameras(cam_extrinsics, cam_intrinsics)

    return image_name_list, depth_name_list, intrinsics_list, pose_list

    
    


    
    
    


def create_npz(data_path, save_path):
    
    # Get the image pair index
    imgs_pairs = get_imagepair_index(data_path)
    
    # Get the image/depth path
    image_name_list, depth_name_list,intr_list,pose_list = readSceneInfo(data_path) 
    
    # write data into npz file
    assert len(image_name_list) == len(depth_name_list) == len(intr_list) == len(pose_list), \
        "image/depth/intrinsics/poses must have the same number of elements"
        
    pair_infos = np.array(imgs_pairs, dtype=object)
    image_paths = np.array(image_name_list, dtype=object)
    depth_paths = np.array(depth_name_list, dtype=object)
    
    intrinsics = np.array(intr_list)
    poses = np.array(pose_list)
    
    np.savez(
        save_path,
        pair_infos=pair_infos,
        image_paths=image_paths,
        depth_paths=depth_paths,
        intrinsics=intrinsics,
        poses=poses
    )
    



if __name__ == "__main__":
    data_path = "../../datasets/head"
    save_path = "./head_7scene.npz"
    create_npz(data_path, save_path)    





