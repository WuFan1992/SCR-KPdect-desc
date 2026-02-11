import os
import sys
import numpy as np
import random
from colmap_reader import read_points3D_binary, read_extrinsics_binary, read_intrinsics_binary,qvec2rotmat
from itertools import combinations
from utils.scene_utils import focal2fov, getIntrinsic, getExtrinsic

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

# 从 img_id 里，根据training set 和 testing set。 在同一个set 里面构建匹配对，并且每一个匹配对都有topk 个reference image,
# 且无论是训练还是测试，所有的reference 都来自training set



def construct_pairs_ref(path, train_idx_list, topK: int):
    
    point3d_path = os.path.join(path, "sparse/0/points3D.bin")
    _, _, _, _, img_ids, _  = read_points3D_binary(point3d_path)
    
    pair_list = []
    ref_list = []
    train_idx_list = set(train_idx_list)
    
    for p_id in img_ids:
        imgs = img_ids[p_id]   # image ids observing this 3D point
        imgs_in_train_idx = [x for x in imgs if x in train_idx_list]
        
        if len(imgs_in_train_idx) < topK + 2:      # a pair + topk references
            continue
        imgs_in_train_idx_shuffle =  imgs_in_train_idx.copy()
        random.shuffle(imgs_in_train_idx_shuffle)
        
        pair = (imgs_in_train_idx_shuffle[0], imgs_in_train_idx_shuffle[1])
        
        rest =  imgs_in_train_idx_shuffle[2:]
        reference = random.sample(rest, topK)
        
        pair_list.append(pair)
        ref_list.append(reference)
        
    return   pair_list, ref_list

      
    
    

def readColmapCameras(cam_extrinsics, cam_intrinsics, test_images_name):

    max_id = max(cam_extrinsics.keys())
    image_name_list = [""] * (max_id+1)  # ex : img_id=768  image_name = 767.color.png 
    depth_name_list = [""] * (max_id+1)
    intrinsics_list = [np.zeros((3,3)) for _  in range(max_id+1)]
    pose_list = [np.zeros((4,4)) for _  in range(max_id+1)]
    
    train_idx_list = []
    test_idx_list = []

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        img_name = extr.name
        
        # If the img_name is in the test_images_name, put it into a test id list, other wise in a train id list
        if img_name in test_images_name:
            test_idx_list.append(extr.id)
        else:
            train_idx_list.append(extr.id)
        
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
    res = {
            "image_name_list": image_name_list,
            "depth_name_list": depth_name_list,
            "intrinsics_list": intrinsics_list,
            "pose_list": pose_list,
            "train_idx_list": train_idx_list,
            "test_idx_list": test_idx_list,
    }
    
    return res

def readSceneInfo(path):
    
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    
    if os.path.exists(os.path.join(path, "sparse/0", "list_test.txt")):
        
        # 7scenes
        with open(os.path.join(path, "sparse/0", "list_test.txt")) as f:
            test_images = f.readlines()
            test_images = [x.strip() for x in test_images]
    else:
        test_images = []
    

    res = readColmapCameras(cam_extrinsics, cam_intrinsics, test_images)

    return res
   

def create_npz(data_path, save_path):
    
    
    # Get the image/depth path
    res = readSceneInfo(data_path) 
    
    pair_list, ref_list = construct_pairs_ref(data_path, res["train_idx_list"], topK=5) 
    
    # write data into npz file
    assert len(res["image_name_list"]) == len(res["depth_name_list"]) == len(res["intrinsics_list"]) == len(res["pose_list"]), \
        "image/depth/intrinsics/poses must have the same number of elements"
        
    pair_infos = np.array(pair_list, dtype=object)
    ref_infos =  np.array(ref_list, dtype=object)
    print("ref infos = ", ref_infos)
    image_paths = np.array(res["image_name_list"], dtype=object)
    depth_paths = np.array(res["depth_name_list"], dtype=object)
    
    intrinsics = np.array(res["intrinsics_list"])
    poses = np.array(res["pose_list"])

    np.savez(
        save_path,
        pair_infos=pair_infos,
        ref_infos=ref_infos,
        image_paths=image_paths,
        depth_paths=depth_paths,
        intrinsics=intrinsics,
        poses=poses
    )
    

if __name__ == "__main__":
    data_path = "../../../datasets/head"
    save_path = "./head_7scene.npz"
    create_npz(data_path, save_path)    





