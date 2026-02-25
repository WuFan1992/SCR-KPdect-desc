import os
import sys
import numpy as np
import random
import torch
from modules.dataset.sevenscene.colmap_reader import read_points3D_binary, read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from itertools import combinations
from modules.utils.scene_utils import focal2fov, getIntrinsic, getExtrinsic

"""
At the project root path:
    python -m modules.dataset.sevenscene.create_npz

"""

def select_topk_poses(T_query, T_list, K=5, rot_weight=0.5):
    """
    T_query: [4,4] numpy array
    T_list: list of [4,4] numpy arrays
    K: number of closest poses to select
    rot_weight: weight for rotation distance
    
    return: indices of top-K closest poses in T_list
    """
    t_dist_list = []
    r_dist_list = []

    R_q = T_query[:3, :3]
    t_q = T_query[:3, 3]

    # 遍历 list
    for T in T_list:
        R_i = T[:3, :3]
        t_i = T[:3, 3]

        # 平移距离
        t_dist = np.linalg.norm(t_i - t_q)
        t_dist_list.append(t_dist)

        # 旋转距离
        R_rel = R_i @ R_q.T
        trace = np.trace(R_rel)
        cos_theta = np.clip((trace - 1) / 2, -1.0, 1.0)
        r_dist = np.arccos(cos_theta)
        r_dist_list.append(r_dist)

    t_dist_arr = np.array(t_dist_list)
    r_dist_arr = np.array(r_dist_list)
    dist = t_dist_arr + rot_weight * r_dist_arr
    
    N = len(dist)

    if N == 0:
        return []

    K = min(K, N)

    # 选 top-K
    topk_indices = np.argpartition(dist, K-1)[:K]
    topk_indices = topk_indices[np.argsort(dist[topk_indices])]

    return topk_indices

def build_topk_poses_from_train(i, j, pose_list, train_img_index, K=5, rot_weight=0.5, mode='train'):
    """
    i, j: indices in pose_list
    pose_list: list of [4,4] numpy arrays
    train_img_index: list of indices in pose_list to consider for top-K
    K: number of closest poses to select
    rot_weight: weight for rotation distance
    mode: 'train' or 'test'

    return:
        T_query: pose_list[i]
        topk_indices: list of indices in pose_list corresponding to top-K closest poses
    """
    # 1️⃣ 查询 pose
    T_query = pose_list[i]

    # 2️⃣ 构建候选 pose list
    if mode == 'train':
        # 排除 i 和 j
        filtered_indices = [idx for idx in train_img_index if idx not in (i, j)]
    elif mode == 'test':
        # 不排除 i 和 j
        filtered_indices = list(train_img_index)
    else:
        raise ValueError(f"Invalid mode '{mode}', must be 'train' or 'test'")

    filtered_pose_list = [pose_list[int(idx)] for idx in filtered_indices]

    # 3️⃣ 查找 top-K
    topk_in_filtered = select_topk_poses(T_query, filtered_pose_list, K=K, rot_weight=rot_weight)

    # 4️⃣ 映射回 pose_list 的原索引
    topk_indices = [filtered_indices[int(idx)] for idx in topk_in_filtered]

    return T_query, topk_indices

def build_train_samples(data_path, train_idx_list, pose_list, topK):
    
    point3d_path = os.path.join(data_path, "sparse/0/points3D.bin")
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
        
        pair = (int(imgs_in_train_idx_shuffle[0]), int(imgs_in_train_idx_shuffle[1]))
        
        
        pair0, pair0_topK = build_topk_poses_from_train(pair[0], pair[1], pose_list, imgs_in_train_idx, topK )
        pair1, pair1_topK = build_topk_poses_from_train(pair[1], pair[0], pose_list, imgs_in_train_idx, topK)
        
        pair_topK = (pair0_topK, pair1_topK)
        pair_list.append(pair)
        ref_list.append(pair_topK)
        
    return pair_list, ref_list   #  ref_list = [([2 , 7, 18 ], [13,15,18]), ([xxx], [xxxx]),.....]

"""
def build_test_samples(data_path, train_idx_list, pose_list, topK):
    
    point3d_path = os.path.join(data_path, "sparse/0/points3D.bin")
    _, _, _, _, img_ids, _  = read_points3D_binary(point3d_path)
    
    query_list = []
    ref_list = []
    train_idx_list = set(train_idx_list)
    
    for p_id in img_ids:
        imgs = img_ids[p_id]   # image ids observing this 3D point
        imgs_in_train_idx = [x for x in imgs if x in train_idx_list]
        imgs_in_test_idx = [y for y in imgs if y not in train_idx_list]
        
        
        # check the data availability
        if len(imgs_in_train_idx) < topK + 2:      # a pair + topk references
            continue
        
        if len(imgs_in_test_idx) == 0:
            continue
        
        for test_img_idx in imgs_in_test_idx:
            query, q_topK = build_topk_poses_from_train(int(test_img_idx), int(test_img_idx), pose_list,imgs_in_train_idx, topK, mode='test')
            
        
        query_list.append(query)
        ref_list.append(q_topK)
        
    return query_list, ref_list  #[[2,13,18], [12,14,15], ....]  
"""
def build_test_samples(data_path, train_idx_list, pose_list, topK):

    point3d_path = os.path.join(data_path, "sparse/0/points3D.bin")
    _, _, _, _, img_ids, _  = read_points3D_binary(point3d_path)

    train_idx_list = set(train_idx_list)

    # 1️⃣ 收集所有 test 图像（去重）
    all_test_imgs = set()

    for p_id in img_ids:
        imgs = img_ids[p_id]
        for img in imgs:
            if img not in train_idx_list:
                all_test_imgs.add(img)

    query_list = []
    ref_list = []

    # 2️⃣ 只对每个 test 图像算一次
    for test_img_idx in all_test_imgs:

        # 找与它共享3D点的train图像
        related_train = set()

        for p_id in img_ids:
            imgs = img_ids[p_id]
            if test_img_idx in imgs:
                for img in imgs:
                    if img in train_idx_list:
                        related_train.add(img)

        if len(related_train) < topK:
            continue

        query, q_topK = build_topk_poses_from_train(
            int(test_img_idx),
            int(test_img_idx),
            pose_list,
            list(related_train),
            K=topK,
            mode='test'
        )

        query_list.append(query)
        ref_list.append(q_topK)

    return query_list, ref_list
      

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

"""
Construct one (query image + k*reference images)
 
"""
def construct_query_ref(path, train_idx_list, topK: int):
    
    point3d_path = os.path.join(path, "sparse/0/points3D.bin")
    _, _, _, _, img_ids, _  = read_points3D_binary(point3d_path)
    
    query_list = []
    ref_list = []
    train_idx_list = set(train_idx_list)
    
    for p_id in img_ids:
        imgs = img_ids[p_id]   # image ids observing this 3D point
        imgs_in_train_idx = [x for x in imgs if x in train_idx_list]
        imgs_in_test_idx = [y for y in imgs if y not in train_idx_list]
        
        
        # check the data availability
        if len(imgs_in_train_idx) < topK + 2:      # a pair + topk references
            continue
        
        if len(imgs_in_test_idx) == 0:
            continue
        
        for test_img_idx in imgs_in_test_idx:
            query = test_img_idx
            #For each query, take the whole reference images in the training data
            rest = imgs_in_train_idx
            random.shuffle(rest)
            reference = random.sample(rest, topK)
            query_list.append(query)
            ref_list.append(reference)
    
    # Keep the unique index in query (each query image participate only once the test) work for all pytorch version 
    seen = set()
    unique_indices = []

    for i, q in enumerate(query_list):
        if q not in seen:
            seen.add(q)
            unique_indices.append(i)

    query_list = [query_list[i] for i in unique_indices]
    ref_list   = [ref_list[i] for i in unique_indices]

    return query_list, ref_list


      
    
    

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
    # Training pair + references
    #train_pair_list, train_ref_list = construct_pairs_ref(data_path, res["train_idx_list"], topK=5) 
    train_pair_list, train_ref_list = build_train_samples(data_path, res["train_idx_list"], res["pose_list"], topK=5) 
    # Testing query + reference
    query_list, query_ref_list = build_test_samples(data_path, res["train_idx_list"], res["pose_list"],  topK=5)
    
    # write data into npz file
    assert len(res["image_name_list"]) == len(res["depth_name_list"]) == len(res["intrinsics_list"]) == len(res["pose_list"]), \
        "image/depth/intrinsics/poses must have the same number of elements"
        
    train_pair_infos = np.array(train_pair_list, dtype=object)
    train_ref_infos =  np.array(train_ref_list, dtype=object)
    query_infos = np.array(query_list, dtype=object)
    query_ref_infos =  np.array(query_ref_list, dtype=object)
    image_paths = np.array(res["image_name_list"], dtype=object)
    depth_paths = np.array(res["depth_name_list"], dtype=object)
    
    intrinsics = np.array(res["intrinsics_list"])
    poses = np.array(res["pose_list"])

    np.savez(
        save_path,
        train_pair_infos=train_pair_infos,
        train_ref_infos=train_ref_infos,
        test_query_infos=query_infos,
        test_query_ref_infos=query_ref_infos,
        image_paths=image_paths,
        depth_paths=depth_paths,
        intrinsics=intrinsics,
        poses=poses
    )
    

if __name__ == "__main__":
    data_path = "datasets/head"
    save_path = "weights/head_7scene_train_test.npz"
    create_npz(data_path, save_path)    





