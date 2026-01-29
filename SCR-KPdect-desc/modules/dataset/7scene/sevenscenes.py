import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from scene_utils import crop_by_intrinsic
from reader import load_one_img

class BaseDataset(Dataset):
    def __init__(self,
                 cfg,
                 transform,
                 mode='train',
                 augment_fn=None,
                 **kwargs):
        """    
        """
        super().__init__()
        self.ref_topk = cfg.ref_topk
        self.root_dir = cfg.root_dir
        self.pad_image = cfg.pad_image
        self.mode = mode
        self.img_K = None
        self.depth_K = None
        self.load_depth = cfg.load_depth
        # prepare scene_info and pair_info
        self.scene_info = np.load(cfg.npz_path, allow_pickle=True)
        self.pair_infos = self.scene_info['pair_infos'].copy()
        

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None
        self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125)
        
        self.crop_img_func = None
        self.crop_depth_func = None
        
        self.transform = transform
        

    def __len__(self):
        return len(self.pair_infos)
    
    def load_query(self, idx, base_dir):
        
        pose_tensor= []
        K_tensor=[]
        depth_tensor=[]
        img_tensor=[]
        img, depth, pose, K = load_one_img(base_dir, self.scene_info, idx, read_img=True)
        
        if self.crop_img_func is not None:  # 如果要裁剪img，同时更新相机内参
            img, K = self.crop_img_func(img, K)
        if self.crop_depth_func is not None: #如果要裁剪深度图，同时更新相机内参
            depth, K = self.crop_depth_func(depth, K)
        
        img, depth, pose, K = self.transform(img, depth, pose, K)
        
        pose_tensor.append(pose)
        K_tensor.append(K)
        depth_tensor.append(depth)
        img_tensor.append(img)
        
        pose_tensor = np.stack(pose_tensor).astype(np.float32)  # 将list 变成 nadrray
        K_tensor = np.stack(K_tensor).astype(np.float32)
        depth_tensor = np.stack(depth_tensor).astype(np.float32)
        
        result = {
            "pose": pose_tensor,
            "K": K_tensor,
            "depth": depth_tensor,
            "img": np.stack(img_tensor).astype(np.float32).transpose(0, 3, 1, 2),
        }
        return result
        
    def load_references(self, idx, base_dir):
        
        pose_tensor= []
        K_tensor=[]
        depth_tensor=[]
        img_tensor=[]
        idx = int(idx)
        # Get the reference index for "idx"th data
        ref_idxs = self.scene_info["ref_infos"][idx]
        
        # For each idx
        for idx in ref_idxs:
            idx=int(idx)
            img, depth, pose, K = load_one_img(base_dir, self.scene_info, idx, read_img=True)
        
            if self.crop_img_func is not None:  # 如果要裁剪img，同时更新相机内参
                img, K = self.crop_img_func(img, K)
            if self.crop_depth_func is not None: #如果要裁剪深度图，同时更新相机内参
                depth, K = self.crop_depth_func(depth, K)
            
            img, depth, pose, K = self.transform(img, depth, pose, K)
            
            pose_tensor.append(pose)
            K_tensor.append(K)
            depth_tensor.append(depth)
            img_tensor.append(img)

            if len(pose_tensor) == self.ref_topk:
                break
        if self.pad_image and len(pose_tensor) < self.ref_topk:
            pose_tensor = pose_tensor + [pose_tensor[0]] * (
                self.ref_topk - len(pose_tensor)
            )
            K_tensor = K_tensor + [K_tensor[0]] * (self.ref_topk - len(K_tensor))
            depth_tensor = depth_tensor + [depth_tensor[0]] * (
                self.ref_topk - len(depth_tensor)
            )
            img_tensor = img_tensor + [img_tensor[0]] * (
                self.ref_topk - len(img_tensor)
            )
        
        pose_tensor = np.stack(pose_tensor).astype(np.float32)  # 将list 变成 nadrray
        K_tensor = np.stack(K_tensor).astype(np.float32)
        depth_tensor = np.stack(depth_tensor).astype(np.float32)

        result = {
            "pose": pose_tensor,
            "K": K_tensor,
            "depth": depth_tensor,
            "img": np.stack(img_tensor).astype(np.float32).transpose(0, 3, 1, 2),  
        } 
        
        return result
        
            
        

    def __getitem__(self, idx):
        (idx0, idx1) = self.pair_infos[idx % len(self)]
        idx, idx0, idx1 = int(idx), int(idx0), int(idx1)
        # Get the pair training 
        base_dir = osp.join(self.root_dir, "datasets/head/images") 

        res_image0 = self.load_query(idx0, base_dir)
        res_image1 = self.load_query(idx1, base_dir)
        # -------- intrinsics (numpy) --------
        K_0 = np.array(res_image0["K"][0], dtype=np.float32)  # (3, 3)
        K_1 = np.array(res_image1["K"][0], dtype=np.float32)  # (3, 3)

        # -------- poses (numpy) --------
        T0 = np.array(res_image0['pose'][0], dtype=np.float32)
        T1 = np.array(res_image1['pose'][0], dtype=np.float32)

        # T_0to1 = T1 * inv(T0)
        T_0to1 = np.matmul(T1, np.linalg.inv(T0)).astype(np.float32)[:4, :4]  # (4, 4)

        # T_1to0 = inverse(T_0to1)
        T_1to0 = np.linalg.inv(T_0to1).astype(np.float32)  # (4, 4)

        pair_data = {
            'image0': np.array(res_image0["img"][0], dtype=np.float32),    # (C, H, W)
            'depth0': np.array(res_image0["depth"][0], dtype=np.float32),  # (H, W)
            'image1': np.array(res_image1["img"][0], dtype=np.float32),
            'depth1': np.array(res_image1["depth"][0], dtype=np.float32),

            'T_0to1': T_0to1,   # (4, 4)
            'T_1to0': T_1to0,   # (4, 4)

            'K0': K_0,         # (3, 3)
            'K1': K_1,         # (3, 3)

            'dataset_name': '7 scene',
            'pair_id': idx,
            'pair_names': (
            self.scene_info['image_paths'][idx0],
            self.scene_info['image_paths'][idx1]
            ),
        }
        
        # Get the reference images
        res_ref = self.load_references(idx,base_dir)
        
        return pair_data, res_ref
        
        
        






class SevenSceneDataset(BaseDataset):
    def __init__(self,
                 cfg,
                 transform,
                 mode='train',
                 augment_fn=None):
        super().__init__(cfg, transform, mode, augment_fn)

        self.depth_K = np.asarray(
            [[585, 0, 320], [0, 585, 240], [0, 0, 1]], dtype=np.float32
        )
        self.img_K = np.asarray(
            [[525, 0, 320], [0, 525, 240], [0, 0, 1]], dtype=np.float32
        )
        self.crop_img_func = self.crop_img  # 7 scene 的样本是480x640 而网络的输出固定为192x256 所以需要resize ，但是
                                            # 单纯resize 会破坏几何机构，所以需要crop 操作

    def crop_img(self, img, K=None, no_none=None):
        return crop_by_intrinsic(img, self.img_K, self.depth_K), self.depth_K