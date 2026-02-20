import torch
import cv2
import numpy as np
import yaml
import os.path as osp

from modules.arch.DSMNet import dsm_net
from modules.arch.KPNet import KPNet
from modules.utils.reader import load_one_img
from modules.utils.transform import *
from modules.utils.scene_utils import *



class Visualer():
    def __init__(self, cfg, kpnet_model_path):
        # Load SDR network
        self.sdr_model = dsm_net(cfg.MODEL).cuda()
    
        # Load the keypoint detection net
        self.kpnet = KPNet().cuda()
        self.kpnet.load_state_dict(torch.load(kpnet_model_path, weights_only=True))
        
        # Load npz file
        self.scene_info = np.load(cfg.TRAIN.DATASET.npz_path, allow_pickle=True)
        
        self.root_dir = cfg.TRAIN.DATASET.root_dir
        
        self.transform = Compose(
            [
                Resize(
                    (
                        cfg.MODEL.TRANSFORM.train_resize_h,
                        cfg.MODEL.TRANSFORM.train_resize_w,
                    )
                ),
                Normalize(
                    scale=cfg.MODEL.TRANSFORM.scale,
                    mean=cfg.MODEL.TRANSFORM.mean,
                    std=cfg.MODEL.TRANSFORM.std,
                ),
            ]
        )
        
        self.normalize = Normalize(
                    scale=cfg.MODEL.TRANSFORM.scale,
                    mean=cfg.MODEL.TRANSFORM.mean,
                    std=cfg.MODEL.TRANSFORM.std,
                )
        
        self.depth_K = np.asarray(
            [[585, 0, 320], [0, 585, 240], [0, 0, 1]], dtype=np.float32
        )
        self.img_K = np.asarray(
            [[525, 0, 320], [0, 525, 240], [0, 0, 1]], dtype=np.float32
        )
        self.crop_img_func = self.crop_img  # 7 scene 的样本是480x640 而网络的输出固定为192x256 所以需要resize ，但是
                                            # 单纯resize 会破坏几何机构，所以需要crop 操作
                                            
        self.crop_depth_func = None
        self.ref_topk = cfg.TRAIN.DATASET.ref_topk
        self.pad_image = cfg.TRAIN.DATASET.pad_image

    def crop_img(self, img, K=None, no_none=None):
        return crop_by_intrinsic(img, self.img_K, self.depth_K), self.depth_K
        

        
        
    def load_query(self, idx, base_dir):
        
        pose_tensor= []
        K_tensor=[]
        depth_tensor=[]
        img_tensor=[]
        img_ori_tensor=[]
        img, depth, pose, K = load_one_img(base_dir, self.scene_info, idx, read_img=True)
        
        if self.crop_img_func is not None:  # 如果要裁剪img，同时更新相机内参
            img, K = self.crop_img_func(img, K)
        if self.crop_depth_func is not None: #如果要裁剪深度图，同时更新相机内参
            depth, K = self.crop_depth_func(depth, K)
        
        
        img_ori, _, _, _ = self.normalize(img, depth, pose, K)
        img, depth, pose, K = self.transform(img, depth, pose, K)
        
        
        pose_tensor.append(pose)
        K_tensor.append(K)
        depth_tensor.append(depth)
        img_tensor.append(img)
        img_ori_tensor.append(img_ori)
        
        pose_tensor = np.stack(pose_tensor).astype(np.float32)  # 将list 变成 nadrray
        K_tensor = np.stack(K_tensor).astype(np.float32)
        depth_tensor = np.stack(depth_tensor).astype(np.float32)
        
        img_array = np.stack(img_tensor).astype(np.float32)        # [B, H, W, C]
        img_array = img_array.transpose(0, 3, 1, 2)                # [B, C, H, W]
        
        img_ori_array = np.stack(img_ori_tensor).astype(np.float32)        # [B, H, W, C]
        img_ori_array = img_ori_array.transpose(0, 3, 1, 2)                # [B, C, H, W]
        
        pose_tensor = np.expand_dims(pose_tensor, axis=1)
        K_tensor = np.expand_dims(K_tensor, axis=1)
        depth_tensor = np.expand_dims(depth_tensor, axis=1)
        img_array = np.expand_dims(img_array, axis=1)
        img_ori_array = np.expand_dims(img_ori_array, axis=1)
 
       
        result = {
            "pose": pose_tensor,
            "K": K_tensor,
            "depth": depth_tensor,
            "img": img_array,
            "img_ori": img_ori_tensor
        }
        return result
    
    def load_references(self, idx, base_dir):
        
        pose_tensor= []
        K_tensor=[]
        depth_tensor=[]
        img_tensor=[]
        img_ori_tensor=[]
        idx = int(idx)
        # Get the reference index for "idx"th data
        ref_idxs = self.scene_info["test_query_ref_infos"][idx]

        
        # For each idx
        for idx in ref_idxs:
            idx=int(idx)
            img, depth, pose, K = load_one_img(base_dir, self.scene_info, idx, read_img=True)

        
            if self.crop_img_func is not None:  # 如果要裁剪img，同时更新相机内参
                img, K = self.crop_img_func(img, K)
            if self.crop_depth_func is not None: #如果要裁剪深度图，同时更新相机内参
                depth, K = self.crop_depth_func(depth, K)
            
            img_ori, _, _, _ = self.normalize(img, depth, pose, K)
            
            img, depth, pose, K = self.transform(img, depth, pose, K)
            

            
            pose_tensor.append(pose)
            K_tensor.append(K)
            depth_tensor.append(depth)
            img_tensor.append(img)
            img_ori_tensor.append(img_ori)

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
        img_ori_array = np.stack(img_ori_tensor).astype(np.float32) 
        

        result = {
            "pose": pose_tensor,
            "K": K_tensor,
            "depth": depth_tensor,
            "img": np.stack(img_tensor).astype(np.float32).transpose(0, 3, 1, 2), 
            "img_ori": np.stack(img_ori_array).astype(np.float32).transpose(0, 3, 1, 2), 
        } 
        
        
        
        return result
        
            
        
    def run_inf(self):
        # get the test image idx
        test_query_infos = self.scene_info['test_query_infos'].copy()
        base_dir = osp.join(self.root_dir, "datasets/head/images") 
        
        for test_idx in test_query_infos:
            
            test_idx = int(test_idx)
            # Load the query 
            query_img =  self.load_query(test_idx, base_dir)
            
            q_img =  np.array(query_img["img"][0], dtype=np.float32)    # (C, H, W)
            q_K = np.array(query_img["K"][0], dtype=np.float32)  # (3, 3)
            q_T = np.array(query_img['pose'][0], dtype=np.float32)
            q_img_ori = np.array(query_img["img_ori"][0], dtype=np.float32)    # (C, H, W)
            depth =  np.array(query_img["depth"][0], dtype=np.float32)  # (H, W)
            

            
            q_img =  torch.from_numpy(np.expand_dims(q_img, axis=0)).cuda()
            q_K =  torch.from_numpy(np.expand_dims(q_K, axis=0)).cuda()
            q_T =  torch.from_numpy(np.expand_dims(q_T, axis=0)).cuda()
            q_img_ori =  torch.from_numpy(np.expand_dims(q_img_ori, axis=0)).cuda()
            depth =  torch.from_numpy(np.expand_dims(depth, axis=0)).cuda()
            
            
            
            # Load reference
            ref_imgs = self.load_references(test_idx, base_dir)
            s_img = ref_imgs["img"]  # (N L 3 H W)
            s_T = ref_imgs["pose"]
            s_K = ref_imgs["K"]
            s_depth = ref_imgs["depth"]
            s_img_ori = ref_imgs["img_ori"]
            
            s_img =  torch.from_numpy(np.expand_dims(s_img, axis=(0))).cuda()
            s_K =  torch.from_numpy(np.expand_dims(s_K, axis=(0))).cuda()
            s_T =  torch.from_numpy(np.expand_dims(s_T, axis=(0))).cuda()
            s_img_ori =  torch.from_numpy(np.expand_dims(s_img_ori, axis=(0))).cuda()
            s_depth =  torch.from_numpy(np.expand_dims(s_depth, axis=(0))).cuda()


            
            with torch.no_grad():
                _, _, _, _, _, _, q_feat_list0 = self.sdr_model(
                q_img,
                depth,
                q_T,
                q_K,
                s_img,
                s_depth,
                s_T,
                s_K,
                s_T[:,0, :, :],
                )
                
                # Prepare the display 
                ref_img = s_img[:,1:2 , : , : ]
                ref_depth = s_depth[: , 1:2 , : , : ]
                ref_T = s_T[:, 1:2 , : , : ]
                ref_K = s_K[:, 1:2 , : , : ]
                
                
                
                _, _, _, _, _, _, q_feat_list1 = self.sdr_model(
                ref_img,
                ref_depth,
                q_T,
                q_K,
                s_img,
                s_depth,
                s_T,
                s_K,
                s_T[:,0, :, :],
                )
                
                
                description_map0, invariance_map0,_ = self.kpnet(q_img_ori,q_feat_list0)
                description_map1, invariance_map1,_ = self.kpnet(s_img_ori[:,1,:,:],q_feat_list1)
                
                print("description_map 0 = ", description_map0)
                
            
            

def draw_matches(img1, img2, pts1, pts2, line_color=(0, 255, 0)):
    
    """
    img1, img2: OpenCV 读取的 BGR 图像 (numpy array)
    pts1, pts2: list 或 torch tensor，形状为 (N, 2)
                每个元素为 (x, y)
    """

    # 如果是 torch tensor，转为 numpy
    if isinstance(pts1, torch.Tensor):
        pts1 = pts1.cpu().numpy()
    if isinstance(pts2, torch.Tensor):
        pts2 = pts2.cpu().numpy()

    pts1 = np.array(pts1).astype(int)
    pts2 = np.array(pts2).astype(int)

    # 获取图像尺寸
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 创建拼接画布
    height = max(h1, h2)
    width = w1 + w2
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # 放置两张图
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    # 画匹配点和连线
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        # 第二张图的点需要平移
        x2_shifted = x2 + w1

        # 画点
        cv2.circle(canvas, (x1, y1), 4, (0, 0, 255), -1)
        cv2.circle(canvas, (x2_shifted, y2), 4, (0, 0, 255), -1)

        # 画线
        cv2.line(canvas, (x1, y1), (x2_shifted, y2), line_color, 1)

    return canvas




class Dict2Obj(dict):
    def __getattr__(self, name):
        value = self.get(name)
        if isinstance(value, dict):
            value = Dict2Obj(value)
            self[name] = value
        return value

def load_config(yaml_path: str):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)
    return Dict2Obj(cfg_dict)

if __name__ == "__main__":
    yaml_path = "modules/dataset/sevenscene/7scenes.yaml"
    cfg = load_config(yaml_path)
    visualer = Visualer(cfg, "trained_model/first_training_2000.pth")
    visualer.run_inf()

