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

import torch.nn.functional as F

import matplotlib.pyplot as plt



import torchvision.utils as vutils



class InterpolateSparse2d(nn.Module):
    """ Efficiently interpolate tensor at given sparse 2D positions. """ 
    def __init__(self, mode = 'bicubic', align_corners = False): 
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(self, x, H, W):  # 把像素坐标转换到 [-1, 1] 区间
        """ Normalize coords to [-1,1]. """
        return 2. * (x/(torch.tensor([W-1, H-1], device = x.device, dtype = x.dtype))) - 1.

    def forward(self, x, pos, H, W):
        """
        Input
            x: [B, C, H, W] feature tensor
            pos: [B, N, 2] tensor of positions
            H, W: int, original resolution of input 2d positions -- used in normalization [-1,1]

        Returns
            [B, N, C] sampled channels at 2d positions
        """
        grid = self.normgrid(pos, H, W).unsqueeze(-2).to(x.dtype)
        x = F.grid_sample(x, grid, mode = self.mode , align_corners = False)
        return x.permute(0,2,3,1).squeeze(-2)


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
        
        self.interpolator = InterpolateSparse2d('bicubic')
        
        
        self.top_k = 4096
        

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
    
    def NMS(self, x, threshold = 0.05, kernel_size = 5):
        B, _, H, W = x.shape
        pad=kernel_size//2
        local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)
     
        pos = (x == local_max) & (x > threshold)
        
        pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]

        pad_val = max([len(x) for x in pos_batched])
        pos = torch.zeros((B, pad_val, 2), dtype=torch.long, device=x.device)
        
        #Pad kpts and build (B, N, 2) tensor
        for b in range(len(pos_batched)):
            pos[b, :len(pos_batched[b]), :] = pos_batched[b]
        
        return pos
    
    def NMS_min(self, x, threshold=0.05, kernel_size=5):
        B, _, H, W = x.shape
        pad = kernel_size // 2

        # 用负号实现 MinPool
        local_min = -nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=1,
            padding=pad
        )(-x)

        # 极小值条件
        pos_mask = (x == local_min) & (x < threshold)

        pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos_mask]

        pad_val = max([len(k) for k in pos_batched]) if len(pos_batched) > 0 else 0

        pos = torch.zeros((B, pad_val, 2), dtype=torch.long, device=x.device)

        for b in range(len(pos_batched)):
            pos[b, :len(pos_batched[b]), :] = pos_batched[b]

        return pos
    
    
    
    def match(self, feats1, feats2, min_cossim = 0.82):
        cossim = feats1 @ feats2.t()
        cossim_t = feats2 @ feats1.t()
        
        _, match12 = cossim.max(dim=1)
        _, match21 = cossim_t.max(dim=1)
        
        idx0 = torch.arange(len(match12), device=match12.device)
        mutual = match21[match12] == idx0
        
        if min_cossim > 0:
            cossim, _ = cossim.max(dim=1)
            good = cossim > min_cossim
            idx0 = idx0[mutual & good]
            idx1 = match12[mutual & good]
        else:
            idx0 = idx0[mutual]
            idx1 = match12[mutual]
        
        return idx0, idx1
    
    def detectAndCompute(self, x, q_feat_list, top_k = None):
        """
			Compute sparse keypoints & descriptors. Supports batched mode.

			input:
				x -> torch.Tensor(B, C, H, W): grayscale or rgb image
				top_k -> int: keep best k features
			return:
				List[Dict]: 
					'keypoints'    ->   torch.Tensor(N, 2): keypoints (x,y)
					'scores'       ->   torch.Tensor(N,): keypoint scores
					'descriptors'  ->   torch.Tensor(N, 64): local features
     """
        if top_k is None: top_k = self.top_k
        # 如果输入是 NHWC，转成 NCHW
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        
        B, _, _H1, _W1 = x.shape
        
        description_map, invariance_map = self.kpnet(x,q_feat_list)
        

       
        
        inv_map_orisize =  F.interpolate(
                    invariance_map,
                    scale_factor=8,
                    mode='bilinear',      # 推荐 heatmap 用 bilinear
                    align_corners=False
                )
        
        description_map = F.normalize(description_map, dim=1)
        
        heat_min = inv_map_orisize.amin(dim=(2, 3), keepdim=True)
        heat_max = inv_map_orisize.amax(dim=(2, 3), keepdim=True)
        heat = (inv_map_orisize - heat_min) / (heat_max - heat_min + 1e-6)
        
        print("heat min: ", heat.min().item())
        print("heat max: ", heat.max().item())
        
        #heat = inv_map_orisize.squeeze().detach().cpu().numpy()
        #heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
        
        
        plt.figure(figsize=(6, 6))
        plt.imshow(heat.squeeze().detach().cpu().numpy(), cmap='jet')
        plt.colorbar()
        plt.axis('off')
        plt.show()

		#Convert logits to heatmap and extract kpts
        
        #mkpts = self.NMS(inv_map_orisize, threshold = 0.99, kernel_size=5)
        
        mkpts = self.NMS_min(heat, threshold = 0.3, kernel_size=5)

        
        


        # ===== 1️⃣ 处理图像 =====
        # image: torch.Size([480, 640, 3])
        img = x[0].permute(1,2,0).detach().cpu().numpy()

        


        # 如果是 float 且范围 0~1，需要转成 uint8
        if img.dtype != np.uint8:
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            img = (img * 255).astype(np.uint8)

        # OpenCV 用 BGR，如果你原图是 RGB：
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # ===== 2️⃣ 处理关键点 =====
        # keypoints: [1, N, 2]
        kpts = mkpts[0].detach().cpu().numpy().astype(np.int32)
        img_draw = img.copy()


        
        _nearest = InterpolateSparse2d('nearest')
        
        scores = _nearest(inv_map_orisize, mkpts, _H1, _W1).squeeze(-1)
        scores[torch.all(mkpts == 0, dim=-1)] = -1
                
        #Select top-k features
        idxs = torch.argsort(-scores)
        mkpts_x  = torch.gather(mkpts[...,0], -1, idxs)[:, :self.top_k]
        mkpts_y  = torch.gather(mkpts[...,1], -1, idxs)[:, :self.top_k]
        
        mkpts = torch.cat([mkpts_x[...,None], mkpts_y[...,None]], dim=-1)
        scores = torch.gather(scores, -1, idxs)[:, :self.top_k]
        
        ########################
        
        mkpts_x = mkpts_x[0].detach().cpu().numpy().astype(np.int32)
        mkpts_y = mkpts_y[0].detach().cpu().numpy().astype(np.int32)

        H, W = img_draw.shape[:2]

        mask = (
            (mkpts_x >= 0) & (mkpts_x < W) &
            (mkpts_y >= 0) & (mkpts_y < H)
        )

        mkpts_x = mkpts_x[mask]
        mkpts_y = mkpts_y[mask]

        #img_draw[mkpts_y, mkpts_x] = (0,255,0)
        for x_pt, y_pt in zip(mkpts_x, mkpts_y):
            cv2.circle(img_draw, (x_pt, y_pt), radius=3, color=(0,255,0), thickness=-1)

        cv2.imshow("kpts", img_draw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
        
        
		#Interpolate descriptors at kpts positions
        feats = self.interpolator(description_map, mkpts, H = _H1, W = _W1)
        
		#L2-Normalize
        feats = F.normalize(feats, dim=-1)
        
        valid = scores > 0
        
                
        return [  
				   {'keypoints': mkpts[b][valid[b]],
					'scores': scores[b][valid[b]],
					'descriptors': feats[b][valid[b]]} for b in range(B) 
			   ]
            
        
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
            
            A = q_img.squeeze(0).squeeze(0).unsqueeze(0)
            all_imgs = torch.cat([A, torch.from_numpy(s_img).cuda()], dim=0)

            grid = vutils.make_grid(all_imgs, nrow=3, normalize=True)

            plt.imshow(grid.permute(1,2,0).cpu())
            plt.axis("off")
            plt.show() 
            
            
            
            
            s_img =  torch.from_numpy(np.expand_dims(s_img, axis=(0))).cuda()
            s_K =  torch.from_numpy(np.expand_dims(s_K, axis=(0))).cuda()
            s_T =  torch.from_numpy(np.expand_dims(s_T, axis=(0))).cuda()
            s_img_ori =  torch.from_numpy(np.expand_dims(s_img_ori, axis=(0))).cuda()
            s_depth =  torch.from_numpy(np.expand_dims(s_depth, axis=(0))).cuda()

            self.sdr_model.eval()
            self.kpnet.eval()
            
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
                ref_T,
                ref_K,
                s_img,
                s_depth,
                s_T,
                s_K,
                s_T[:,0, :, :],
                )
                
                
                #description_map0, invariance_map0 = self.kpnet(q_img_ori,q_feat_list0)
                #description_map1, invariance_map1 = self.kpnet(s_img_ori[:,1,:,:],q_feat_list1)
                out0 = self.detectAndCompute(q_img_ori,q_feat_list0)[0]
                
                #print("out0 =", out0)
                
                
                #out1 = self.detectAndCompute(s_img_ori[:,1,:,:],q_feat_list1)[0]
                

                
                #idxs0, idxs1 = self.match(out0['descriptors'], out1['descriptors'], min_cossim=-1)
                
                #print("idxs 0 = ", idxs0)
                
                
                

                
                

                
            
            

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
    visualer = Visualer(cfg, "trained_model/sdr_kpdetect_2000.pth")
    visualer.run_inf()

