import torch
import torch.utils.data as data
import os.path as osp


from modules.utils.transform import *
from modules.dataset.sevenscene.sevenscenes import *
from modules.dataset.sevenscene import sevenscenes_warper
from modules.training.losses import *
from tqdm import tqdm


class Trainer(object):
    def __init__(self, cfg, model, kpnet):
        self.reproj_loss = cfg.TRAIN.reproj_loss
        self.reproj_loss_scale = cfg.TRAIN.reproj_loss_scale
        self.reproj_loss_start = cfg.TRAIN.reproj_loss_start

        transform = Compose(
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
        
        normalize = Normalize(
                    scale=cfg.MODEL.TRANSFORM.scale,
                    mean=cfg.MODEL.TRANSFORM.mean,
                    std=cfg.MODEL.TRANSFORM.std,
                )
   
        self.dataset = SevenSceneDataset(cfg.TRAIN.DATASET, transform, normalize)

        self.data_loader = data.DataLoader(
            self.dataset,
            cfg.TRAIN.batch_size,
            num_workers=0,
            shuffle=True,
            pin_memory=True,
            
        )

        self.model = model 
        self.kpnet = kpnet
        self.niter = 0
        self.data_loader_iter = iter(self.data_loader)  # 把数据变成迭代器，方便使用next 一个一个获取
        
        self.optimizer = torch.optim.Adam(list(self.kpnet.parameters()), lr=1e-3)
        self.progress_bar = tqdm(range(0, 32000), desc="Training progress")
        

    def train_iters(self, iter_num):
        
        
        if True:
            try:
                q, r = next(self.data_loader_iter)
            except StopIteration:
                self.data_loader_iter = iter(self.data_loader)
                q, r = next(self.data_loader_iter)
                
                
            positive_md_coarse = sevenscenes_warper.spvs_coarse(q,4) 
            
            #Check if batch is corrupted with too few correspondences
            """
            is_corrupted = False
            for p in positive_md_coarse:
                if len(p) < 30:
                    is_corrupted = True
                    
            if is_corrupted:
                return  # set to continue when add the n_steps 
            
            """
            # Pair images
            q0_img = q["image0"].cuda()  # (N T 3 H W)
            q0_img_ori = q["image0_ori"].cuda()
            q0_Tcw = q["T0"].cuda()
            q0_K = q["K0"].cuda()
            q0_depth = q["depth0"].cuda()
            
            q1_img = q["image1"].cuda()  # (N T 3 H W)
            q1_img_ori = q["image1_ori"].cuda()
            q1_Tcw = q["T1"].cuda()
            q1_K = q["K1"].cuda()
            q1_depth = q["depth1"].cuda()
            
            
            
           # Reference images (for coordinates estimation)
            s_img = r["img"].cuda()  # (N L 3 H W)
            s_Tcw = r["pose"].cuda()
            s_K = r["K"].cuda()
            s_depth = r["depth"].cuda()
            
            # Scene Coordinates Estimation
            losses0, metrics0, pred_coords0, gt_coords0, _, _, q_feat_list0 = self.model(
                q0_img,
                q0_depth,
                q0_Tcw,
                q0_K,
                s_img,
                s_depth,
                s_Tcw,
                s_K,
                s_Tcw[:, 0, :, :],
            )
            
            
            losses1, metrics1, pred_coords1, gt_coords1, _, _, q_feat_list1 = self.model(
                q1_img,
                q1_depth,
                q1_Tcw,
                q1_K,
                s_img,
                s_depth,
                s_Tcw,
                s_K,
                s_Tcw[:, 0, :, :],
            )
            
            # Keypoint Detection & Description
            description_map0, invariance_map0,keypoints0 = self.kpnet(q0_img_ori,q_feat_list0)
            description_map1, invariance_map1,keypoints1 = self.kpnet(q1_img_ori,q_feat_list1)
            
                
            loss_items = []
            
            for b in range(len(positive_md_coarse)):
                
                if len(positive_md_coarse[b]) < 20:
                    continue
                #Get positive correspondencies
                pts0, pts1 = positive_md_coarse[b][:, :2], positive_md_coarse[b][:, 2:]

                #Grab features at corresponding idxs   feats1 size = [B, C, Hc, Wc]
                m0 = description_map0[b, :, pts0[:,1].long(), pts0[:,0].long()].permute(1,0) # 从特征图里面采样descriptor   [M,C]
                m1 = description_map1[b, :, pts1[:,1].long(), pts1[:,0].long()].permute(1,0)

                #grab invariance map at corresponding idxs   hmap.shape = [B, 1, Hc, Wc] 
                h0 = invariance_map0[b, 0, pts0[:,1].long(), pts0[:,0].long()]
                h1 = invariance_map1[b, 0, pts1[:,1].long(), pts1[:,0].long()]
                
                #Compute losses
                loss_ds, conf = dual_softmax_loss(m0, m1)
                loss_view = weighted_distance_loss(h0,h1,q0_Tcw[b], q0_Tcw[b])
                
                
                loss = loss_ds + loss_view
                loss_items.append(loss)
                
            loss_mean = sum(loss_items) / len(loss_items)
            self.optimizer.zero_grad()
            loss_mean.backward()
            self.optimizer.step()
                
            print("loss = ", loss_mean.item())
                
                
                
                
                
                
                
                
                
                


            
            
            
            

            
            

