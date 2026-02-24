import torch
import torch.utils.data as data
import os
from torch.utils.tensorboard import SummaryWriter

from modules.utils.transform import *
from modules.dataset.sevenscene.sevenscenes import *
from modules.dataset.sevenscene import sevenscenes_warper
from modules.training.losses import *
from tqdm import tqdm
import time

import matplotlib.pyplot as plt
def save_overlay(image, heatmap, save_path):
    """
    image: (1,H,W) or (3,H,W)
    heatmap: (1,H,W)
    """
    img = image.detach().cpu().permute(1,2,0).numpy()
    heat = heatmap.squeeze().detach().cpu().numpy()

    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)

    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap='gray')
    plt.imshow(heat, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.savefig(save_path, dpi=150)
    plt.close()


class Trainer(object):
    def __init__(self, cfg, model, kpnet, cpkt_save_path, save_ckpt_every = 1000):
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
        self.progress_bar = tqdm(range(0, cfg.TRAIN.model_save_iters), desc="Training progress")
        
        self.writer = SummaryWriter(cpkt_save_path + f'/logdir/scr_kpdect_' + time.strftime("%Y_%m_%d-%H_%M_%S"))
        
        self.save_ckpt_every = save_ckpt_every
        os.makedirs(cpkt_save_path, exist_ok=True)
        os.makedirs(cpkt_save_path + '/logdir', exist_ok=True)
        
        self.cpkt_save_path = cpkt_save_path
        

    def train_iters(self, iter_num):
        
        for i in range(iter_num):
            try:
                q, r = next(self.data_loader_iter)
            except StopIteration:
                # If StopIteration is raised, create a new iterator.
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
            with torch.no_grad():
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
                
                q_feat_list0 = [f.detach() for f in q_feat_list0]
                q_feat_list1 = [f.detach() for f in q_feat_list1]
                

                
            
            # Keypoint Detection & Description
            description_map0, invariance_map0 = self.kpnet(q0_img_ori,q_feat_list0)
            description_map1, invariance_map1 = self.kpnet(q1_img_ori,q_feat_list1)
            
                
            loss_items = []
            loss_dss =[]
            loss_views = []
            
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
                #loss_ds, conf = dual_softmax_loss(m0, m1)
                loss_ds, conf = weighted_dual_softmax_loss(m0, m1, h0, h1)
                loss_view = invariance_pose_loss(m0,m1,h0,h1,q0_Tcw[b], q1_Tcw[b], self.kpnet.beta)
                
                
                
                loss = loss_ds + loss_view
                loss_items.append(loss)
                loss_dss.append(loss_ds)
                loss_views.append(loss_view)
                
            if len(loss_items) > 0:    
                loss_mean = sum(loss_items) / len(loss_items)
                loss_ds_mean = sum(loss_dss) / len(loss_dss)
                loss_view_mean = sum(loss_views) / len(loss_views)
                self.optimizer.zero_grad()
                loss_mean.backward()
                self.optimizer.step()
                self.writer.add_scalar('train_loss_patches/l1_loss', loss_mean.item(), i)
                self.writer.add_scalar('train_loss_patches/ds_loss', loss_ds_mean.item(), i)
                self.writer.add_scalar('train_loss_patches/view_loss', loss_view_mean.item(), i)
                
            
            img0 = q0_img_ori[0].permute(2,0,1).clone()
            img0 = (img0 - img0.min()) / (img0.max() - img0.min())
            if i % 25 == 0:
                save_overlay(img0, invariance_map0[0], f"save_img/overlay_{i}.png")
                
                
            
            
            if (i+1) % self.save_ckpt_every == 0:
                print("saving iter ", i+1)
                torch.save(self.kpnet.state_dict(), self.cpkt_save_path  + "/sdr_kpdetect_" + str(iter_num) + ".pth")
                
            
                
            with torch.no_grad():
                # Progress bar
                if i % 10 == 0:
                    self.progress_bar.set_postfix({"Loss": f"{loss_mean.item():.{7}f}"})
                    print("loss_ds = ", loss_ds.item())
                    print("loss_view = ", loss_view.item())
                    self.progress_bar.update(10)
                    
            
            del loss_items
            del description_map0, description_map1
            del invariance_map0, invariance_map1
                
                
                
                
                
                
                
                
                
                
                


            
            
            
            

            
            

