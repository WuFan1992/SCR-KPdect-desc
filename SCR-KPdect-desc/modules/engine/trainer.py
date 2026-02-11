import torch
import torch.utils.data as data
import os.path as osp


from modules.utils.transform import *
from modules.dataset.sevenscene.sevenscenes import *
from modules.dataset.sevenscene import sevenscenes_warper


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
        

              

    def train_iters(self, iter_num):
        if True:
            try:
                q, r = next(self.data_loader_iter)
            except StopIteration:
                self.data_loader_iter = iter(self.data_loader)
                q, r = next(self.data_loader_iter)
                
                
            positive_md_coarse = sevenscenes_warper.spvs_coarse(q,4) 

            
            """ 
                
            

            q_img = q["image0"].cuda()  # (N T 3 H W)
            q_img_ori = q["image0_ori"].cuda()
            
            q_Tcw = q["T0"].cuda()
            q_K = q["K0"].cuda()
            q_depth = q["depth0"].cuda()
           
            s_img = r["img"].cuda()  # (N L 3 H W)
            s_Tcw = r["pose"].cuda()
            s_K = r["K"].cuda()
            s_depth = r["depth"].cuda()
            

            losses, metrics, pred_coords, gt_coords, _, _, q_feat_list = self.model(
                q_img,
                q_depth,
                q_Tcw,
                q_K,
                s_img,
                s_depth,
                s_Tcw,
                s_K,
                s_Tcw[:, 0, :, :],
            )
            
            description_map, invariance_map,keypoints = self.kpnet(q_img_ori,q_feat_list)
            
            
            
            
            
            print("description map shape = ", description_map.shape)
            print("invariance map shape = ", invariance_map.shape)
            print("keypoint map shape = ", keypoints.shape)
            
            """

