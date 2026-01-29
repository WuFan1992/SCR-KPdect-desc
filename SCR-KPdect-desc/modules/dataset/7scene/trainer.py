import torch
import torch.utils.data as data
import os.path as osp

from dataset import *
from transform import *
from sevenscenes import *


class Trainer(object):
    def __init__(self, cfg):
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
   
        self.dataset = SevenSceneDataset(cfg.TRAIN.DATASET, transform)

        self.data_loader = data.DataLoader(
            self.dataset,
            cfg.TRAIN.batch_size,
            num_workers=0,
            shuffle=True,
            pin_memory=True,
            
        )


        self.niter = 0
        self.data_loader_iter = iter(self.data_loader)  # 把数据变成迭代器，方便使用next 一个一个获取
      

    def train_iters(self, iter_num):
        if True:
            try:
                q, r = next(self.data_loader_iter)
            except StopIteration:
                self.data_loader_iter = iter(self.data_loader)
                q, r = next(self.data_loader_iter)

            q_img = q["image0"].cuda()  # (N T 3 H W)
           

            s_img = r["img"].cuda()  # (N L 3 H W)

            
            print("size(q_img) = ", q_img.shape)
            print("size(s_img) = ", s_img.shape)

