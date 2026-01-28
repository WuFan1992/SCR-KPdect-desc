import torch
import torch.utils.data as data
import os.path as osp

from dataset import *
from transform import *


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
   
        self.dataset = VideoDataset7scene(cfg.TRAIN.DATASET, transform)

        self.data_loader = data.DataLoader(
            self.dataset,
            cfg.TRAIN.batch_size,
            num_workers=cfg.TRAIN.workers,
            shuffle=True,
            pin_memory=True,
        )

        self.base_lr = cfg.TRAIN.base_lr
        self.lr_steps1 = cfg.TRAIN.lr_steps1
        self.lr_steps2 = cfg.TRAIN.lr_steps2

        self.niter = 0
        self.data_loader_iter = iter(self.data_loader)  # 把数据变成迭代器，方便使用next 一个一个获取
      
    def adjust_learning_rate(self):
        if self.niter in self.lr_steps1:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * 0.5

        if self.niter in self.lr_steps2:
            if self.niter == self.lr_steps2[0]:
                self.reproj_loss_start = True
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.base_lr * 0.5
            elif self.niter in self.lr_steps2:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = param_group["lr"] * 0.5

    def train_iters(self, iter_num):
        if True:
            try:
                q, r = next(self.data_loader_iter)
            except StopIteration:
                self.data_loader_iter = iter(self.data_loader)
                q, r = next(self.data_loader_iter)

            q_img = q["img"].cuda()  # (N T 3 H W)
            q_Tcw = q["Tcw"].cuda()
            q_K = q["K"].cuda()
            q_depth = q["depth"].cuda()

            s_img = r["img"].cuda()  # (N L 3 H W)
            s_Tcw = r["Tcw"].cuda()
            s_K = r["K"].cuda()
            s_depth = r["depth"].cuda()
            
            print("size(q_img) = ", q_img.shape)
            print("size(s_img) = ", s_img.shape)

