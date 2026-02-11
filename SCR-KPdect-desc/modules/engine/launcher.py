import yaml
import torch
import os.path as osp
from modules.engine.trainer import Trainer
from modules.arch.DSMNet import dsm_net
from modules.arch.KPNet import KPNet


class Launcher(object):
    def __init__(self, cfg, test_only=False):
        self.cfg = cfg

        self.trainer = None
        self.model = dsm_net(cfg.MODEL).cuda()
        self.kpnet = KPNet().cuda()
        
        if not test_only:
            self.trainer = Trainer(cfg, self.model, self.kpnet)

        self.tester = None

    def run_train(self):
        self.trainer.train_iters(self.cfg.TRAIN.model_save_iters)
        
    def save_model(self, log_dir):
        save_params = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.trainer.optimizer.state_dict(),
            "train_niter": self.trainer.niter,
            "test_niter": self.tester.niter,
            "reproj_loss_start": self.trainer.reproj_loss_start,
        }
        checkpoint_name = "checkpoint.pth-{}".format(self.trainer.niter)
        with open(osp.join(log_dir, "latest.txt"), "w") as f:
            f.write(checkpoint_name)
        torch.save(save_params, osp.join(log_dir, checkpoint_name))

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
    launcher = Launcher(cfg)  
    launcher.run_train()   

