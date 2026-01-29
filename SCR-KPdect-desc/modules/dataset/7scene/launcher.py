import torch
import yaml
from trainer import Trainer



class Launcher(object):
    def __init__(self, cfg, test_only=False):
        self.cfg = cfg

        self.trainer = None
        if not test_only:
            self.trainer = Trainer(cfg)

    def run_train(self):
        self.trainer.train_iters(self.cfg.TRAIN.model_save_iters)

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
    yaml_path = "./7scenes.yaml"
    cfg = load_config(yaml_path)
    launcher = Launcher(cfg)  
    launcher.run_train()   

