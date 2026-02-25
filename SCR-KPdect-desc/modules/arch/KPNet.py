import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicLayer(nn.Module):
	"""
	  Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
	"""
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
		super().__init__()
		self.layer = nn.Sequential(
									  nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = bias),
									  nn.BatchNorm2d(out_channels, affine=False),
									  nn.ReLU(inplace = True),
									)

	def forward(self, x):
	  return self.layer(x)


class KPNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)
        self.block_fusion =  nn.Sequential(
										BasicLayer(256, 256, stride=1),
										BasicLayer(256, 256, stride=1),
										nn.Conv2d (256, 256, 1, padding=0)
									 )
        self.invariance_head = nn.Sequential(
										BasicLayer(256, 256, 1, padding=0),
										BasicLayer(256, 256, 1, padding=0),
										nn.Conv2d (256, 1, 1)
									)
        self.beta = nn.Parameter(torch.tensor(0.1)) 
    
    def forward(self, x, q_feat_list):
        

        # don't backprop through normalization
        with torch.no_grad():
            x = self.norm(x.mean(dim=1, keepdim=True))

        # 目标尺寸（取最后一个特征图）
        target_size = q_feat_list[-1][0].shape[-2:]


        # resize + 累加
        contact_feat = sum(
            F.interpolate(
            q_feat,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        for q_feat in q_feat_list
        )
        
        #print("contact_feat min:", contact_feat.min().item())
        #print("contact_feat max:", contact_feat.max().item())
        
        
        description_map = self.block_fusion(contact_feat)
        invariance_map = self.invariance_head(description_map)
        

        
        return description_map, invariance_map
        
        
        
        
        
        
        
        
        
        
		    
        
        
        
        

  
    


