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
										nn.Conv2d (256, 1, 1),
										nn.Sigmoid()
									)
        self.keypoint_head = nn.Sequential(
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										nn.Conv2d (64, 65, 1),
									)
        
    def _unfold2d(self, x, ws = 2):
        """
			Unfolds tensor in 2D with desired ws (window size) and concat the channels
	    """
        B, C, H, W = x.shape
        x = x.unfold(2,  ws , ws).unfold(3, ws,ws)                             \
			.reshape(B, C, H//ws, W//ws, ws**2)
        
        return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)
    
    
    def forward(self, x, q_feat_list):
        
        # BHWC → BCHW
        x = x.permute(0, 3, 1, 2)

        # don't backprop through normalization
        with torch.no_grad():
            x = self.norm(x.mean(dim=1, keepdim=True))

        # 目标尺寸（取最后一个特征图）
        target_size = q_feat_list[-1][0].shape[-2:]

        # resize + 累加
        contact_feat = sum(
            F.interpolate(q_feat[0].unsqueeze(0),
                  size=target_size,
                  mode='bilinear',
                  align_corners=False)
        for q_feat in q_feat_list
        )
            
        
        description_map = self.block_fusion(contact_feat)
        invariance_map = self.invariance_head(description_map)
        
        keypoints = self.keypoint_head(self._unfold2d(x, ws=8)) #Keypoint map logits
        
        return description_map, invariance_map,keypoints
        
        
        
        
        
        
        
        
        
        
		    
        
        
        
        

  
    


