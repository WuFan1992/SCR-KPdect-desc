import torch
import numpy as np
import pdb

debug_cnt = -1

def make_batch(augmentor, difficulty = 0.3, train = True):
    Hs = []
    img_list = augmentor.train if train else augmentor.test
    dev = augmentor.device
    batch_images = []

    with torch.no_grad(): # we dont require grads in the augmentation
        for b in range(augmentor.batch_size):
            rdidx = np.random.randint(len(img_list))
            img = torch.tensor(img_list[rdidx], dtype=torch.float32).permute(2,0,1).to(augmentor.device).unsqueeze(0)
            batch_images.append(img)

        batch_images = torch.cat(batch_images)

        p1, H1 = augmentor(batch_images, difficulty)
        p2, H2 = augmentor(batch_images, difficulty, TPS = True, prob_deformation = 0.7)

    return p1, p2, H1, H2


def plot_corrs(p1, p2, src_pts, tgt_pts):
    import matplotlib.pyplot as plt
    p1 = p1.cpu()
    p2 = p2.cpu()
    src_pts = src_pts.cpu() ; tgt_pts = tgt_pts.cpu()
    rnd_idx = np.random.randint(len(src_pts), size=200)
    src_pts = src_pts[rnd_idx, ...]
    tgt_pts = tgt_pts[rnd_idx, ...]

    #Plot ground-truth correspondences
    fig, ax = plt.subplots(1,2,figsize=(18, 12))
    colors = np.random.uniform(size=(len(tgt_pts),3))
    #Src image
    img = p1
    for i, p in enumerate(src_pts):
        ax[0].scatter(p[0],p[1],color=colors[i])
    ax[0].imshow(img.permute(1,2,0).numpy()[...,::-1])

    #Target img
    img2 = p2
    for i, p in enumerate(tgt_pts):
        ax[1].scatter(p[0],p[1],color=colors[i])
    ax[1].imshow(img2.permute(1,2,0).numpy()[...,::-1])
    plt.show()


def get_corresponding_pts(p1, p2, H, H2, augmentor, h, w, crop = None):
    '''
        Get dense corresponding points
    '''
    global debug_cnt
    negatives, positives = [], []

    with torch.no_grad():
        #real input res of samples
        rh, rw = p1.shape[-2:]
        ratio = torch.tensor([rw/w, rh/h], device = p1.device)

        (H, mask1) = H
        (H2, src, W, A, mask2) = H2

        #Generate meshgrid of target pts
        x, y = torch.meshgrid(torch.arange(w, device=p1.device), torch.arange(h, device=p1.device), indexing ='xy')
        mesh = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
        target_pts = mesh.view(-1, 2) * ratio

        #Pack all transformations into T
        for batch_idx in range(len(p1)):
            with torch.no_grad():
                T = (H[batch_idx], H2[batch_idx], 
                    src[batch_idx].unsqueeze(0), W[batch_idx].unsqueeze(0), A[batch_idx].unsqueeze(0))
                #We now warp the target points to src image
                src_pts = (augmentor.get_correspondences(target_pts, T) ) #target to src 
                tgt_pts = (target_pts)
            
                #Check out of bounds points
                mask_valid = (src_pts[:, 0] >=0) & (src_pts[:, 1] >=0) & \
                            (src_pts[:, 0] < rw) & (src_pts[:, 1] < rh)

                negatives.append( tgt_pts[~mask_valid] )            
                tgt_pts = tgt_pts[mask_valid]
                src_pts = src_pts[mask_valid]


                #Remove invalid pixels
                mask_valid =    mask1[batch_idx, src_pts[:,1].long(), src_pts[:,0].long()]  & \
                                mask2[batch_idx, tgt_pts[:,1].long(), tgt_pts[:,0].long()]
                tgt_pts = tgt_pts[mask_valid]
                src_pts = src_pts[mask_valid]

                # limit nb of matches if desired
                if crop is not None:
                    rnd_idx = torch.randperm(len(src_pts), device=src_pts.device)[:crop]
                    src_pts = src_pts[rnd_idx]
                    tgt_pts = tgt_pts[rnd_idx]

                if debug_cnt >=0 and debug_cnt < 4:
                    plot_corrs(p1[batch_idx], p2[batch_idx], src_pts , tgt_pts )
                    debug_cnt +=1

                src_pts = (src_pts / ratio)
                tgt_pts = (tgt_pts / ratio)

                #Check out of bounds points
                padto = 10 if crop is not None else 2
                mask_valid1 = (src_pts[:, 0] >= (0 + padto)) & (src_pts[:, 1] >= (0 + padto)) & \
                             (src_pts[:, 0] < (w - padto)) & (src_pts[:, 1] < (h - padto))
                mask_valid2 = (tgt_pts[:, 0] >= (0 + padto)) & (tgt_pts[:, 1] >= (0 + padto)) & \
                             (tgt_pts[:, 0] < (w - padto)) & (tgt_pts[:, 1] < (h - padto))
                mask_valid = mask_valid1 & mask_valid2
                tgt_pts = tgt_pts[mask_valid]
                src_pts = src_pts[mask_valid]         

                #Remove repeated correspondences
                lut_mat = torch.ones((h, w, 4), device = src_pts.device, dtype = src_pts.dtype) * -1
                # src_pts_np = src_pts.cpu().numpy()
                # tgt_pts_np = tgt_pts.cpu().numpy()
                try:
                    lut_mat[src_pts[:,1].long(), src_pts[:,0].long()] = torch.cat([src_pts, tgt_pts], dim=1)
                    mask_valid = torch.all(lut_mat >= 0, dim=-1)
                    points = lut_mat[mask_valid]
                    positives.append(points)
                except:
                    pdb.set_trace()
                    print('..')

    return negatives, positives


def crop_patches(tensor, coords, size = 7):
    '''
        Crop [size x size] patches around 2D coordinates from a tensor.
    '''
    B, C, H, W = tensor.shape

    x, y = coords[:, 0], coords[:, 1]
    y = y.view(-1, 1, 1)
    x = x.view(-1, 1, 1)
    halfsize = size // 2
    # Create meshgrid for indexing
    x_offset, y_offset = torch.meshgrid(torch.arange(-halfsize, halfsize+1), torch.arange(-halfsize, halfsize+1), indexing='xy')
    y_offset = y_offset.to(tensor.device)
    x_offset = x_offset.to(tensor.device)

    # Compute indices around each coordinate
    y_indices = (y + y_offset.view(1, size, size)).squeeze(0) + halfsize
    x_indices = (x + x_offset.view(1, size, size)).squeeze(0) + halfsize

    # Handle out-of-boundary indices with padding
    tensor_padded = torch.nn.functional.pad(tensor, (halfsize, halfsize, halfsize, halfsize), mode='constant')

    # Index tensor to get patches
    patches = tensor_padded[:, :, y_indices, x_indices] # [B, C, N, H, W]
    return patches

def subpix_softmax2d(heatmaps, temp = 0.25):
    N, H, W = heatmaps.shape
    heatmaps = torch.softmax(temp * heatmaps.view(-1, H*W), -1).view(-1, H, W)
    x, y = torch.meshgrid(torch.arange(W, device =  heatmaps.device ), torch.arange(H, device =  heatmaps.device ), indexing = 'xy')
    x = x - (W//2)
    y = y - (H//2)
    #pdb.set_trace()
    coords_x = (x[None, ...] * heatmaps)
    coords_y = (y[None, ...] * heatmaps)
    coords = torch.cat([coords_x[..., None], coords_y[..., None]], -1).view(N, H*W, 2)
    coords = coords.sum(1)

    return coords


def check_accuracy(X, Y, pts1 = None, pts2 = None, plot=False):
    with torch.no_grad():
        #dist_mat = torch.cdist(X,Y)
        dist_mat = X @ Y.t()
        nn = torch.argmax(dist_mat, dim=1)
        #nn = torch.argmin(dist_mat, dim=1)
        correct = nn == torch.arange(len(X), device = X.device)

        if pts1 is not None and plot:
            import matplotlib.pyplot as plt
            canvas = torch.zeros((60, 80),device=X.device)
            pts1 = pts1[~correct]
            canvas[pts1[:,1].long(), pts1[:,0].long()] = 1
            canvas = canvas.cpu().numpy()
            plt.imshow(canvas), plt.show()

        acc = correct.sum().item() / len(X)
        return acc

def get_nb_trainable_params(model):
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	nb_params = sum([np.prod(p.size()) for p in model_parameters])
 
	print('Number of trainable parameters: {:d}'.format(nb_params))
 
 
 
# Calculate Pose difference 

def so3_log(R):
    """
    R: (..., 3, 3)
    return: (..., 3)  rotation vector
    """
    cos_theta = (R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)

    wx = R[..., 2, 1] - R[..., 1, 2]
    wy = R[..., 0, 2] - R[..., 2, 0]
    wz = R[..., 1, 0] - R[..., 0, 1]

    w = torch.stack([wx, wy, wz], dim=-1)

    small_angle = theta < 1e-5

    A = torch.where(
        small_angle,
        0.5 - theta**2 / 12,
        theta / (2 * torch.sin(theta))
    )

    omega = A.unsqueeze(-1) * w
    return omega


def se3_log(T):
    """
    T: (..., 4, 4)
    return: (..., 6)  (omega, v)
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]

    omega = so3_log(R)
    theta = torch.norm(omega, dim=-1, keepdim=True)

    small_angle = theta < 1e-5

    wx = torch.zeros_like(R)
    wx[..., 0, 1] = -omega[..., 2]
    wx[..., 0, 2] =  omega[..., 1]
    wx[..., 1, 0] =  omega[..., 2]
    wx[..., 1, 2] = -omega[..., 0]
    wx[..., 2, 0] = -omega[..., 1]
    wx[..., 2, 1] =  omega[..., 0]

    I = torch.eye(3, device=T.device).expand_as(R)

    theta_sq = theta ** 2
    A = torch.where(
        small_angle,
        1 - theta_sq / 6,
        torch.sin(theta) / theta
    )
    B = torch.where(
        small_angle,
        0.5 - theta_sq / 24,
        (1 - torch.cos(theta)) / theta_sq
    )

    V = I + B.unsqueeze(-1) * wx + \
        ((1 - A) / theta_sq).unsqueeze(-1) * (wx @ wx)

    V_inv = torch.linalg.pinv(V)
    v = (V_inv @ t.unsqueeze(-1)).squeeze(-1)

    return torch.cat([omega, v], dim=-1)


def pose_se3_error(T1, T2):
    """
    返回 SE(3) 李代数误差的 L2 范数
    """
    T_rel = torch.linalg.inv(T1) @ T2
    xi = se3_log(T_rel)    
    return torch.norm(xi, dim=-1)


def invariance_to_logvar(v, eps=1e-6):
    """
    v: (N,) invariance map, must be positive
    return: log variance s
    """
    return - torch.log(v + eps)