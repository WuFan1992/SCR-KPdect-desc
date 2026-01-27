
import numpy as np
import math

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def getIntrinsic(FoVx, width, height):
    K = np.eye(3)
    focal_length = fov2focal(FoVx, width)
    K[0, 0] = K[1, 1] = focal_length
    K[0, 2] = width / 2
    K[1, 2] = height / 2
    return K

def getExtrinsic(R, t):
    pose = np.eye(4)
    pose[:3,:3] = R
    pose[:3,3] = t
    return pose
