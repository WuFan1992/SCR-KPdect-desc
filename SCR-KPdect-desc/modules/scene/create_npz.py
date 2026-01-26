import os
from colmap_reader import read_points3D_binary
from itertools import combinations
from scene_utils import pairwise_combinations, remove_duplicates_imgids

def get_imagepair_index(path):
    """
     Get the image pair index 
    """

    point3d_path = os.path.join(path, "sparse/0/points3D.bin")
    _, _, _, _, img_ids, _  = read_points3D_binary(point3d_path)
    
    pair_set = set()   # 用集合直接去重

    for p_id in img_ids:
        imgs = img_ids[p_id]   # image ids observing this 3D point
        # 生成两两组合
        for a, b in combinations(imgs, 2):
            # 无序对规范化
            if a < b:
                pair_set.add((a, b))
            else:
                pair_set.add((b, a))
    # 转回 list[list]
    imgs_pairs = [list(pair) for pair in pair_set]
    return imgs_pairs
    
    
    
    
    


def create_npz(path):
    
    # Get the image pair index
    imgs_pairs = get_imagepair_index(path)

    
    # Get the image path 
    
    # Get the depth path
    
    # Get the intrincsics 
    
    # Get the pose
    
    # write data into npz file 



if __name__ == "__main__":
    path = "../../datasets/head"
    create_npz(path)    





