import torch
from itertools import combinations

def pairwise_combinations(nums):
    """
    nums: List[int]
    return: List[List[int]]  # all unordered unique pairs
    """
    return [list(pair) for pair in combinations(nums, 2)]

def remove_duplicates_imgids(pair_list):
    """
    pair_list: List[List[int]]
        Each element has length 2, e.g. [[a, b], [c, d], ...]
    
    return:
        A list with unordered duplicates removed
        ([a, b] and [b, a] are considered the same)
    """
    x = torch.tensor(pair_list)   # shape: (N, 2)

    # Sort each row so that [a, b] and [b, a] become the same representation
    x_sorted, _ = torch.sort(x, dim=1)  # shape: (N, 2)

    # Remove duplicates row-wise
    unique_x_sorted, indices = torch.unique(
        x_sorted,
        dim=0,
        return_inverse=False,
        return_counts=False,
        sorted=False
    )

    # Convert back to Python list
    return unique_x_sorted.tolist()
