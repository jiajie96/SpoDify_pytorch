import numpy as np
import os
from scipy import linalg
import tntorch as tn
import torch
import time


def load_and_stack_tensors(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    
    all_files = []
    
    for file in files:
        file_path = os.path.join(folder_path, file)
        tensor = np.load(file_path)
        
        if tensor.shape != (130, 130, 130):
            raise ValueError(f"Tensor in file {file} does not have the required shape (130, 130, 130), but has shape {tensor.shape}.")
        
        all_files.append(tensor)
    
    if not all_files:
        raise ValueError("No tensors found in the folder.")
        
    tensor_stack = np.asarray(all_files)
    
    return tensor_stack


def compute2DSVD(tensor_stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the 2D SVD decomposition of a given tensor stack.

    Args:
        tensor_stack: A numpy array representing the stack of tensors to be decomposed.

    Returns:
        A tuple of three numpy arrays: the left singular vectors, the singular values, and the right singular vectors.
    """
    X = tensor_stack.reshape(-1, tensor_stack.shape[1]*tensor_stack.shape[2]*tensor_stack.shape[3])
    u2, s, v2 = linalg.svd(X @ X.T)
    u = X.T @ u2 / np.sqrt(s)
    return u2, np.sqrt(s), u.T

def computeTT(tensor_stack: np.ndarray, ranks: list[int]) -> tn.Tensor:
    """
    Computes the Tensor Train (TT) decomposition of a given tensor stack.

    Args:
        tensor_stack (np.ndarray): A numpy array representing the stack of tensors to be decomposed.
        ranks (list[int]): A list of integers specifying the TT ranks for the decomposition.

    Returns:
        tn.Tensor: The TT representation of the input tensor stack.
    """
    b = tn.Tensor(tensor_stack, ranks_tt=ranks)
    return b

