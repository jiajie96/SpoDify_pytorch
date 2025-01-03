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


def compute2DSVD(tensor_stack):
    X = tensor_stack.reshape(-1, 130 * 130 * 130)
    XXT = np.dot(X, X.T)
    print(XXT.shape)
    u2, s, v2= linalg.svd(XXT)
    u = np.dot(X.T, u2) / np.sqrt(s)
    return u2, np.sqrt(s), u.T

def computeTT(tensor_stack, ranks):
    b = tn.Tensor(tensor_stack, ranks_tt = ranks)
    return b

