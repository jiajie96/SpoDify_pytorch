import numpy as np
import os
import pywt
from tqdm import tqdm
from multiprocessing import Pool
import argparse
import trimesh
import mesh2sdf

    
num_cores = os.cpu_count()
print(f"This machine has {num_cores} CPU cores.")

wavelet = pywt.Wavelet('coif1')

def preprocess_shape_to_SDF(input_folder, filename):
    """
    Preprocess a 3D mesh read from an STL file.
    
    Args:
        mesh_file (str): Path to the STL file containing the 3D mesh.
        
    Returns:
        np.ndarray: Truncated signed distance field (SDF) of the shape, with resolution 256^3.
    """
    # Load the 3D mesh using trimesh
    mesh_file = input_folder +'/' + filename + '.obj'
    mesh = trimesh.load(mesh_file)
    
    # Normalize the mesh to fit within the [-0.5, 0.5]^3 bounding box
    vertices = mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 1/ max(bbmax - bbmin)
    vertices = (vertices - center) * scale
    mesh.vertices = vertices
    # Compute the SDF using mesh2sdf.compute
    sdf = mesh2sdf.compute(vertices, mesh.faces, 256, fix=True, level=2/256, return_mesh=False)
    
    # Truncate the distance values in the SDF to the range [-0.1, +0.1]
    #sdf = np.clip(sdf, -0.1, 0.1)
    
    return sdf


def process_mesh(filename, SDF_dir, WAV_dir, args):
    
    sdf = preprocess_shape_to_SDF(args.mesh_path, filename)
    if args.activation:
        sdf = (1/2) * np.tanh(sdf) - (1/2)

    np.save(os.path.join(SDF_dir, filename), sdf)
    
    transform = pywt.dwtn(sdf, wavelet, mode='symmetric', axes=None)
    np.save(os.path.join(WAV_dir,filename),transform['aaa'])
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_path', type=str, default='/home/sagemaker-user/ExtractData/Planes', help='Path to the mesh files')
    parser.add_argument('--activation', type=bool, default=True, help='decide wether to add a tanh function before wavelet decomposition or not')
    args = parser.parse_args()
    
    
    local_folder = args.mesh_path
    activation = args.activation
    
    cluster_dir = os.path.join(local_folder, 'clustered_data')
    SDF_dir = os.path.join(local_folder, 'SDFs_data')
    WAV_dir = os.path.join(local_folder, 'WAVs_data')

    
    all_filenames = np.load(os.path.join(cluster_dir, 'filenames.npy')) 
    closest_clusters_indices =  np.load(os.path.join(cluster_dir, 'closest_clusters_indices.npy')) 
    
    files = all_filenames[closest_clusters_indices].split('.')[0]
    
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_mesh, files), total=len(files)))
    
    
    
if __name__ == '__main__':
    main()