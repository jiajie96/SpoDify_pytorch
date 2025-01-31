import os
import numpy as np
from tqdm import tqdm
import point_cloud_utils as pcu
from trimesh import load_mesh
from typing import List, Tuple
from multiprocessing import Pool
import scipy
import numexpr as ne
from sklearn.cluster import KMeans

import argparse


def read_meshes(local_folder: str, target_num_vertices: int = 4096) -> Tuple[np.ndarray, List[str]]:
    """
    Reads all obj files in a given folder, samples the vertices of each mesh so that it has the target number of vertices, and returns the vertices and the filenames.
    
    Args:
        local_folder (str): The folder containing the obj files.
        target_num_vertices (int, optional): The number of vertices each mesh should have. Defaults to 4096.
    
    Returns:
        Tuple[np.ndarray, List[str]]: A tuple containing the vertices of the meshes (as a numpy array) and the filenames of the meshes (as a list of strings).
    """
    meshes_within_folder = []
    filenames_within_folder = []

    try:
        file_count = len([f for f in os.listdir(local_folder) if f.endswith('.obj')])
        with tqdm(total=file_count, desc="Reading meshes") as pbar:
            for filename in os.listdir(local_folder):
                if filename.endswith('.obj'):
                    mesh_path = os.path.join(local_folder, filename)
                    mesh = load_mesh(mesh_path)

                    bbmin = mesh.vertices.min(0)
                    bbmax = mesh.vertices.max(0)
                    center = (bbmin + bbmax) * 0.5
                    scale = 2.0 / (bbmax - bbmin).max()
                    mesh.vertices = (mesh.vertices - center) * scale

                    sampled_vertices = mesh.vertices[np.linspace(0, len(mesh.vertices) - 1, target_num_vertices, dtype=int)]
                    meshes_within_folder.append(sampled_vertices)
                    filenames_within_folder.append(filename.replace('.obj', ''))
                    pbar.update(1)
    except Exception as e:
        print(f"Error reading meshes: {e}")

    return np.array(meshes_within_folder), filenames_within_folder

def compute_diffusion_maps_without_alphas(X, n_eigenpairs=10):
    L = X
    d = np.sum(L, axis=1)
    
    D = np.diag(d)
    M = np.linalg.pinv(D) @ L
    
    n_samples, n_features = M.shape
    
    # Compute the eigendecomposition of M
    scipy_eigvec_solver = scipy.sparse.linalg.eigsh

    solver_kwargs = {
        "k": n_eigenpairs,
        "which": "LM",
        "v0": np.ones(n_samples),
        "tol": 1e-14,
    }

    solver_kwargs["sigma"] = None

    if scipy.sparse.issparse(M) and M.data.dtype.kind not in ["fdFD"]:
        M = M.asfptype()
    elif isinstance(M, np.ndarray) and M.dtype.kind != "f":
        M = M.astype(float)

    eigenvalues, eigenvectors = scipy_eigvec_solver(M, **solver_kwargs)
    
    return eigenvalues, eigenvectors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_path', type=str, default='/data/Planes', help='Path to the mesh files')
    parser.add_argument('--n_eigenpairs', type=int, default=64, help='Number of eigenpairs to compute')
    parser.add_argument('--n_clusters', type=int, default=1000, help='Number of clusters to form')
    args = parser.parse_args()
    
    
    local_folder = args.mesh_path
    n_eigenpairs = args.n_eigenpairs
    n_clusters = args.n_clusters
    
    cluster_dir = os.path.join(local_folder, 'clustered_data')
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)
        
    
    all_meshes, all_filenames = read_meshes(local_folder)
    print(len(all_filenames), len(all_meshes))
    
    # Compute the pairwise Hausdorff distance matrix in parallel
    def compute_chamfer_distance(i_j):
        i,j = i_j
        return pcu.chamfer_distance(np.float64(all_meshes[i]), np.float64(all_meshes[j]))

    with Pool() as p:
        chamfer_matrix = np.array(list(tqdm(p.imap_unordered(compute_chamfer_distance, [(i, j) for i in range(num_meshes) for j in range(i, num_meshes)]), total=num_meshes*(num_meshes+1)//2)))

    # Reconstruct the full Hausdorff distance matrix
    num_meshes = len(all_meshes)

    full_chamfer_matrix = np.zeros((num_meshes, num_meshes))
    full_chamfer_matrix[np.triu_indices(num_meshes)] = chamfer_matrix
    full_chamfer_matrix = full_chamfer_matrix + full_chamfer_matrix.T - np.diag(np.diag(full_chamfer_matrix))

    print("Pairwise Chamfer Distance Matrix:")
    print(full_chamfer_matrix)
    
    # Save the Hausdorff distance matrix and the filenames
    np.save(os.path.join(cluster_dir, 'selected_1k_airplanes_chamfer_matrix_s3.npy'), full_chamfer_matrix)
    np.save(os.path.join(cluster_dir, 'selected_1k_airplanes_filenames_s3.npy'), all_filenames) 
    
    expr_dict = {}
    expr_dict["D"] = full_chamfer_matrix
    expr_dict["eps"] = 1.0

    expr = "exp((- 1 / (2*eps)) * D)"
    kernel_matrix = ne.evaluate(expr, expr_dict)
    
    eigenvalues, eigenvectors =  compute_diffusion_maps_without_alphas(kernel_matrix, n_eigenpairs = n_eigenpairs)
    eigenvectors /= np.linalg.norm(eigenvectors, axis=0)[np.newaxis, :]
    idx = np.argsort(eigenvalues)
    idx = idx[::-1]
    sorted_eigenvalues, sorted_eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
    print(sorted_eigenvalues.shape, sorted_eigenvectors.shape)
    diffusion_embeedings  = sorted_eigenvectors  * sorted_eigenvalues[None, :]
    
    np.save(os.path.join(cluster_dir, 'selected_1k_airplanes_64Evs_diffusion_embeedings_cd_s3.npy'), diffusion_embeedings)
    
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=0).fit(diffusion_embeedings)
    centroids = kmeans.cluster_centers_
    closest_meshes = []
    closest_indices = []
    for centroid in centroids:
        distances = [np.linalg.norm(centroid - mesh) for mesh in diffusion_embeedings]
        closest_index = np.argmin(distances)
        closest_indices.append(closest_index)
        #closest_meshes.append(all_meshes[closest_index])
        
    np.save(os.path.join(cluster_dir, 'selected_1k_Clusters_closest_indices_s3.npy'), closest_indices)
    
    
if __name__ == '__main__':
    main()