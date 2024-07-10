import cv2
import open3d as o3d
import numpy as np
import cupy as cp
from initial_mesh_gen import generate_initial_mesh, get_scaled_depth_map
from scipy.sparse import lil_matrix, csr_matrix
import time
import os
from torch.optim import SGD  # Example optimizer, you might choose others
import torch

# Load initial mesh and depth map
def load_initial_data(depth_file, mesh_file):
    depth_map = get_scaled_depth_map(depth_file)
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    return depth_map, mesh

# Formulation of the Optimization Problem

def compute_data_fidelity_term(mesh, depth_map):
    # Data fidelity term: Ldepth and Ltracking
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    # Assuming we have depth values for each vertex
    # Initial depth values from the depth map (visual odometry)
    initial_depth_values = depth_map[vertices[:, 1].astype(int), vertices[:, 0].astype(int)]
    print(f"vertices {vertices.shape} initial_depth_vlues{len(initial_depth_values)}")
    
    # Convert to cupy arrays for GPU computation
    vertices_cp = cp.asarray(vertices)
    initial_depth_values_cp = cp.asarray(initial_depth_values)
    
    # Compute inverse depths from the depth map values
    # vξ - current inverse depth of vertices being optimized
    # vz - initial inverse depth from VO (used for Ltracking)
    vξ = cp.asarray(1.0 / (vertices[:, 2] + 1e-5))  # Current estimated inverse depth
    vz = cp.asarray(1.0 / (initial_depth_values_cp + 1e-5))  # Initial inverse depth from VO

    # Compute barycentric coordinates matrix A
    A = compute_barycentric_coordinates(vertices, faces, depth_map.shape)
    A_cupy = cp.sparse.csr_matrix(A)  # Convert scipy sparse matrix to cupy sparse matrix
    A_vξ = A_cupy @ vξ
    A_vξ = A_vξ[:initial_depth_values_cp.size]  # Align dimensions with depth_values_cp

    # Depth fidelity term: Ldepth(vξ) = ∑d∈D |advξ - bd|
    b = cp.asarray(1.0 / (initial_depth_values_cp + 1e-5))  # Measured inverse depth from depth map
    Ldepth = cp.sum(cp.abs(A_vξ - b))

    # Tracking fidelity term: Ltracking(vξ) = ∑v∈V |vξ - vz|
    Ltracking = cp.sum(cp.abs(vξ - vz))
    
    return Ldepth, Ltracking

def compute_data_fidelity_term_backup(mesh, depth_map):
    # Data fidelity term: Ldepth and Ltracking
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    # Assuming we have depth values for each vertex
    depth_values = depth_map[vertices[:, 1].astype(int), vertices[:, 0].astype(int)]
    
    # Convert to cupy arrays
    vertices_cp = cp.asarray(vertices)
    depth_values_cp = cp.asarray(depth_values)
    
    # Ldepth(vξ) = ∑d∈D |advξ - bd|
    A = compute_barycentric_coordinates(vertices, faces, depth_map.shape)
    vξ = cp.asarray(1.0 / (vertices[:, 2] + 1e-5))  # Inverse depth
    b = cp.asarray(1.0 / (depth_values + 1e-5))     # Measured inverse depth
    A_cupy = cp.sparse.csr_matrix(A)  # Convert scipy sparse matrix to cupy sparse matrix
    # Ensure proper dimension alignment
    A_vξ = A_cupy @ vξ
    A_vξ = A_vξ[:depth_values_cp.size]  # Align dimensions with depth_values_cp
    Ldepth = cp.sum(cp.abs(A_vξ - b))
    
    # Ltracking(vξ) = ∑v∈V |vξ - vz|
    # v_z = vξ.copy()  # Assuming initial v_z is the same as vξ for simplicity
    v_z = cp.asarray(1.0 / (vertices[:, 2] + 1e-5))  # Assuming v_z should be initialized correctly
    Ltracking = cp.sum(cp.abs(vξ - v_z))
    
    return Ldepth, Ltracking


def edge_length(vi, vj):
    """ Calculate the Euclidean distance between two vertices. """
    return cp.linalg.norm(vi - vj)

def compute_angle_between_normals(ni, nj):
    """ Compute the cosine of the angle between two normals. """
    dot_product = cp.dot(ni, nj)
    norms = cp.linalg.norm(ni) * cp.linalg.norm(nj)
    if norms > 0:
        return dot_product / norms
    else:
        return 1.0  # Return 1 when one of the normals is a zero vector

def compute_smoothness_term(mesh, w):
    mesh.compute_vertex_normals()
    
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals)

    vξ = cp.asarray(1.0 / (vertices[:, 2] + 1e-5))
    vertices_cp = cp.asarray(vertices)
    normals_cp = cp.asarray(normals)

    NLTGV2 = 0
    for e in faces:
        i, j, k = e
        vi, vj, vk = vertices_cp[e[0]], vertices_cp[e[1]], vertices_cp[e[2]]
        ni, nj, nk = normals_cp[e[0]], normals_cp[e[1]], normals_cp[e[2]]
        viξ = vξ[e[0]]
        vjξ = vξ[e[1]]
        vkξ = vξ[e[2]]

        # Calculate edge weights based on length
        length_ij = edge_length(vi, vj)
        length_ik = edge_length(vi, vk)
        e_alpha_ij = 1.0 / length_ij if length_ij > 0 else 1.0
        e_alpha_ik = 1.0 / length_ik if length_ik > 0 else 1.0

        # Calculate e_beta based on angle between normals
        angle_ij = compute_angle_between_normals(ni, nj)
        angle_ik = compute_angle_between_normals(ni, nk)
        e_beta_ij = 1.0 - angle_ij  # Scale based on the angle; larger angle leads to larger e_beta
        e_beta_ik = 1.0 - angle_ik

        De_vi_vj = cp.array([e_alpha_ij * (viξ - vjξ - cp.dot(ni, vi - vj)), e_beta_ij * (ni[0] - nj[0]), e_beta_ij * (ni[1] - nj[1])])
        De_vi_vk = cp.array([e_alpha_ik * (viξ - vkξ - cp.dot(ni, vi - vk)), e_beta_ik * (ni[0] - nk[0]), e_beta_ik * (ni[1] - nk[1])])

        NLTGV2 += cp.linalg.norm(De_vi_vj, 1)
        NLTGV2 += cp.linalg.norm(De_vi_vk, 1)

    return NLTGV2


def compute_smoothness_term_backup(mesh, w):

    mesh.compute_vertex_normals()
    # Smoothness term: NLTGV2(ξ)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals)  # Correctly obtain normals from mesh
    vξ = cp.asarray(1.0 / (vertices[:, 2] + 1e-5))  # Inverse depth
    vertices_cp = cp.asarray(vertices)
    normals_cp = cp.asarray(normals)
    w_cp = cp.asarray(w)
    # vξ = 1.0 / (vertices[:, 2] + 1e-5)  # Inverse depth
    
    # Define the smoothing term as described in the paper
    # Here, we use placeholders for the necessary computations
    NLTGV2 = 0
    for e in faces:
        vi = vertices_cp[e[0]]
        vj = vertices_cp[e[1]]
        vk = vertices_cp[e[2]]

        viξ = vξ[e[0]]
        vjξ = vξ[e[1]]
        vkξ = vξ[e[2]]

        # Normal vectors (assumed for simplicity, may need actual computation)
        ni = normals_cp[e[0]]
        nj = normals_cp[e[1]]
        nk = normals_cp[e[2]]

        # Using the proper formula for De and NLTGV2
        De_vi_vj = cp.array([viξ - vjξ - cp.dot(ni, vi - vj), ni[0] - nj[0], ni[1] - nj[1]])
        De_vi_vk = cp.array([viξ - vkξ - cp.dot(ni, vi - vk), ni[0] - nk[0], ni[1] - nk[1]])
        
        NLTGV2 += cp.linalg.norm(De_vi_vj, 1)
        NLTGV2 += cp.linalg.norm(De_vi_vk, 1)

        # Add the smoothing term contribution for each edge in the mesh
        # NLTGV2 += cp.linalg.norm(vi - vj) + cp.linalg.norm(vi - vk) + cp.linalg.norm(vj - vk)
        # NLTGV2 += cp.linalg.norm(viw - vjw)
    
    return NLTGV2

def compute_total_loss(Ldepth, Ltracking, NLTGV2, lambda_val=1.0):
    # Total loss: L = Lsmooth + λLdata
    Ldata = Ldepth + Ltracking
    Lsmooth = NLTGV2
    L = Lsmooth + lambda_val * Ldata
    return L

def compute_barycentric_coordinates(vertices, faces, depth_shape):
    # Placeholder function to compute the barycentric coordinates matrix A
    # Normally, this would involve complex geometry processing
    # Function to compute the barycentric coordinates matrix A
    num_pixels = depth_shape[0] * depth_shape[1]
    num_vertices = len(vertices)
    
    # Create a sparse matrix to store barycentric coordinates
    A = lil_matrix((num_pixels, num_vertices))
    
    for face in faces:
        tri_vertices = vertices[face]
        # Compute barycentric coordinates for each triangle (simplified placeholder)
        for vertex in face:
            pixel_x = int(vertices[vertex, 0])
            pixel_y = int(vertices[vertex, 1])
            if 0 <= pixel_x < depth_shape[1] and 0 <= pixel_y < depth_shape[0]:
                pixel_idx = pixel_y * depth_shape[1] + pixel_x
                A[pixel_idx, vertex] = 1.0  # Simplified placeholder
    
    return A.tocsr()
    # return cp.array(A)


def compute_vertex_normals(vertices, triangles):
    # Compute normals for the mesh
    normals = np.zeros_like(vertices)
    for i in range(triangles.shape[0]):
        v1, v2, v3 = triangles[i]
        triangle = vertices[[v1, v2, v3]]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= np.linalg.norm(normal)
        normals[[v1, v2, v3]] += normal
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
    return normals

def primal_dual_optimization_backup_wrongans(mesh, depth_map, max_iterations=10, sigma=0.1, tau=0.1, theta=1.0, lambda_val=1.0):
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, requires_grad=True)
    triangles = np.asarray(mesh.triangles)
    
    num_vertices = vertices.shape[0]
    num_pixels = depth_map.shape[0] * depth_map.shape[1]
    depth_map_tensor = torch.tensor(depth_map, dtype=torch.float32)

    w = np.zeros((num_vertices, 3))  # Placeholder for normals
    
    # Placeholder dual variables
    q = torch.zeros(vertices.shape[0], dtype=torch.float32, requires_grad=True)
    p = torch.zeros(depth_map_tensor.numel(), dtype=torch.float32, requires_grad=True)

    optimizer = SGD([vertices, q, p], lr=0.1)

    # optimizer = SGD([vertices], lr=0.1)

    for iteration in range(max_iterations):
        optimizer.zero_grad()

        # Compute loss components
        Ldepth, Ltracking = compute_data_fidelity_term(mesh, depth_map_tensor.numpy())
        NLTGV2 = compute_smoothness_term(mesh, vertices.detach().numpy())
        # Convert loss components back to tensors
        Ldepth = torch.tensor(Ldepth, dtype=torch.float32, requires_grad=True)
        Ltracking = torch.tensor(Ltracking, dtype=torch.float32, requires_grad=True)
        NLTGV2 = torch.tensor(NLTGV2, dtype=torch.float32, requires_grad=True)

        # Calculate total loss
        L = Ldepth + lambda_val * (Ltracking + NLTGV2)
        L.requires_grad_()  # Ensure the total loss requires grad

        # Compute gradient of loss
        L.backward()

        with torch.no_grad():
            # Update vertices positions
            vertices -= tau * vertices.grad
            vertices += theta * (vertices - vertices.grad)
        
        optimizer.step()

        # Update mesh and compute normals
        updated_vertices = vertices.detach().cpu().numpy()
        mesh.vertices = o3d.utility.Vector3dVector(updated_vertices)
        w = compute_vertex_normals(updated_vertices, triangles)  # Update normals
        mesh.vertex_normals = o3d.utility.Vector3dVector(w)  # Set normals in mesh

        # Save the mesh at each iteration
        #file_path = f"./mesh_iteration_{iteration:03d}.ply"
        o3d.io.write_triangle_mesh(f"/home/kaiying/Downloads/Meshesti/mesh/iteration_mesh_{iteration:03d}.ply", mesh)
        #print(f"Saved mesh to {file_path}")

    return mesh


def primal_dual_optimization(mesh, depth_map, max_iterations=10, sigma=0.1, tau=0.1, theta=1.0, lambda_val=1.0):
    vertices = np.asarray(mesh.vertices)
    num_vertices = vertices.shape[0]
    num_pixels = depth_map.shape[0] * depth_map.shape[1]
    vξ = cp.asarray(1.0 / (vertices[:, 2] + 1e-5))  # Initial inverse depth
    vξ_old = vξ.copy()
    w = np.zeros((num_vertices, 2))  # Initialize normal vectors
    # Initialize dual variables
    y = cp.zeros(num_pixels, dtype=cp.float32)
    
    # Create directory for iteration meshes if it does not exist
    # os.makedirs('mesh', exist_ok=True)
    
    # Iteratively apply the primal-dual updates
    for iteration in range(max_iterations):
        # Dual Ascent Step
        A = compute_barycentric_coordinates(vertices, np.asarray(mesh.triangles), depth_map.shape)
        A_cupy = cp.sparse.csr_matrix(A)
        K = A_cupy
        K_bar = K @ vξ  # K_bar = K \bar{x}
        # K_bar = K_bar[:y.size]  # Ensure dimensions match
        # y = (y + sigma * K_bar) / (1 + sigma)
        print("y.shape: ", y.shape, "K_bar.shape: ", K_bar)
        # y = (y + sigma * K_bar) / (1 + sigma) 
        y = (y + sigma * K_bar) / cp.maximum(1, cp.abs(y + sigma * K_bar))
        # y = (y + sigma * K_bar[:num_vertices]) / (1 + sigma)
        
        # Primal Descent Step
        Ldepth, Ltracking = compute_data_fidelity_term(mesh, depth_map)
        NLTGV2 = compute_smoothness_term(mesh, w)
        L = compute_total_loss(Ldepth, Ltracking, NLTGV2, lambda_val)
        # print("Total Loss: ", L)
        K_star_y = K.T @ y  # K* y
        # vξ = (vξ - tau * K_star_y) / (1 + tau * L)
        # vξ = (vξ - tau * K_star_y[:num_vertices]) / (1 + tau * L)
        vξ = (vξ - tau * K_star_y[:num_vertices]) / (1 + tau)
        
        # Extra Gradient Step
        vξ_bar = vξ + theta * (vξ - vξ_old)
        vξ_old = vξ.copy()




        # Inside the primal descent step
        K_star_y = K.T @ y  # Compute K transpose y
        gradient = tau * K_star_y[:num_vertices]  # Assuming this slices correctly according to your dimensions

        print(f"Iteration {iteration}: Gradient max/min = {cp.max(gradient)}, {cp.min(gradient)}")

        vξ_new = vξ - gradient
        change = cp.linalg.norm(vξ_new - vξ)
        print(f"Iteration {iteration}: vξ change magnitude = {change}")

        if change < 0.1:
            print("Small or no change in vξ, consider adjusting tau or examining gradient calculations")
        # vξ = vξ_new





        
        # Update the mesh vertices with the optimized inverse depths
        #vertices[:, 2] = 1.0 / vξ.get()
        # Update the mesh vertices with the optimized inverse depths
        new_depths = 1.0 / vξ.get()
        if np.any(np.isnan(new_depths)) or np.any(np.isinf(new_depths)):
            print(f"Iteration {iteration}: Invalid depths detected, stopping optimization.")
            break
        # Clamping the depths to avoid extreme values
        # Normalize the new depths to be within 0 to 255
        # print(f"vertices[:3]: {vertices[:3]} \n new_depths[:3]_before normalization: {new_depths[:3]}")
        new_depths = (new_depths - np.min(new_depths)) / (np.max(new_depths) - np.min(new_depths)) * 255
        new_depths = np.clip(new_depths, 0, 255)
        
        # print("vertices[:3]: ", vertices[:3], "new_depths[:3]: ", new_depths[:3])
        print(f"vertices[:3]: {vertices[:3]} \n new_depths[:3]: {new_depths[:3]}")
        vertices[:, 2] = new_depths   # Normalize the new depths
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        
        for e in mesh.triangles:
            vi = vertices[e[0]]
            vj = vertices[e[1]]
            vk = vertices[e[2]]

            normal = np.cross(vj - vi, vk - vi)
            normal = normal / np.linalg.norm(normal)
            w[e[0]] = normal[:2]
            w[e[1]] = normal[:2]
            w[e[2]] = normal[:2]

        # Save the mesh at each iteration
        o3d.io.write_triangle_mesh(f"/home/kaiying/Downloads/Meshesti/mesh/iteration_mesh_{iteration:03d}.ply", mesh)
        #print(f"Iteration {iteration}: Total Loss = {L}")
        # Inside the loop of primal_dual_optimization function
        print(f"Iteration {iteration}: vξ changes = {cp.linalg.norm(vξ - vξ_old)}")
        print(f"K_bar max/min: {cp.max(K_bar)}, {cp.min(K_bar)}")
        print(f"Depth values changes: {np.max(new_depths) - np.min(new_depths)}")
        print(f"Iteration {iteration}: Ldepth = {Ldepth}, Ltracking = {Ltracking}, NLTGV2 = {NLTGV2}, Total Loss = {L}")

    return mesh



def main():
    # File paths
    depth_file = '/home/kaiying/Downloads/Meshesti/Depth_data_for_Task 4.pgm'
    mesh_file = '/home/kaiying/Downloads/Meshesti/mesh/initial_mesh_1.ply'

    # Load initial data
    time1 = time.time()
    depth_map, mesh = load_initial_data(depth_file, mesh_file)
    # Apply Primal-Dual Optimization
    optimized_mesh = primal_dual_optimization(mesh, depth_map)
    
    # Save the final optimized mesh
    o3d.io.write_triangle_mesh("/home/kaiying/Downloads/Meshesti/mesh/optimized_mesh.ply", optimized_mesh)

    # Compute terms for optimization
    #Ldepth, Ltracking = compute_data_fidelity_term(mesh, depth_map)
    #NLTGV2 = compute_smoothness_term(mesh)
    #total_loss = compute_total_loss(Ldepth, Ltracking, NLTGV2)

    # Print the total loss
    #print("Total Loss: ", total_loss)
    time2 = time.time()
    print("time cost: ", time2 - time1)

if __name__ == "__main__":
    main()
