import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
from scipy.spatial import Delaunay

def read_depth_image(file_path):
    """Reads a depth image in PGM format."""
    return cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

def scale_depth_image(depth_image):
    """Scales the depth image to maintain original depth variations."""
    depth_min = np.min(depth_image)
    depth_max = np.max(depth_image)
    depth_image_scaled = (depth_image - depth_min) / (depth_max - depth_min) * 255.0
    return depth_image_scaled.astype(np.float32)

def normalize_depth_image(depth_image):
    """Normalizes the depth image to a range of 0 to 1."""
    depth_min = np.min(depth_image)
    depth_max = np.max(depth_image)
    return (depth_image - depth_min) / (depth_max - depth_min)

def convert_to_grayscale(image):
    """Converts a single-channel depth image to 8-bit grayscale."""
    grayscale_image = (image * 255).astype(np.uint8)
    return grayscale_image

def compute_normals(depth_image_normalized):
    """Computes normals from the depth image."""
    gradient_x = cv2.Sobel(depth_image_normalized, cv2.CV_64F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(depth_image_normalized, cv2.CV_64F, 0, 1, ksize=5)
    normals = np.dstack((-gradient_x, -gradient_y, np.ones_like(depth_image_normalized)))
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normals /= norms
    return normals

def detect_keypoints(depth_image_normalized, normals, max_points):
    """Detects keypoints in the depth image using weighted sampling based on normals."""
    height, width = depth_image_normalized.shape
    keypoints = []

    weights = np.linalg.norm(normals[:, :, :2], axis=2)
    weights = weights / np.sum(weights)
    probabilities = weights.flatten()

    indices = np.arange(height * width)
    sampled_indices = np.random.choice(indices, size=max_points, replace=False, p=probabilities)

    for idx in sampled_indices:
        y = idx // width
        x = idx % width
        keypoints.append((x, y, depth_image_normalized[y, x]))

    return np.array(keypoints)

def sample_keypoints(keypoints, depth_image_normalized, max_points):
    """Samples keypoints to a maximum number."""
    if len(keypoints) > max_points:
        indices = np.random.choice(len(keypoints), max_points, replace=False)
        keypoints = keypoints[indices]
    keypoints_3d = np.array([[kp[0], kp[1], depth_image_normalized[kp[1], kp[0]]] for kp in keypoints])
    return keypoints_3d

def create_edges(keypoints_3d):
    """Creates edges using Delaunay triangulation."""
    points_2d = keypoints_3d[:, :2]
    delaunay = Delaunay(points_2d)
    edges = set()
    for simplex in delaunay.simplices:
        edges.add(tuple(sorted([simplex[0], simplex[1]])))
        edges.add(tuple(sorted([simplex[1], simplex[2]])))
        edges.add(tuple(sorted([simplex[2], simplex[0]])))
    return np.array(list(edges))

def create_open3d_mesh(vertices, edges):
    """Creates an Open3D mesh from vertices and edges."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # Generate triangles using Delaunay triangulation on 2D points
    delaunay_triangles = Delaunay(vertices[:, :2])
    triangles = delaunay_triangles.simplices

    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh

    # Convert edges to triangles for Open3D mesh
    triangles = []
    for edge in edges:
        idx1, idx2 = edge
        if idx1 < len(vertices) - 1 and idx2 < len(vertices) - 1:
            next_idx1 = (idx1 + 1) % len(vertices)
            next_idx2 = (idx2 + 1) % len(vertices)
            if idx1 != next_idx1 and idx2 != next_idx2 and idx1 != idx2:
                triangles.append([idx1, idx2, next_idx1])
                triangles.append([idx2, next_idx2, next_idx1])

    valid_triangles = []
    for tri in triangles:
        if len(set(tri)) == 3 and all(i < len(vertices) for i in tri):
            valid_triangles.append(tri)

    mesh.triangles = o3d.utility.Vector3iVector(valid_triangles)
    mesh.compute_vertex_normals()
    return mesh


def save_open3d_mesh(mesh, filename):
    """Saves the Open3D mesh to a file."""
    if not os.path.exists('/home/kaiying/Downloads/Meshesti/mesh'):
        os.makedirs('/home/kaiying/Downloads/Meshesti/mesh')
    o3d.io.write_triangle_mesh(os.path.join('/home/kaiying/Downloads/Meshesti/mesh', filename), mesh)

def display_depth_image(depth_image_normalized):
    """Displays the normalized depth image."""
    plt.imshow(depth_image_normalized, cmap='gray')
    plt.title('Normalized Depth Map')
    plt.show()

def display_mesh(vertices, edges):
    """Displays the generated 3D mesh."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='r', marker='o')

    for edge in edges:
        point1 = vertices[edge[0]]
        point2 = vertices[edge[1]]
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], 'b-')

    plt.title('Initial 3D Mesh')
    plt.show()

def generate_initial_mesh():
    depth_image_path = '/home/kaiying/Downloads/Meshesti/Depth_data_for_Task 4.pgm'
    max_points = 1000  # Maximum number of keypoints to sample

    # Read and normalize depth image
    depth_image = read_depth_image(depth_image_path)
    #depth_image_normalized = normalize_depth_image(depth_image)
    depth_image_scaled = scale_depth_image(depth_image)

    # Compute normals
    #normals = compute_normals(depth_image_normalized)
    normals = compute_normals(depth_image_scaled)
    keypoints_3d = detect_keypoints(depth_image_scaled, normals, max_points)

    #keypoints_3d = detect_keypoints(depth_image_normalized, normals, max_points)

    #grayscale_image = convert_to_grayscale(depth_image_normalized)

    # Detect and sample keypoints
    #keypoints = detect_keypoints(grayscale_image)
    #keypoints_3d = sample_keypoints(keypoints, depth_image_normalized, max_points)

    # Create edges
    edges = create_edges(keypoints_3d)

    # Create and save Open3D mesh
    o3d_mesh = create_open3d_mesh(keypoints_3d, edges)
    save_open3d_mesh(o3d_mesh, 'initial_mesh_1.ply')

    # Display the Open3D mesh (optional)
    #o3d.visualization.draw_geometries([o3d_mesh])

def get_scaled_depth_map(depth_image_path):
    # Read and normalize depth image
    depth_image = read_depth_image(depth_image_path)
    #depth_image_normalized = normalize_depth_image(depth_image)
    depth_image_scaled = scale_depth_image(depth_image)
    return depth_image_scaled

    max_points = 1000

    normals = compute_normals(depth_image_scaled)
    keypoints_3d = detect_keypoints(depth_image_scaled, normals, max_points)

    # Create a sampled depth map
    sampled_depth_map = np.zeros_like(depth_image_scaled)
    for keypoint in keypoints_3d:
        x, y, z = int(keypoint[0]), int(keypoint[1]), keypoint[2]
        sampled_depth_map[y, x] = z

    #return sampled_depth_map

def sample_points_with_normals(mesh, num_samples=1000):
    # Convert mesh to point cloud
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_samples)
    # Compute normals for the point cloud
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return np.asarray(pcd.points), np.asarray(pcd.normals)

def create_depth_map_from_samples(samples, normals, depth_shape):
    # Create an empty depth map
    depth_map = np.zeros(depth_shape, dtype=np.float32)
    # Project the 3D points onto the 2D depth map
    for (x, y, z), (nx, ny, nz) in zip(samples, normals):
        ix, iy = int(x), int(y)
        if 0 <= ix < depth_shape[1] and 0 <= iy < depth_shape[0]:
            depth_map[iy, ix] = z
    return depth_map

def main():
    generate_initial_mesh()

if __name__ == "__main__":
    main()
