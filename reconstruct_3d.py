import numpy as np
import cv2
import open3d as o3d

def load_data(reconstruction_path):
    tstamps = np.load(f"reconstructions/{reconstruction_path}/tstamps.npy")
    images = np.load(f"reconstructions/{reconstruction_path}/images.npy")
    disps = np.load(f"reconstructions/{reconstruction_path}/disps.npy")
    poses = np.load(f"reconstructions/{reconstruction_path}/poses.npy")
    intrinsics = np.load(f"reconstructions/{reconstruction_path}/intrinsics.npy")
    return tstamps, images, disps, poses, intrinsics

def disparity_to_point_cloud(disparity, intrinsic, baseline):
    h, w = disparity.shape
    f = intrinsic[0, 0]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    Q = np.zeros((4, 4))
    Q[0, 0] = 1
    Q[0, 3] = -cx
    Q[1, 1] = 1
    Q[1, 3] = -cy
    Q[2, 3] = f
    Q[3, 2] = 1 / baseline

    points = cv2.reprojectImageTo3D(disparity, Q)
    mask = disparity > 0  # Only use valid points
    points = points[mask]
    return points

def apply_pose_to_point_cloud(point_cloud, pose):
    point_cloud_hom = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
    point_cloud_transformed = (pose @ point_cloud_hom.T).T[:, :3]
    return point_cloud_transformed

def main(reconstruction_path, baseline):
    # Load the data
    tstamps, images, disps, poses, intrinsics = load_data(reconstruction_path)

    # Generate and align point clouds
    aligned_point_clouds = []
    for i, disp in enumerate(disps):
        point_cloud = disparity_to_point_cloud(disp, intrinsics[i], baseline)
        aligned_point_cloud = apply_pose_to_point_cloud(point_cloud, poses[i])
        aligned_point_clouds.append(aligned_point_cloud)

    # Combine all point clouds into a single point cloud
    all_points = np.vstack(aligned_point_clouds)
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(all_points)

    # Estimate normals and create mesh
    point_cloud_o3d.estimate_normals()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud_o3d, depth=9)

    # Visualize the point cloud and mesh
    o3d.visualization.draw_geometries([point_cloud_o3d], window_name='Point Cloud')
    o3d.visualization.draw_geometries([mesh], window_name='Mesh')

    # Save the point cloud and mesh
    o3d.io.write_point_cloud("reconstruction.ply", point_cloud_o3d)
    o3d.io.write_triangle_mesh("reconstruction_mesh.ply", mesh)

if __name__ == "__main__":
    reconstruction_path = 'your_reconstruction_path'  # Change this to your path
    baseline = 0.54  # Example baseline, in meters
    main(reconstruction_path, baseline)
