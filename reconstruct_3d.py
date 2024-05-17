import numpy as np
import cv2
import open3d as o3d
import argparse

def load_data(reconstruction_path):
    tstamps = np.load(f"{reconstruction_path}/tstamps.npy")
    images = np.load(f"{reconstruction_path}/images.npy")
    disps = np.load(f"{reconstruction_path}/disps.npy")
    poses = np.load(f"{reconstruction_path}/poses.npy")
    intrinsics = np.load(f"{reconstruction_path}/intrinsics.npy")
    return tstamps, images, disps, poses, intrinsics

def disparity_to_point_cloud(disparity, intrinsic, baseline):
    """
    Convert a disparity map to a 3D point cloud using the intrinsic parameters and baseline.
    :param disparity: Disparity map
    :param intrinsic: Intrinsic camera parameters (fx, fy, cx, cy)
    :param baseline: Distance between the stereo cameras
    :return: 3D points
    """
    h, w = disparity.shape
    fx, fy, cx, cy = intrinsic

    Q = np.zeros((4, 4))
    Q[0, 0] = 1.0 / fx
    Q[0, 3] = -cx / fx
    Q[1, 1] = 1.0 / fy
    Q[1, 3] = -cy / fy
    Q[2, 3] = 1.0
    Q[3, 2] = -1.0 / baseline

    points = cv2.reprojectImageTo3D(disparity, Q)
    mask = disparity > 0  # Only use valid points
    points = points[mask]
    return points

def apply_pose_to_point_cloud(point_cloud, pose):
    """
    Apply the pose transformation to the point cloud.
    :param point_cloud: 3D points
    :param pose: 4x4 transformation matrix
    :return: Transformed 3D points
    """
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
    parser = argparse.ArgumentParser(description="3D Reconstruction from disparity maps and camera poses.")
    parser.add_argument("reconstruction_path", type=str, help="Path to the reconstruction data directory")
    parser.add_argument("--baseline", type=float, default=0.54, help="Baseline distance between stereo cameras (in meters)")
    args = parser.parse_args()

    main(args.reconstruction_path, args.baseline)
