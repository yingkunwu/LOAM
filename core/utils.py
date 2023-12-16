import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


def numpy2pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    return pcd


def get_transform_mat(pose):
    r = Rotation.from_euler('zyx', pose[3:], degrees=False)
    r = r.as_matrix()

    t = pose[:3]

    trans_mat = np.eye(4)
    trans_mat[:3, :3] = r
    trans_mat[:3, 3] = t

    return trans_mat


def transform(T, points):
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = (T @ points_homogeneous.T).T
    transformed_points = transformed_points[:, :3]

    return transformed_points


def matrix_dot_product(A, B):
    """
    Calculates the dot product between two matrices A and B.
    Returns array like:
    [   np.dot(A[0], B[0]),
        np.dot(A[1], B[1]),
        np.dot(A[2], B[2]),
        ...
        np.dot(A[M - 1], B[M - 1])  ]

    Parameters:
        A (numpy.ndarray): [M, 3]
        B (numpy.ndarray): [M, 3]

    Returns:
        The dot product of A and B (numpy.ndarray): [M, 1]
    """
    assert A.shape == B.shape
    return np.einsum('ij,ij->i', A, B)


def downsample_points(points, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downpcd = pcd.voxel_down_sample(voxel_size)

    return np.asarray(downpcd.points)


def downsample_pcd(pcd, voxel_size):
    return pcd.voxel_down_sample(voxel_size)
