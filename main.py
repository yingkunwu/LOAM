import open3d as o3d
import numpy as np

from loader import LoadKITTIData
from core.laser_odometry import OdometryEstimator
from core.laser_mapping import LiDARMapper


def visualize(pcd):
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    data_path = 'data/KITTI/'
    sequence = '00'
    loader = LoadKITTIData(data_path, sequence)

    odometry_estimator = OdometryEstimator()
    lidar_mapper = LiDARMapper()

    for idx, (pcd, scan_start, scan_end) in enumerate(loader):
        print("========================================"
              "========================================")
        print("Processing Frame:", idx)
        print("----------------------------------------"
              "----------------------------------------")
        T, less_sharp_points, less_flat_points = \
            odometry_estimator.estimate(pcd, scan_start, scan_end)
        world = lidar_mapper.append_undistorted(
            T, pcd, less_sharp_points, less_flat_points)
        print("========================================"
              "========================================")

        print(np.asarray(world.points).shape)
