import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

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

    targ_x, targ_y = [], []
    pred_x, pred_y = [], []
    world = None

    for idx, (pcd, scan_start, scan_end, pose) in enumerate(loader):
        if idx == 150:
            visualize(world)
            break

        targ_x.append(pose[0, -1])
        targ_y.append(pose[1, -1])

        print("========================================"
              "========================================")
        print("Processing Frame:", idx)
        print("----------------------------------------"
              "----------------------------------------")

        T, less_sharp_points, less_flat_points = \
            odometry_estimator.estimate(pcd, scan_start, scan_end)
        world = lidar_mapper.append_undistorted(
            T, pcd, less_sharp_points, less_flat_points)

        pred_pose = lidar_mapper.get_pose()
        pred_x.append(pred_pose[0, -1])
        pred_y.append(pred_pose[1, -1])

        print("Number of points in world LiDAR map:",
              np.asarray(world.points).shape)

        print("========================================"
              "========================================")

    # Plot the pose on the xy-plane
    plt.plot(targ_x, targ_y, label='target')
    plt.plot(pred_x, pred_y, label='estimated')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()
    plt.title('LiDAR Odometry')
    plt.savefig('result.png')
    plt.close()
