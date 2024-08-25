import tqdm
import open3d as o3d
import matplotlib.pyplot as plt

from loader import LoadKITTIData
from core.utils import numpy2pcd


def visualize(pcd):
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    data_path = 'data/KITTI/'
    sequence = '00'
    loader = LoadKITTIData(data_path, sequence)

    pred_x, pred_y = [], []

    world = o3d.geometry.PointCloud()
    for idx, (pcd, scan_start, scan_end, pose) in \
            tqdm.tqdm(enumerate(loader), total=len(loader)):
        pcd = numpy2pcd(pcd)
        pcd = pcd.transform(pose)
        # pcd = pcd.voxel_down_sample(voxel_size=0.2)

        world += pcd

        if idx == 30:
            world = world.voxel_down_sample(voxel_size=0.1)
            visualize(world)
            break

        pred_x.append(pose[0, -1])
        pred_y.append(pose[1, -1])

    # Plot the pose on the xy-plane
    plt.plot(pred_x, pred_y, label='estimated')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()
    plt.show()
