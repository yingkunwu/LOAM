import os
import numpy as np


class LoadKITTIData:
    def __init__(self, data_path, sequence):
        self.folder_path = os.path.join(
            data_path, 'odometry', 'sequences', sequence, 'velodyne')
        self.pcds_list = os.listdir(self.folder_path)
        self.pcds_list.sort()

        self.frame_idx = list(range(len(self.pcds_list)))

        pose_path = os.path.join(data_path, 'odometry', 'poses')
        self.poses = self._load_poses(pose_path, sequence)

        # Number of laser scans per frame
        self.NUM_SCANS = 64
        # Laser scan period
        self.SCAN_PERIOD = 0.1
        # Distance threshold to remove close points
        self.DISTANCE_THRES = 4

    def __iter__(self):
        self.count = 0
        return self

    def __len__(self):
        return len(self.pcds_list)

    def __next__(self):
        if self.count == len(self.frame_idx):
            raise StopIteration
        idx = self.frame_idx[self.count]
        self.count += 1

        path = os.path.join(self.folder_path, self.pcds_list[idx])
        pcd = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]

        # generate scan ids for each point based on pitch angle
        scan_ids = self._get_scan_ids(pcd)

        # calculate relative time for each point in a scan
        rel_time = self._get_rel_time(pcd)

        # append scan ids and relative time info to each point
        scan_info = scan_ids + self.SCAN_PERIOD * rel_time
        pcd = np.hstack((pcd, scan_info[:, np.newaxis]))

        # remove unreliable points
        pcd, scan_ids = self._remove_unreliable_points(pcd, scan_ids)

        # create index by the number of points in each laser scan
        scan_start = np.zeros(self.NUM_SCANS, dtype=int)
        scan_end = np.zeros(self.NUM_SCANS, dtype=int)
        _, elem_cnt = np.unique(scan_ids, return_counts=True)
        start = 0
        for ind, cnt in enumerate(elem_cnt):
            scan_start[ind] = start
            start += cnt
            scan_end[ind] = start

        sorted_ind = np.argsort(scan_ids, kind='stable')
        pcd = pcd[sorted_ind]

        return pcd, scan_start, scan_end, self.poses[idx]

    def _load_poses(self, pose_file, sequence):
        R_transform = np.array([
            [0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]
        ])

        pose_file = os.path.join(pose_file, sequence + '.txt')

        # Read and parse the poses
        poses = []
        try:
            with open(pose_file, 'r') as f:
                lines = f.readlines()
                if self.frame_idx is not None:
                    lines = [lines[i] for i in self.frame_idx]

                for line in lines:
                    T = np.fromstring(line, dtype=float, sep=' ')
                    T = T.reshape(3, 4)
                    T = np.vstack((T, [0, 0, 0, 1]))
                    T = R_transform @ T
                    poses.append(T)

        except FileNotFoundError:
            print('Ground truth poses are not available for sequence ' +
                  self.sequence + '.')

        return poses

    def _remove_unreliable_points(self, pcd, scan_ids):
        # remove points that are too close to the lidar
        dists = np.sum(pcd[:, :3] ** 2, axis=1)
        valid_mask1 = dists > self.DISTANCE_THRES ** 2

        # only keep points with scan id in a reasonable range [0, NUM_SCANS)
        valid_mask2 = np.logical_and(scan_ids >= 0, scan_ids < self.NUM_SCANS)

        valid_mask = np.logical_and(valid_mask1, valid_mask2)
        pcd = pcd[valid_mask]
        scan_ids = scan_ids[valid_mask]

        return pcd, scan_ids

    def _get_rel_time(self, pcd):
        start_ori = -np.arctan2(pcd[0, 1], pcd[0, 0])
        end_ori = -np.arctan2(pcd[-1, 1], pcd[-1, 0]) + 2 * np.pi
        if end_ori - start_ori > 3 * np.pi:
            end_ori -= 2 * np.pi
        elif end_ori - start_ori < np.pi:
            end_ori += 2 * np.pi

        half_passed = False
        num_points = pcd.shape[0]
        rel_time = np.zeros(num_points)

        for i in range(num_points):
            ori = -np.arctan2(pcd[i, 1], pcd[i, 0])
            if not half_passed:
                if ori < start_ori - np.pi / 2:
                    ori += 2 * np.pi
                elif ori > start_ori + np.pi * 3 / 2:
                    ori -= 2 * np.pi
                if ori - start_ori > np.pi:
                    half_passed = True
            else:
                ori += 2 * np.pi
                if ori < end_ori - np.pi * 3 / 2:
                    ori += 2 * np.pi
                elif ori > end_ori + np.pi / 2:
                    ori -= 2 * np.pi
            rel_time[i] = (ori - start_ori) / (end_ori - start_ori)
        return rel_time

    def _get_scan_ids(self, pcd):
        # calculate pitch of each lidar point
        depth = np.linalg.norm(pcd, axis=1)
        pitch = np.arcsin(pcd[:, 2] / depth)

        # LiDAR vfov:
        # +2 up to -24.8 down with 64 equally spaced angular subdivisions
        fov_down = -24.8 / 180.0 * np.pi
        fov = (abs(-24.8) + abs(2.0)) / 180.0 * np.pi
        scan_ids = (pitch + abs(fov_down)) / fov
        scan_ids *= self.NUM_SCANS
        scan_ids = np.floor(scan_ids).astype(np.int32)

        return scan_ids
