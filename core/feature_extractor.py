import numpy as np

from .utils import downsample_points


class FeatureExtractor:
    def __init__(self):
        # Number of segments to split every scan for feature detection
        self.N_SEGMENTS = 6
        # Number of sharp points to pick from point cloud
        self.PICKED_NUM_SHARP = 2
        # Number of less sharp points to pick from point cloud
        self.PICKED_NUM_LESS_SHARP = 20
        # Number of less sharp points to pick from point cloud
        self.PICKED_NUM_FLAT = 4
        # Threshold to split sharp and flat points
        self.CURVATURE_THRES = 0.1
        # Radius of points for curvature analysis
        self.FEATURES_REGION = 5

    def calc_curvatures(self, cloud):
        kernel = [1, 1, 1, 1, 1, -10, 1, 1, 1, 1, 1]
        assert len(kernel) == 2 * self.FEATURES_REGION + 1

        # Values outside the signal boundary have no effect after convlution
        curvatures = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, 'valid'), 0, cloud[:, :3])
        curvatures = np.sum(curvatures ** 2, axis=1)
        curvatures /= (
            np.linalg.norm(
                cloud[self.FEATURES_REGION:-self.FEATURES_REGION], axis=1)
            * self.FEATURES_REGION * 2
        )
        curvatures = np.pad(curvatures, self.FEATURES_REGION)
        return curvatures

    def extract(self, laser_cloud, scan_start, scan_end):
        keypoints_sharp = []
        keypoints_less_sharp = []
        keypoints_flat = []
        keypoints_less_flat = []

        cloud_label = np.zeros((laser_cloud.shape[0]))
        cloud_neighbors_picked = np.zeros((laser_cloud.shape[0]))

        cloud_curvatures = self.calc_curvatures(laser_cloud)
        cloud_neighbors_picked = self.remove_occluded(cloud_neighbors_picked,
                                                      laser_cloud)

        for i in range(scan_end.shape[0]):
            if i < 45:
                continue
            s = scan_start[i] + self.FEATURES_REGION
            e = scan_end[i] - self.FEATURES_REGION - 1
            if e - s < self.N_SEGMENTS:
                continue

            for j in range(self.N_SEGMENTS):
                sp = s + (e - s) * j // self.N_SEGMENTS
                ep = s + (e - s) * (j + 1) // self.N_SEGMENTS - 1
                segments_curvatures = cloud_curvatures[sp:ep + 1]
                sort_indices = np.argsort(segments_curvatures)

                largest_picked_num = 0
                for k in reversed(range(ep - sp + 1)):
                    ind = sort_indices[k] + sp

                    if cloud_neighbors_picked[ind] == 0 \
                            and cloud_curvatures[ind] > self.CURVATURE_THRES:
                        largest_picked_num += 1
                        if largest_picked_num <= self.PICKED_NUM_SHARP:
                            keypoints_sharp.append(laser_cloud[ind])
                            keypoints_less_sharp.append(laser_cloud[ind])
                            cloud_label[ind] = 2
                        elif largest_picked_num <= self.PICKED_NUM_LESS_SHARP:
                            keypoints_less_sharp.append(laser_cloud[ind])
                            cloud_label[ind] = 1
                        else:
                            break

                        cloud_neighbors_picked = \
                            self.mark_as_picked(laser_cloud,
                                                cloud_neighbors_picked, ind)

                smallest_picked_num = 0
                for k in range(ep - sp + 1):
                    ind = sort_indices[k] + sp

                    if cloud_neighbors_picked[ind] == 0 \
                            and cloud_curvatures[ind] < self.CURVATURE_THRES:
                        smallest_picked_num += 1
                        cloud_label[ind] = -1
                        keypoints_flat.append(laser_cloud[ind])

                        if smallest_picked_num >= self.PICKED_NUM_FLAT:
                            break

                        cloud_neighbors_picked = \
                            self.mark_as_picked(laser_cloud,
                                                cloud_neighbors_picked, ind)

                for k in range(sp, ep + 1):
                    if cloud_label[k] <= 0 \
                            and cloud_curvatures[k] < self.CURVATURE_THRES:
                        keypoints_less_flat.append(laser_cloud[k])

        keypoints_sharp = np.asarray(keypoints_sharp)
        keypoints_less_sharp = np.asarray(keypoints_less_sharp)
        keypoints_flat = np.asarray(keypoints_flat)
        keypoints_less_flat = np.asarray(keypoints_less_flat)

        keypoints_less_flat = downsample_points(keypoints_less_flat, 0.2)

        return keypoints_sharp, keypoints_less_sharp, \
            keypoints_flat, keypoints_less_flat

    def mark_as_picked(self, laser_cloud, cloud_neighbors_picked, ind):
        cloud_neighbors_picked[ind] = 1

        p1 = laser_cloud[ind - self.FEATURES_REGION + 1:
                         ind + self.FEATURES_REGION + 2, :3]
        p2 = laser_cloud[ind - self.FEATURES_REGION:
                         ind + self.FEATURES_REGION + 1, :3]

        sq_dist = np.sum((p1 - p2) ** 2, axis=1)

        for i in range(1, self.FEATURES_REGION + 1):
            if sq_dist[i + self.FEATURES_REGION] > 0.05:
                break
            cloud_neighbors_picked[ind + i] = 1

        for i in range(-self.FEATURES_REGION, 0):
            if sq_dist[i + self.FEATURES_REGION] > 0.05:
                break
            cloud_neighbors_picked[ind + i] = 1

        return cloud_neighbors_picked

    def remove_occluded(self, cloud_neighbors_picked, cloud):
        num_points = cloud.shape[0]
        start = self.FEATURES_REGION
        end = num_points - self.FEATURES_REGION - 1

        depth = np.linalg.norm(cloud[:, :3], axis=1)
        for j in range(start, end):
            prev_point = cloud[j - 1, :3]
            curr_point = cloud[j, :3]
            next_point = cloud[j + 1, :3]

            # distance's square between current point and next point
            diff_next = np.sum((curr_point - next_point) ** 2)
            if diff_next > 0.1:
                if depth[j] > depth[j + 1]:
                    depth_diff = np.linalg.norm(
                        next_point - curr_point * (depth[j + 1] / depth[j])
                    )
                    if depth_diff / depth[j + 1] < 0.1:
                        cloud_neighbors_picked[
                            j - self.FEATURES_REGION:j + 1] = 1
                else:
                    depth_diff = np.linalg.norm(
                        next_point * (depth[j] / depth[j + 1]) - curr_point
                    )
                    if depth_diff / depth[j] < 0.1:
                        cloud_neighbors_picked[
                            j + 1:j + self.FEATURES_REGION + 2] = 1

            # distance's square between current point and previous point
            diff_prev = np.sum((curr_point - prev_point) ** 2)

            if (diff_next > 0.0002 * depth[j] * depth[j]
                    and diff_prev > 0.0002 * depth[j] * depth[j]):
                cloud_neighbors_picked[j] = 1

        return cloud_neighbors_picked
