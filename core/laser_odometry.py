import numpy as np
import open3d as o3d

from .feature_extractor import FeatureExtractor
from .optimizer import LOAMOptimizer
from .utils import numpy2pcd


class OdometryEstimator:
    def __init__(self):
        self.inited = False
        self.last_less_sharp_points = None
        self.last_less_flat_points = None

        self.feature_extractor = FeatureExtractor()
        self.optimizer = LOAMOptimizer()

        self.DISTANCE_SQ_THRESHOLD = 25
        self.SCAN_VICINITY = 2.5

    def estimate(self, pcd, scan_start, scan_end):
        print("Feature Extraction:")
        sharp_points, less_sharp_points, flat_points, less_flat_points = \
            self.feature_extractor.extract(pcd, scan_start, scan_end)
        print("    Sharp count: {:5g}, Less sharp count: {:5g}"
              .format(len(sharp_points), len(less_sharp_points)))
        print("    Flat count: {:5g}, Less flat count: {:5g}"
              .format(len(flat_points), len(less_flat_points)))

        if not self.inited:
            self.inited = True
            T = np.eye(4)
        else:
            print("Find Correspondences for Odometry:")
            edge_corresp = self.find_edge_correspondences(sharp_points)
            surface_corresp = self.find_surface_correspondences(flat_points)

            T = self.optimizer.run(edge_corresp, surface_corresp, "odometry")

        self.last_less_sharp = numpy2pcd(less_sharp_points.copy())
        self.last_less_flat = numpy2pcd(less_flat_points.copy())

        self.last_less_sharp_scan = less_sharp_points[:, 3].copy()
        self.last_less_flat_scan = less_flat_points[:, 3].copy()

        return T, less_sharp_points, less_flat_points

    def find_edge_correspondences(self, sharp_points):
        edge_points = []
        edge_1 = []
        edge_2 = []

        last_less_sharp_points = np.asarray(self.last_less_sharp.points)
        corner_last_tree = o3d.geometry.KDTreeFlann(self.last_less_sharp)

        for i in range(len(sharp_points)):
            point_sel = sharp_points[i, :3]
            _, ind, dist = corner_last_tree.search_knn_vector_3d(point_sel, 1)
            closest_ind = -1
            min_ind2 = -1

            # find the closest point to the closest_ind in the last frame
            if dist[0] < self.DISTANCE_SQ_THRESHOLD:
                closest_ind = ind[0]
                closest_scan_id = self.last_less_sharp_scan[ind[0]]
                min_sq_dist2 = self.DISTANCE_SQ_THRESHOLD

                point_sq_dist_list = np.sum(
                    (last_less_sharp_points - point_sel) ** 2, axis=1)

                for j in range(closest_ind + 1,
                               len(self.last_less_sharp_scan)):
                    if self.last_less_sharp_scan[j] <= closest_scan_id:
                        continue
                    if self.last_less_sharp_scan[j] > \
                            closest_scan_id + self.SCAN_VICINITY:
                        break

                    point_sq_dist = point_sq_dist_list[j]
                    if point_sq_dist < min_sq_dist2:
                        min_sq_dist2 = point_sq_dist
                        min_ind2 = j

                for j in range(closest_ind - 1, -1, -1):
                    if self.last_less_sharp_scan[j] >= closest_scan_id:
                        continue
                    if self.last_less_sharp_scan[j] < \
                            closest_scan_id - self.SCAN_VICINITY:
                        break

                    if point_sq_dist < min_sq_dist2:
                        min_sq_dist2 = point_sq_dist
                        min_ind2 = j

                if min_ind2 >= 0:
                    edge_points.append(sharp_points[i, :3])
                    edge_1.append(last_less_sharp_points[closest_ind])
                    edge_2.append(last_less_sharp_points[min_ind2])

        print("    edge_points   : {:4g}, edge_1   : {:4g}, edge_2   : {:4g},"
              .format(len(edge_points), len(edge_1), len(edge_2)))

        edge_points = np.asarray(edge_points)
        edge_1 = np.asarray(edge_1)
        edge_2 = np.asarray(edge_2)

        return edge_points, edge_1, edge_2

    def find_surface_correspondences(self, flat_points):
        surface_points = []
        surface_1 = []
        surface_2 = []
        surface_3 = []

        last_less_flat_points = np.asarray(self.last_less_flat.points)
        surf_last_tree = o3d.geometry.KDTreeFlann(self.last_less_flat)

        for i in range(len(flat_points)):
            point_sel = flat_points[i, :3]
            _, ind, dist = surf_last_tree.search_knn_vector_3d(point_sel, 1)
            closest_ind = -1
            min_ind2 = -1
            min_ind3 = -1

            if dist[0] < self.DISTANCE_SQ_THRESHOLD:
                closest_ind = ind[0]
                closest_scan_id = self.last_less_flat_scan[ind[0]]
                min_sq_dist2 = self.DISTANCE_SQ_THRESHOLD
                min_sq_dist3 = self.DISTANCE_SQ_THRESHOLD

                point_sq_dist_list = np.sum(
                    (last_less_flat_points - point_sel) ** 2, axis=1)

                for j in range(closest_ind + 1,
                               len(self.last_less_flat_scan)):
                    if self.last_less_flat_scan[j] > \
                            closest_scan_id + self.SCAN_VICINITY:
                        break

                    point_sq_dist = point_sq_dist_list[j]
                    if self.last_less_flat_scan[j] <= \
                            closest_scan_id and point_sq_dist < min_sq_dist2:
                        min_sq_dist2 = point_sq_dist
                        min_ind2 = j
                    elif point_sq_dist < min_sq_dist3:
                        min_sq_dist3 = point_sq_dist
                        min_ind3 = j

                for j in range(closest_ind-1, -1, -1):
                    if self.last_less_flat_scan[j] < \
                            closest_scan_id - self.SCAN_VICINITY:
                        break

                    if self.last_less_flat_scan[j] >= \
                            closest_scan_id and point_sq_dist < min_sq_dist2:
                        min_sq_dist2 = point_sq_dist
                        min_ind2 = j
                    elif point_sq_dist < min_sq_dist3:
                        min_sq_dist3 = point_sq_dist
                        min_ind3 = j

                if min_ind2 >= 0 and min_ind3 >= 0:
                    surface_points.append(flat_points[i, :3])
                    surface_1.append(last_less_flat_points[closest_ind])
                    surface_2.append(last_less_flat_points[min_ind2])
                    surface_3.append(last_less_flat_points[min_ind3])

        print("    surface_points: {:4g}, surface_1: {:4g}, surface_2: {:4g}, "
              "surface_3: {:4g}"
              .format(len(surface_points), len(surface_1), len(surface_2),
                      len(surface_3)))

        surface_points = np.asarray(surface_points)
        surface_1 = np.asarray(surface_1)
        surface_2 = np.asarray(surface_2)
        surface_3 = np.asarray(surface_3)

        return surface_points, surface_1, surface_2, surface_3
