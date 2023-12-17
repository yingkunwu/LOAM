import numpy as np
import open3d as o3d

from .optimizer import LOAMOptimizer
from .utils import numpy2pcd, transform, downsample_points, downsample_pcd


class LiDARMapper:
    def __init__(self):
        self.init = False
        self.trans_world = np.eye(4)
        self.world = None
        self.all_edges = None
        self.all_surfaces = None

        self.optimizer = LOAMOptimizer()

        self.COVARIANCE_CNT = 5
        self.WORLD_VOXEL_SIZE = 0.2
        self.EDGE_VOXEL_SIZE = 0.2
        self.SURFACE_VOXEL_SIZE = 0.4

        self.CLOUD_DEPTH = 50
        self.CLOUD_WIDTH = 50
        self.CLOUD_HEIGHT = 10

    def get_pose(self):
        curr_pose = self.trans_world.copy()
        return curr_pose

    def append_undistorted(self, TL, cloud, edge_points, surface_points):
        if not self.init:
            self.init = True
            self.world = numpy2pcd(cloud)
            self.all_edges = numpy2pcd(edge_points)
            self.all_surfaces = numpy2pcd(surface_points)
        else:
            TW = TL @ self.trans_world

            edge_points = downsample_points(
                edge_points[:, :3], self.EDGE_VOXEL_SIZE)
            surface_points = downsample_points(
                surface_points[:, :3], self.SURFACE_VOXEL_SIZE)

            # transform current points to the global frame
            trans_cloud = transform(TW, cloud[:, :3])
            trans_edge_points = transform(TW, edge_points[:, :3])
            trans_surface_points = transform(TW, surface_points[:, :3])

            # create KDTree for fast search
            edges_kdtree = o3d.geometry.KDTreeFlann(self.all_edges)
            surfaces_kdtree = o3d.geometry.KDTreeFlann(self.all_surfaces)

            print("Find Correspondences for Mapping:")

            # search for points on edge lines in the surrounded points set
            edges, edge_A, edge_B = [], [], []
            for ind in range(len(edge_points)):
                point = trans_edge_points[ind]
                _, idx, dists = edges_kdtree.search_knn_vector_3d(
                    point, self.COVARIANCE_CNT)
                # check the distance to the largest nearest neighbors
                if dists[-1] < 1:
                    surrounded_pcd = self.all_edges.select_by_index(idx)
                    status, point_a, point_b = \
                        self.extract_from_edge(surrounded_pcd)
                    if status:
                        edges.append(point)
                        edge_A.append(point_a)
                        edge_B.append(point_b)

            # search for points on planar patches in the surrounded points set
            surfaces, surface_A, surface_B = [], [], []
            for ind in range(len(surface_points)):
                point = trans_surface_points[ind]
                _, idx, dists = surfaces_kdtree.search_knn_vector_3d(
                    point, self.COVARIANCE_CNT)
                # check the distance to the largest nearest neighbors
                if dists[-1] < 1:
                    surrounded_pcd = self.all_surfaces.select_by_index(idx)
                    status, coeff = self.extract_from_surface(surrounded_pcd)
                    if status:
                        surfaces.append(point)
                        surface_A.append(coeff[:3])
                        surface_B.append(coeff[3])

            print("    edges   : {:5g}, edge_A   : {:5g}, edge_B   : {:5g},"
                  .format(len(edges), len(edge_A), len(edge_B)))
            print("    surfaces: {:5g}, surface_A: {:5g}, surface_B: {:5g},"
                  .format(len(surfaces), len(surface_A), len(surface_B)))

            if len(edges) > 50 and len(surfaces) > 50:
                edge_factors = (np.vstack(edges),
                                np.vstack(edge_A),
                                np.vstack(edge_B))
                surface_factors = (np.vstack(surfaces),
                                   np.vstack(surface_A),
                                   np.vstack(surface_B))
                T = self.optimizer.run(edge_factors, surface_factors,
                                       "mapping")

                # update the global point clouds
                trans_cloud = transform(T, trans_cloud)
                trans_edge_points = transform(T, trans_edge_points)
                trans_surface_points = transform(T, trans_surface_points)

                centroid = np.mean(trans_cloud, axis=0)

                # remove points that are too far away from the centroid
                world_points = self._clip_points(
                    np.asarray(self.world.points), centroid)
                all_edges_points = self._clip_points(
                    np.asarray(self.all_edges.points), centroid)
                all_surfaces_points = self._clip_points(
                    np.asarray(self.all_surfaces.points), centroid)

                self.world = \
                    numpy2pcd(np.vstack(
                        [world_points, trans_cloud]
                    ))
                self.all_edges = \
                    numpy2pcd(np.vstack(
                        [all_edges_points, trans_edge_points]
                    ))
                self.all_surfaces = \
                    numpy2pcd(np.vstack(
                        [all_surfaces_points, trans_surface_points]
                    ))

                # optimize the global pose -> TW
                self.trans_world = T @ TW
            else:
                print("Not enough points for mapping")
                self.trans_world = TW

        # downsample point clouds
        self.world = \
            downsample_pcd(self.world, self.WORLD_VOXEL_SIZE)
        self.all_edges = \
            downsample_pcd(self.all_edges, self.EDGE_VOXEL_SIZE)
        self.all_surfaces = \
            downsample_pcd(self.all_surfaces, self.SURFACE_VOXEL_SIZE)

        return self.world

    def _clip_points(self, points, centroid):
        points = points[
            (points[:, 0] >= centroid[0] - self.CLOUD_DEPTH)
            & (points[:, 0] <= centroid[0] + self.CLOUD_DEPTH)
            & (points[:, 1] >= centroid[1] - self.CLOUD_WIDTH)
            & (points[:, 1] <= centroid[1] + self.CLOUD_WIDTH)
            & (points[:, 2] >= centroid[2] - self.CLOUD_HEIGHT)
            & (points[:, 2] <= centroid[2] + self.CLOUD_HEIGHT)
        ]
        return points

    def extract_from_edge(self, pcd):
        if not isinstance(pcd, o3d.geometry.PointCloud):
            assert False, "only support o3d.geometry.PointCloud"

        centroid, covariance = pcd.compute_mean_and_covariance()

        vals, vecs = np.linalg.eig(covariance)
        idx = vals.argsort()
        vals = vals[idx]
        vecs = vecs[:, idx]

        unit_direction = vecs[:, 2]

        if vals[2] > 3 * vals[1]:
            point_a = centroid + 0.1 * unit_direction
            point_b = centroid - 0.1 * unit_direction
            return True, point_a, point_b
        else:
            return False, None, None

    def extract_from_surface(self, pcd):
        if not isinstance(pcd, o3d.geometry.PointCloud):
            assert False, "only support o3d.geometry.PointCloud"

        points = np.asarray(pcd.points)
        b = -np.ones((self.COVARIANCE_CNT,))

        surf_normal = np.linalg.lstsq(points, b, rcond=None)[0]
        surf_norm = np.linalg.norm(surf_normal)
        coeff = np.append(surf_normal, 1.0) / surf_norm

        points_ = np.hstack((points, np.ones((points.shape[0], 1))))
        plane_residual = np.abs(points_ @ coeff.reshape(4, 1))

        if np.any(plane_residual > 0.2):
            return None, None
        else:
            return True, coeff
