import numpy as np
from scipy.optimize import least_squares

from .utils import matrix_dot_product, get_transform_mat, transform


class LOAMOptimizer:
    def __init__(self, loss='huber', tolerance=1e-10):
        # Objective function for minimization
        self.loss = loss
        # Tolerance for termination by the change of the independent variables
        self.tolerance = tolerance

        self.edge_factors = None
        self.surface_factors = None

    def run(self, edge_factors, surface_factors, action):
        self.edge_factors = edge_factors
        self.surface_factors = surface_factors

        if action == "odometry":
            assert len(self.edge_factors) == 3, \
                "Invalid number of edge factors"
            assert len(self.surface_factors) == 4, \
                "Invalid number of surface factors"

            resid_function = self.resid_function_1
        elif action == "mapping":
            assert len(self.edge_factors) == 3, \
                "Invalid number of edge factors"
            assert len(self.surface_factors) == 3, \
                "Invalid number of surface factors"

            resid_function = self.resid_function_2
        else:
            raise NotImplementedError("Invalid action")

        # initialize pose in quaternion form
        init_pose = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
        opt_solution = least_squares(resid_function, init_pose,
                                     loss=self.loss, xtol=self.tolerance)
        T = get_transform_mat(opt_solution.x)
        print('    {} optimization error: {}\n'
              .format(action, np.mean(resid_function(opt_solution.x))))

        del self.edge_factors
        del self.surface_factors

        return T

    def resid_function_1(self, x):
        T = get_transform_mat(x)

        # Transform points in the local frame to the previous frame
        aligned_edges = transform(T, self.edge_factors[0])
        aligned_surfaces = transform(T, self.surface_factors[0])

        # Compute the distance from points to line
        nu = np.cross(aligned_edges - self.edge_factors[1],
                      aligned_edges - self.edge_factors[2])
        de = self.edge_factors[1] - self.edge_factors[2]
        edge_resid = \
            np.linalg.norm(nu, axis=1) / (np.linalg.norm(de, axis=1) + 1e-15)

        # Compute the distance from points to plane
        normal = np.cross(self.surface_factors[1] - self.surface_factors[2],
                          self.surface_factors[1] - self.surface_factors[3])
        normal /= (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-15)
        dh = matrix_dot_product(aligned_surfaces - self.surface_factors[1],
                                normal)
        surface_resid = np.abs(dh)

        # Concatenate the residuals
        resid = np.concatenate((edge_resid, surface_resid))
        return resid

    def resid_function_2(self, x):
        T = get_transform_mat(x)

        # Project points in the local frame to the global frame
        aligned_edges = transform(T, self.edge_factors[0])
        aligned_surfaces = transform(T, self.surface_factors[0])

        # Compute the distance from points to line
        nu = np.cross(aligned_edges - self.edge_factors[1],
                      aligned_edges - self.edge_factors[2])
        de = self.edge_factors[1] - self.edge_factors[2]
        edge_resid = \
            np.linalg.norm(nu, axis=1) / (np.linalg.norm(de, axis=1) + 1e-15)

        # Compute the distance from points to plane
        # aligned_surfaces -> (x, y, z)
        # self.surface_factors[1] -> (a, b, c)
        # self.surface_factors[2] -> (d, )
        # surface_resid -> |ax + by + cz + d|
        du = \
            matrix_dot_product(aligned_surfaces, self.surface_factors[1]) \
            + self.surface_factors[2].reshape((-1,))
        surface_resid = np.abs(du)

        resid = np.concatenate((edge_resid, surface_resid))
        return resid
