import unittest

import numpy as np

from experiment import OBSTACLE_RADIUS, make_data, segment_clearance
from lomap import cosine_neighbors, pca_project, project_trajectory


class ProjectionTests(unittest.TestCase):
    def test_neighbors_use_cosine_similarity(self):
        bank = np.array([[30.0, 0.0], [1.0, 1.0], [-1.0, 0.0]])
        np.testing.assert_array_equal(cosine_neighbors(np.array([1.0, 0.0]), bank, 2), [0, 1])

    def test_pca_projects_onto_an_affine_subspace(self):
        neighbors = np.array([[-2.0, 5.0], [2.0, 5.0], [0.0, 4.9], [0.0, 5.1]])
        projected, rank, retained = pca_project(np.array([1.2, 8.0]), neighbors, 0.99)
        self.assertEqual(rank, 1)
        self.assertGreaterEqual(retained, 0.99)
        np.testing.assert_allclose(projected, [1.2, 5.0])

    def test_noise_alignment_and_hard_endpoints(self):
        bank = np.array([
            [[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            [[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]],
        ])
        sigma = 0.6
        prediction = np.array([[0.0, 0.0], [0.0, -2.0], [0.0, 0.0]])
        sample = np.array([[-1.0, 0.0], [0.0, -0.4], [1.0, 0.0]])
        neighbor_noise = np.ones((1, 3, 2))
        result = project_trajectory(sample, prediction, bank, neighbor_noise, sigma, 0.99)
        np.testing.assert_array_equal(result.neighbor_ids, [0])
        np.testing.assert_allclose(result.trajectory[1], [0.6, 1.4])
        np.testing.assert_array_equal(result.trajectory[[0, -1]], sample[[0, -1]])
        self.assertEqual(result.rank, 0)

    def test_collision_checks_between_waypoints(self):
        paths = np.array([[[-1.0, 0.0], [1.0, 0.0]], [[-1.0, 0.4], [1.0, 0.4]]])
        np.testing.assert_allclose(segment_clearance(paths), [0.0, 0.4])

    def test_clean_bank_and_references_are_feasible(self):
        data = make_data(20261120)
        for key in ("bank", "clean"):
            self.assertTrue(np.all(segment_clearance(data[key]) > OBSTACLE_RADIUS))


if __name__ == "__main__":
    unittest.main()
