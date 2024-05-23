import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
import numpy as np
from pycorrelator import distances_to_target, point_offset, rotate_radec_about_axis
from pycorrelator import great_circle_distance


class TestAngularDistance(unittest.TestCase):

    # 1. Distance between the North and South celestial poles.
    def test_distance_north_south_poles(self):
        self.assertTrue(np.isclose(distances_to_target((0, 90), np.array([0, -90])), 180.0)[0])

    # 2. Distance between a random point and the North celestial pole.
    def test_distance_random_point_north_pole(self):
        random_point = (45, 45)
        expected_distance = 45
        self.assertTrue(np.isclose(distances_to_target(random_point, np.array([(0, 90)])), expected_distance)[0])

    # 3. Distance between a random point and the South celestial pole.
    def test_distance_random_point_south_pole(self):
        random_point = (45, 45)
        expected_distance = 135
        self.assertTrue(np.isclose(distances_to_target(random_point, np.array([(0, -90)])), expected_distance)[0])

    # 4. Two points with one on RA = 359° and the other on RA = 1° but with the same declination.
    def test_distance_near_wraparound_points(self):
        point1 = (359, 45)
        point2 = (1, 45)
        expected_distance = 1.4141776609521082
        self.assertTrue(np.isclose(distances_to_target(point1, np.array([point2])), expected_distance)[0])

    # 5. Two points with one on RA = 0° and the other on RA = 360° but with the same declination.
    def test_distance_wraparound_points(self):
        point1 = (0, 45)
        point2 = (360, 45)
        expected_distance = 0
        self.assertTrue(np.isclose(distances_to_target(point1, np.array([point2])), expected_distance)[0])


class TestPointOffset(unittest.TestCase):

    def setUp(self):
        self.initial_point = (45, 30)
        self.distances = [10, 20, 30, 40, 50]
        self.directions = [0, 45, 90, 135, 180]

    def test_point_offsets(self):
        for distance in self.distances:
            for direction in self.directions:
                new_point = point_offset(self.initial_point, distance, direction)
                computed_distance = distances_to_target(self.initial_point, np.array([new_point]))[0]
                self.assertTrue(np.isclose(computed_distance, distance, atol=1e-4))


class TestPointOffsetExtremeCases(unittest.TestCase):

    def setUp(self):
        self.directions = [0, 45, 90, 135, 180, 225, 270, 315]
        self.random_ra = np.random.uniform(0, 360)
        self.random_dec = np.random.uniform(-90, 90)
        self.random_point = (self.random_ra, self.random_dec)

    def check_distance(self, point1, point2, expected_distance):
        computed_distance = distances_to_target(point1, np.array([point2]))[0]
        self.assertTrue(np.isclose(computed_distance, expected_distance, atol=1e-4))

    # 1. Starting from the North celestial pole and moving in any direction.
    def test_north_pole_offsets(self):
        north_pole = (0, 90)
        for direction in self.directions:
            new_point = point_offset(north_pole, 10, direction)
            self.check_distance(north_pole, new_point, 10)

    # 2. Starting from the South celestial pole and moving in any direction.
    def test_south_pole_offsets(self):
        south_pole = (0, -90)
        for direction in self.directions:
            new_point = point_offset(south_pole, 10, direction)
            self.check_distance(south_pole, new_point, 10)

    # 3. Moving exactly 180° in distance from the random point in various directions.
    def test_opposite_point_offsets(self):
        for direction in self.directions:
            opposite_point = point_offset(self.random_point, 180, direction)
            self.check_distance(self.random_point, opposite_point, 180)

    # 4. Moving a very small distance from the random point in various directions.
    def test_tiny_distance_offsets(self):
        tiny_distance = 1e-6
        for direction in self.directions:
            new_point = point_offset(self.random_point, tiny_distance, direction)
            self.check_distance(self.random_point, new_point, tiny_distance)

    # 5. Moving in a direction of exactly RA = 0° or RA = 360° from the random point.
    def test_wraparound_direction_offsets(self):
        new_point_0 = point_offset(self.random_point, 10, 0)
        new_point_360 = point_offset(self.random_point, 10, 360)
        self.check_distance(self.random_point, new_point_0, 10)
        self.check_distance(self.random_point, new_point_360, 10)


class TestRotateRADEC(unittest.TestCase):

    # 1. Test No Rotation
    def test_no_rotation(self):
        ra, dec = 45, 45
        axis_ra, axis_dec = 30, 60
        theta = 0
        new_ra, new_dec = rotate_radec_about_axis(ra, dec, axis_ra, axis_dec, theta)
        self.assertTrue(np.isclose(ra, new_ra, atol=1e-5) and np.isclose(dec, new_dec, atol=1e-5))

    # 2. Test 180° Rotation about the pole
    def test_rotation_about_pole(self):
        ra, dec = 45, 0
        axis_ra, axis_dec = 0, 90
        theta = 180
        new_ra, new_dec = rotate_radec_about_axis(ra, dec, axis_ra, axis_dec, theta)
        self.assertTrue(np.isclose((ra + 180) % 360, new_ra, atol=1e-5) and np.isclose(dec, new_dec, atol=1e-5))

    # 3. Test Perpendicular Rotation
    def test_perpendicular_rotation(self):
        ra, dec = 120, 0
        axis_ra, axis_dec = 30, 60
        theta = 45
        new_ra, new_dec = rotate_radec_about_axis(ra, dec, axis_ra, axis_dec, theta)
        expected_distance = theta
        actual_distance = great_circle_distance(ra, dec, new_ra, new_dec)
        self.assertTrue(np.isclose(actual_distance, expected_distance, atol=1e-5))

    # 4. Test Rotation About Itself
    def test_rotation_about_itself(self):
        ra, dec = 30, 60
        axis_ra, axis_dec = ra, dec
        theta = 45
        new_ra, new_dec = rotate_radec_about_axis(ra, dec, axis_ra, axis_dec, theta)
        self.assertTrue(np.isclose(ra, new_ra, atol=1e-5) and np.isclose(dec, new_dec, atol=1e-5))

    # 5. Test Full Rotation
    def test_full_rotation(self):
        ra, dec = 45, 45
        axis_ra, axis_dec = 30, 60
        theta = 360
        new_ra, new_dec = rotate_radec_about_axis(ra, dec, axis_ra, axis_dec, theta)
        self.assertTrue(np.isclose(ra, new_ra, atol=1e-5) and np.isclose(dec, new_dec, atol=1e-5))

    # 6. Test South Pole Rotation
    def test_south_pole_rotation(self):
        ra, dec = 45, 0
        axis_ra, axis_dec = 0, -90
        theta = 180
        new_ra, new_dec = rotate_radec_about_axis(ra, dec, axis_ra, axis_dec, theta)
        self.assertTrue(np.isclose((ra + 180) % 360, new_ra, atol=1e-5) and np.isclose(-dec, new_dec, atol=1e-5))

    # 7. Test Small Angle Rotation
    def test_small_angle_rotation(self):
        ra, dec = 120, 0
        axis_ra, axis_dec = 30, 60
        theta = 0.001
        new_ra, new_dec = rotate_radec_about_axis(ra, dec, axis_ra, axis_dec, theta)
        actual_distance = great_circle_distance(ra, dec, new_ra, new_dec)
        self.assertTrue(np.isclose(actual_distance, theta, atol=1e-5))

    # 8. Test Multiple Rotations
    def test_multiple_rotations(self):
        ra, dec = 120, 0
        axis_ra, axis_dec = 30, 60
        theta1 = 45
        theta2 = 90
        new_ra1, new_dec1 = rotate_radec_about_axis(ra, dec, axis_ra, axis_dec, theta1)
        new_ra2, new_dec2 = rotate_radec_about_axis(new_ra1, new_dec1, axis_ra, axis_dec, theta2)
        new_ra_combined, new_dec_combined = rotate_radec_about_axis(ra, dec, axis_ra, axis_dec, theta1 + theta2)
        self.assertTrue(np.isclose(new_ra2, new_ra_combined, atol=1e-5)
                        and np.isclose(new_dec2, new_dec_combined, atol=1e-5))


if __name__ == '__main__':
    unittest.main()
