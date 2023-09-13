from SphericalMatch.Toolbox_Spherical import distances_to_target, point_offset, rotate_radec_about_axis
from SphericalMatch.Toolbox_Spherical import great_circle_distance
import numpy as np


def test_angular_distance():
    # 1. Distance between the North and South celestial poles.
    assert np.isclose(distances_to_target((0, 90), np.array([0, -90])), 180.0)[0]

    # 2. Distance between a random point and the North celestial pole.
    random_point = (45, 45)
    expected_distance = 45  # DEC of North pole (90) minus DEC of random point (45)
    assert np.isclose(distances_to_target(random_point, np.array([(0, 90)])), expected_distance)[0]

    # 3. Distance between a random point and the South celestial pole.
    expected_distance = 135  # 180 minus previous expected distance
    assert np.isclose(distances_to_target(random_point, np.array([(0, -90)])), expected_distance)[0]

    # 4. Two points with one on RA = 359° and the other on RA = 1° but with the same declination.
    point1 = (359, 45)
    point2 = (1, 45)
    expected_distance = 1.4141776609521082  # Updated expected value based on the Haversine formula
    assert np.isclose(distances_to_target(point1, np.array([point2])), expected_distance)[0]

    # 5. Two points with one on RA = 0° and the other on RA = 360° but with the same declination.
    point1 = (0, 45)
    point2 = (360, 45)  # This should effectively be the same as 0° in RA
    expected_distance = 0
    assert np.isclose(distances_to_target(point1, np.array([point2])), expected_distance)[0]

    print("All tests passed!")


def test_point_offset():
    # Choose a random initial point
    initial_point = (45, 30)

    # Define the desired angular distances and directions for testing
    distances = [10, 20, 30, 40, 50]
    directions = [0, 45, 90, 135, 180]  # Sample directions in degrees

    for distance in distances:
        for direction in directions:
            # Generate the new point
            new_point = point_offset(initial_point, distance, direction)

            # Calculate the actual angular distance between the initial point and the new point
            computed_distance = distances_to_target(initial_point, np.array([new_point]))[0]

            # Check if the computed distance is close to the desired distance
            assert np.isclose(computed_distance, distance,
                              atol=1e-4), f"Expected {distance}, but got {computed_distance}"

    print("All tests passed!")


def test_point_offset_extreme_cases():
    # Generate a truly random initial point
    random_ra = np.random.uniform(0, 360)
    random_dec = np.random.uniform(-90, 90)
    random_point = (random_ra, random_dec)

    directions = [0, 45, 90, 135, 180, 225, 270, 315]  # Define the directions for testing

    # 1. Starting from the North celestial pole and moving in any direction.
    north_pole = (0, 90)
    for direction in directions:
        new_point = point_offset(north_pole, 10, direction)
        computed_distance = distances_to_target(north_pole, np.array([new_point]))[0]
        assert np.isclose(computed_distance, 10, atol=1e-4)

    # 2. Starting from the South celestial pole and moving in any direction.
    south_pole = (0, -90)
    for direction in directions:
        new_point = point_offset(south_pole, 10, direction)
        computed_distance = distances_to_target(south_pole, np.array([new_point]))[0]
        assert np.isclose(computed_distance, 10, atol=1e-4)

    # 3. Moving exactly 180° in distance from the random point in various directions.
    for direction in directions:
        opposite_point = point_offset(random_point, 180, direction)
        computed_distance = distances_to_target(random_point, np.array([opposite_point]))[0]
        assert np.isclose(computed_distance, 180, atol=1e-4)

    # 4. Moving a very small distance from the random point in various directions.
    tiny_distance = 1e-6
    for direction in directions:
        new_point = point_offset(random_point, tiny_distance, direction)
        computed_distance = distances_to_target(random_point, np.array([new_point]))[0]
        assert np.isclose(computed_distance, tiny_distance, atol=1e-8)

    # 5. Moving in a direction of exactly RA = 0° or RA = 360° from the random point.
    new_point_0 = point_offset(random_point, 10, 0)
    new_point_360 = point_offset(random_point, 10, 360)
    computed_distance_0 = distances_to_target(random_point, np.array([new_point_0]))[0]
    computed_distance_360 = distances_to_target(random_point, np.array([new_point_360]))[0]
    assert np.isclose(computed_distance_0, 10, atol=1e-4)
    assert np.isclose(computed_distance_360, 10, atol=1e-4)

    print(f"Random starting point: RA = {random_ra:.2f}, DEC = {random_dec:.2f}")
    print("All updated random extreme tests passed!")


def test_rotate_radec_about_axis():
    # 1. Test No Rotation
    ra, dec = 45, 45
    axis_ra, axis_dec = 30, 60
    theta = 0
    new_ra, new_dec = rotate_radec_about_axis(ra, dec, axis_ra, axis_dec, theta)
    assert np.isclose(ra, new_ra, atol=1e-5) and np.isclose(dec, new_dec, atol=1e-5)

    # 2. Test 180° Rotation about Pole
    ra, dec = 45, 0  # Point on the equator
    axis_ra, axis_dec = 0, 90  # North pole
    theta = 180
    new_ra, new_dec = rotate_radec_about_axis(ra, dec, axis_ra, axis_dec, theta)
    assert np.isclose((ra + 180) % 360, new_ra, atol=1e-5) and np.isclose(dec, new_dec, atol=1e-5)

    # 3. Test Perpendicular Rotation
    ra, dec = 120, 0  # Point on the great circle perpendicular to the axis (30, 60)
    axis_ra, axis_dec = 30, 60
    theta = 45
    new_ra, new_dec = rotate_radec_about_axis(ra, dec, axis_ra, axis_dec, theta)
    expected_distance = theta
    actual_distance = great_circle_distance(ra, dec, new_ra, new_dec)
    assert np.isclose(actual_distance, expected_distance, atol=1e-5)

    # 4. Test Rotation About Itself
    ra, dec = 30, 60
    axis_ra, axis_dec = ra, dec
    theta = 45
    new_ra, new_dec = rotate_radec_about_axis(ra, dec, axis_ra, axis_dec, theta)
    assert np.isclose(ra, new_ra, atol=1e-5) and np.isclose(dec, new_dec, atol=1e-5)

    # 5. Test Full Rotation
    ra, dec = 45, 45
    axis_ra, axis_dec = 30, 60
    theta = 360
    new_ra, new_dec = rotate_radec_about_axis(ra, dec, axis_ra, axis_dec, theta)
    assert np.isclose(ra, new_ra, atol=1e-5) and np.isclose(dec, new_dec, atol=1e-5)

    # 6. Test South Pole Rotation
    ra, dec = 45, 0
    axis_ra, axis_dec = 0, -90
    theta = 180
    new_ra, new_dec = rotate_radec_about_axis(ra, dec, axis_ra, axis_dec, theta)
    assert np.isclose((ra + 180) % 360, new_ra, atol=1e-5) and np.isclose(-dec, new_dec, atol=1e-5)

    # 7. Test Small Angle Rotation
    ra, dec = 120, 0  # Point on the great circle perpendicular to the axis (30, 60)
    axis_ra, axis_dec = 30, 60
    theta = 0.001
    new_ra, new_dec = rotate_radec_about_axis(ra, dec, axis_ra, axis_dec, theta)
    actual_distance = great_circle_distance(ra, dec, new_ra, new_dec)
    assert np.isclose(actual_distance, theta, atol=1e-5)

    # 8. Test Multiple Rotations
    ra, dec = 120, 0  # Point on the great circle perpendicular to the axis (30, 60)
    axis_ra, axis_dec = 30, 60
    theta1 = 45
    theta2 = 90
    new_ra1, new_dec1 = rotate_radec_about_axis(ra, dec, axis_ra, axis_dec, theta1)
    new_ra2, new_dec2 = rotate_radec_about_axis(new_ra1, new_dec1, axis_ra, axis_dec, theta2)
    new_ra_combined, new_dec_combined = rotate_radec_about_axis(ra, dec, axis_ra, axis_dec, theta1 + theta2)
    assert np.isclose(new_ra2, new_ra_combined, atol=1e-5) and np.isclose(new_dec2, new_dec_combined, atol=1e-5)

    print("All tests for rotate_radec_about_axis passed!")


if __name__ == "__main__":
    test_angular_distance()
    test_point_offset()
    for _ in range(5):
        test_point_offset_extreme_cases()
    test_rotate_radec_about_axis()
