import numpy as np


def distances_to_target(target, points):
    """Compute the great-circle distances from a target point to a list of other points on a sphere.

    This function can also handles a single point as an input.

    Parameters
    ----------
    target : tuple
        (RA, DEC) of the target point.
    points : numpy.ndarray | tuple
        numpy array of shape (n, 2) where n is the number of points. Each row is (RA, DEC) for a point.
        Can also be a single point with shape (2,).

    Returns
    -------
    distances : numpy.ndarray
        Great-circle distances to the target point.
    """

    # If points represent a single point, reshape it
    if points.ndim == 1:
        points = points.reshape(1, -1)

    # Convert degrees to radians
    target_rad = np.radians(target)
    points_rad = np.radians(points)

    # Haversine formula
    delta_ra = points_rad[:, 0] - target_rad[0]
    delta_dec = points_rad[:, 1] - target_rad[1]
    a = np.sin(delta_dec/2.0)**2 + np.cos(target_rad[1]) * np.cos(points_rad[:, 1]) * np.sin(delta_ra/2.0)**2
    distances = 2 * np.arcsin(np.sqrt(a))

    # Convert back to degrees
    return np.degrees(distances)


def great_circle_distance(ra1, dec1, ra2, dec2):
    """Compute the great-circle distance between two points on a sphere using their right ascension and declination.

    Parameters
    ----------
    ra1 : float
        Right ascension of the first point in degrees.
    dec1 : float
        Declination of the first point in degrees.
    ra2 : float
        Right ascension of the second point in degrees.
    dec2 : float
        Declination of the second point in degrees.

    Returns
    -------
    distance : float
        Angular distance between the two points in degrees.
    """
    target = np.array([ra1, dec1])
    point = np.array([ra2, dec2])
    return distances_to_target(target, point)[0]


def point_offset(ra_dec, angular_distance, theta):
    """Give a point that is a given angular distance away from a specified point on the celestial sphere.

    Parameters
    ----------
    ra_dec : tuple
        (RA, DEC) in degrees for the initial point.
    angular_distance : float
        Distance in degrees to move from the initial point.
    theta : float
        Direction in degrees counter-clockwise from the positive DEC axis when viewed from the center of the celestial sphere.

    Returns
    -------
    new_point : tuple
        (RA, DEC) in degrees for the point after offset.
    
    Note
    ----
    The direction specified by theta is counter-clockwise when viewed from the center of the celestial sphere, looking outwards.
    If visualizing from a point above the North Celestial Pole, the direction will appear clockwise.
    """

    # Convert all angles to radians
    ra1, dec1 = np.radians(ra_dec)
    d = np.radians(angular_distance)
    theta = np.radians(theta)

    dec2 = np.arcsin(np.sin(dec1) * np.cos(d) + np.cos(dec1) * np.sin(d) * np.cos(theta))
    ra2 = ra1 + np.arctan2(np.sin(d) * np.sin(theta),
                           np.cos(dec1) * np.cos(d) - np.sin(dec1) * np.sin(d) * np.cos(theta))

    # Ensure ra2 is in [0, 2*pi] range
    ra2 = ra2 % (2 * np.pi)

    # Convert back to degrees
    ra2, dec2 = np.degrees([ra2, dec2])
    return ra2, dec2


def radec_to_cartesian(ra, dec):
    """Convert Right Ascension and Declination to Cartesian coordinates.

    Parameters
    ----------
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.

    Returns
    -------
    np.array
        Cartesian coordinates [x, y, z].
    """
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return np.array([x, y, z]).T


def cartesian_to_radec(cartesian_coords):
    """Convert Cartesian coordinates to Right Ascension and Declination.

    Parameters
    ----------
    cartesian_coords : np.array
        Array of Cartesian coordinates [x, y, z] SHOULD BE NORMALIZED.

    Returns
    -------
    tuple
        (RA, DEC) in degrees.
    """
    x, y, z = cartesian_coords.T
    if not np.allclose(np.linalg.norm(cartesian_coords, axis=-1), 1, atol=1e-9):
        raise ValueError("[x, y, z] must be a normalized vector")
    ra = np.degrees(np.arctan2(y, x)) % 360
    dec = np.degrees(np.arcsin(z))
    return ra, dec


def rodrigues_rotation(v, k, theta):
    """Rotate a vector using Rodrigues' rotation formula.

    Parameters
    ----------
    v : np.array
        Vector to be rotated.
    k : np.array
        Unit vector indicating the axis of rotation.
    theta : float
        Angle of rotation in degrees.

    Returns
    -------
    np.array
        Rotated vector.
    """
    theta_rad = np.radians(theta)
    v_rot = v * np.cos(theta_rad)
    v_rot += np.cross(k, v) * np.sin(theta_rad)
    v_rot += k * np.dot(v, k.T) * (1 - np.cos(theta_rad))
    return v_rot


def rotate_radec_about_axis(ra, dec, axis_ra, axis_dec, theta):
    """Rotate a point (or points) in celestial coordinates about a specified axis.

    Given a point (or an array of points) defined by its Right Ascension and Declination, 
    this function rotates it about an arbitrary axis (defined by its own RA and Dec) by a 
    specified angle.

    Parameters
    ----------
    ra : float or np.array
        Right Ascension of the point(s) to be rotated. Can be a single value or an array of values.
    dec : float or np.array
        Declination of the point(s) to be rotated. Can be a single value or an array of values.
    axis_ra : float
        Right Ascension of the rotation axis. Expected to be a scalar.
    axis_dec : float
        Declination of the rotation axis. Expected to be a scalar.
    theta : float
        Angle of rotation in degrees. Expected to be a scalar.

    Returns
    -------
    tuple | tuple[numpy.array]
        If `ra` and `dec` are scalars: Returns a tuple (rotated_RA, rotated_Dec) of scalar values.
        If `ra` and `dec` are arrays: Returns a tuple of arrays (rotated_RAs, rotated_Decs).
    """
    scalar = False
    if np.isscalar(ra):
        ra = np.array([ra])
        scalar = True
    if np.isscalar(dec):
        dec = np.array([dec])
        scalar = True
    if np.isscalar(axis_ra):
        axis_ra = np.array([axis_ra])
    if np.isscalar(axis_dec):
        axis_dec = np.array([axis_dec])
    if not np.isscalar(theta):
        raise ValueError("theta must be a scalar")
    v = radec_to_cartesian(ra, dec)
    k = radec_to_cartesian(axis_ra, axis_dec)
    v_rot = rodrigues_rotation(v, k, theta)
    rotated = cartesian_to_radec(v_rot)
    if scalar:
        return rotated[0][0], rotated[1][0]
    return rotated


def generate_random_point(n, seed=None):
    """Generate random points in Right Ascension and Declination uniformly distributed on the celestial sphere.

    Parameters
    ----------
    n : int
        Number of random points to generate.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    tuple
        (RA, DEC) arrays in degrees.
    """
    np.random.seed(seed)
    points = np.random.randn(3, n)
    norms = np.linalg.norm(points, axis=0)
    points /= norms
    ra, dec = cartesian_to_radec(points.T)
    return ra, dec
