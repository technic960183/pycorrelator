import numpy as np
from scipy.spatial.distance import euclidean
from .utilities_spherical import point_offset


'''
This script provides a detailed analysis of the discrepancies between Euclidean 
and angular distances on a spherical coordinate system, particularly examining 
how these discrepancies occur for various declinations and angular distances.

The script has two primary functionalities:

1. Compute Relative Error:
   For a given declination and angular distance, the script calculates the maximum 
   relative error between the Euclidean and angular distances using the function:
      compute_error(declination, distance)
   This relative error highlights the deviation of the Euclidean approximation 
   from the actual angular distance. Though primarily intended for error 
   visualization, this function can also be utilized in other scripts, serving as 
   an API for determining relative errors based on specific declination and 
   angular distance inputs.

2. Visualize Errors: (Removed for brevity, see the original script in the backup branch)
   The script generates three main plots to enhance the understanding of these errors:

   a. Relative Error in Euclidean Distance vs. Declination (Top-left subplot):
      This plot shows how the relative error between Euclidean and angular distances 
      varies with declination for different angular distances. Initially, the relative 
      error grows slowly nearly as a constant. As declination increases, the error 
      follows the trend of the transformation:
          transformed value = (1 - cos(Dec)) / cos(Dec)
      A reference line representing this transformation is plotted to offer a baseline 
      comparison. This curve helps in illustrating how the error aligns with this 
      cosine transformation.

      Limits:
      -------
      - For minimal angular separations (close to 0°), the relative error follows a power-law:
          error ≈ 1.2694 * 10^(-5) * d^2 
        where d represents the angular distance in degrees.
      
      - Approaching an angular separation of 75°, the relative error aligns with:
          error ≈ (1 - cos(θ)) / cos(θ)
      
      These limiting behaviors illuminate the approximation error extremes across 
      angular separations, guiding modeling efforts for errors across all angles.

   b. Angle Corresponding to Max Relative Error vs. Declination (Bottom-left subplot):
      This plot identifies the direction, represented by θ (theta), where the maximum 
      discrepancy or relative error occurs for different declinations. θ is defined 
      as the direction in degrees counter-clockwise from the positive DEC axis when 
      viewed from the center of the celestial sphere.

   c. Fit of Relative Error vs. Angular Distance (Right subplot):
      Focusing on a near-zero declination, this subplot visualizes how the relative 
      error varies with different angular distances. It contrasts observed data 
      with a fitted curve, revealing the inherent relationship between relative 
      error and angular distance at this specific declination.


Usage:
- Execute the script to generate the three plots described above. (Removed for brevity)
- Import compute_error() to calculate the relative error for a given declination
    and angular distance.

Note:
At low declinations, the maximum relative error typically arises when moving 
in a diagonal direction around 45° from the DEC axis. This direction captures 
more of the sphere's curvature compared to strictly horizontal or vertical 
movements. As declination increases, the direction corresponding to the most 
pronounced error shifts, becoming predominantly horizontal (θ = 90°).
'''


def compute_error(declination, distance):
    '''
    Purpose: Compute the relative error in Euclidean distance given declination and angular distance.

    Parameters:
    - declination: float, the declination in degrees
    - distance: float, the angular distance in degrees

    Returns:
    - error: float, the computed relative error defined as (Euclidean - angular) / angular.
    '''

    theta_values = np.linspace(0, 90, 100)  # 100 sampling points for different directions
    max_error, _ = compute_max_relative_error(declination, distance, theta_values)

    return max_error

def compute_max_relative_error(dec, distances, theta_values):
    origin = (180, dec)
    offset_points_theta = np.array(point_offset(origin, distances, theta_values))
    euclidean_distances_theta = np.array([euclidean(origin, offset_point)
                                          for offset_point in offset_points_theta.T])
    relative_errors = np.abs((euclidean_distances_theta - distances) / distances)
    max_relative_error = np.max(relative_errors)
    angle_of_max_error = theta_values[np.argmax(relative_errors)]
    return max_relative_error, angle_of_max_error
