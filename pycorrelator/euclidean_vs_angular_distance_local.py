import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import expit as sigmoid_function
from scipy.spatial.distance import euclidean
from scipy.stats import linregress
from .toolbox_spherical import point_offset

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

2. Visualize Errors:
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
- Execute the script to generate the three plots described above.
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


def blended_error(theta, a, b, d):
    '''
    Note: This function is not used in the script, but it is included here for reference.
    '''
    weight = sigmoid_function(a * (theta - b * d))
    power_law_term = 1.2694230e-05 * d**2
    cosine_term = (1 - np.cos(np.deg2rad(theta))) / np.cos(np.deg2rad(theta))
    return weight * power_law_term + (1 - weight) * cosine_term


def compute_max_relative_error(dec, distances, theta_values):
    origin = (180, dec)
    offset_points_theta = np.array(point_offset(origin, distances, theta_values))
    euclidean_distances_theta = np.array([euclidean(origin, offset_point)
                                          for offset_point in offset_points_theta.T])
    relative_errors = np.abs((euclidean_distances_theta - distances) / distances)
    max_relative_error = np.max(relative_errors)
    angle_of_max_error = theta_values[np.argmax(relative_errors)]
    return max_relative_error, angle_of_max_error


if __name__ == '__main__':
    declinations = np.geomspace(5e-5, 75, 40)
    theta_values = np.linspace(0, 90, 50)
    distances = [0.1, 1, 5]

    fig = plt.figure(figsize=(11, 7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.6], height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, 0])  # Top-left
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # Bottom-left
    ax3 = fig.add_subplot(gs[:, 1])  # Entire right column

    min_error = []
    for d in distances:
        max_errors = []
        angles_of_max_errors = []
        for dec in declinations:
            max_error, angle_of_max_error = compute_max_relative_error(dec, d, theta_values)
            max_errors.append(max_error)
            angles_of_max_errors.append(angle_of_max_error)
        min_error.append(min(max_errors))

        ax1.loglog(declinations, max_errors, marker='o', linestyle='-', label=f'Distance = {d}°')
        ax2.semilogx(declinations, angles_of_max_errors, marker='o', linestyle='-', label=f'Distance = {d}°')

    min_error = min(min_error)
    cos_ref = np.cos(np.deg2rad(declinations))
    cos_ref = (1 - cos_ref) / cos_ref
    ax1.loglog(
        declinations[cos_ref > min_error],
        cos_ref[cos_ref > min_error],
        'k--', label='Transformed cos(Dec)')

    # Max Relative Error plot configurations
    ax1.set_xlabel('Declination (degrees)')
    ax1.set_ylabel('Maximum Relative Euclidean Error')
    ax1.set_title('Relative Error in Euclidean Distance vs. Declination')
    ax1.legend()
    ax1.grid(True, which="both", ls="--")

    # Angle of Max Relative Error plot configurations
    ax2.set_xlabel('Declination (degrees)')
    ax2.set_ylabel('Theta (degrees) at Maximum Error')
    ax2.set_title('Angle Corresponding to Max Relative Error vs. Declination')
    ax2.grid(True, which="both", ls="--")

    small_dec = 1e-5
    distances_ax3 = np.geomspace(1e-2, 10, 100)
    errors_at_small_dec = []

    for d in distances_ax3:
        max_error, _ = compute_max_relative_error(small_dec, d, theta_values)
        errors_at_small_dec.append(max_error)

    log_distances = np.log10(distances_ax3)
    log_errors = np.log10(errors_at_small_dec)
    slope, intercept, _, _, _ = linregress(log_distances, log_errors)
    a = 10**intercept  # Result: 1.2694230e-05
    b = slope         # Result: 1.9999953

    def fitted_power_law(x, a, b):
        return a * x**b

    predicted_errors = fitted_power_law(distances_ax3, a, b)

    ax3.loglog(distances_ax3, errors_at_small_dec, 'o-', label=f'Relative Error at Dec = {small_dec}')
    ax3.loglog(distances_ax3, predicted_errors, 'r--', label=f'Fitted Curve: y = {a:.4e} x^{b:.2f}')
    ax3.set_xlabel('Angular Distance (degrees)')
    ax3.set_ylabel('Relative Error at Dec close to 0')
    ax3.set_title('Limiting Behavior of Relative Error')
    ax3.grid(True, which="both", ls="--")
    ax3.legend()

    plt.tight_layout()
    plt.show()
