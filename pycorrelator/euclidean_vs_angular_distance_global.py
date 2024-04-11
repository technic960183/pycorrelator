import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski
from .toolbox_spherical import distances_to_target, point_offset


'''
This script is designed to explore and visualize the relationship between Euclidean 
and angular distances on a spherical coordinate system, particularly when approximating
angular distances with Euclidean distances. The primary aim is to understand how these 
two types of distances correlate with each other for different angles (theta) emanating 
from a fixed point (the origin) on the sphere.

Two main types of plots are generated:

1. Euclidean vs. Angular Distance: This plot aims to show how the Euclidean distance 
   from the origin to a point varies with the angular distance for different values 
   of theta. 

2. Relative Error in Distance Measures: This plot focuses on quantifying the deviation 
   between the Euclidean and angular distances as a function of the angular distance 
   for different theta values. The deviation is expressed as a relative error.

Optional: The script can also calculate and plot the maximum possible relative error 
across a range of theta values, providing insights into the worst-case scenario of 
using Euclidean distance as a proxy for angular distance.

Note:
    The relative error for theta > 45 degrees is bounded by theta = 45 degrees. 
    For theta < 45, the peak of the relative error becomes larger when theta approaches 0, 
    and the angular distance of passing through the error of theta = 45 degrees becomes 
    further for larger theta (bounded by 90 degrees).
'''


if __name__ == '__main__':
    # Initialize
    theta_values = [5, 20, 45, 70, 85]
    origin = (180, 0)
    num_points = 100
    end_distance = 179

    # Toggle between logspace and linspace
    use_logspace = False  # Change this to False for linspace
    # To calculate the maximum relative error or not
    calculate_max_relative_error = False

    if use_logspace:
        distances = np.logspace(-2, np.log10(end_distance), num_points)
    else:
        distances = np.linspace(1, end_distance, num_points)

    # Initialize a 2-subplot figure
    fig, axs = plt.subplots(2, 1, figsize=(11, 7))

    # Loop through each theta value for plotting
    for i, theta in enumerate(theta_values):
        offset_points_theta = np.array([point_offset(origin, d, theta) for d in distances])
        minkowski_distances_theta = np.array([minkowski(point, origin, p=2) for point in offset_points_theta])
        angular_distances_theta = distances_to_target(origin, offset_points_theta)

        axs[0].plot(angular_distances_theta, minkowski_distances_theta, marker='.',
                    linestyle='-', linewidth=1, label=f'$\\theta={theta}$')
        axs[1].plot(angular_distances_theta, np.abs(
            (minkowski_distances_theta - angular_distances_theta) / angular_distances_theta),
            marker='.', linestyle='-', linewidth=1, label=f'$\\theta={theta}$')

    # Calcualte the possible maximum relative error regardless of theta
    if calculate_max_relative_error:
        theta_list = list(np.linspace(0, 90, 202))[1:-1]
        relative_error_list = []
        for i, theta in enumerate(theta_list):
            offset_points_theta = np.array([point_offset(origin, d, theta) for d in distances])
            minkowski_distances_theta = np.array([minkowski(point, origin, p=2) for point in offset_points_theta])
            relative_error_list.append((minkowski_distances_theta - distances) / distances)
        max_relative_error = np.max(np.array(relative_error_list), axis=0)
        axs[1].plot(distances, max_relative_error, marker='.', linestyle='-', linewidth=1, label=f'Maximum')

    # Configure the subplots
    if use_logspace:
        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        axs[1].set_xscale('log')
    axs[1].set_yscale('log')

    axs[0].set_xlabel('Angular Distance (degrees)')
    axs[0].set_ylabel('Euclidean Distance')
    axs[0].grid(True, which="both", ls="--")
    axs[0].legend()
    axs[0].set_title('Euclidean vs. Angular Distance centered at (180, 0)')

    axs[1].set_xlabel('Angular Distance (degrees)')
    axs[1].set_ylabel('Relative Error')
    axs[1].grid(True, which="both", ls="--")
    axs[1].legend()
    axs[1].set_title('Relative Error in Distance Measures')

    plt.tight_layout()
    plt.show()
