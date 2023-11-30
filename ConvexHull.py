import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull


def find_y_range_for_x(x, hull_points):  # Should be optimized
    y_values = []
    for i in range(len(hull_points)):
        x1, y1 = hull_points[i]
        x2, y2 = hull_points[(i + 1) % len(hull_points)]

        if (x1 <= x <= x2) or (x2 <= x <= x1):
            if x2 - x1 == 0:
                y = y1
            else:
                slope = (y2 - y1) / (x2 - x1)
                y = slope * (x - x1) + y1
            y_values.append(y)

    if y_values:
        return min(y_values), max(y_values)
    else:
        return None, None


def inner_grid_points(input_points):
    """
    Given a set of input points, return the list of grid points inside the convex hull.

    Parameters:
    - input_points: numpy array of shape (N, 2) where N is the number of points.

    Returns:
    - List of tuples where each tuple is an (x, y) coordinate of a grid point inside the convex hull.
    """
    if input_points.shape[0] < 3:
        return []

    # Compute convex hull
    hull = ConvexHull(input_points)

    # Determine grid points inside the convex hull
    grid_points_inside_hull = []
    min_x, max_x = int(input_points[:, 0].min()), int(input_points[:, 0].max())
    for x in range(min_x, max_x + 1):
        y_min, y_max = find_y_range_for_x(x, input_points[hull.vertices])
        if y_min is not None and y_max is not None:
            for y in range(int(np.ceil(y_min)), int(np.floor(y_max)) + 1):
                grid_points_inside_hull.append((x, y))

    return grid_points_inside_hull


def inner_grid_points_margin(input_points):
    """
    Given a set of input points, return the list of grid points inside the convex hull and the margin in y-axis.

    Parameters:
    - input_points: numpy array of shape (N, 2) where N is the number of points.

    Returns:
    - List of tuples where each tuple is an (x, y) coordinate of a grid point inside the convex hull and the margin.
    """
    if input_points.shape[0] < 3:
        return []

    # Compute convex hull
    hull = ConvexHull(input_points)

    # Determine grid points inside the convex hull
    grid_points_inside_hull = []
    min_x, max_x = int(input_points[:, 0].min()), int(input_points[:, 0].max())
    for x in range(min_x, max_x + 1):
        y_min, y_max = find_y_range_for_x(x, input_points[hull.vertices])
        if y_min is not None and y_max is not None:
            for y in range(int(np.floor(y_min)), int(np.ceil(y_max)) + 1):
                grid_points_inside_hull.append((x, y))

    return grid_points_inside_hull


if __name__ == "__main__":
    # Set up the initial plotting window
    plt.figure(figsize=(10, 6))
    plt.ion()  # Turn on interactive mode for real-time plotting

    while True:
        current_seed = np.random.randint(0, 1e5)  # Random seed
        np.random.seed(current_seed)

        # Randomly determine the number of points between 10 and 30
        num_points = np.random.randint(10, 31)

        # Generate sample points
        neighbor_Ra = np.random.randint(0, 100, num_points)
        neighbor_Dec = np.random.randint(0, 100, num_points)
        points = np.array(list(zip(neighbor_Ra, neighbor_Dec)))

        # Determine grid points inside the convex hull using the inner_grid_points function
        grid_points_inside_hull = inner_grid_points(points)

        # Update the plot
        plt.clf()  # Clear the current figure
        plt.plot(points[:, 0], points[:, 1], 'o')
        if len(points) > 2:  # Only plot the convex hull if there are more than 2 points
            hull = ConvexHull(points)
            plt.plot(np.append(points[hull.vertices, 0], points[hull.vertices[0], 0]),
                     np.append(points[hull.vertices, 1], points[hull.vertices[0], 1]), 'r--', lw=2)
        grid_points_inside_hull = np.array(grid_points_inside_hull)
        x = grid_points_inside_hull[:, 0]
        y = grid_points_inside_hull[:, 1]
        plt.xticks(np.arange(int(min(x)), int(max(x)) + 1, 1))
        plt.yticks(np.arange(int(min(y)), int(max(y)) + 1, 1))
        plt.scatter(x, y, c='g', s=10, marker='s', alpha=0.2)
        plt.title(f'Grid Points Inside the Convex Hull (Seed: {current_seed})')
        plt.xlabel('neighbor_Ra')
        plt.ylabel('neighbor_Dec')
        plt.grid(True)
        plt.draw()  # Redraw the updated plot

        # Pause for 0.5 seconds
        plt.pause(0.5)

    plt.ioff()  # Turn off interactive mode
