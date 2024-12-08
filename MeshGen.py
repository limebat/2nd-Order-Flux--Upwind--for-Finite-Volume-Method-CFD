
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import Delaunay

def generate_uniform_triangular_mesh(radius, target_num_points):
    """
    Generate a uniform triangular mesh for a circle with a given number of points.
    
    Parameters:
    -----------
    radius : float
        Radius of the circle
    target_num_points : int
        Desired number of points in the mesh
    
    Returns:
    --------
    coordinates : numpy.ndarray
        Array of (x, y) coordinates for all nodes
    elements : numpy.ndarray
        Array of triangular element node connections
    """
    #  ESTIMATE
    area_of_circle = np.pi * (radius ** 2)
    approximate_triangle_area = area_of_circle / target_num_points
    triangle_side_length = np.sqrt((4 * approximate_triangle_area) / np.sqrt(3))
    
    # Hexagonal grid generation
    coordinates = []
    hex_width = triangle_side_length
    hex_height = triangle_side_length * math.sqrt(3) / 2

    # Generate points in a grid, then filter for circle
    for j in range(int(2 * radius / hex_height) + 2):
        for i in range(int(2 * radius / hex_width) + 2):
            # Alternate rows offset
            x = i * hex_width + (j % 2) * (hex_width / 2)
            y = j * hex_height

            # Translate to center the mesh
            x -= radius
            y -= radius

            # Check if point is inside circle
            if x*x + y*y <= radius*radius:
                coordinates.append([x, y])

    coordinates = np.array(coordinates)

    # Adjust points on the boundary to lie exactly on the circle
    for i, (x, y) in enumerate(coordinates):
        distance_to_center = np.sqrt(x**2 + y**2)
        if np.abs(distance_to_center - radius) < triangle_side_length:
            scale_factor = radius / distance_to_center
            coordinates[i] = [x * scale_factor, y * scale_factor]

    # Add additional boundary points to refine the edge triangles
    num_additional_points = len(coordinates)
    for i in range(num_additional_points):
        x, y = coordinates[i]
        if np.sqrt(x**2 + y**2) > radius - triangle_side_length:
            angle = np.arctan2(y, x)
            new_x = x + (triangle_side_length / 2) * np.cos(angle + np.pi / 6)
            new_y = y + (triangle_side_length / 2) * np.sin(angle + np.pi / 6)
            if np.sqrt(new_x**2 + new_y**2) <= radius:
                coordinates = np.vstack([coordinates, [new_x, new_y]])

    # Perform Delaunay triangulation
    delaunay_tri = Delaunay(coordinates)
    elements = delaunay_tri.simplices

    return coordinates, elements

def write_mesh_files(coordinates, elements, target_num_points):
    """
    Write coordinates and elements to files.
    
    Parameters:
    -----------
    coordinates : numpy.ndarray
        Array of (x, y) coordinates
    elements : numpy.ndarray
        Array of triangular element node connections
    prefix : str, optional
        Prefix for output files (default is 'circle')
    """
    # Write coordinates file
    
    np.savetxt(f'mesh/coordinates_{target_num_points}.txt', coordinates, 
               fmt='%.6f', delimiter=',')
    
    # Write elements file (adding 1 to make node ids 1-indexed)
    np.savetxt(f'mesh/elements_{target_num_points}.txt', elements, 
               fmt='%d', delimiter=',')
    
    print(f"Mesh files 'coordinates.txt' and 'elements.txt' generated.")

def visualize_mesh(coordinates, elements, target_num_points):
    """
    Visualize the triangular mesh.
    
    Parameters:
    -----------
    coordinates : numpy.ndarray
        Array of (x, y) coordinates
    elements : numpy.ndarray
        Array of triangular element node connections
    """
    plt.figure(figsize=(10, 10))
    
    # Plot the triangulation
    plt.triplot(coordinates[:, 0], coordinates[:, 1], elements, 
                color='blue', linewidth=0.5, alpha=0.5)
    
    # Plot nodes
    plt.scatter(coordinates[:, 0], coordinates[:, 1], 
                c='red', s=10, zorder=2)
    
    plt.title(f'Uniform Triangular Mesh of Circle at Approx. {target_num_points} Points ({len(elements)} Actual)')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('circle_uniform_mesh.png', dpi=300)
    plt.close()
    print("Mesh visualization saved as 'circle_uniform_mesh.png'")


radius = 1.0
target_num_points = 250

# Generate, save, and visualize mesh
coordinates, elements = generate_uniform_triangular_mesh(
    radius, 
    target_num_points  # Adjust the number of points as needed
)
write_mesh_files(coordinates, elements, target_num_points)
visualize_mesh(coordinates, elements, target_num_points)

# Optional: print some stats
print(f"Number of nodes: {len(coordinates)}")
print(f"Number of elements: {len(elements)}")
