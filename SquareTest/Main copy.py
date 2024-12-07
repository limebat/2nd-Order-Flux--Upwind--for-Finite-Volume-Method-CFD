import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def load_mesh(coordinates_file, elements_file):
    points = np.loadtxt(coordinates_file, delimiter=',')
    elements = np.loadtxt(element_file_path, delimiter=',').astype(int) - 1
    
    num_elements = len(elements)
    areas = np.zeros(num_elements)
    circumcenters = np.zeros((num_elements, 2))
    
    for i, triangle in enumerate(elements):
        A, B, C = points[triangle]
        areas[i] = 0.5 * np.abs(np.cross(B-A, C-A))
        
        D = 2 * (A[0] * (B[1]-C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
        circumcenter_x = ((A[0]**2 + A[1]**2) * (B[1] - C[1]) + (B[0]**2 + B[1]**2) * (C[1] - A[1]) + (C[0]**2 + C[1]**2) * (A[1] - B[1])) / D
        circumcenter_y = ((A[0]**2 + A[1]**2) * (C[0] - B[0]) + (B[0]**2 + B[1]**2) * (A[0] - C[0]) + (C[0]**2 + C[1]**2) * (B[0] - A[0])) / D
        circumcenters[i] = np.array([circumcenter_x, circumcenter_y])
        
    return points, elements, areas, circumcenters

def calculate_edge_length(i, j, elements, points):
    shared_vertices = set(elements[i]).intersection(set(elements[j]))
    
    if len(shared_vertices) == 2:
        v1, v2 = list(shared_vertices)
        return np.linalg.norm(points[v1] - points[v2])
    
    return 0

def analytical_solution(points, elements, circumcenters):
    T_elements = np.zeros(len(elements))
    for i, triangle in enumerate(elements):
        x = circumcenters[i, 0]
        y = circumcenters[i, 1]
        T_circumcenter = 0
        for n in range(1, 100, 2):
            for m in range(1, 100, 2):
                T_circumcenter += (16 / (np.pi**4 * n * m * (n**2 + m**2))) * np.sin(n * np.pi * x) * np.sin(m * np.pi * y)
        T_elements[i] = T_circumcenter
    return T_elements

def find_central_vertical_line(T_elements, T_elements_analytical):
    points, elements, areas, circumcenters = load_mesh(coordinate_file_path, element_file_path)
    central_points = []
    central_T_numerical = []
    central_T_analytical = []
    
    for i, element in enumerate(elements):
        x_coords = points[element, 0]
        y_coords = points[element, 1]
        
        # Check if this element intersects x=0.5
        if np.min(x_coords) <= 0.5 <= np.max(x_coords):
            central_T_numerical.append(T_elements[i])
            central_T_analytical.append(T_elements_analytical[i])
            central_points.append(circumcenters[i])
    
    return central_points, central_T_numerical, central_T_analytical

def Jacobi(T, neighbors, boundary_elements, circumcenters, elements, points, areas, Q, lambda_value):
    T_new = np.copy(T)
    num_elements = len(elements)
    
    for i in range(num_elements):
        neighbor_flux_sum = 0
        total_flux_weight = 0
        for j in neighbors[i]:
            length_ij = calculate_edge_length(i, j, elements, points)
            distance_ij = np.linalg.norm(circumcenters[i] - circumcenters[j])
            
            weight = (length_ij / distance_ij) 
            neighbor_flux_sum += T[j] * weight 
            total_flux_weight += weight 
            
        if boundary_elements[i]:  # BC CONTRIBUTORS
            T_ghost = -T[i]  # DIRICHLET CONDITIONS
            # FLUX FACTOR ACCOUNTING WITH GHOST CELL -- USING PREVIOUS WEIGHT ASSUMING NEAR - IDENTICAL CELL TO PRIOR NEIGHBORS
            weight = (length_ij / distance_ij)
            neighbor_flux_sum += T_ghost * weight
            total_flux_weight += weight 
            
        T_new[i] = neighbor_flux_sum / total_flux_weight - Q * areas[i] / (lambda_value * total_flux_weight)
        
    return T_new

def GS(T, neighbors, boundary_elements, circumcenters, elements, points, areas, Q, lambda_value):
    num_elements = len(elements)
    
    for i in range(num_elements):
        neighbor_flux_sum = 0
        total_flux_weight = 0
        for j in neighbors[i]:
            length_ij = calculate_edge_length(i, j, elements, points)
            distance_ij = np.linalg.norm(circumcenters[i] - circumcenters[j])
            weight = length_ij / distance_ij
            neighbor_flux_sum += T[j] * weight
            total_flux_weight += weight
            
        if boundary_elements[i]:
            T_ghost = -T[i]
            weight = length_ij / distance_ij
            neighbor_flux_sum += T_ghost * weight
            total_flux_weight += weight
            
        T[i] = neighbor_flux_sum / total_flux_weight - Q * areas[i] / (lambda_value * total_flux_weight)
        
    return T


def get_extended_neighbors(neighbors, i, levels=2):
    """ 
    Function to get extended neighbors for a given element i up to specified levels.
    """
    extended_neighbors = set(neighbors[i])
    next_level_neighbors = set(neighbors[i])
    
    for _ in range(levels - 1):
        current_neighbors = list(next_level_neighbors)
        for neighbor in current_neighbors:
            next_level_neighbors.update(neighbors[neighbor])
        extended_neighbors.update(next_level_neighbors)

    # Remove the current element itself
    extended_neighbors.discard(i)
    return list(extended_neighbors)

def SOR(T, neighbors, boundary_elements, circumcenters, elements, points, areas, Q, lambda_value, w=1.3, levels=2):
    num_elements = len(elements)
    
    for i in range(num_elements):
        neighbor_flux_sum = 0
        total_flux_weight = 0
        
        # Get extended neighbors up to specified levels
        extended_neighbors = get_extended_neighbors(neighbors, i, levels)
        
        for j in neighbors[i]:
            if len(extended_neighbors) >= 9:
                length_ij = calculate_edge_length(i, j, elements, points)
                distance_ij = np.linalg.norm(circumcenters[i] - circumcenters[j])
                level = 1 if j in neighbors[i] else 2  # Determine the level of the neighbor
                
                # Adjust weight based on the level
                weight = (length_ij / distance_ij) * (3. / 3  if level == 1 else -1.0/3 ) #-1./ 6 )
                neighbor_flux_sum += T[j] * weight
                total_flux_weight += weight
                third_order = 1. #2. / 3
            else:
                #Swap to frst order instead of not satisfied:
                for j in neighbors[i]:
                    length_ij = calculate_edge_length(i, j, elements, points)
                    distance_ij = np.linalg.norm(circumcenters[i] - circumcenters[j])
                    weight = length_ij / distance_ij
                    neighbor_flux_sum += T[j] * weight
                    total_flux_weight += weight
            
        if boundary_elements[i]:
            T_ghost = -T[i]
            weight = length_ij / distance_ij
            neighbor_flux_sum += T_ghost * weight
            total_flux_weight += weight
            
            
        T[i] = (1 - w) * T[i] + w * (neighbor_flux_sum / total_flux_weight - Q * areas[i] / (lambda_value * total_flux_weight))
        
    return T


def solve_eqn(coordinates_file, elements_file, lambda_value, Q, tol=1e-6, max_iter=20000):
    points, elements, areas, circumcenters = load_mesh(coordinates_file, elements_file)
    num_elements = len(elements)
    
    T = np.zeros(num_elements)
    boundary_nodes = np.where((points[:,0]==0) | (points[:,0]==1) | (points[:,1]==0) | (points[:,1]==1))[0]
    
    #INITIALIZE ALL ELEMENTS AND THEIR INDEXES FOR ELEMENTS
    boundary_elements = np.zeros(num_elements, dtype=bool)
    nhat_BC = np.zeros(num_elements)

    # FIND ELEMENTS THAT HAVE 2 OR MORE BOUNDARY NODES
    for i, element in enumerate(elements):
        shared_boundary_nodes = np.isin(element, boundary_nodes)
        if np.sum(shared_boundary_nodes) >= 2:  # 2 NODES SHARED MINIMUM TO BE CONSIDERED A BOUNDARY ELEMENT
            boundary_elements[i] = True
            for node_index in element:
                x, y = points[node_index]
                if ((x == 0) | (y == 0)):  # NEGATIVE WEST \ SOUTH
                    nhat_BC[i] = -1
                    break
                elif ((x == 1) | (y == 1)):  # POSITIVE EAST \ NORTH
                    nhat_BC[i] = 1
                    break
    # BC ELEMENTAL TEMPERATURE = 0 - DIRICHLET CONDITION
    T[boundary_elements] = 0
    
    #PRE-COMPUTE ij NEIGHBORS FOR OPTIMIZED FOR LOOP
    neighbors = {}
    for i in range(num_elements):
        neighbors[i] = []
        for j in range(num_elements):
            length_ij = calculate_edge_length(i, j, elements, points)
            if length_ij > 0: #ONLY ADD NEIGHBORS SHARING i -> j
                neighbors[i].append(j)
    
    residual = []
    
    
    for iter_num in range(max_iter):
        #T_new = np.copy(T)
        T_old = np.copy(T)
        T = SOR(T, neighbors, boundary_elements, circumcenters, elements, points, areas, Q, lambda_value)

        residual.append(np.linalg.norm(T_old - T))
        
        if residual[-1] < tol:
            print('Converged at iteration: {iter_num}')
            break
        if iter_num % 10 == 0:
            print(f'Iteration {iter_num}: max(T) = {np.max(T):.6f}, min(T) = {np.min(T):.6f}, avg(T) = {np.mean(T):.6f}')
        
        #T = np.copy(T_new)
        
    return points, elements, T, residual, circumcenters

coordinate_file_path = r'mesh\coordinates_20.input'
element_file_path = r'mesh\elements_20.input'

lambda_value = 1
Q = -1



points, elements, T_elements, residual, circumcenters = solve_eqn(coordinate_file_path, element_file_path, lambda_value, Q)

triang = tri.Triangulation(points[:, 0], points[:, 1], elements)

plt.plot(residual, label='SOR Residual')
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Residual')
plt.title('Resdiual vs. Iterations (Gauss Seidel w/ SOR, $\omega$=1.3)')
plt.grid(True)
plt.legend()
plt.show()

plt.tripcolor(triang, facecolors=T_elements, cmap='viridis', edgecolors='k')
plt.colorbar(label='Temperature')
plt.title('Numerical Temperature Per Element')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

T_elements_analytical = analytical_solution(points, elements, circumcenters)

plt.tripcolor(triang, facecolors=T_elements_analytical, cmap='viridis', edgecolors='k')
plt.colorbar(label='Temperature')
plt.title('Analytical Temperature Distribution Per Element')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# FIND T_i AT ALL MIDPOINT ELEMENTS
central_points, central_T_numerical, central_T_analytical = find_central_vertical_line(T_elements, T_elements_analytical)

central_y = [point[1] for point in central_points]

# NOW ORGANIZE ALL IN ASCENDING ORDER
central_y = np.array(central_y)
central_T_numerical = np.array(central_T_numerical)
central_T_analytical = np.array(central_T_analytical)

sorted_indices = np.argsort(central_y)

central_y_sorted = central_y[sorted_indices]
central_T_numerical_sorted = central_T_numerical[sorted_indices]
central_T_analytical_sorted = central_T_analytical[sorted_indices]

#X = 0.5 SOLUTION
plt.plot(central_y_sorted, central_T_numerical_sorted, 'b-', label='Numerical Solution')
plt.plot(central_y_sorted, central_T_analytical_sorted, 'r--', label='Analytical Solution')
plt.xlabel('y')
plt.ylabel('Temperature')
plt.title('Temperature Distribution along Central Vertical Line (x=0.5)')
plt.legend()
plt.grid(True)
plt.show()

print('For the last hw part, write down this value for L2 Error:' , np.sqrt(np.mean((T_elements - T_elements_analytical)**2)))

