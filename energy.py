import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import minimize,basinhopping
from geometry import Vertex, Edge

class EnergyModel:
    def __init__(self, model):
        self.model = model
        self.vertices   = model.vertices
        self.vertices0  = model.vertices0
        self.hexagons = model.hexagons
        self.edges = model.edges
        self.preferred_area = model.preferred_area
        self.K_area = model.K_area
        self.K_peri = model.K_peri
        self.gamma = model.gamma   
        self.fric = model.fric
        self.energies = {}
        self.energies_edges = {}

    def dissipation(self):
        dot_product_sum = 0
        # Iterate through each vertex, assuming the keys match in both dictionaries
        for vid, vertex in self.vertices.items():
            vertex0 = self.vertices0[vid]  # Directly access the matching vertex in vertices0
            # Compute the 2D displacement vector components
            dx = vertex.x - vertex0.x
            dy = vertex.y - vertex0.y

            # Compute the dot product of the displacement vector with itself (2D)
            dot_product = dx**2 + dy**2
            
            dot_product_sum += self.fric*dot_product
        return dot_product_sum

    def compute_energy(self):
        """
        Compute the energy of each hexagon based on its area deviation from the preferred area.
        """
        self.energies = {}
        for hex_id, vertex_ids in self.hexagons.items():
            # Convert vertex IDs to (x, y) coordinates
            vertices = [(self.vertices[v_id].x, self.vertices[v_id].y) for v_id in vertex_ids[:]]  # Use slice to correct indexing if necessary
            actual_area      = self.calculate_hexagon_area(vertices)
            actual_perimiter = self.calculate_hexagon_perimeter(vertices)
            # Calculate energy as the square of the difference between actual and preferred areas
            energy =    self.K_area*(actual_area - self.preferred_area) ** 2     + self.K_peri*actual_perimiter**2 
            self.energies[hex_id] = energy 


    def compute_energy_edges(self):
        """Compute the total energy as the sum of the lengths of all edges."""
        for edge in self.edges:
            dx = edge.start.x - edge.end.x
            dy = edge.start.y - edge.end.y
            # If considering 3D, include z coordinate difference
            edge_length = math.sqrt(dx**2 + dy**2 )
            if (self.model.is_edge_boundary(edge)):
                self.energies_edges[edge] = self.gamma*edge_length
            else:
                self.energies_edges[edge] = 0.0

    def compute_total_energy(self):
        """
        Compute the total energy of the system by summing up the energy of each cell.
        """
        self.compute_energy() #First contribution on areas 
        self.compute_energy_edges() #Second contribution on edges 
        if self.energies and self.energies_edges:  # Check if energies have been computed
            return sum(self.energies.values()) + sum(self.energies_edges.values())
        else:
            print("Energy not computed for cells and/or edges. Please run compute_energy first.")
            return None

    def compute_rayleigh(self):
        """
        Compute the rayleigh function  of the system by summing up the total energy and dissipation.
        """
        return self.compute_total_energy() + self.dissipation()

    def objective_function(self, vertex_positions):
        # Recalculate the total energy of the system with the updated positions
        for i, v_id in enumerate(self.vertices):
            self.vertices[v_id].x, self.vertices[v_id].y = vertex_positions[2*i], vertex_positions[2*i + 1]
        
        return self.compute_rayleigh()
    
    def calculate_hexagon_area(self,vertices):
        num_vertices = len(vertices)
        area = 0.0
        for i in range(num_vertices):
            j = (i + 1) % num_vertices
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        area = abs(area) / 2.0
        return area
    

    def derivative_area_wrt_vertices(self):
        n = len(self.vertices)
        dA_dx = {}
        dA_dy = {}
        
        vertex_ids = list(self.vertices.keys())
        for i, v_id in enumerate(vertex_ids):
            prev_id = vertex_ids[i-1]
            next_id = vertex_ids[(i+1) % n]
            
            x_k_minus_1, y_k_minus_1 = self.vertices[prev_id].x, self.vertices[prev_id].y
            x_k_plus_1, y_k_plus_1 = self.vertices[next_id].x, self.vertices[next_id].y
            
            dA_dx[v_id] = 0.5 * (y_k_plus_1 - y_k_minus_1)
            dA_dy[v_id] = 0.5 * (x_k_minus_1 - x_k_plus_1)

        return dA_dx, dA_dy
    


    def derivative_perimeter_wrt_vertices(self):
        n = len(self.vertices)
        dP_dx = {}
        dP_dy = {}
        
        vertex_ids = list(self.vertices.keys())
        for i, v_id in enumerate(vertex_ids):
            prev_id = vertex_ids[i-1]
            next_id = vertex_ids[(i+1) % n]
            
            # Extract coordinates for current vertex and its adjacent vertices
            x_k, y_k = self.vertices[v_id].x, self.vertices[v_id].y
            x_k_minus_1, y_k_minus_1 = self.vertices[prev_id].x, self.vertices[prev_id].y
            x_k_plus_1, y_k_plus_1 = self.vertices[next_id].x, self.vertices[next_id].y
            
            # Calculate lengths of the adjacent sides to vertex k
            L_k_minus_1 = ((x_k - x_k_minus_1)**2 + (y_k - y_k_minus_1)**2)**0.5
            L_k_plus_1 = ((x_k_plus_1 - x_k)**2 + (y_k_plus_1 - y_k)**2)**0.5
            
            # Calculate derivatives with respect to x and y for vertex k
            dP_dx[v_id] = (x_k - x_k_minus_1) / L_k_minus_1 + (x_k - x_k_plus_1) / L_k_plus_1
            dP_dy[v_id] = (y_k - y_k_minus_1) / L_k_minus_1 + (y_k - y_k_plus_1) / L_k_plus_1
        
        return dP_dx, dP_dy



    def derivatives_of_energy_with_area(self):
        K_area = self.K_area  # Assuming K=1 for simplicity; adjust as needed
        K_peri = self.K_peri         
        gradients = {}

        # Initialize derivatives of area with respect to x and y for all vertices
        dA_dx, dA_dy = self.derivative_area_wrt_vertices()
        dP_dx, dP_dy = self.derivative_perimeter_wrt_vertices()
        for hex_id, vertex_ids in self.hexagons.items():
            # Convert vertex IDs to (x, y) coordinates for this hexagon
            hexagon = [(self.vertices[v_id].x, self.vertices[v_id].y) for v_id in vertex_ids]
            # Calculate the current area of the hexagon
            A = self.calculate_hexagon_area(hexagon)
            P = 0 #self.calculate_hexagon_perimeter(hexagon) 
            # Calculate energy derivatives for each vertex in the hexagon
            for v_id in vertex_ids:
                grad_x  = 2 * K_area * (A - self.preferred_area) * dA_dx[v_id]
                grad_y  = 2 * K_area * (A - self.preferred_area) * dA_dy[v_id]
                grad_x_P= 2 * K_peri*P * dP_dx[v_id]
                grad_y_P= 2 * K_peri*P * dP_dy[v_id]               
                gradients[v_id] = (grad_x + grad_x_P, grad_y + grad_y_P)

            
        # Convert gradients dict to a flattened array to match the optimization variables' structure
        flattened_gradients = np.zeros(2 * len(self.vertices))
        for i, v_id in enumerate(self.vertices.keys()):
            if v_id in gradients:
                flattened_gradients[2 * i] = gradients[v_id][0]  # Gradient wrt x
                flattened_gradients[2 * i + 1] = gradients[v_id][1]  # Gradient wrt y

        return gradients


    def calculate_hexagon_perimeter(self,vertices):
        perimeter = 0
        num_vertices = len(vertices)
        for i in range(num_vertices):
            j = (i + 1) % num_vertices  # Ensures that the last vertex connects back to the first
            dx = vertices[i][0] - vertices[j][0]
            dy = vertices[i][1] - vertices[j][1]
            distance = (dx**2 + dy**2)**0.5
            perimeter += distance
        return perimeter


