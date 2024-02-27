import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import minimize,basinhopping
from geometry import Vertex, Edge

class EnergyModel:
    def __init__(self, model):
        self.vertices = model.vertices
        self.hexagons = model.hexagons
        self.preferred_area = model.preferred_area
        self.energies = {}


    def compute_energy(self):
        """
        Compute the energy of each hexagon based on its area deviation from the preferred area.
        """
        self.energies = {}
        for hex_id, vertex_ids in self.hexagons.items():
            # Convert vertex IDs to (x, y) coordinates
            vertices = [(self.vertices[v_id].x, self.vertices[v_id].y) for v_id in vertex_ids[:-1]]  # Use slice to correct indexing if necessary
            actual_area = self.calculate_hexagon_area(vertices)
            # Calculate energy as the square of the difference between actual and preferred areas
            energy = (actual_area - self.preferred_area) ** 2
            self.energies[hex_id] = energy

    def compute_total_energy(self):
        """
        Compute the total energy of the system by summing up the energy of each cell.
        """
        self.compute_energy()
        if self.energies:  # Check if energies have been computed
            return sum(self.energies.values())
        else:
            print("Energy not computed for cells. Please run compute_energy first.")
            return None
        
    def objective_function(self, vertex_positions):
        # Recalculate the total energy of the system with the updated positions
        for i, v_id in enumerate(self.vertices):
            self.vertices[v_id].x, self.vertices[v_id].y = vertex_positions[2*i], vertex_positions[2*i + 1]
        
        self.compute_energy()
        return self.compute_total_energy()
    
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



    def derivatives_of_energy_with_area_(self,vertices):

        self.energies = {}
        A0 = self.preferred_area

        for hex_id, vertex_ids in self.hexagons.items():
            # Convert vertex IDs to (x, y) coordinates
            vertices = [(self.vertices[v_id].x, self.vertices[v_id].y) for v_id in vertex_ids[:-1]]  # Use slice to correct indexing if necessary
            A = self.calculate_hexagon_area(vertices)
            dA_dx, dA_dy = self.derivatives_of_area(vertices)
            # Calculate energy as the square of the difference between actual and preferred areas
            # Compute the derivatives of energy with respect to x and y coordinates
            dE_dx = [2 *  (A - A0) * dAdx for dAdx in dA_dx]
            dE_dy = [2 *  (A - A0) * dAdy for dAdy in dA_dy]
            return dE_dx, dE_dy



    def derivatives_of_energy_with_area(self):
        gradients = {}

        dA_dx, dA_dy = self.derivative_area_wrt_vertices()

        current_area = self.calculate_hexagon_area([(self.vertices[v_id].x, self.vertices[v_id].y) for v_id in self.vertices])
        

        for v_id in self.vertices:
            vertex = self.vertices[v_id]
            grad_x = 2 * (current_area - self.preferred_area) * dA_dx[v_id]
            grad_y = 2 * (current_area - self.preferred_area) * dA_dy[v_id]

            gradients[v_id] = (grad_x, grad_y)

        return gradients


