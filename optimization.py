import numpy as np
from scipy.optimize import minimize
from scipy.optimize import minimize

class Optimiser:
    def __init__(self, energy_model):
        self.energy_model = energy_model
        self.optimized_positions = []

    def update_vertex_positions(self, vertex_positions):
        """
        Update the positions of vertices in the energy model based on optimization results.
        """
        for i, v_id in enumerate(self.energy_model.vertices.keys()):
            self.energy_model.vertices[v_id].x, vertex_positions[2*i]
            self.energy_model.vertices[v_id].y = vertex_positions[2*i + 1]
            



    def minimize_energies(self, learning_rate=0.01, tolerance=1e-6, max_iterations=10000):
        # Flatten vertex positions into a single array for the optimizer
        initial_positions = np.array([coord for vertex in self.energy_model.vertices.values() for coord in (vertex.x, vertex.y)])

        # The objective function now directly computes the energy or metric to minimize using the energy model
        def objective_function(vertex_positions):
            # Update the model's vertices based on the optimization variables
            self.update_vertex_positions(vertex_positions)
            
            # Compute the total energy for the current configuration using the energy model
            return self.energy_model.objective_function(vertex_positions)

        # The optimization routine using Nelder-Mead
        result = minimize(fun=objective_function,
                          x0=initial_positions,
                          method='Nelder-Mead',
                          options={'disp': True, 'maxiter': max_iterations, 'fatol': tolerance})

        # Update the model with the optimized positions
        self.optimized_positions = result.x


    def minimize_energies_with_jacobian(self, learning_rate=0.01, tolerance=1e-6, max_iterations=10000):
        # Initial positions of vertices as a flattened array for optimization
        initial_positions = np.array([coord for vertex in self.energy_model.vertices.values() for coord in (vertex.x, vertex.y)])

        # Define the objective function that calculates the energy
        def objective_function(vertex_positions):
            # Update vertex positions in the model to the current optimization state
            self.update_vertex_positions(vertex_positions)
            # Compute and return the total energy for the current configuration
            return self.energy_model.objective_function(vertex_positions)
        
        # Define the Jacobian (gradient) of the objective function
        def jacobian(vertex_positions):
            # Update vertex positions in the model to the current optimization state
            self.update_vertex_positions(vertex_positions)
            # Assuming you have a method that calculates the derivative of energy with respect to positions
            gradients = self.energy_model.derivatives_of_energy_with_area()
            # Flatten the gradient values to match the optimization variables' structure
            flattened_gradients = np.array([grad for _, grad in gradients.items()]).flatten()
            print("Gradients:", flattened_gradients)
            if np.all(flattened_gradients == 0):
                print("Warning: Initial gradient is zero.")
            return flattened_gradients

        # Call the optimization routine with the objective function and its Jacobian
        result = minimize(fun=objective_function,
                        x0=initial_positions,
                        method='CG',
                        jac=jacobian,  # This is optional but recommended if you have the gradient function.
                        options={'disp': True, 'maxiter': 100000})

        # After optimization, update the model with the optimized positions
        self.optimized_positions = result.x


    def minimize_energies_gradient_descent(self, learning_rate=0.001, tolerance=1e-6, max_iterations=10000):
        for iteration in range(max_iterations):
            # Compute the energy gradients for current configuration
            gradients = self.energy_model.derivatives_of_energy_with_area()
            
            # Track the change in vertex positions for this iteration to check for convergence
            max_position_change = 0
            
            # Update the positions of vertices based on the computed gradients
            for v_id, grad in gradients.items():
                vertex = self.energy_model.vertices[v_id]
                
                # Calculate the updates for x and y positions
                update_x = -learning_rate * grad[0]
                update_y = -learning_rate * grad[1]
                
                # Update vertex positions
                vertex.x += update_x
                vertex.y += update_y
                
                # Track the largest position change
                max_position_change = max(max_position_change, abs(update_x), abs(update_y))
            
            # Recompute the energies after updating vertex positions
            self.energy_model.compute_energy()
            total_energy = self.energy_model.compute_total_energy()
            
            # Output some information for monitoring
            print(f"Iteration {iteration + 1}, Total Energy: {total_energy}, Max Position Change: {max_position_change}")
            
            # Check for convergence (if the maximum position change is below the tolerance)
            if max_position_change < tolerance:
                print("Convergence achieved.")
                break
        else:
            print("Max iterations reached without convergence.")
        
        # After the optimization loop, store the most updated coordinates
        # This ensures the model reflects the current state of all vertices
        self.optimized_positions = np.array([coord for vertex in self.energy_model.vertices.values() for coord in (vertex.x, vertex.y)])

