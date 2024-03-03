import numpy as np
from scipy.optimize import minimize
from scipy.optimize import minimize
import copy


class Optimiser:
    def __init__(self, energy_model):
        self.energy_model = energy_model
        self.optimized_positions = []


            

    def minimize_energies(self, learning_rate=0.01, tolerance=1e-6, max_iterations=10000):
        # Flatten vertex positions into a single array for the optimizer
        initial_positions = np.array([coord for vertex in self.energy_model.vertices.values() for coord in (vertex.x, vertex.y)])
        #Update initial condition for the iteration+1 time step
        self.energy_model.vertices0 = copy.deepcopy(self.energy_model.vertices)
        # The objective function now directly computes the energy or metric to minimize using the energy model
        def objective_function(vertex_positions):
            # Compute the total energy for the current configuration using the energy model
            return self.energy_model.objective_function(vertex_positions)

        #The optimization routine using Nelder-Mead
        # result = minimize(fun=objective_function,
        #                    x0=initial_positions,
        #                    method='Nelder-Mead',
        #                    options={'disp': True, 'maxiter': max_iterations, 'fatol': tolerance})

        result = minimize(
            fun=objective_function,
            x0=initial_positions,
            method='L-BFGS-B',
            jac="3-point",
            options={
                'disp': True,
                'maxiter': 1000000,
                'maxfun': 600000,   # Increase the maximum number of function evaluations
                'ftol': 1e-5,  # Tighten the function tolerance
                'gtol': 1e-5   ,  # Gradient tolerance
                'maxls': 200,   # Increase the maximum number of line search steps
            })

        # Update the model with the optimized positions
        # Check if the optimization was successful
        if not result.success:
            raise RuntimeError("Optimization did not converge successfully.")
        else: 
            print("Optimization converge successfully.")
        self.optimized_positions = result.x
    
    def minimize_energies_with_jacobian(self, learning_rate=0.01, tolerance=1e-6, max_iterations=10000):
        # Initial positions of vertices as a flattened array for optimization
        initial_positions = np.array([coord for vertex in self.energy_model.vertices.values() for coord in (vertex.x, vertex.y)])
        
        # Example of fixing vertices with specific IDs
        fixed_vertex_ids = [ 'vertex_1.2408_0.0', 'vertex_0.6204_1.0746']  # IDs of vertices to fix

        # Create bounds for each vertex
        bounds_ = []
        for i, (v_id, vertex) in enumerate(self.energy_model.vertices.items()):
            if v_id in fixed_vertex_ids:
                # Fix vertex by setting its bounds to its current position
                bounds_.append((vertex.x, vertex.x))
                bounds_.append((vertex.y, vertex.y))
            else:
                # Allow vertex to move freely within a wide range
                bounds_.append((None, None))  # Assuming the optimization method supports None bounds
                bounds_.append((None, None))


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
            if np.all(flattened_gradients == 0):
                print("Warning: Initial gradient is zero.")
            return flattened_gradients

        # Call the optimization routine with the objective function and its Jacobian
        result = minimize(
            fun=objective_function,
            x0=initial_positions,
            method='CG',
            jac=jacobian,
            bounds=bounds_,
            options={
                'disp': False,
                'maxiter': 100000,
                'ftol': 1e-6,  # Tighten the function tolerance
                'gtol': 1e-6   ,  # Gradient tolerance
                'maxls': 200,   # Increase the maximum number of line search steps
            }
        )
        # Check if the optimization was successful
        if not result.success:
            raise RuntimeError("Optimization did not converge successfully.")
        else: 
            print("Optimization converge successfully.")

        # After optimization, update the model with the optimized positions
        self.optimized_positions = result.x

