import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize,basinhopping
from geometry import Vertex, Edge, VertexModel
from energy import EnergyModel
from optimization import Optimiser
from visualization import visualize_model,create_video_from_images,write_vertex_model_to_vtk
import  os

# Check if the directory exists
output_dir = "output_dir"
if not os.path.exists(output_dir):
    # If the directory does not exist, create it
    os.makedirs(output_dir)


def evolve_system_with_time_increment(model, initial_area, target_area_start, target_area_end, area_increment, radius, alpha, tol, max_iter):
    fig, ax = plt.subplots(figsize=(10, 6))

    current_target_area = target_area_start 
    output_dir = "images"  # Define the directory where images will be saved
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    # Generate the initial hexagonal grid and compute energies
    #model.generate_hexagonal_circle(circle_radius=radius, hex_area=initial_area)
    model.generate_hexagonal_grid(8, 8 , radius=math.sqrt((2 * initial_area) / (3 * math.sqrt(3))))
    #Get the value fo friction
    fric = model.get_fric()
    adaptive_time_step = 1.2
    # Visualize initial state
    model.visualize(ax=ax, style='ro--', label='Initial State')
    iteration = 0  # Initialize iteration counter
    while iteration <= 100000:
        # Visualize the current model state
        #visualize_model(model, iteration, output_dir=output_dir)
        write_vertex_model_to_vtk(model, filename=f"output_dir/hexagonal_grid_{iteration}.vtk")
        # Update the model with the current target area
        model.set_preferred_area(4.0)
        model.set_fric(fric)
        # Initialize the energy model with the current state of the model
        energy_model = EnergyModel(model)
        
        # Print the current state: total energy, target area, and iteration number
        print("                                                                  ")
        print("******************************************************************")
        print(f" Iteration: {iteration}")
        print(f" Target Area: {current_target_area}")
        print(f" Total Energy: {energy_model.compute_total_energy()}")
        print(f" Circularity : {model.calculate_circularity()}")
        print(f" Friction    : {model.get_fric()}")
        # Initialize the optimizer with the current energy model
        optimizer = Optimiser(energy_model)
        
        
        # Perform energy minimization with specified parameters
        optimizer.minimize_energies(learning_rate=alpha, tolerance=tol, max_iterations=max_iter)


        # Prepare for the next iteration
        # Increment the target area
        current_target_area += area_increment
        # Increment the iteration counter
        iteration += 1
        #Adaptive time step should only effect the friction parameter
        fric = model.get_fric()/adaptive_time_step
        print("******************************************************************")
        print("                                                                  ")
    
    energy    = EnergyModel(model)
    model.update_vertex_positions(optimizer.optimized_positions)
    print(" ********* The total energy in the last time increment is  ********* ", energy.compute_total_energy(), " and circularity is ", model.calculate_circularity())

    # Visualize final state after all increments
    model.visualize(ax=ax, style='bo-', label='Final State')
    plt.legend()
    plt.show()



def main():
    # Parameters for the model and simulation
    initial_area = 4
    target_area_start = initial_area  # Starting target area
    target_area_end = 4.0   # Ending target area
    alpha = 0.001
    area_increment = 0.1    # Increment in target area per iteration
    radius = 2.2
    tol = 1E-3
    max_iter = 1E5
    #Material parameters
    K_area  = 1.
    K_peri  = 0.
    gamma   = 1.0
    fric    = 1.0
    # Create the model instance
    model = VertexModel()
    model.set_K_area(K_area)
    model.set_K_peri(K_peri)
    model.set_gamma(gamma)
    model.set_fric(fric)
    # Evolve the system with time increment
    evolve_system_with_time_increment(model, initial_area, target_area_start, target_area_end, area_increment, radius, alpha, tol, max_iter)

    # Calculate area increment
    delta_alpha = alpha * initial_area

    # Calculate total number of increments
    total_increments = (target_area_end - initial_area) / delta_alpha

    create_video_from_images(input_dir="images", output_file="model_evolution.mp4", framerate=total_increments/20)
    print("Evolution of the system completed.")

if __name__ == "__main__":
    main()










