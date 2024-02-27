import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize,basinhopping
from geometry import Vertex, Edge, VertexModel
from energy import EnergyModel
from optimization import Optimiser
from visualization import visualize_model,create_video_from_images
import  os


def evolve_system_with_time_increment(model, initial_area, target_area_start, target_area_end, area_increment, radius, alpha, tol, max_iter):
    fig, ax = plt.subplots(figsize=(10, 6))

    current_target_area = target_area_start 
    output_dir = "images"  # Define the directory where images will be saved
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    
    # Generate the initial hexagonal grid and compute energies
    #model.generate_hexagonal_circle(circle_radius=radius, hex_area=initial_area)
    model.generate_hexagonal_grid(1, 2, radius=math.sqrt((2 * initial_area) / (3 * math.sqrt(3))))
    # Visualize initial state
    model.visualize(ax=ax, style='ro--', label='Initial State')
    iteration = 0  # Initialize iteration counter
    while current_target_area <= 6.001:
        model.set_preferred_area(current_target_area)
        energy    = EnergyModel(model)
        print(" ********* The total energy at this time increments is  ********* ", energy.compute_total_energy(), "  the value of area is ", current_target_area, " and the increment number is ",iteration,"**************************")
        optimise =  Optimiser(energy)
        optimise.minimize_energies_gradient_descent(learning_rate=alpha, tolerance=tol, max_iterations=max_iter)
        model.update_vertex_positions(optimise.optimized_positions)
        # Increment the target area for the next iteration
        current_target_area += area_increment
        visualize_model(model, iteration, output_dir=output_dir)
         # Update for the next iteration
        current_target_area += area_increment
        iteration += 1
    energy    = EnergyModel(model)
    print(" ********* The total energy in the last time increment is  ********* ", energy.compute_total_energy())
    # Visualize final state after all increments
    model.visualize(ax=ax, style='bo-', label='Final State')
    plt.legend()
    plt.show()


def main():
    # Parameters for the model and simulation
    initial_area = 4
    target_area_start = initial_area  # Starting target area
    target_area_end = 6   # Ending target area
    alpha = 0.001
    area_increment = 0.01    # Increment in target area per iteration
    radius = 2.2
    tol = 1E-8
    max_iter = 1000000

    # Create the model instance
    model = VertexModel()

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










