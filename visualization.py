import os
import matplotlib.pyplot as plt
import subprocess
import vtk
from vtk.util.numpy_support import numpy_to_vtk







def visualize_model(model, iteration, output_dir="images"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots()
    # Plotting each hexagon's vertices and edges
    for hex_id, vertex_ids in model.hexagons.items():
        hex_vertices = [model.vertices[v_id] for v_id in vertex_ids]  # Get the actual Vertex objects

        x_values = [vertex.x for vertex in hex_vertices]
        y_values = [vertex.y for vertex in hex_vertices]
        
        # Complete the hexagon by connecting the last vertex to the first
        x_values.append(x_values[0])
        y_values.append(y_values[0])
        
        # Plot vertices
        ax.plot(x_values, y_values, 'o-', color='blue')  # Plots vertices and edges for each hexagon
    
    # Adjust the figure
    ax.set_aspect('equal', 'box')
    plt.axis('off')  # Hide axes for a cleaner look
    
    # Save the figure
    fig_path = os.path.join(output_dir, f"model_state_{iteration:04d}.png")
    fig.savefig(fig_path)
    plt.close(fig)  # Free up memory by closing the figure
    
    return fig_path  # Optional: return the path of the saved figure




def create_video_from_images(input_dir="images", output_file="model_evolution.mp4", framerate=1):
    # Construct the ffmpeg command to compile images into a video
    ffmpeg_command = [
        'ffmpeg',
        '-framerate', str(framerate),  # Number of frames per second
        '-i', os.path.join(input_dir, 'model_state_%04d.png'),  # Input file pattern
        '-c:v', 'libx264',  # Video codec
        '-pix_fmt', 'yuv420p',  # Pixel format
        '-r', '30',  # Output frame rate
        output_file
    ]
    
    # Execute ffmpeg command
    subprocess.run(ffmpeg_command, check=True)

    # Construct the VLC command to play the video
    vlc_command = ['/Applications/VLC.app/Contents/MacOS/VLC', output_file]

    # Execute VLC command
    subprocess.run(vlc_command, check=True)

# Note: This function assumes that `ffmpeg` and `VLC` are installed and accessible from the command line.
# You might need to specify the full path to the ffmpeg and VLC executables if they're not in your system's PATH.



def write_vertex_model_to_vtk(vertex_model, filename="hexagonal_grid.vtk"):
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()

    # Mapping from vertex ID to point index in VTK
    vertex_id_to_vtk_index = {}

    for hex_id, hex_vertices in vertex_model.hexagons.items():
        # VTK polygon for the current hexagon
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(len(hex_vertices))

        for i, vertex_id in enumerate(hex_vertices):
            vertex = vertex_model.vertices[vertex_id]
            if vertex_id not in vertex_id_to_vtk_index:
                vtk_index = points.InsertNextPoint(vertex.x, vertex.y, vertex.z)
                vertex_id_to_vtk_index[vertex_id] = vtk_index
            else:
                vtk_index = vertex_id_to_vtk_index[vertex_id]

            polygon.GetPointIds().SetId(i, vtk_index)

        cells.InsertNextCell(polygon)

    # Create an unstructured grid and set its points and cells
    unstructured_grid = vtk.vtkUnstructuredGrid()
    unstructured_grid.SetPoints(points)
    unstructured_grid.SetCells(vtk.VTK_POLYGON, cells)

    # Write the unstructured grid to a VTK file
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(unstructured_grid)
    writer.Write()




