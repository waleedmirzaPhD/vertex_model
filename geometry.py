import matplotlib.pyplot as plt
import math
import numpy as np



class Vertex:
    def __init__(self, id, x, y, z=0):
        self.id = id
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Vertex(id={self.id}, x={self.x}, y={self.y}, z={self.z})"

class Edge:
    def __init__(self, start_vertex, end_vertex):
        self.start = start_vertex
        self.end = end_vertex

    def __repr__(self):
        return f"Edge(start={self.start.id}, end={self.end.id})"

class VertexModel:
    def __init__(self):
        self.vertices = {}
        self.vertices0= {} # New attribute to track previous nodal positions
        self.edges = []
        self.hexagons = {}  # New attribute to track hexagons
        self.preferred_area = None
        self.K_area = 0
        self.K_peri = 0
        self.gamma = 0       
        self.fric = 0  
    def set_preferred_area(self, value):
        """Set the preferred area for the model."""
        self.preferred_area = value


    def set_K_area(self, value):
        """Set the area constant K_area."""
        self.K_area = value

    def set_K_peri(self, value):
        """Set the perimeter constant K_peri."""
        self.K_peri = value

    def set_gamma(self, value):
        """Set the gamma constant."""
        self.gamma = value


    def set_fric(self, value):
        """Set the gamma constant."""
        self.fric = value

    def get_preferred_area(self):
        """Get the preferred area for the model."""
        return self.preferred_area 


    def get_K_area(self):
        """Get the area constant K_area."""
        return self.K_area 

    def get_K_peri(self):
        """Get the perimeter constant K_peri."""
        return self.K_peri 

    def get_gamma(self):
        """Get the gamma constant."""
        return self.gamma 

    def get_fric(self):
        """Get the gamma constant."""
        return self.fric 


    



    def add_vertex(self, id, x, y, z=0):
        if id not in self.vertices:
            self.vertices[id] = Vertex(id, x, y, z)
        else:
            print(f"Vertex with id {id} already exists.")

    def add_edge(self, start_id, end_id):
        if start_id in self.vertices and end_id in self.vertices:
            edge = Edge(self.vertices[start_id], self.vertices[end_id])
            self.edges.append(edge)
        else:
            print("One or both vertices not found.")

    def __repr__(self):
        return f"VertexModel(Vertices={len(self.vertices)}, Edges={len(self.edges)})"


    def generate_hexagonal_circle(self, circle_radius, hex_area):
        """
        Generate a circular arrangement of hexagons.
        - circle_radius: Radius of the circle to be filled with hexagons.
        - hex_radius: Radius of each hexagon.
        """
        hex_radius =  math.sqrt((2 * hex_area) / (3 * math.sqrt(3)))
        vert_spacing = math.sqrt(3) * hex_radius  # Vertical spacing between hexagon centers
        horiz_spacing = 3 * hex_radius / 2  # Horizontal spacing between hexagon centers

        # Calculate the number of hexagons along the diameter
        diameter_in_hexes = int(circle_radius / horiz_spacing) * 2
        center_x, center_y = circle_radius, circle_radius  # Assuming circle's center is at (circle_radius, circle_radius)

        for x in range(-diameter_in_hexes, diameter_in_hexes + 1):
            for y in range(-diameter_in_hexes, diameter_in_hexes + 1):
                hex_center_x = center_x + x * horiz_spacing
                # Offset every other column to fit hexagons snugly
                hex_center_y = center_y + y * vert_spacing + (x % 2) * vert_spacing / 2

                # Calculate distance from the center of the circle to the center of the hexagon
                distance_to_center = math.sqrt((hex_center_x - center_x) ** 2 + (hex_center_y - center_y) ** 2)

                # Only add the hexagon if its center is within the specified circle's radius
                if distance_to_center <= circle_radius:
                    self.add_hexagon(hex_center_x, hex_center_y, hex_radius)


    def add_hexagon(self, center_x, center_y, radius):
        """
        Add a single hexagon to the model, ensuring that vertices are not duplicated
        and adjacent hexagons share vertices.
        """
        angle_deg = 60
        angle_rad = math.radians(angle_deg)
        hex_vertices = []

        for i in range(6):
            x = center_x + radius * math.cos(i * angle_rad)
            y = center_y + radius * math.sin(i * angle_rad)

            # Generate a unique ID for the vertex based on its coordinates
            # Consider rounding the coordinates to a fixed precision if necessary
            vertex_id = f"vertex_{round(x, 4)}_{round(y, 4)}"

            # Add the vertex if it does not already exist
            if vertex_id not in self.vertices:
                self.add_vertex(vertex_id, x, y)

            hex_vertices.append(vertex_id)

        # Add edges between consecutive vertices in the hexagon
        for i in range(len(hex_vertices)):
            start_vertex_id = hex_vertices[i]
            end_vertex_id = hex_vertices[(i + 1) % len(hex_vertices)]  # Ensures the last vertex connects to the first
            self.add_edge(start_vertex_id, end_vertex_id)

        # Store the hexagon's vertices using a unique ID for the hexagon itself
        hexagon_id = f"hex_{center_x}_{center_y}"
        self.hexagons[hexagon_id] = hex_vertices



    def visualize(self, ax=None, style='bo-', label='State'):
        """
        Visualize the graph on the given Matplotlib axis with the specified style and add a legend only for the first edge.

        Parameters:
        - ax: Matplotlib axis object. If None, creates a new figure and axis.
        - style: A string defining the style and color of the plot.
        - label: Label for the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        # Track whether the legend has been added
        legend_added = False
        
        for edge in self.edges:
            x_values = [edge.start.x, edge.end.x]
            y_values = [edge.start.y, edge.end.y]
            if not legend_added:
                ax.plot(x_values, y_values, style, label=label)  # Add label only for the first edge
                legend_added = True  # Prevent further labels
            else:
                ax.plot(x_values, y_values, style)  # Plot without label for subsequent edges

        ax.set_title('Vertex Model Visualization')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.axis('equal')
        ax.grid(True)
        ax.legend()

    def generate_hexagonal_grid(self, grid_width, grid_height, radius):
        """
        Generate a hexagonal grid.
        - grid_width, grid_height: Dimensions of the grid in terms of the number of hexagons.
        - radius: Distance from the center to any vertex of the hexagons.
        """
        vert_spacing = math.sqrt(3) * radius
        horiz_spacing = 1.5 * radius
        for row in range(grid_height):
            for col in range(grid_width):
                center_x = col * horiz_spacing
                # Offset for every other row
                center_y = row * vert_spacing
                if col % 2 == 1:
                    center_y += vert_spacing / 2
                self.add_hexagon(center_x, center_y, radius)


    def edge_id(self, start_vertex_id, end_vertex_id):
        """Generate a unique ID for an edge based on the start and end vertex IDs."""
        # Sort the vertex IDs to ensure consistency (undirected edge)
        return tuple(sorted([start_vertex_id, end_vertex_id]))

    def count_edge_occurrences(self):
        """Count occurrences of each edge among all hexagons."""
        edge_occurrences = {}
        for hex_id, hex_vertices in self.hexagons.items():
            num_vertices = len(hex_vertices)
            for i in range(num_vertices):
                # Assuming hex_vertices stores IDs of vertices
                current_vertex_id = hex_vertices[i]
                next_vertex_id = hex_vertices[(i + 1) % num_vertices]
                edge = self.edge_id(current_vertex_id, next_vertex_id)
                if edge in edge_occurrences:
                    edge_occurrences[edge] += 1
                else:
                    edge_occurrences[edge] = 1
        return edge_occurrences

    def is_edge_boundary(self, edge):
        """Determine if an edge is a boundary edge."""
        edge_occurrences = self.count_edge_occurrences()
        edge_tuple = self.edge_id(edge.start.id, edge.end.id)
        # An edge is a boundary if it occurs only once among all hexagons
        return edge_occurrences.get(edge_tuple, 0) == 1
    

    def calculate_boundary_perimeter(self):
        """Calculate the total perimeter of the tissue using only boundary edges."""
        total_perimeter = 0
        for edge in self.edges:
            if self.is_edge_boundary(edge):
                dx = edge.start.x - edge.end.x
                dy = edge.start.y - edge.end.y
                edge_length = (dx**2 + dy**2)**0.5
                total_perimeter += edge_length
        return total_perimeter
    

    def calculate_hexagon_area(self, hexagon_vertices):
        """Calculate the area of a hexagon using the Shoelace formula."""
        n = len(hexagon_vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            # Correctly access the x and y attributes of the Vertex objects
            xi, yi = self.vertices[hexagon_vertices[i]].x, self.vertices[hexagon_vertices[i]].y
            xj, yj = self.vertices[hexagon_vertices[j]].x, self.vertices[hexagon_vertices[j]].y
            area += xi * yj
            area -= xj * yi
        area = abs(area) / 2.0
        return area

    def calculate_total_tissue_area(self):
        """Calculate the total area of the tissue by summing the areas of all hexagons."""
        total_area = 0.0
        for hexagon_vertices in self.hexagons.values():
            hexagon_area = self.calculate_hexagon_area(hexagon_vertices)
            total_area += hexagon_area
        return total_area
    
    def calculate_circularity(self):
        """Calculate the circularity of the entire tissue."""
        total_area = self.calculate_total_tissue_area()
        total_perimeter = self.calculate_boundary_perimeter()
        if total_perimeter == 0:
            return 0  # Avoid division by zero
        circularity = (4 * math.pi * total_area) / (total_perimeter ** 2)
        return circularity


    def update_vertex_positions(self, vertex_positions):
        for i, v_id in enumerate(self.vertices):
            self.vertices[v_id].x, self.vertices[v_id].y = vertex_positions[2*i], vertex_positions[2*i + 1]

