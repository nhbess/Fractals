import pyvista as pv
import numpy as np

# Example list of 3D arrays with ones and zeros
list_of_arrays = [np.random.randint(0, 2, (10, 10, 10)) for _ in range(100)]  # Replace with your actual data

# Initialize the PyVista plotter and open the GIF file for writing
plotter = pv.Plotter(off_screen=True)  # Use off_screen=True to avoid displaying the window
plotter.open_gif('animation.gif', duration=100)  # Adjust duration per frame as needed

# Loop over each array in the list
for data in list_of_arrays:
    # Get the coordinates of the voxels where the value is 1
    points = np.argwhere(data == 1)
    
    # Create a PyVista PolyData object with these points
    cloud = pv.PolyData(points)
    
    # Create a 3D glyph (cube) for each point (voxel)
    glyph = cloud.glyph(scale=False, geom=pv.Cube())
    
    # Clear previous data
    plotter.clear()
    
    # Add the glyphs to the plotter
    plotter.add_mesh(glyph, color="blue", show_edges=True)
    
    # Set the plot title
    plotter.add_text(f"3D Grid Visualization", font_size=12)
    
    # Capture the frame
    plotter.write_frame()

# Close the GIF file
plotter.close()
