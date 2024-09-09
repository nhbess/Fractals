import pyvista as pv
import numpy as np
import imageio
import glob
import os

# Example list of 3D arrays with ones and zeros
list_of_arrays = [np.random.randint(0, 2, (10, 10, 10)) for _ in range(5)]  # Replace with your actual data

# Directory to save frames
os.makedirs('frames', exist_ok=True)

# Initialize the PyVista plotter
plotter = pv.Plotter(off_screen=True)  # Use off_screen=True to avoid displaying the window

# Loop over each array in the list
for i, data in enumerate(list_of_arrays):
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
    
    # Save the frame
    plotter.screenshot(f'frames/frame_{i:03d}.png')

# Create a video from the frames
with imageio.get_writer('animation.mp4', fps=10) as writer:
    for filename in sorted(glob.glob('frames/frame_*.png')):
        image = imageio.imread(filename)
        writer.append_data(image)
