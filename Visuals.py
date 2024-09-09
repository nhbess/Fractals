import pyvista as pv
import numpy as np
from tqdm import tqdm
import os
import pyvista as pv
import numpy as np
from tqdm import tqdm

pv.global_theme.allow_empty_mesh = True

def make_gif(data:np.array, filename:str = 'animation.gif', duration:int = 100, title:str = None) -> None:
    
    plotter = pv.Plotter(off_screen=True)  # Use off_screen=True to avoid displaying the window
    
    #check if folder MEDIA exists else create it
    if not os.path.exists('_MEDIA'):
        os.mkdir('_MEDIA')
    plotter.open_gif(f'_MEDIA/{filename}.gif', duration=duration)  # Adjust duration per frame as needed

    # Loop over each array in the list
    print('Creating GIF...')


    # Assuming `data` is a list of 3D numpy arrays and `plotter` is a PyVista Plotter instance

    for d in tqdm(data):
        # Get the coordinates of the voxels where the value is 1
        points = np.argwhere(d == 1)

        # Create a PyVista PolyData object with these points
        cloud = pv.PolyData(points)

        # Create a 3D glyph (cube) for each point (voxel)
        glyph = cloud.glyph(scale=False, geom=pv.Cube())

        # Clear previous data
        plotter.clear()

        # Add the glyphs to the plotter
        plotter.add_mesh(glyph, color="blue", show_edges=True, opacity=1)

        # Key light (primary light source)
        plotter.add_light(pv.Light(position=(3, 3, 5), color='white', intensity=0.5))

        # Fill light (secondary light source to soften shadows)
        plotter.add_light(pv.Light(position=(-3, -3, 3), color='white', intensity=0.8))

        # Back light (to separate the object from the background)
        plotter.add_light(pv.Light(position=(3, -3, -3), color='white', intensity=0.5))

        # Ambient light (base illumination)
        plotter.add_light(pv.Light(position=(0, 0, 0), color='white', intensity=0.2))

        # Enable shadows
        #plotter.enable_shadows()

        # Set the plot title
        if title:
            plotter.add_text(title, font_size=12)

        # Capture the frame
        plotter.write_frame()
        
    plotter.close()

#data = [np.ones((20, 20, 20)) for _ in range(1)]
#make_gif(data, 'test', 100, 'test')