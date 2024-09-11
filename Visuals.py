import pyvista as pv
import numpy as np
from tqdm import tqdm
import os
import imageio
import sys

pv.global_theme.allow_empty_mesh = True

def create_visualization(data:np.array, 
                         filename:str = 'animation', 
                         duration:int = 100, 
                         title:str = None, 
                         gif:bool = False, 
                         video:bool = False,
                         rotate:bool = False,
                         colors:np.array = None) -> None:
    
    if not gif and not video: raise ValueError('At least one of gif or video must be True')
    
    plotter = pv.Plotter(off_screen=True)  # Use off_screen=True to avoid displaying the window

    if not os.path.exists('_MEDIA'):
        os.mkdir('_MEDIA')

    if gif: plotter.open_gif(f'_MEDIA/{filename}.gif', duration=duration)  # Adjust duration per frame as needed
    frames = []
    fps = 1000 / duration
    num_frames = len(data)

    print('Creating visualization...')

    for i,d in enumerate(tqdm(data)):

        points = np.argwhere(d == 1)
        cloud = pv.PolyData(points)
        glyph = cloud.glyph(scale=False, geom=pv.Cube())

        plotter.clear()
        if colors is not None: color = colors[i]
        else: color = 'blue'

        plotter.add_mesh(glyph, color=color, show_edges=True, opacity=1)

        # LIGHTING
        plotter.add_light(pv.Light(position=(3, 3, 5), color='white', intensity=0.5))
        plotter.add_light(pv.Light(position=(-3, -3, 3), color='white', intensity=0.8))
        plotter.add_light(pv.Light(position=(3, -3, -3), color='white', intensity=0.5))
        plotter.add_light(pv.Light(position=(0, 0, 0), color='white', intensity=0.2))

        # CAMERA
        plotter.reset_camera()
        if rotate: plotter.camera.azimuth = float(360 * i / num_frames)

        if title: plotter.add_text(title, font_size=12)

        if gif: plotter.write_frame()
        if video:
            img = plotter.screenshot(return_img=True)
            frames.append(img)

    if gif: plotter.close()
    if video:
        with imageio.get_writer(f'_MEDIA/{filename}.mp4', fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)

# Example usage:
if __name__ == '__main__':
    data = [np.random.randint(0, 2, (3, 3, 3)) for _ in range(50)]  # Example data
    create_visualization(data, 'test_gif', 100, 'Test GIF', gif=True, video=False, rotate=True)
    create_visualization(data, 'test_video', 100, 'Test Video', gif=False, video=True, rotate=True)
