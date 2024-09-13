import os
import sys

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.animation import FFMpegWriter, PillowWriter
from tqdm import tqdm

pv.global_theme.allow_empty_mesh = True

def create_visualization_pyvista(data:np.array, 
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
        else: color = 'yellow'
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



def create_visualization_grid(data: np.array, 
                              filename: str = 'animation', 
                              duration: int = 100,  
                              gif: bool = False, 
                              video: bool = False) -> None:
    
    if not gif and not video:
        raise ValueError('At least one of gif or video must be True')
    
    sizes = np.shape(data[0])
    fig = plt.figure()
    fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    

    def update(frame):
        ax.clear()
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True        
        ax.set_axis_off()

        ax.axis('off')
        fig.add_axes(ax)
        ax.imshow(data[frame], cmap='inferno', norm=plt.Normalize(0, 1))
        
    # Creating the animation object
    ani = plt.matplotlib.animation.FuncAnimation(
        fig, update, frames=len(data), interval=duration, repeat=False
    )
    
    # Save as GIF if required
    if gif:
        gif_path = f"{filename}.gif"
        ani.save(gif_path, writer=PillowWriter(fps=1000//duration))
        print(f"GIF saved as {gif_path}")
    
    # Save as video if required
    if video:
        video_path = f"{filename}.mp4"
        ani.save(video_path, writer=FFMpegWriter(fps=1000//duration))
        print(f"Video saved as {video_path}")
    
    plt.close(fig)



if __name__ == '__main__':
    pass
    #data = [np.random.randint(0, 2, (3, 3, 3)) for _ in range(50)]  # Example data
    #create_visualization_pyvista(data, 'test_video', 100, 'Test Video', gif=False, video=True, rotate=True)
    #data = [np.random.randint(0, 2, (3, 3)) for _ in range(10)]  # Example data
    #create_visualization_grid(data, 'test_video', 100, gif=False, video=True)