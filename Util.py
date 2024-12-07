import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import requests
import io

def load_simple_image_as_numpy_array(image_path):
    image = PIL.Image.open(image_path)
    
    if image.mode == 'RGBA':
        image_array = np.array(image)
        alpha_channel = image_array[:, :, 3]
        image_array = alpha_channel
        image_array = np.where(image_array > 0, 1, 0)
    
    return image_array

def load_image_as_numpy_array(image_path, normalize=True, black_and_white=False, binary=False, sensibility=0.5):
    image = PIL.Image.open(image_path).convert("RGB")  # Ensure image is in RGB mode
    image_array = np.array(image)
    if normalize:
        image_array = image_array / 255.0
    if black_and_white:
        image_array = np.mean(image_array, axis=-1)
    if binary:
        image_array = np.where(image_array > sensibility, 1, 0)
    return image_array


def x0_sampling(dist, nb_params):
    if dist == "U[0,1]":
        return np.random.rand(nb_params)
    elif dist == "U[-1,1]":
        return 2 * np.random.rand(nb_params) - 1
    elif dist == "N[0,1]":
        return np.random.randn(nb_params)
    else:
        raise ValueError("Distribution not available")


# Load and preprocess emoji image
def load_image(url, size=64):
    r = requests.get(url)
    img = PIL.Image.open(io.BytesIO(r.content))
    img.thumbnail((size, size), PIL.Image.LANCZOS)
    img = np.float32(img) / 255.0
    img[..., :3] *= img[..., 3:]
    img = np.array(img)
    return img


def load_emoji(emoji, size, code=None):
    if code is None:
        code = hex(ord(emoji))[2:].lower()
    url = "https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true" % code
    return load_image(url, size)


def emoji_to_numpy(emoji, size, remove_alpha=True):
    img_tensor = load_emoji(emoji, size)
    img_np = img_tensor.numpy()
    img_np = img_np
    if remove_alpha:
        img_np = img_np[:3]
    return img_np