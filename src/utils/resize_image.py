import numpy as np
from PIL import Image
from skimage import io, transform


def resize_image(pic_path: str):
    img = Image.open(pic_path)
    width, height = img.size
    if width > 512:
        percent = 512/width
        width = 512
        height = int(height*percent)
    elif height > 768:
        percent = 768/height
        height = 768
        width = int(width*percent)

    source_image = np.array(img)
    source_image = transform.resize(source_image, (height, width, 3))
    io.imsave(pic_path, source_image)
