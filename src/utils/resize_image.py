import numpy as np
import cv2
from os.path import splitext
from skimage import io, transform


def resize_image(pic_path: str):
    img = cv2.imread(pic_path)
    width, height, channels = img.shape
    if channels < 4:
        print('converting to png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    if width > 512:
        percent = 512/width
        width_new = 512
        height_new = int(height*percent)
    elif height > 768:
        percent = 768/height
        height_new = 768
        width_new = int(width*percent)
    else:
        height_new = height
        width_new = width

    source_image = img.astype(np.uint8)
    source_image = transform.resize(source_image, (width_new, height_new))
    io.imsave(f'{splitext(pic_path)[0]}.png', source_image)
