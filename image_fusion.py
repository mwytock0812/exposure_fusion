import os
import cv2
import numpy as np
import QualityMeasures as qm

from glob import glob


def readImages(image_dir, ext_list=[], resize=False):
    """ This function reads in input images from a image directory

    Args:
        image_dir (str): The image directory to get images from.

        ext_list (list): (Optional) List of additional image file extensions
                         to read from the input folder. (The function always
                         returns images with extensions: bmp, jpeg, jpg, png,
                         tif, tiff)

        resize (bool): (Optional) If True, downsample the images by 1/4th.

    Returns:
        images(list): List of images in image_dir. Each image in the list is of
                      type numpy.ndarray.

    """
    # The main file extensions. Feel free to add more if you want to test an
    # image format of yours that is not listed (Make sure OpenCV can read it).
    extensions = ["bmp", "jpeg", "jpg", "png", "tif", "tiff"] + ext_list
    search_paths = [os.path.join(image_dir, '*.' + ext) for ext in extensions]
    image_files = sorted(reduce(list.__add__, map(glob, search_paths)))
    images = [cv2.imread(f, cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR)
              for f in image_files]
    images = [i * 1./255 for i in images]
    images = [np.float32(i) for i in images]

    if resize:
        images = [img[::4, ::4] for img in images]

    return images


if __name__ == "__main__":
    np.random.seed()
    image_dir = "input"
    rgb_images = readImages(image_dir, resize=False)
    gray_images = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in rgb_images]

    rgb_stack = np.concatenate([i[..., np.newaxis] for i in rgb_images], axis=3)
    gray_stack = np.concatenate([g[..., np.newaxis] for g in gray_images], axis=2)

    quality = qm.QualityMeasures(rgb_images, gray_images, rgb_stack, gray_stack)

    cv2.pyrUp(rgb_images[0])
    import pdb
    pdb.set_trace()
    # fused = np.median(rgb_stack, axis=3)
    # grayscale = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("fused.jpg", fused)
    # cv2.imwrite("grayscale.jpg", grayscale)
