import os
import cv2
import numpy as np
import QualityMeasures as qm
import ImageBlender as ib

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
    if resize:
        images = [img[::4, ::4] for img in images]
    return images


if __name__ == "__main__":
    np.random.seed()
    image_dir = "input"
    images = readImages(image_dir, resize=False)

    quality = qm.QualityMeasures(images)

    blend = ib.ImageBlender(images, quality.weights)

    # fused = np.median(rgb_stack, axis=3)
    # grayscale = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("contrast.jpg", quality.contrast)
    #cv2.imwrite("saturation.jpg", quality.saturation)
    #cv2.imwrite("exposedness.jpg", quality.exposedness)
    counter = 0
    for weight in quality.weights:
        cv2.imwrite("weight_map%s.jpg" % counter, weight)
        counter += 1

    cv2.imwrite("contrast.jpg", np.uint8(quality.contrast[0] / quality.contrast[0].max() * 255))
    cv2.imwrite("saturation.jpg", np.uint8(quality.saturation[0] / quality.saturation[0].max() * 255))
    cv2.imwrite("exposedness.jpg", np.uint8(quality.exposedness[0] / quality.exposedness[0].max() * 255))

    cv2.imwrite("naive_result.jpg", quality.naive_result)
    cv2.imwrite("fused.jpg", blend.final)
    # cv2.imwrite("grayscale.jpg", grayscale)
