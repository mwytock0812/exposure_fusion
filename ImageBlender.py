import cv2
import numpy as np
import scipy.signal


class ImageBlender:
    def __init__(self, images, weights):
        self.gaussian_pyr_images = self.gaussian_pyramid(images)
        self.laplacian_pyr_images = self.laplacian_pyramid(self.gaussian_pyr_images)
        self.gaussian_pyr_weights = self.gaussian_pyramid(weights)
        self.blended_pyramid = self.blend(self.gaussian_pyr_weights,
                                          self.laplacian_pyr_images)
        self.collapsed = self.collapse(self.blended_pyramid)
        self.final = self.convert_final_image(self.collapsed)

    def fix_dimensions(self, images):
        """Ensures even-numbered dimensions."""
        fixed = []
        for image in images:
            if image.shape[0] % 2 == 1:
                # Copy last row if num rows is off
                last_row = image[-1, :, ...]
                last_row = last_row[np.newaxis, :, ...]
                image = np.concatenate((image, last_row), axis=0)
            if image.shape[1] % 2 == 1:
                # Copy last col if num cols is off
                last_col = image[:, -1, ...]
                last_col = last_col[:, np.newaxis, ...]
                image = np.concatenate((image, last_col), axis=1)
            fixed.append(image)
        return fixed

    def gaussian_pyramid(self, images, levels=1):
        """Constructs a set of Gaussian pyramids.
        images: List of images.
        level: Integer number of levels to include in the pyramid.

        returns: List of lists with the outer list indices corresponding
        to each input image and the inner list's indices corresponding
        to each level of the pyramid.
        """
        images = self.fix_dimensions(images)
        gaussian_pyramids = []
        print images[0].shape
        for i in range(len(images)):
            image = images[i]
            pyramid = [image]
            for l in range(levels):
                image = cv2.pyrDown(image)
                pyramid.append(image)
            gaussian_pyramids.append(pyramid)
        return gaussian_pyramids

    def laplacian_pyramid(self, gauss_pyramids):
        """Constructs a set of Laplacian pyramids.
        pyramids: A list of pyramids (which are themselves lists).

        returns: List of lists with the outer list indices corresponding
        to each input pyramid and the inner list's indices corresponding
        to each level of the pyramid.
        """
        laplacian_pyramids = []
        for p in range(len(gauss_pyramids)):
            pyramid = gauss_pyramids[p]
            laplacian_pyramid = [pyramid[-1]]
            for l in range(len(pyramid)-1, 0, -1):
                gauss_expanded = cv2.pyrUp(pyramid[l])
                print l, "gauss_expanded.shape:", gauss_expanded.shape
                laplacian = cv2.subtract(pyramid[l-1], gauss_expanded)
                laplacian_pyramid.append(laplacian)
            laplacian_pyramids.append(laplacian_pyramid[::-1])
        return laplacian_pyramids

    def convert_to_float(self, pyramids):
        output = []
        for pyramid in pyramids:
            levels = []
            for level in pyramid:
                levels.append(np.float32(level))
            output.append(pyramid)
        return output

    def blend(self, gauss_pyr_weights, lapl_pyr_images):
        gauss_pyr_weights = self.convert_to_float(gauss_pyr_weights)
        lapl_pyr_images = self.convert_to_float(lapl_pyr_images)
        num_images = len(gauss_pyr_weights)
        num_levels = len(gauss_pyr_weights[0])
        result_pyramid = []
        # Revise code in this function
        for l in range(num_levels):
            level = np.zeros(lapl_pyr_images[0][l].shape, dtype=np.uint8)
            for i in range(num_images):
                gauss_pyr = gauss_pyr_weights[i][l] / 255.
                gp = np.dstack((gauss_pyr, gauss_pyr, gauss_pyr))
                lp_gp = cv2.multiply(lapl_pyr_images[i][l], gp, dtype=cv2.CV_8UC3)
                level = cv2.add(level, lp_gp)
            result_pyramid.append(level)
        return result_pyramid

    def collapse(self, pyramid):
        """Collapses a list of blended pyramids."""
        if len(pyramid) == 1:
            return pyramid[0]
        image = pyramid.pop()
        expanded = expand(image)
        if expanded.shape[0] > pyramid[-1].shape[0]:
            expanded = expanded[:-1, :]
        if expanded.shape[1] > pyramid[-1].shape[1]:
            expanded = expanded[:, :-1]
        pyramid[-1] = pyramid[-1] + expanded
        return self.collapse(pyramid)

    def convert_final_image(self, image):
        return np.uint8(image / image.max() * 255)

def expand(image):
    # WRITE YOUR CODE HERE.
    upsampled = np.zeros(shape=(image.shape[0] * 2, image.shape[1] * 2, 3), dtype=np.float32)
    for c in range(3):
        upsampled[::2, ::2, c] = image[:, :, c]
        kernel = generatingKernel(0.4)
        upsampled[:, :, c] = scipy.signal.convolve2d(upsampled[:, :, c], kernel, 'same') * 4
    return upsampled

def generatingKernel(parameter):
    """ Return a 5x5 generating kernel based on an input parameter.

    Note: This function is provided for you, do not change it.

    Args:
      parameter (float): Range of value: [0, 1].

    Returns:
      numpy.ndarray: A 5x5

    """
    kernel = np.array([0.25 - parameter / 2.0, 0.25, parameter,
                       0.25, 0.25 - parameter /2.0])
    return np.outer(kernel, kernel)
