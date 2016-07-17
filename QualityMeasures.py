import cv2
import numpy as np


class QualityMeasures:
    def __init__(self, rgb_images, gray_images, rgb_stack, gray_stack):
        """Generates quality measures from a list of float32 images.

        The RGB image stack is concatenated along axis=3, while gray is
        concatenated along axis=2.
        """
        # Consider checking number of channels
        self.contrast = self.get_contrast(gray_images)
        self.saturation = self.get_saturation(rgb_images)
        self.exposedness = self.get_exposedness(rgb_images)

        self.weights = self.get_weights(self.contrast,
                                        self.saturation,
                                        self.exposedness)

        self.naive_result = self.get_result(self.weights, rgb_images)


    def get_contrast(self, gray_images):
        """

        Note: This returns a gray image with only one channel.
        """
        gaussian_blur = [cv2.GaussianBlur(g, (3, 3), 0) for g in gray_images]
        laplacian = [cv2.Laplacian(gb, ddepth=cv2.CV_32F, ksize=3,
                                   borderType=cv2.BORDER_REFLECT101)
                     for gb in gaussian_blur]
        abs_laplacian = [np.absolute(l) for l in laplacian]
        return abs_laplacian

    def get_saturation(self, rgb_images):
        """

        Note: This returns a gray image with only one channel.
        """
        saturation = [np.std(i, axis=2) for i in rgb_images]
        return saturation

    def get_exposedness(self, rgb_images, sigma=0.2):
        def weigh_intensities(image):
            weights = np.ones(shape=image.shape[0:2])
            for channel in range(3):
                weights *= np.exp(-1 * ((image[:, :, channel] - 0.5) ** 2) / ((2 * sigma) ** 2))
            return weights

        exposedness = [weigh_intensities(i) for i in rgb_images]
        return exposedness

    def compute_W(self, contrast, saturation, exposedness, w_con, w_sat, w_exp):
        W = (contrast ** w_con) * (saturation ** w_sat) * (exposedness ** w_exp)
        return W

    def get_weights(self, contrast, saturation, exposedness,
                    w_con=1, w_sat=1, w_exp=1):
        weight_list = [self.compute_W(contrast[i],
                                      saturation[i],
                                      exposedness[i],
                                      w_con,
                                      w_sat,
                                      w_exp) for i in range(len(contrast))]

        qm_stack = np.concatenate([i[..., np.newaxis] for i in weight_list], axis=2)
        inverted_weights = qm_stack.sum(axis=2) ** -1

        norm_weight_list = []
        for k in range(len(weight_list)):
            norm_weight_list.append(inverted_weights * weight_list[k])

        return norm_weight_list

    def get_result(self, weights, images):
        results = []
        for i in range(len(images)):
            output = np.zeros(shape=images[i].shape)
            output = weights[i][:, :, np.newaxis] * images[i]
            results.append(output)
        result_stack = np.concatenate([i[..., np.newaxis] for i in results], axis=3)
        result = result_stack.sum(axis=3)
        return result
