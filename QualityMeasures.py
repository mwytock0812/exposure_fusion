import cv2
import numpy as np


class QualityMeasures:
    def __init__(self, rgb_images, gray_images):
        """Generates quality measures and weights from a list of float32 images.

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

    def normalize_image_list(self, image_list):
        axes = len(image_list[0].shape)
        stack = np.concatenate([i[..., np.newaxis] for i in image_list], axis=axes)
        maximum = stack.max()
        image_list = [i / maximum for i in image_list]
        return image_list

    def get_contrast(self, gray_images):
        """

        Note: This returns a gray image with only one channel.
        """
        # Unsure whether to blur first
        # gaussian_blur = [cv2.GaussianBlur(g, (3, 3), 0) for g in gray_images]
        # laplacian = [cv2.Laplacian(gb, ddepth=cv2.CV_32F, ksize=3,
        #                            borderType=cv2.BORDER_REFLECT101)
        #              for gb in gaussian_blur]

        laplacian = [cv2.Laplacian(g, ddepth=cv2.CV_32F, ksize=3,
                                   borderType=cv2.BORDER_REFLECT101)
                     for g in gray_images]
        abs_laplacian = [np.absolute(l) for l in laplacian]
        norm_laplacian = self.normalize_image_list(abs_laplacian)
        return norm_laplacian

    def get_saturation(self, rgb_images):
        """

        Note: This returns a gray image with only one channel.
        """
        saturation = [np.std(i, axis=2) for i in rgb_images]
        norm_saturation = self.normalize_image_list(saturation)
        return norm_saturation

    def get_exposedness(self, rgb_images, sigma=0.2):
        def weigh_intensities(image):
            weights = np.ones(shape=image.shape[0:2])
            for channel in range(3):
                weights *= np.exp(-1 * ((image[:, :, channel] - 0.5) ** 2) / (2 * (sigma ** 2)))
            return weights

        exposedness = [weigh_intensities(i) for i in rgb_images]
        norm_exposedness = self.normalize_image_list(exposedness)
        return norm_exposedness

    def compute_W(self, contrast, saturation, exposedness, w_con, w_sat, w_exp):
        def replace_zeros(array):
            array[array == 0] = 2 ** -149
            return array

        contrast = replace_zeros(contrast)
        saturation = replace_zeros(saturation)
        exposedness = replace_zeros(exposedness)

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
            weights = np.uint8(inverted_weights * weight_list[k] * 255)
            norm_weight_list.append(weights)

        return norm_weight_list

    def get_result(self, weights, images):
        results = []
        for i in range(len(images)):
            output = np.zeros(shape=images[i].shape)
            output = weights[i][:, :, np.newaxis] * images[i]
            results.append(output)
        result_stack = np.concatenate([i[..., np.newaxis] for i in results], axis=3)
        result = result_stack.sum(axis=3)
        result = np.uint8(result / result.max() * 255)
        return result
