import cv2
import numpy as np


class QualityMeasures:
    def __init__(self, images):
        """Generates quality measures and weights from a list of uint8 images.
        """
        # Consider checking number of channels
        self.float_images = [np.float32(i) / 255 for i in images]
        self.contrast = self.get_contrast(self.float_images)
        self.saturation = self.get_saturation(self.float_images)
        self.exposedness = self.get_exposedness(self.float_images)

        self.weights = self.get_weights(self.contrast,
                                        self.saturation,
                                        self.exposedness)
        self.naive_result = self.get_result(self.weights, images)

    def normalize_image_list(self, image_list):
        axes = len(image_list[0].shape)
        stack = np.concatenate([i[..., np.newaxis] for i in image_list], axis=axes)
        maximum = stack.max()
        image_list = [i / maximum for i in image_list]
        return image_list

    def get_contrast(self, images):
        """

        Note: This returns a gray image with only one channel.
        """
        # Unsure whether to blur first
        # gaussian_blur = [cv2.GaussianBlur(g, (3, 3), 0) for g in gray_images]
        # laplacian = [cv2.Laplacian(gb, ddepth=cv2.CV_32F, ksize=3,
        #                            borderType=cv2.BORDER_REFLECT101)
        #              for gb in gaussian_blur]

        gray = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in images]
        laplacian = [cv2.Laplacian(g, ddepth=cv2.CV_32F) for g in gray]
        abs_laplacian = [np.absolute(l) for l in laplacian]
        norm_laplacian = self.normalize_image_list(abs_laplacian)
        return abs_laplacian

    def get_saturation(self, images):
        """

        Note: This returns a gray image with only one channel.
        """
        saturation = [i.std(axis=2, dtype=np.float32) for i in images]
        norm_saturation = self.normalize_image_list(saturation)
        return saturation

    def get_exposedness(self, images, sigma=0.2):
        def weigh_intensities(image):
            weights = np.prod(np.exp(-1 * ((image - 0.5) ** 2) / (2 * sigma)), axis=2, dtype=np.float32)
            return weights

        exposedness = [weigh_intensities(i) for i in images]
        norm_exposedness = self.normalize_image_list(exposedness)
        return exposedness

    def compute_W(self, contrast, saturation, exposedness, w_con, w_sat, w_exp):
        W_c = contrast ** w_con + 1
        W_s = saturation ** w_sat + 1
        W_e = exposedness ** w_exp + 1
        W = W_c * W_s * W_e
        return W

    def get_weights(self, contrast, saturation, exposedness,
                    w_con=1, w_sat=1, w_exp=1):
        weight_list = [self.compute_W(contrast[i],
                                      saturation[i],
                                      exposedness[i],
                                      w_con,
                                      w_sat,
                                      w_exp) for i in range(len(contrast))]
        sum_weights = np.zeros(shape=weight_list[0].shape[:2], dtype=np.float32)
        for weights in weight_list:
            sum_weights += weights
        nonzero = sum_weights > 0

        norm_weight_list = []
        for k in range(len(weight_list)):
            weight_list[k][nonzero] /= sum_weights[nonzero]
            norm_weight_list.append(np.uint8(weight_list[k] * 255))

        return norm_weight_list

    def get_result(self, weights, images):
        weights = [np.float32(w) / 255 for w in weights]
        weighted_results = []
        for i in range(len(images)):
            output = weights[i][:, :, np.newaxis] * images[i]
            weighted_results.append(output)
        result = np.zeros(shape=images[i].shape, dtype=np.float32)
        for i in range(len(weighted_results)):
            result += weighted_results[i]
        result = np.uint8(result)
        return result
