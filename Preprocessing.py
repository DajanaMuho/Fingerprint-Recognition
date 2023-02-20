import numpy as np
from skimage.color import rgb2gray
import cv2
import math
from skimage.morphology import skeletonize as skelt
from skimage.morphology import thin
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def find_pixel_value(block, algorithm, filter_type='mean'):
    m = 0
    if algorithm == 'filter':
        m_flat = np.array(block).flatten()
        m_sorted = sorted(m_flat)
        block_length = len(m_sorted)
        index = (block_length - 1) // 2
        variance_global = np.var(m_flat)
        if filter_type == 'mean':
            m = np.mean(block, dtype=np.float32)
        if filter_type == 'geometric':
            m_prod = np.prod(np.array(block).flatten())
            m = int(m_prod ** (1 / block.size))
        if filter_type == 'harmonic':
            m = len(block) / np.sum(1 / block)
        if filter_type == 'weighted_mean':
            weights = np.full(block.size, 0.5)
            m = int(np.average(m_flat, weights=weights))
        if filter_type == 'median' or filter_type == 'midpoint':
            m_array = m_sorted if filter_type == 'median' else m_flat
            m_length = len(m_array)
            index = (m_length - 1) // 2
            if m_length % 2:
                m = m_array[index]
            else:
                m = (m_array[index] + m_array[index + 1]) / 2.0
        if filter_type == 'weighted_median':
            weights = np.full(block.size, 2)
            data, weights = np.array(m_flat).squeeze(), np.array(weights).squeeze()
            s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
            midpoint = 0.5 * sum(s_weights)
            if any(weights > midpoint):
                w_median = (data[weights == np.max(weights)])[0]
            else:
                cs_weights = np.cumsum(s_weights)
                idx = np.where(cs_weights <= midpoint)[0][-1]
                if cs_weights[idx] == midpoint:
                    w_median = np.mean(s_data[idx:idx + 2])
                else:
                    w_median = s_data[idx + 1]
            m = w_median
        if filter_type == 'max':
            m = m_sorted[len(m_sorted) - 1]
        if filter_type == 'min':
            m = m_sorted[0]
        if filter_type == 'alfa_trimmed_mean':
            alpha = 0.1
            a = round(alpha * len(m_flat))
            if a == 0:
                m = sum(m_flat) / len(m_flat)
                return m
            trimmed_list = m_sorted[a:-a]
            if len(trimmed_list) == 0:
                m = 0
                return m
            m = sum(trimmed_list) / len(trimmed_list)
        if filter_type == 'adaptive_mean':
            variance_local = np.var(m_flat)
            median_local = np.median(m_flat)
            new_block = block - ((variance_global / variance_local) * (block - median_local))
            if  math.isnan(new_block[1][1]):
                m = 0
            else:
                m = new_block[1][1]
    if algorithm == 'sauvola':
        k = 0.2
        R = 128
        mean = np.mean(block)
        std = np.std(block)
        m = mean * (1 + k * ((std / R) - 1))

    return m


def convolution(img, algorithm, filter_type='mean'):
    image_copy = img.copy()
    k_size = 3
    row, col = img.shape[:2]
    new_image = np.zeros([row, col], dtype=np.int)
    kernel = np.ones((k_size, k_size), np.float32) / k_size * k_size
    for i in range(0, row - k_size):
        for j in range(0, col - k_size):
            block = image_copy[i:i + k_size, j:j + k_size]
            if algorithm == 'sauvola':
                threshold = find_pixel_value(block, algorithm)
                if image_copy[i, j] < threshold:
                    new_image[i, j] = 0
                else:
                    new_image[i, j] = 255
            else:
                new_image[i + k_size - 2][j + k_size - 2] = find_pixel_value(block, algorithm, filter_type)

    return new_image


def convolve_filter(img, kernel):
    row, col = img.shape[:2]
    k = kernel.shape[0]
    target_row_size = row - k
    target_col_size = col - k
    convolved_img = np.zeros(shape=(target_row_size, target_col_size))

    for i in range(target_row_size):
        for j in range(target_col_size):
            block = img[i:i + k, j:j + k]
            convolved_img[i, j] = np.sum(np.multiply(block, kernel))

    return convolved_img


def subtract_images(image_1, image_2):
    return image_1 - image_2


def apply_augmentation(test_images, technique):
    new_images = []
    for test_image in test_images:
        if technique == 'rotation':
            new_images.append(cv2.flip(test_image, 1))
            # test_image = np.array(test_image)
            # angle = 25
            # angle = int(np.random.uniform(-angle, angle))
            # h, w = test_image.shape[:2]
            # M = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, 1)
            # new_images.append(cv2.warpAffine(test_image, M, (w, h)))
    return new_images


def convert_to_8bit(images):
    return [cv2.normalize(np.array(img), None, 0, 255, cv2.NORM_MINMAX).astype('uint8') for img in images]


def normalize_pixel(x, v0, v, m, m0):
    dev_coeff = math.sqrt((v0 * ((x - m) ** 2)) / v)
    return m0 + dev_coeff if x > m else m0 - dev_coeff

def plot_filters(list_img, labels):
    fig, axes = plt.subplots(2, 6)
    for i in range(2):
        for j in range(6):
            axes[i][j].title.set_text(labels[i][j]), axes[i][j].title.set_size(7)
            axes[i][j].imshow(list_img[i][j], cmap='gray')  # , interpolation='nearest', aspect='auto')
            axes[i][j].axis("off")
    plt.show()

def plot_sharpening(original_image, filter_image, label, alfa=0.1):
    fig, ax = plt.subplots(nrows=2, ncols=3)

    ax[0][0].imshow(original_image, cmap='gray')
    ax[0][0].set_title('Original image', fontdict={'fontsize': 8})
    ax[0][0].axis('off')

    ax[0][1].imshow(filter_image, cmap='gray')
    ax[0][1].set_title(label + ' image', fontdict={'fontsize': 8})
    ax[0][1].axis('off')

    detail = original_image - filter_image
    ax[0][2].imshow(detail, cmap='gray')
    ax[0][2].set_title('Detail', fontdict={'fontsize': 8})
    ax[0][2].axis('off')

    ax[1][0].imshow(original_image, cmap='gray')
    ax[1][0].axis('off')

    ax[1][1].imshow(detail, cmap='gray')
    ax[1][1].axis('off')

    filter_im = original_image + (alfa * detail)
    ax[1][2].imshow(filter_im, cmap='gray')
    ax[1][2].axis('off')
    plt.show()

def get_fft(img):
    return np.fft.fftshift(np.fft.fft2(img))


def get_i_fft(f_img):
    return np.fft.ifft2(np.fft.ifftshift(f_img))


def get_magnitude(f_img):
    return 20 * np.log(np.abs(f_img))


def get_phase(f_img):
    return np.angle(f_img)


def get_combined_image(f_img):
    combined = np.multiply(np.abs(f_img), np.exp(1j * get_phase(f_img)))
    return np.real(get_i_fft(combined))

def plot_transform_images(gray_image, transformed_image, inverse_transformed_image):
    fig = plt.figure(figsize=(5, 4))

    fig.add_subplot(2, 3, 1)
    plt.title('Original Grayscale Image', fontdict={'fontsize': 8})
    plt.imshow(gray_image, cmap='gray')
    plt.axis("off")

    fig.add_subplot(2, 3, 2)
    plt.title('FFT (shifted)', fontdict={'fontsize': 8})
    plt.imshow(np.abs(transformed_image), cmap='gray')
    plt.axis("off")

    fig.add_subplot(2, 3, 4)
    plt.title('Magnitude', fontdict={'fontsize': 8})
    plt.imshow(get_magnitude(transformed_image), cmap='gray')
    plt.axis("off")

    fig.add_subplot(2, 3, 5)
    plt.title('Phase', fontdict={'fontsize': 8})
    plt.imshow(get_phase(transformed_image), cmap='gray')
    plt.axis("off")

    fig.add_subplot(2, 3, 3)
    plt.title('Combined Image', fontdict={'fontsize': 8})
    plt.imshow(get_combined_image(transformed_image), cmap='gray')
    plt.axis("off")

    fig.add_subplot(2, 3, 6)
    plt.title('Inverse FFT (shifted)', fontdict={'fontsize': 8})
    plt.imshow(get_combined_image(transformed_image), cmap='gray')
    plt.axis("off")

    plt.show()


def plot_enhancement(img, fft_img, alpha):
    new_magnitude = np.abs(fft_img) ** alpha
    combined = np.multiply(new_magnitude, np.exp(1j * get_phase(fft_img)))
    combined_img = np.real(get_i_fft(combined))

    fig = plt.figure(figsize=(5, 4))

    fig.add_subplot(2, 3, 1)
    plt.title('Original Grayscale Image', fontdict={'fontsize': 8})
    plt.imshow(img, cmap='gray')
    plt.axis("off")

    fig.add_subplot(2, 3, 2)
    plt.title('Magnitude ^ Alpha', fontdict={'fontsize': 8})
    plt.imshow(np.abs(new_magnitude), cmap='gray')
    plt.axis("off")

    fig.add_subplot(2, 3, 4)
    plt.title('Magnitude', fontdict={'fontsize': 8})
    plt.imshow(get_magnitude(fft_img), cmap='gray')
    plt.axis("off")

    fig.add_subplot(2, 3, 5)
    plt.title('Phase', fontdict={'fontsize': 8})
    plt.imshow(get_phase(fft_img), cmap='gray')
    plt.axis("off")

    fig.add_subplot(2, 3, (3, 6))
    plt.title('Combined Image', fontdict={'fontsize': 8})
    plt.imshow(combined_img, cmap='gray')
    plt.axis("off")

    plt.show()


class Preprocessing:
    def __init__(self, images):
        self.original_images = images
        self.denoise_images = []
        self.sharpened_images = []
        self.enhanced_images = []
        self.segmented_images = []
        self.original_segmented_images = []

        self.normalized_images = []
        self.gabor_images = []
        self.binary_images = []
        self.gabor_list_images = []
        self.skeleton_images = []

    # Color space
    def convert_to_grayscale(self):
        gray_scale_images = []
        for image in self.original_images:
            gray_scale_images.append(rgb2gray(image))
        self.original_images = gray_scale_images
        return gray_scale_images

    # Denoise
    def apply_filter(self):
        denoise_images = []
        for image in self.original_images:
            new_image = convolve_filter(image, np.ones((3, 3), dtype="float") * (1.0 / (3 * 3)))
            denoise_images.append(new_image)
        self.denoise_images = denoise_images
        MEAN_IMG = convolution(self.original_images[0], 'filter', 'mean')

        GEOMETRIC_IMG = convolution(self.original_images[0], 'filter', 'geometric')

        HARMONIC_IMG = convolution(self.original_images[0], 'filter', 'harmonic')

        WEIGHTED_MEAN_IMG = convolution(self.original_images[0], 'filter',  'weighted_mean')

        MEDIAN_IMG = convolution(self.original_images[0], 'filter',  'median')

        MIDPOINT_IMG = convolution(self.original_images[0], 'filter',  'midpoint')

        WEIGHTED_MEDIAN_IMG = convolution(self.original_images[0], 'filter', 'weighted_median')

        MAX_IMG = convolution(self.original_images[0], 'filter',  'max')

        MIN_IMG = convolution(self.original_images[0], 'filter',  'min')

        ALFA_TRIMMED_IMG = convolution(self.original_images[0], 'filter', 'alfa_trimmed_mean')

        ADAPTIVE_MEAN = convolution(self.original_images[0], 'filter', 'adaptive_mean')

        img_list = [[self.original_images[0], MEAN_IMG, WEIGHTED_MEAN_IMG, GEOMETRIC_IMG, HARMONIC_IMG, MEDIAN_IMG,],
                    [WEIGHTED_MEDIAN_IMG, MAX_IMG, MIN_IMG, MIDPOINT_IMG, ALFA_TRIMMED_IMG, ADAPTIVE_MEAN]]

        img_list_label = [['Original Image', 'Mean', 'Weighted Mean', 'Geometric', 'Harmonic', 'Median',],
                          ['Weighted Median', 'Max', 'Min', 'Midpoint', 'Alfa Trimmed Mean', 'Adaptive Mean']]
        plot_filters(img_list, img_list_label)

        plot_sharpening(self.original_images[0], MEAN_IMG, 'Mean')

        FINGER_IMG_GRAY_fft = get_fft(self.original_images[0])
        FINGER_IMG_GRAY_i_fft = get_i_fft(FINGER_IMG_GRAY_fft)

        plot_transform_images(self.original_images[0], FINGER_IMG_GRAY_fft, FINGER_IMG_GRAY_i_fft)
        plot_enhancement(self.original_images[0], FINGER_IMG_GRAY_fft, 0.7)
        return denoise_images

    # Sharpening
    def apply_sharpening(self, alpha=3):
        sharpened_images = []
        for image in self.denoise_images:
            sharpen_filter = np.array(([0, -1, 0],
                                       [-1, 5, -1],
                                       [0, -1, 0]), dtype="int")
            # sharpen = convolve2d(self.enhanced_images[i], sharpen_filter, 'same', boundary='fill', fillvalue = 0)
            # sharpen = cv2.filter2D(src=self.enhanced_images[i], ddepth=-1, kernel=sharpen_filter)
            # detailed_image = subtract_images(self.enhanced_images[i], self.denoise_images[i])
            # sharpen = self.enhanced_images[i] + (alpha * detailed_image)
            sharpen = convolve_filter(image, sharpen_filter)
            sharpen[sharpen < 0] = 0
            sharpened_images.append(sharpen)
        self.sharpened_images = sharpened_images
        return sharpened_images

    # Enhancement
    def apply_enhancement(self):
        transformed_images = []
        gamma = 3.2
        for image in self.sharpened_images:
            gamma_corrected = np.array(255 * (image / 255) ** gamma, dtype='uint8')
            c = 255 / np.log(1 + np.max(gamma_corrected))
            new_image = c * (np.log(gamma_corrected + 1))
            transformed_images.append(new_image)
        self.enhanced_images = transformed_images

        # # Visualization
        # fig = plt.figure(figsize=(5, 4))
        # k = 1
        # img = self.sharpened_images[0]
        # for gamma in [0.1, 0.5, 1.2, 2.2, 3.2]:
        #     gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')
        #     fig.add_subplot(2, 5, k)
        #     plt.title('Gamma=' + str(gamma))
        #     plt.imshow(gamma_corrected, cmap='gray')
        #     fig.add_subplot(2, 5, k + 5)
        #     plt.title('Gamma=' + str(gamma))
        #     plt.plot(sorted(img.flatten()), sorted(gamma_corrected.flatten()))
        #     k = k + 1
        # plt.show()

        return transformed_images

    # Segmentation
    def apply_segmentation(self):
        segmented_images = []
        original_segmented_images = []
        for image in self.original_images:
            new_image = convolution(image, 'sauvola')
            original_segmented_images.append(new_image)
        for image in self.enhanced_images: # self.normalized_images: -PIPELINE I
            new_image = convolution(image, 'sauvola')
            segmented_images.append(new_image)

        self.segmented_images = segmented_images
        self.original_segmented_images = original_segmented_images
        return segmented_images, original_segmented_images

    def apply_normalization(self, m0=float(100), v0=float(100)):
        normalized_images = []
        for image in self.original_images:
            m = np.mean(image)
            v = np.std(image) ** 2
            (y, x) = image.shape
            normalize_image = image.copy()
            for i in range(x):
                for j in range(y):
                    normalize_image[j, i] = normalize_pixel(image[j, i], v0, v, m, m0)
            normalized_images.append(normalize_image)

        self.normalized_images = normalized_images
        return normalized_images

    def apply_gabor_filter(self):
        gabor_images = {
            '0': [],
            '45': [],
            '90': [],
        }
        ksize = 31
        thetas = {0, 45, 90}

        for image in self.original_segmented_images:
            # gabor_image = np.zeros_like(image)
            for theta in thetas:
                kern = cv2.getGaborKernel((ksize, ksize), 8.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
                kern /= 1.5 * kern.sum()
                gabor_image = cv2.filter2D(image, cv2.CV_8UC3, kern)
                gabor_images[str(theta)].append(gabor_image)

        self.gabor_images = gabor_images['90']
        self.gabor_list_images = gabor_images
        return gabor_images['90'], gabor_images

    def convert_to_binary(self):
        binary_images = []
        for image in self.gabor_images:
            ret, gabor_image_binarized = cv2.threshold(image / 255, 0.3, 255, cv2.THRESH_BINARY)
            binary_images.append(gabor_image_binarized)

        self.binary_images = binary_images
        return binary_images

    def skeletonize(self):
        skeleton_images = []
        for image in self.binary_images:
            img = np.zeros_like(image)
            img[image == 0] = 1.0
            skeleton_image = np.zeros_like(image)
            skeleton = skelt(image / 255)
            skeleton_image[skeleton] = 255
            cv2.bitwise_not(skeleton_image, skeleton_image)
            skeleton_images.append(skeleton_image)
            # skeleton_images.append(cv2.ximgproc.thinning(image))
        self.skeleton_images = skeleton_images
        return skeleton_images
