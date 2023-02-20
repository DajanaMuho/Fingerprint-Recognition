import numpy as np
import pandas as pd
import scipy.stats as stats
import cv2
from skimage.transform import resize
from skimage.feature import hog
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def stats_calculations(data_images):
    mean = []
    std_deviation = []
    smoothness = []
    skewness = []
    kurtosis = []
    entropy = []
    mapped_labels = []
    for i, data in data_images.iterrows():
        image = data['Image']
        mapped_labels.append(data['Mapped Label'])
        mean.append(np.mean(image))
        std_deviation.append(np.std(image))
        variance = np.var(image)
        smoothness.append(1 - 1 / 1 + variance ** 2)
        skewness.append(stats.skew(image.flatten()))
        kurtosis.append(stats.kurtosis(image.flatten()))
        entropy.append(stats.entropy(image.flatten()))

    return pd.DataFrame(
        {'Mapped Label': mapped_labels, 'Mean': mean, 'Standard Deviation': std_deviation, 'Skewness': skewness,
         'Kurtosis': kurtosis, 'Entropy': entropy})


class FeatureExtraction:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def first_order_statistic(self):
        df_train = stats_calculations(self.train_data)
        df_test = stats_calculations(self.test_data)
        return df_train, df_test

    def color_histogram_calculation(self):
        df = pd.DataFrame([], columns=['Filename', 'Correlation'])
        test_img = cv2.imread(self.test_data['Filename'].values[0])
        hist_test = cv2.calcHist(test_img, [0, 1, 2], None, [256, 256, 256], [0, 256] + [0, 256] + [0, 256],
                                 accumulate=False)
        for i, data in self.train_data.iterrows():
            train_img = cv2.imread(data['Filename'])
            hist_train = cv2.calcHist(train_img, [0, 1, 2], None, [256, 256, 256], [0, 256] + [0, 256] + [0, 256],
                                      accumulate=False)
            correlation = cv2.compareHist(hist_test, hist_train, cv2.HISTCMP_CORREL)
            df.loc[len(df.index)] = [data['Filename'], correlation]
        return df

    def sift(self):
        sift = cv2.xfeatures2d.SIFT_create()
        train_key_points = []
        train_descriptors = []
        test_key_points = []
        test_descriptors = []
        for i, data in self.train_data.iterrows():
            train_key_point, train_descriptor = sift.detectAndCompute(data['Image'], None)
            train_key_points.append(train_key_point)
            train_descriptors.append(train_descriptor)
        for i, data in self.train_data.iterrows():
            test_key_point, test_descriptor = sift.detectAndCompute(data['Image'], None)
            test_key_points.append(test_key_point)
            test_descriptors.append(test_descriptor)

        return train_key_points, train_descriptors, test_key_points, test_descriptors

    def hog(self):
        train_hog_features = []
        test_hog_features = []
        for i, data in self.train_data.iterrows():
            img = data['Image']
            resized_img = resize(img, (128 * 4, 64 * 4))
            fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                visualize=True)
            # #VISUALIZE
            # plt.subplot(1, 3, 1)
            # plt.title('Original Image', fontdict={'fontsize': 8})
            # plt.imshow(img, cmap='gray')
            # plt.axis('off')
            #
            # plt.subplot(1, 3, 2)
            # plt.title('Resized Image', fontdict={'fontsize': 8})
            # plt.imshow(resized_img, cmap='gray')
            # plt.axis('off')
            #
            # plt.subplot(1, 3, 3)
            # plt.title('HOG Features', fontdict={'fontsize': 8})
            # plt.imshow(hog_image, cmap='gray')
            # plt.axis('off')
            # plt.show()

            train_hog_features.append(fd)
        for i, data in self.test_data.iterrows():
            img = data['Image']
            resized_img = resize(img, (128 * 4, 64 * 4))
            fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                visualize=True)
            test_hog_features.append(fd)
        self.train_data = pd.concat([self.train_data, pd.DataFrame(train_hog_features)], axis=1)
        self.test_data = pd.concat([self.test_data, pd.DataFrame(test_hog_features)], axis=1)
        return self.train_data, self.test_data
