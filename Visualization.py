import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm


def draw_matches(matched_data):
    result = cv2.drawMatches(matched_data['test_image'], matched_data['train_key_points'], matched_data['train_image'],
                             matched_data['test_key_points'], matched_data['match_points'], None)
    plt.imshow(result)
    plt.axis("off")
    plt.show()


def plot_image(original_image, augmented_image):
    fig = plt.figure(figsize=(5, 4))

    fig.add_subplot(1, 2, 1)
    plt.title('Original Image', fontdict={'fontsize': 8})
    plt.imshow(original_image, cmap='gray')
    plt.axis("off")

    fig.add_subplot(1, 2, 2)
    plt.title('Augmented Image', fontdict={'fontsize': 8})
    plt.imshow(augmented_image, cmap='gray')
    plt.axis("off")
    plt.show()


def plot_classes(df):
    classes = ['A', 'L', 'R', 'T', 'W']
    mapped_label = {
        'A': 0,
        'L': 1,
        'R': 2,  # Right Loop
        'T': 3,  # Tented Arch
        'W': 4,  # Whirl
    }
    label_names = {
        'A': 'Arch',
        'L': 'Left Loop',
        'R': 'Right Loop',
        'T': 'Tented Arch',
        'W':  'Whirl'
    }
    dim = len(classes)
    fig, axes = plt.subplots(1, dim)
    fig.subplots_adjust(0, 0, 2, 2)
    for idx, i in enumerate(classes):
        dum = df[df['Label'] == i]
        random_num = np.random.choice(dum.index)
        label = df.loc[random_num]['Label']
        axes[idx].imshow(cv2.imread(df.loc[random_num]['Path']))
        axes[idx].set_title("CLASS: " + label_names[label] + "\n" + "LABEL:" + str(mapped_label[label]))
        axes[idx].axis('off')
    plt.show()


class Visualization:
    def __init__(self, original_images, normalized_images, segmented_images, original_segmented_images, gabor_images,
                 gabor_list_images, binary_images, skeleton_images, denoise_images, sharpened_images, enhanced_images):
        self.original_images = original_images
        self.denoise_images = denoise_images
        self.sharpened_images = sharpened_images
        self.segmented_images = segmented_images
        self.enhanced_images = enhanced_images
        self.original_segmented_images = original_segmented_images

        self.normalized_images = normalized_images
        self.gabor_images = gabor_images
        self.gabor_list_images = gabor_list_images
        self.binary_images = binary_images
        self.skeleton_images = skeleton_images

    def plot_transformation_pipeline_II(self):
        for i in range(len(self.original_images)):
            fig = plt.figure(figsize=(5, 4))

            fig.add_subplot(2, 6, 1)
            plt.title('Original Image', fontdict={'fontsize': 8})
            plt.imshow(self.original_images[i], cmap='gray')
            plt.axis("off")
            fig.add_subplot(2, 6, 7)
            plt.imshow(self.original_images[i][50:100], cmap='gray')
            plt.axis("off")

            fig.add_subplot(2, 6, 2)
            plt.title('Denoise Image', fontdict={'fontsize': 8})
            plt.imshow(self.denoise_images[i], cmap='gray')
            plt.axis("off")
            fig.add_subplot(2, 6, 8)
            plt.imshow(self.denoise_images[i][50:100], cmap='gray')
            plt.axis("off")

            fig.add_subplot(2, 6, 3)
            plt.title('Sharpened Image', fontdict={'fontsize': 8})
            plt.imshow(self.sharpened_images[i], cmap='gray')
            plt.axis("off")
            fig.add_subplot(2, 6, 9)
            plt.imshow(self.sharpened_images[i][50:100], cmap='gray')
            plt.axis("off")

            fig.add_subplot(2, 6, 4)
            plt.title('Enhanced Image', fontdict={'fontsize': 8})
            plt.imshow(self.enhanced_images[i], cmap='gray')
            plt.axis("off")
            fig.add_subplot(2, 6, 10)
            plt.imshow(self.enhanced_images[i][50:100], cmap='gray')
            plt.axis("off")

            fig.add_subplot(2, 6, 5)
            plt.title('Segmented Image', fontdict={'fontsize': 8})
            plt.imshow(self.segmented_images[i], cmap='gray')
            plt.axis("off")
            fig.add_subplot(2, 6, 11)
            plt.imshow(self.segmented_images[i][50:100], cmap='gray')
            plt.axis("off")

            fig.add_subplot(2, 6, 6)
            plt.title('Segmented on Original Image', fontdict={'fontsize': 8})
            plt.imshow(self.original_segmented_images[i], cmap='gray')
            plt.axis("off")
            fig.add_subplot(2, 6, 12)
            plt.imshow(self.original_segmented_images[i][50:100], cmap='gray')
            plt.axis("off")

            plt.show()

    def plot_gabor_images(self):
        for i in range(len(self.gabor_list_images['0'])):
            fig = plt.figure(figsize=(5, 4))

            fig.add_subplot(2, 4, 1)
            plt.title('Original Image', fontdict={'fontsize': 8})
            plt.imshow(self.original_images[i], cmap='gray')
            plt.axis("off")
            fig.add_subplot(2, 4, 5)
            plt.imshow(self.original_images[i][50:100], cmap='gray')
            plt.axis("off")

            fig.add_subplot(2, 4, 2)
            plt.title('Gabor filter without orientation error', fontdict={'fontsize': 8})
            plt.imshow(self.gabor_list_images['0'][i], cmap='gray')
            plt.axis("off")
            fig.add_subplot(2, 4, 6)
            plt.imshow(self.gabor_list_images['0'][i][50:100], cmap='gray')
            plt.axis("off")

            fig.add_subplot(2, 4, 3)
            plt.title('Gabor filter  with 45° orientation error', fontdict={'fontsize': 8})
            plt.imshow(self.gabor_list_images['45'][i], cmap='gray')
            plt.axis("off")
            fig.add_subplot(2, 4, 7)
            plt.imshow(self.gabor_list_images['45'][i][50:100], cmap='gray')
            plt.axis("off")

            fig.add_subplot(2, 4, 4)
            plt.title('Gabor filter  with 90° orientation error', fontdict={'fontsize': 8})
            plt.imshow(self.gabor_list_images['90'][i], cmap='gray')
            plt.axis("off")
            fig.add_subplot(2, 4, 8)
            plt.imshow(self.gabor_list_images['90'][i][50:100], cmap='gray')
            plt.axis("off")
            plt.show()

    def plot_transformation(self):
        for i in range(len(self.original_images)):
            fig =  plt.figure(figsize=(5, 4))

            fig.add_subplot(2, 6, 1)
            plt.title('Original Image', fontdict={'fontsize': 8})
            plt.imshow(self.original_images[i], cmap='gray')
            plt.axis("off")
            fig.add_subplot(2, 6, 7)
            plt.imshow(self.original_images[i][50:100], cmap='gray')
            plt.axis("off")

            fig.add_subplot(2, 6, 2)
            plt.title('Normalized Image', fontdict={'fontsize': 8})
            plt.imshow(self.normalized_images[i], cmap='gray')
            plt.axis("off")
            fig.add_subplot(2, 6, 8)
            plt.imshow(self.normalized_images[i][50:100], cmap='gray')
            plt.axis("off")

            fig.add_subplot(2, 6, 3)
            plt.title('Segmented Image', fontdict={'fontsize': 8})
            plt.imshow(self.original_segmented_images[i], cmap='gray')
            plt.axis("off")
            fig.add_subplot(2, 6, 9)
            plt.imshow(self.original_segmented_images[i][50:100], cmap='gray')
            plt.axis("off")

            fig.add_subplot(2, 6, 4)
            plt.title('Gabor Image with 90° orientation error', fontdict={'fontsize': 8})
            plt.imshow(self.gabor_images[i], cmap='gray')
            plt.axis("off")
            fig.add_subplot(2, 6, 10)
            plt.imshow(self.gabor_images[i][50:100], cmap='gray')
            plt.axis("off")

            fig.add_subplot(2, 6, 5)
            plt.title('Binary image of Gabor', fontdict={'fontsize': 8})
            plt.imshow(self.binary_images[i], cmap='gray')
            plt.axis("off")
            fig.add_subplot(2, 6, 11)
            plt.imshow(self.binary_images[i][50:100], cmap='gray')
            plt.axis("off")

            fig.add_subplot(2, 6, 6)
            plt.title('Skeleton Image', fontdict={'fontsize': 8})
            plt.imshow(self.skeleton_images[i], cmap='gray')
            plt.axis("off")
            fig.add_subplot(2, 6, 12)
            plt.imshow(self.skeleton_images[i][50:100], cmap='gray')
            plt.axis("off")

            plt.show()
