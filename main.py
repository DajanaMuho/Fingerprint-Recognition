from skimage.io import imread
from skimage.color import rgb2gray
import Preprocessing
import Visualization
import Detection
import Classification
import FeatureExtraction
import os
import glob
import pandas as pd

cwd = os.getcwd()


# -------------------------------------------------- HELP METHODS ------------------------------------------------------
def read_txt_file(directory, file):
    mapped_label = {
        'A': 0,  # Arch
        'L': 1,  # Left Loop
        'R': 2,  # Right Loop
        'T': 3,  # Tented Arch
        'W': 4,  # Whirl
    }
    with open(os.path.join(directory, file), 'r') as t:
        content = t.readlines()
        gender = content[0].rsplit(' ')[1][0]
        img_name = content[2].rsplit(' ')[1][:-4] + '.png'
        img_path = os.path.join(directory, img_name)
        label = content[1].rsplit(' ')[1][0]
        mapped_label = mapped_label[label]
    return label, img_name, img_path, gender, mapped_label


def read_dataset(directory):
    labels = []
    img_names = []
    img_paths = []
    genders = []
    images = []
    mapped_labels = []
    for file in os.listdir(directory):
        # TODO: Remove this condition to include all images
        if file == 'f0002_05.png':
            break
        if file.endswith('.txt'):
            label, img_name, img_path, gender, mapped_label = read_txt_file(directory, file)
            labels.append(label)
            img_names.append(img_name)
            img_paths.append(img_path)
            genders.append(gender)
            mapped_labels.append(mapped_label)
        else:
            images.append(imread(os.path.join(directory, file)))

    df = pd.DataFrame()
    df['Path'] = img_paths
    df['Filename'] = img_names
    df['Label'] = labels
    df['Mapped Label'] = mapped_labels
    df['Gender'] = gender
    df['Image'] = images
    return df


def read_test_data(file, directory='./DATASET/'):
    images = imread(os.path.join(directory, file + '.png'))
    label, img_name, img_path, gender, mapped_label = read_txt_file(directory, file + '.txt')

    df = pd.DataFrame()
    df['Path'] = [img_path]
    df['Filename'] = [img_name]
    df['Label'] = [label]
    df['Mapped Label'] = [mapped_label]
    df['Gender'] = [gender]
    df['Image'] = [images]
    return df


def preprocess_images(images):
    preprocessing = Preprocessing.Preprocessing(images)

    normalized_images = preprocessing.apply_normalization()
    segmented_images, original_segmented_images = preprocessing.apply_segmentation()
    gabor_images, gabor_list_images = preprocessing.apply_gabor_filter()
    binary_images = preprocessing.convert_to_binary()
    skeleton_images = preprocessing.skeletonize()

    return normalized_images, segmented_images, original_segmented_images, gabor_images, gabor_list_images, \
           binary_images, skeleton_images


def preprocess_images_pipelineII(images):
    preprocessing = Preprocessing.Preprocessing(images)
    # FINGER_IMAGES_GRAY = preprocessing.convert_to_grayscale()
    denoise_images = preprocessing.apply_filter()
    sharpened_images = preprocessing.apply_sharpening()
    enhanced_images = preprocessing.apply_enhancement()
    segmented_images, original_segmented_images = preprocessing.apply_segmentation()

    return denoise_images, sharpened_images, enhanced_images, segmented_images, original_segmented_images


def plot_preprocessed_images_pipelineII(finger_images, denoise_images, sharpened_images, enhanced_images,
                                        segmented_images, original_segmented_images):
    visualization = Visualization.Visualization(finger_images, [], segmented_images, original_segmented_images, [], [],
                                                [], [], denoise_images, sharpened_images, enhanced_images,
                                                )
    visualization.plot_transformation_pipeline_II()


def plot_preprocessed_images(finger_images, normalized_images, segmented_images, original_segmented_images,
                             gabor_images, gabor_list_images, binary_images, skeleton_images):
    visualization = Visualization.Visualization(finger_images, normalized_images, segmented_images,
                                                original_segmented_images, gabor_images, gabor_list_images,
                                                binary_images, skeleton_images, [], [], [])
    visualization.plot_gabor_images()
    visualization.plot_transformation()


# ---------------------------------------------- PREPROCESS TRAINING DATA  ---------------------------------------------
TRAIN_DATA = read_dataset("./DATASET/")

# Visualize classification classes, TODO: Run for all images
# Visualization.plot_classes(TRAIN_DATA)

# Preprocess training images - PIPELINE I
TRAIN_NORMALIZED_IMAGES, TRAIN_SEGMENTED_IMAGES, TRAIN_ORIGINAL_SEGMENTED_IMAGES, TRAIN_GABOR_IMAGES, \
TRAIN_GABOR_LIST_IMAGES, TRAIN_BINARY_IMAGES, TRAIN_SKELETON_IMAGES = preprocess_images(TRAIN_DATA['Image'])

# Preprocess training images - PIPELINE II
TRAIN_DENOISE_IMAGES, TRAIN_SHARPENED_IMAGES, TRAIN_ENHANCED_IMAGES, TRAIN_SEGMENTED_IMAGES, \
TRAIN_ORIGINAL_SEGMENTED_IMAGES = preprocess_images_pipelineII(TRAIN_DATA['Image'])

# Visualize transformation
plot_preprocessed_images(TRAIN_DATA['Image'], TRAIN_NORMALIZED_IMAGES, TRAIN_SEGMENTED_IMAGES,
                         TRAIN_ORIGINAL_SEGMENTED_IMAGES,
                         TRAIN_GABOR_IMAGES, TRAIN_GABOR_LIST_IMAGES, TRAIN_BINARY_IMAGES, TRAIN_SKELETON_IMAGES)

plot_preprocessed_images_pipelineII(TRAIN_DATA['Image'], TRAIN_DENOISE_IMAGES, TRAIN_SHARPENED_IMAGES,
                                    TRAIN_ENHANCED_IMAGES, TRAIN_SEGMENTED_IMAGES, TRAIN_ORIGINAL_SEGMENTED_IMAGES)

# Use the transformed data
TRAIN_DATA['Image'] = TRAIN_SEGMENTED_IMAGES

# --------------------------------------------- PREPROCESS TESTING DATA ------------------------------------------------
# test_image_filename = input('Welcome to fingerprint detection app!\n'
#                             'Enter the filename of image to check if it matches any of the image in the database ?')

test_image_filename = 'f0001_01'
TEST_DATA = read_test_data(test_image_filename)

# Preprocess testing images
TEST_NORMALIZED_IMAGES, TEST_SEGMENTED_IMAGES, TEST_ORIGINAL_SEGMENTED_IMAGES, TEST_GABOR_IMAGES, \
TEST_GABOR_LIST_IMAGES, TEST_BINARY_IMAGES, TEST_SKELETON_IMAGES = preprocess_images(TEST_DATA['Image'])

TEST_DENOISE_IMAGES, TEST_SHARPENED_IMAGES, TEST_ENHANCED_IMAGES, TEST_SEGMENTED_IMAGES, \
TEST_ORIGINAL_SEGMENTED_IMAGES = preprocess_images_pipelineII(TEST_DATA['Image'])

# Visualize transformation
plot_preprocessed_images(TEST_DATA['Image'], TEST_NORMALIZED_IMAGES, TEST_SEGMENTED_IMAGES,
                         TEST_ORIGINAL_SEGMENTED_IMAGES,
                         TEST_GABOR_IMAGES, TEST_GABOR_LIST_IMAGES, TEST_BINARY_IMAGES, TEST_SKELETON_IMAGES)

# Use the transformed data
TEST_DATA['Image'] = TEST_SEGMENTED_IMAGES

# ------------------------------------------------- CLASSIFICATION -----------------------------------------------------
feature_extraction = FeatureExtraction.FeatureExtraction(TRAIN_DATA, TEST_DATA)
# TRAIN_DATA_FEATURES, TEST_DATA_FEATURES = feature_extraction.first_order_statistic()
TRAIN_DATA_FEATURES, TEST_DATA_FEATURES = feature_extraction.hog()

classification = Classification.Classification(TRAIN_DATA_FEATURES, TEST_DATA_FEATURES)
classification.run_rf()
classification.run_svm()
classification.run_logistic_regression()

# detection = Detection.Detection(TRAIN_DATA, TEST_DATA)
# svm_accuracy = detection.SVM_Classifier()
# print('Accuracy using SVM and first-order-stats on classifying fingerprints based in gender is: ', svm_accuracy)

# # ---------------------------------------------- COLOR FEATURES --------------------------------------------------------
# detection = Detection.Detection(TRAIN_DATA, TEST_DATA)
# detection.COLOR_MATCH()
#
# # ---------------------------------------------- HISTOGRAM FEATURES ----------------------------------------------------
# detection = Detection.Detection(TRAIN_DATA, TEST_DATA)
# detection.STATS_MATCH()
#
# # ----------------------------------------------------- SIFT -----------------------------------------------------------
# # Convert images to 8bit integer values
TRAIN_DATA_SIFT = TRAIN_DATA.copy()
TEST_DATA_SIFT = TEST_DATA.copy()
TRAIN_DATA_SIFT['Image'] = Preprocessing.convert_to_8bit(TRAIN_BINARY_IMAGES)
TEST_DATA_SIFT['Image'] = Preprocessing.convert_to_8bit(TEST_BINARY_IMAGES)

detection = Detection.Detection(TRAIN_DATA_SIFT, TEST_DATA_SIFT)
matched_data = detection.SIFT()
if len(matched_data) == 0:
    print('Match not found!')
else:
    Visualization.draw_matches(matched_data)
