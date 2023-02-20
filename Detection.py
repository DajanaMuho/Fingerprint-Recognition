import cv2
import numpy as np
from sklearn.svm import SVC
import FeatureExtraction
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def find_best_matching(train_descriptors, test_descriptors):
    # Fast Nearest Neighbour search
    index_params = dict(algorithm=1, trees=10)
    search_params = dict()
    # train_descriptors = np.asarray(train_descriptors, np.float32)
    # test_descriptors = np.asarray(test_descriptors, np.float32)
    matches = cv2.FlannBasedMatcher(index_params, search_params).knnMatch(train_descriptors, test_descriptors, k=2)
    match_points = []
    for p, q in matches:
        if p.distance < 0.9 * q.distance:
            match_points.append(p)
    return match_points


class Detection:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    # Scale Invariant Feature Transform
    def SIFT(self):
        sift = cv2.xfeatures2d.SIFT_create()
        matched_data = {}
        scores = pd.DataFrame([], columns=['Filename', 'Match Score'])
        test_image = self.test_data['Image'].values[0]
        for i, data in self.train_data.iterrows():
            train_key_points, train_descriptors = sift.detectAndCompute(data['Image'], None)
            test_key_points, test_descriptors = sift.detectAndCompute(test_image, None)
            match_points = find_best_matching(train_descriptors, test_descriptors)
            key_points_len = 0
            if len(train_key_points) <= len(test_key_points):
                key_points_len = len(train_key_points)
            else:
                key_points_len = len(test_key_points)

            score = len(match_points) / key_points_len
            scores.loc[len(scores.index)] = [data['Filename'], score]
            if score > 0.95:
                print("MATCH FOUND!, There is: " + '{:.2%}'.format(score) + " match")
                print("Image Filename: " + str(data['Filename']))
                matched_data = {
                    'train_image': data['Image'],
                    'test_image': test_image,
                    'train_key_points': train_key_points,
                    'test_key_points': test_key_points,
                    'match_points': match_points
                }
                break
        print('Matches \n', scores)
        return matched_data

    def STATS_MATCH(self):
        feature_extraction = FeatureExtraction.FeatureExtraction(self.train_data, self.test_data)
        df_train, df_test = feature_extraction.first_order_statistic()
        # print('DF Train \n', df_train)
        # print('DF Test \n', df_test)
        new_df = df_train.iloc[:, 1:6].div(df_test.iloc[:, 1:6])
        print('Matches \n', new_df)
        print('Has 100% match with the filename', df_train.iloc[new_df[new_df.sum(axis=1) / 5 == 1].index]['Filename'])

    def SVM_Classifier(self):
        feature_extraction = FeatureExtraction.FeatureExtraction(self.train_data, self.test_data)
        df_train, df_test = feature_extraction.first_order_statistic()
        x_train, x_test, y_train, y_test = train_test_split(df_train.iloc[:, 1:6], df_train['Gender'])
        svm_model = SVC()
        svm_model.fit(x_train, y_train)
        svm_pred = svm_model.predict(x_test)
        svm_acc = accuracy_score(y_test, svm_pred)
        return svm_acc

    def COLOR_MATCH(self):
        feature_extraction = FeatureExtraction.FeatureExtraction(self.train_data, self.test_data)
        df = feature_extraction.color_histogram_calculation()
        print('Matches \n', df)
