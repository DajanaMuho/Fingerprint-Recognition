from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
import FeatureExtraction
from sklearn.svm import SVC
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import linear_model
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

def calculate_metrics(y_test, y_pred):
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    precision, recall, f_score, support = metrics.precision_recall_fscore_support(y_test, y_pred)
    print('Precision', precision)
    print('Recall', recall)
    print('F-Score', f_score)
    print('Support', support)


class Classification:
    def __init__(self, df, test_input):
        self.df = df
        self.test_input = test_input

    def split_data(self):
        # HISTOGRAM FEATURES
        # X = self.df.drop(['Mapped Label'], axis=1)
        # HOG FEATURES
        X = self.df.drop(['Path', 'Filename', 'Label', 'Mapped Label', 'Gender', 'Image'], axis=1)
        Y = self.df['Mapped Label']
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        print(f"Total number of Images: {len(X)}")
        print(f"Number of Training Images: {len(X_train)}")
        print(f"Number of Test Images: {len(X_test)}")
        return X_train, X_test, y_train, y_test

    def run_rf(self):
        X_train, X_test, y_train, y_test = self.split_data()
        # # Impute data
        # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        # imp_train = imp.fit(X_train)
        # X_train = imp_train.transform(X_train)
        # imp_test = imp.fit(X_train)
        # X_test = imp_test.transform(X_test)

        # rfc=RandomForestClassifier(random_state=42)
        # param_grid = {
        #     'n_estimators': [200, 500],
        #     'max_features': ['auto', 'sqrt', 'log2'],
        #     'max_depth': [4, 5, 6, 7, 8],
        #     'criterion': ['gini', 'entropy']
        # }
        # CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
        # CV_rfc.fit(X_train, y_train)
        # print('BEST PARAMS: ',CV_rfc.best_params_)
        rfc1 = RandomForestClassifier(random_state=42, max_features='auto', n_estimators=200, max_depth=4,
                                      criterion='gini')
        rfc1.fit(X_train, y_train)
        y_pred = rfc1.predict(X_test)
        calculate_metrics(y_test, y_pred)

        # y_pred_test_input = rf.predict(self.test_input)
        # return y_pred_test_input

    def run_svm(self):
        X_train, X_test, y_train, y_test = self.split_data()
        svm = SVC(kernel='linear', C=1, decision_function_shape='ovo')
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        calculate_metrics(y_test, y_pred)

    def run_logistic_regression(self):
        X_train, X_test, y_train, y_test = self.split_data()
        lr = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear')
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        calculate_metrics(y_test, y_pred)