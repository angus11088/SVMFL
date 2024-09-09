import torch
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

class XGBoostClassifier:
    def __init__(self, n_classes=62, max_depth=6, learning_rate=0.1, n_estimators=600):
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.model = None
        self.label_encoder = LabelEncoder()

    def train(self, dataset):
        """
        Train the model.
        :param dataset: Dataset object containing feature and label data.
        """
        # Convert data to NumPy arrays as XGBoost does not directly support PyTorch Tensors
        X = dataset.xs.numpy()
        y = dataset.ys.numpy()

        # Flatten the images if they are not already flat
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        X_train, _, y_train, _ = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        params = {
            'objective': 'multi:softmax',
            'num_class': self.n_classes,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate
        }
        # Train the model
        self.model = xgb.train(params, dtrain, num_boost_round=self.n_estimators)

    def evaluate_accuracy(self, dataset):
        """
        Calculate accuracy on the test set.
        :param dataset: Dataset object containing feature and label data.
        """
        X_test = dataset.xs.numpy()
        y_test = dataset.ys.numpy()

        # Flatten the images if they are not already flat
        if X_test.ndim > 2:
            X_test = X_test.reshape(X_test.shape[0], -1)


        # Identify which test labels are known
        known_labels_mask = np.isin(y_test, self.label_encoder.classes_)
        y_test_known = y_test[known_labels_mask]
        X_test_known = X_test[known_labels_mask]

        if len(y_test_known) > 0:
            dtest = xgb.DMatrix(X_test_known)
            predictions = self.model.predict(dtest)

            # Make sure predictions and y_test_known have the same length
            if len(predictions) != len(y_test_known):
                raise ValueError("The number of predictions must match the number of test labels")

            y_test_encoded = self.label_encoder.transform(y_test_known)
            accuracy = accuracy_score(y_test_encoded, predictions.astype(int))
        else:
            accuracy = None
            print("No known labels in the test set.")

        return accuracy
        # dtest = xgb.DMatrix(X_test)
        # try:
        #     y_test_encoded = self.label_encoder.transform(y_test)
        # except:
        #     known_labels = set(self.label_encoder.classes_)
        #     unseen_labels = set(y_test) - known_labels
        #     mask = [label in known_labels for label in y_test]
        #     y_test_encoded = self.label_encoder.transform(y_test[mask])
        #     predictions = self.model.predict(dtest)[mask]

        # predictions = self.model.predict(dtest)
        # accuracy = accuracy_score(y_test_encoded, predictions.astype(int))
        # return accuracy

    def get_trees(self):
        """
        Return a string description of all trees in the model.
        """
        if self.model:
            return self.model.get_dump()
        else:
            return "Model is not trained yet."
