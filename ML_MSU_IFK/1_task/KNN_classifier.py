import numpy as np

class KNN_classifier:
    def init(self, n_neighbors: int, **kwargs):
        self.n_neighbors = n_neighbors

    def fit(self, X_train: np.array, y_train: np.array):
        self.X_train, self.y_train = X_train, y_train
        pass

    def make_predictions(self, x_test_i: np.array):
        distances = self.euclidian_metric(x_test_i)
        k_nearest_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearest_objects = [self.y_train[i] for i in k_nearest_indices]
    
        return np.bincount(k_nearest_objects).argmax()

    def euclidian_metric(self, x_test_i: np.array):
        return np.sqrt(np.sum((self.X_train - x_test_i)**2, axis=1))

    def predict(self, X_test: np.array):
        predictions = [self.make_predictions(x_test_i) for x_test_i in X_test]
        return predictions
