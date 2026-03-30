import numpy as np
from collections import Counter

class CustomKNeighborsClassifier:
    
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean', normalize=True):
        self.n_neighbors = n_neighbors
        self.weights = weights  # 'uniform' or 'distance'
        self.metric = metric
        self.normalize = normalize
        self.X_train = None
        self.y_train = None
        self.classes_ = None
        self.n_samples_fit_ = None

    def _l2_normalize(self, X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return X / norms

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        if self.normalize:
            X = self._l2_normalize(X)

        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(y)
        self.n_samples_fit_ = X.shape[0]

        return self

    def _calculate_distances(self, X):
        X = np.asarray(X)
        distances = np.zeros((X.shape[0], self.X_train.shape[0]))

        for i, x in enumerate(X):
            for j, x_train in enumerate(self.X_train):
                distances[i, j] = np.sqrt(np.sum((x - x_train) ** 2))

        return distances

    def _get_neighbors(self, distances):
        """Get k nearest neighbors and their distances"""
        neighbors = np.zeros((distances.shape[0], self.n_neighbors), dtype=int)
        neighbor_distances = np.zeros((distances.shape[0], self.n_neighbors))

        for i in range(distances.shape[0]):
            # Get indices of k nearest neighbors (sorted by distance)
            neighbor_indices = np.argpartition(distances[i], self.n_neighbors)[:self.n_neighbors]
            # Sort them by actual distance
            neighbor_indices = neighbor_indices[np.argsort(distances[i][neighbor_indices])]
            neighbors[i] = neighbor_indices
            neighbor_distances[i] = distances[i][neighbor_indices]

        return neighbors, neighbor_distances

    def _calculate_weights(self, neighbor_distances):
        """Calculate weights based on distances"""
        if self.weights == 'uniform':
            # All neighbors have equal weight
            return np.ones_like(neighbor_distances)
        elif self.weights == 'distance':
            # Closer neighbors have more weight
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            weights = 1 / (neighbor_distances + epsilon)
            return weights
        else:
            raise ValueError(f"weights must be 'uniform' or 'distance', got {self.weights}")

    def predict(self, X):
        if self.X_train is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.normalize:
            X = self._l2_normalize(X)

        distances = self._calculate_distances(X)
        neighbors, neighbor_distances = self._get_neighbors(distances)
        weights = self._calculate_weights(neighbor_distances)

        predictions = []
        for i, neighbor_indices in enumerate(neighbors):
            neighbor_labels = self.y_train[neighbor_indices]
            neighbor_weights = weights[i]
            
            # Weighted voting
            weighted_votes = {}
            for label, weight in zip(neighbor_labels, neighbor_weights):
                if label in weighted_votes:
                    weighted_votes[label] += weight
                else:
                    weighted_votes[label] = weight
            
            # Get label with highest weighted vote
            predicted_label = max(weighted_votes, key=weighted_votes.get)
            predictions.append(predicted_label)

        return np.array(predictions)

    def predict_proba(self, X):
        if self.X_train is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.normalize:
            X = self._l2_normalize(X)

        distances = self._calculate_distances(X)
        neighbors, neighbor_distances = self._get_neighbors(distances)
        weights = self._calculate_weights(neighbor_distances)

        probabilities = []
        for i, neighbor_indices in enumerate(neighbors):
            neighbor_labels = self.y_train[neighbor_indices]
            neighbor_weights = weights[i]

            # Weighted probability calculation
            weighted_votes = {}
            total_weight = 0
            for label, weight in zip(neighbor_labels, neighbor_weights):
                if label in weighted_votes:
                    weighted_votes[label] += weight
                else:
                    weighted_votes[label] = weight
                total_weight += weight

            # Normalize to probabilities
            proba = np.zeros(len(self.classes_))
            for j, class_label in enumerate(self.classes_):
                proba[j] = weighted_votes.get(class_label, 0) / total_weight

            probabilities.append(proba)

        return np.array(probabilities)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def get_params(self, deep=True):
        return {
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'metric': self.metric,
            'normalize': self.normalize
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key in ['n_neighbors', 'weights', 'metric', 'normalize']:
                setattr(self, key, value)
        return self