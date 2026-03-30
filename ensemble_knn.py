import numpy as np
from collections import Counter
from custom_knn import CustomKNeighborsClassifier

class EnsembleKNN:
    
    def __init__(self, k_values=[1,3,5], weights='uniform', metric='euclidean'):
        self.k_values = k_values
        self.weights = weights
        self.metric = metric
        self.models = []
        
    def fit(self, X, y):
        self.models = []
        
        print(f"[INFO] EnsembleKNN: Training {len(self.k_values)} models with K={self.k_values}")
        
        for k in self.k_values:
            model = CustomKNeighborsClassifier(
                n_neighbors=k, 
                weights=self.weights, 
                metric=self.metric
            )
            model.fit(X, y)
            self.models.append(model)
            print(f"[INFO] EnsembleKNN: Model with K={k} trained successfully")
            
        return self
    
    def predict(self, X):
        """Predict using majority voting from both models"""
        if not self.models:
            raise ValueError("Models must be fitted before prediction")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get predictions from both models
        all_predictions = []
        for i, model in enumerate(self.models):
            predictions = model.predict(X)
            all_predictions.append(predictions)
            print(f"[DEBUG] EnsembleKNN: Model {i+1} (K={self.k_values[i]}) predicted: {predictions[0]}")
        
        # Transpose to get predictions for each sample
        all_predictions = np.array(all_predictions).T
        
        # Majority voting for each sample
        final_predictions = []
        for sample_predictions in all_predictions:
            # Count votes for each class
            vote_counts = Counter(sample_predictions)
            # Get class with most votes
            predicted_class = max(vote_counts, key=vote_counts.get)
            final_predictions.append(predicted_class)
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        if not self.models:
            raise ValueError("Models must be fitted before prediction")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get probabilities from both models
        all_probas = []
        for model in self.models:
            probas = model.predict_proba(X)
            all_probas.append(probas)
        
        # Average probabilities across both models
        avg_probas = np.mean(all_probas, axis=0)
        
        return avg_probas
    
    def score(self, X, y):
        """Calculate accuracy score"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_params(self, deep=True):
        return {
            'k_values': self.k_values,
            'weights': self.weights,
            'metric': self.metric
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self