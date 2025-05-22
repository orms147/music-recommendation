import pandas as pd
import numpy as np
import os
import pickle
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from config.config import MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseRecommender(ABC):
    """Base abstract class for all recommendation models"""
    
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
        self.is_trained = False
        self.train_time = None
    
    @abstractmethod
    def train(self, data):
        """Train the model with the given data"""
        pass
    
    @abstractmethod
    def recommend(self, input_data, n_recommendations=10):
        """Generate recommendations based on input data"""
        pass
    
    def save(self, filepath=None):
        """Save the model to disk"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(MODELS_DIR, f"{self.name}_{timestamp}.pkl")
            
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
        logger.info(f"Model saved to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath):
        """Load a model from disk"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
            
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def _validate_n_recommendations(self, n):
        """Validate and return a valid number of recommendations"""
        try:
            n = int(n)
            return max(1, n)  # At least 1 recommendation
        except (ValueError, TypeError):
            logger.warning(f"Invalid n_recommendations value: {n}. Using default value 10.")
            return 10
            
    def _log_recommendation_metrics(self, input_data, recommendations):
        """Log metrics about the recommendations (to be implemented by child classes)"""
        pass