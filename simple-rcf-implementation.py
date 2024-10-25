import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
from datetime import datetime, timedelta
import random
import logging
from dataclasses import dataclass
from typing import Optional, Dict

# for logging purposes, configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EnergyPoint:
    """represents an individual energy price data point"""
    timestamp: int
    value: float
    
    def validate(self) -> bool:
        """performs validation on the data point"""
        try:
            if not isinstance(self.timestamp, (int)):
                return False
            if not np.isfinite(self.value):
                return False
            if self.value < 0:  # again, energy prices cannot be negative
                return False
            return True
        except Exception as e:
            logger.error(f"Point validation error: {e}")
            return False

class DataValidator:
    """used for data validation and parameter checking"""
    
    @staticmethod
    def validate_forest_parameters(
        window_size: int,
        num_trees: int,
        sample_size: int
    ) -> bool:
        """validates rcf parameters."""
        try:
            if window_size <= 0:
                raise ValueError("Window size must be positive")
            if num_trees <= 0:
                raise ValueError("Number of trees must be positive")
            if sample_size <= 0:
                raise ValueError("Sample size must be positive")
            if sample_size > window_size:
                raise ValueError("Sample size cannot exceed window size")
            return True
        except Exception as e:
            logger.error(f"Forest parameter validation error: {e}")
            return False
            
    @staticmethod
    def validate_bounding_box(box: Dict) -> bool:
        """validates the bounding box structure."""
        try:
            required_keys = {'min', 'max', 'count'}
            if not all(key in box for key in required_keys):
                return False
            if not isinstance(box['count'], (int, np.integer)):
                return False
            # Allow infinite values during initialization
            if np.isfinite(box['min']) and np.isfinite(box['max']):
                if box['min'] > box['max']:
                    return False
            if box['count'] < 0:
                return False
            return True
        except Exception as e:
            logger.error(f"Bounding box validation error: {e}")
            return False

class EnhancedRandomCutForest:
    def __init__(self, window_size=100, num_trees=10, sample_size=50):
        if not DataValidator.validate_forest_parameters(window_size, num_trees, sample_size):
            raise ValueError("Invalid forest parameters")
            
        self.window_size = window_size
        self.num_trees = num_trees
        self.sample_size = sample_size
        self.points = deque(maxlen=window_size)
        self.bounding_boxes = []
        self.initialize_forest()
    
    def initialize_forest(self):
        """Initialize empty bounding boxes (trees)."""
        try:
            self.bounding_boxes = []
            for _ in range(self.num_trees):
                box = {
                    'min': float('inf'),
                    'max': float('-inf'),
                    'count': 0
                }
                # infitite values should be allowed for initialization
                if not DataValidator.validate_bounding_box(box):
                    raise ValueError("Invalid bounding box initialization")
                self.bounding_boxes.append(box)
        except Exception as e:
            logger.error(f"Forest initialization error: {e}")
            raise
    
    def update_bounding_boxes(self, point: float) -> None:
        """updates bounding boxes with new point"""
        try:
            if not np.isfinite(point):
                raise ValueError("Invalid point value")
                
            for box in self.bounding_boxes:
                if not DataValidator.validate_bounding_box(box):
                    raise ValueError("Invalid bounding box state.")
                    
                if box['count'] < self.sample_size:
                    # handling first point
                    if box['count'] == 0:
                        box['min'] = point
                        box['max'] = point
                    else:
                        box['min'] = min(box['min'], point)
                        box['max'] = max(box['max'], point)
                    box['count'] += 1
                elif random.random() < self.sample_size / (box['count'] + 1):
                    # sampling reservoir
                    box['min'] = min(box['min'], point)
                    box['max'] = max(box['max'], point)
                box['count'] += 1
                
                if not DataValidator.validate_bounding_box(box):
                    raise ValueError("Invalid bounding box after update")
        except Exception as e:
            logger.error(f"Error updating bounding boxes: {e}")
            raise
    
    def compute_anomaly_score(self, point: float) -> float:
        """checks distance from bounding box and computes anomaly score."""
        try:
            if not np.isfinite(point):
                raise ValueError("Invalid point value")
            
            if len(self.points) < 10:  # some initial data to get understanding of the data, can be increased
                return 0.0
            
            scores = []
            for box in self.bounding_boxes:
                if not DataValidator.validate_bounding_box(box):
                    raise ValueError("Invalid bounding box state")
                    
                if box['count'] > 0:
                    if point < box['min']:
                        score = (box['min'] - point) / (box['max'] - box['min']) if box['max'] > box['min'] else 1
                    elif point > box['max']:
                        score = (point - box['max']) / (box['max'] - box['min']) if box['max'] > box['min'] else 1
                    else:
                        score = 0
                    scores.append(score)
            
            return np.mean(scores) if scores else 0
        except Exception as e:
            logger.error(f"Error computing anomaly score: {e}")
            raise

    def update(self, point: float) -> float:
        """add new point to forest and return anomaly score for the said point"""
        try:
            if not np.isfinite(point):
                raise ValueError("Invalid point value")
                
            self.points.append(point)
            self.update_bounding_boxes(point)
            return self.compute_anomaly_score(point)
        except Exception as e:
            logger.error(f"Error updating forest: {e}")
            raise

class EnergyDataSimulator:
    def __init__(self, base_price: float = 50.0):
        if not np.isfinite(base_price) or base_price <= 0:
            raise ValueError("base price must be positive")
            
        self.time = 0
        self.base_price = base_price
        self.last_value = self.base_price
        
    def get_time_features(self) -> tuple[int, bool]:
        """this function tells how to seasonally change the energy price"""
        try:
            hour = self.time % 24
            day = (self.time // 24) % 7
            is_weekend = day >= 5
            return hour, is_weekend
        except Exception as e:
            logger.error(f"Error extracting time features: {e}")
            raise
    
    def generate_seasonal_pattern(self, hour: int, is_weekend: bool) -> float:
        """this function generates the seasonal pattern for the energy price"""
        try:
            if not 0 <= hour < 24:
                raise ValueError("hour must be between 0 and 23")
                
            # daily pattern
            if 8 <= hour <= 20:  # peak hours
                base = self.base_price * 1.3
            else:  # Off-peak hours
                base = self.base_price * 0.8
            
            # weekend adjustment
            if is_weekend:
                base *= 0.9
            
            return base
        except Exception as e:
            logger.error(f"Error generating seasonal pattern: {e}")
            raise
    
    def generate_point(self) -> float:
        """uses the get_time_features and generate_seasonal_pattern functions to generate a single data point with realistic energy price patterns"""
        try:
            hour, is_weekend = self.get_time_features()
            seasonal_base = self.generate_seasonal_pattern(hour, is_weekend)
            
            # add unexpected fluctuations, random noise basically
            random_walk = np.random.normal(0, 1)
            self.last_value = self.last_value * 0.7 + seasonal_base * 0.3 + random_walk
            
            # adding occasional price spikes
            if random.random() < 0.08:  # 8% chance of spike
                spike_multiplier = random.uniform(1.5, 3.0)
                self.last_value *= spike_multiplier
            
            # price can't be negative
            self.last_value = max(0.1, self.last_value)
            
            self.time += 1
            
            # validate point before returning
            point = EnergyPoint(timestamp=self.time, value=self.last_value)
            if not point.validate():
                raise ValueError("Generated invalid point")
                
            return self.last_value
        except Exception as e:
            logger.error(f"Error generating point: {e}")
            raise


# main class responsible for anomaly detection
class EnhancedAnomalyDetector:
    def __init__(self, threshold: float = 0.3, window_size: int = 200):
        if not 0 < threshold < 1:
            raise ValueError("Threshold must be between 0 and 1")
        if window_size <= 0:
            raise ValueError("Window size must be positive")
            
        # create the random cut forest
        self.rcf = EnhancedRandomCutForest()
        self.threshold = threshold
        self.window_size = window_size
        
        # Visualization setup
        self.times = deque(maxlen=window_size)
        self.values = deque(maxlen=window_size)
        self.scores = deque(maxlen=window_size)
        self.anomalies = deque(maxlen=window_size)
        
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.setup_plots()
        
    def setup_plots(self):
        """initialize/setup the plots."""
        try:
            self.ax1.set_title('Energy Price Stream with Anomalies')
            self.ax1.set_xlabel('Time (hours)')
            self.ax1.set_ylabel('Price ($/MWh)')
            
            self.ax2.set_title('Anomaly Scores')
            self.ax2.set_xlabel('Time (hours)')
            self.ax2.set_ylabel('Score')
            self.ax2.axhline(y=self.threshold, color='r', linestyle='--', alpha=0.5)
        except Exception as e:
            logger.error(f"Error setting up plots: {e}")
            raise
        
    def update_plots(self):
        """updates the visualization."""
        try:
            if not self.times or not self.values:
                return
                
            self.ax1.clear()
            self.ax2.clear()
            
            # plot energy price stream
            self.ax1.plot(list(self.times), list(self.values), 'b-', label='Energy Price')
            self.ax1.scatter([self.times[i] for i in range(len(self.times)) if self.anomalies[i]],
                            [self.values[i] for i in range(len(self.values)) if self.anomalies[i]],
                            color='red', label='Anomalies')
            
            # plot anomaly scores
            self.ax2.plot(list(self.times), list(self.scores), 'g-', label='Anomaly Score')
            self.ax2.axhline(y=self.threshold, color='r', linestyle='--', alpha=0.5, label='Threshold')
            
            self.ax1.set_title('Energy Price Stream with Anomalies')
            self.ax2.set_title('Anomaly Scores')
            self.ax1.legend()
            self.ax2.legend()
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)
        except Exception as e:
            logger.error(f"Error updating plots: {e}")
            raise
        
    def process_point(self, value: float, timestamp: int) -> tuple[bool, float]:
        """process a single data point to check for anomalies and append the point as well""" 
        try:
            point = EnergyPoint(timestamp=timestamp, value=value)
            if not point.validate():
                raise ValueError("Invalid input point")
                
            score = self.rcf.update(value)
            is_anomaly = score > self.threshold
            
            self.times.append(timestamp)
            self.values.append(value)
            self.scores.append(score)
            self.anomalies.append(is_anomaly)
            
            self.update_plots()
            return is_anomaly, score
        except Exception as e:
            logger.error(f"Error processing point: {e}")
            raise

def main():
    try:
        simulator = EnergyDataSimulator()
        detector = EnhancedAnomalyDetector()
        
        while True:
            value = simulator.generate_point()
            timestamp = simulator.time
            
            is_anomaly, score = detector.process_point(value, timestamp)
            
            if is_anomaly:
                logger.warning(
                    f"Anomaly detected at hour {timestamp}! "
                    f"Price: ${value:.2f}/MWh, Score: {score:.2f}"
                )
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        logger.info("Stopping data stream...")
        plt.ioff()
        plt.close('all')
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        raise

if __name__ == "__main__":
    main()