import sys
import os
import torch
import numpy as np
from sklearn.metrics import fbeta_score

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import calculate_metrics

def test_fbeta_score():
    print("Testing F-beta score calculation...")
    
    # Case 1: Perfect prediction
    y_true = np.array([0, 1, 0, 1])
    y_pred_logits = np.array([-10.0, 10.0, -10.0, 10.0]) # Sigmoid(10) ~= 1, Sigmoid(-10) ~= 0
    
    metrics = calculate_metrics(y_true, y_pred_logits)
    print(f"Case 1 Metrics: {metrics}")
    assert metrics['F0.5'] == 1.0
    
    # Case 2: Mixed
    # True: [0, 1, 0, 1]
    # Pred: [0, 1, 1, 0] (FP at index 2, FN at index 3)
    # TP=1, FP=1, FN=1
    # Precision = 1/2 = 0.5
    # Recall = 1/2 = 0.5
    # F0.5 = (1 + 0.5^2) * (P * R) / (0.5^2 * P + R)
    #      = 1.25 * 0.25 / (0.25 * 0.5 + 0.5)
    #      = 0.3125 / 0.625 = 0.5
    
    y_true_2 = np.array([0, 1, 0, 1])
    y_pred_logits_2 = np.array([-10.0, 10.0, 10.0, -10.0])
    
    metrics_2 = calculate_metrics(y_true_2, y_pred_logits_2)
    print(f"Case 2 Metrics: {metrics_2}")
    
    expected_fbeta = fbeta_score(y_true_2, [0, 1, 1, 0], beta=0.5)
    print(f"Expected F0.5: {expected_fbeta}")
    
    assert np.isclose(metrics_2['F0.5'], expected_fbeta)
    
    print("F-beta score verification passed!")

if __name__ == "__main__":
    test_fbeta_score()
