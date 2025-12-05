import sys
import os
import torch
import numpy as np
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimize import run_optimization, objective

# Mock dataset and training
class MockData:
    def __init__(self):
        self.x = torch.randn(10, 5)
        self.edge_index = torch.tensor([[0, 1], [1, 0]])
        self.edge_attr = torch.randn(2, 3)
        self.y = torch.tensor([1.0])
        self.smiles = "C"

class MockDataset(list):
    def __init__(self, data_list):
        super().__init__(data_list)
        self.slices = None # PyG dataset attribute

@patch('src.optimize.run_training')
@patch('src.optimize.scaffold_split')
def test_optimization(mock_split, mock_train):
    print("Testing Hyperparameter Optimization...")
    
    # Setup mocks
    mock_dataset = MockDataset([MockData() for _ in range(10)])
    mock_split.return_value = (mock_dataset, mock_dataset, mock_dataset)
    
    # Mock training to return a dummy metric
    # We want to see if Optuna can maximize this
    # Let's make it return random values
    mock_train.side_effect = lambda *args, **kwargs: {'AP': np.random.rand()}
    
    # Run optimization with very few trials
    n_trials = 2
    best_params = run_optimization(mock_dataset, n_trials=n_trials)
    
    assert best_params is not None
    assert 'lr' in best_params
    assert 'hidden_dim' in best_params
    
    print(f"Optimization test passed. Best params: {best_params}")

if __name__ == "__main__":
    try:
        test_optimization()
        print("All optimization tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
