import os
import sys
import shutil
import torch
import numpy as np
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import BASE_DIR, PLOTS_DIR
from src.train import run_training

def test_plot_organization():
    print("Testing plot organization...")
    
    from torch_geometric.data import Data

    # Mock dataset
    def create_mock_data():
        x = torch.randn(10, 5)
        edge_index = torch.randint(0, 10, (2, 20))
        edge_attr = torch.randn(20, 2)
        y = torch.tensor([1.0])
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    mock_dataset = [create_mock_data() for _ in range(10)]
    
    # Mock config
    config = {
        'hidden_dim': 16,
        'heads': 2,
        'layers': 2,
        'dropout': 0.1,
        'lr': 0.01
    }
    
    # Test Case 1: Single Run (fold_idx=None)
    print("\nTest Case 1: Single Run")
    try:
        # We need to mock the model training part to avoid actual training time
        # But run_training is a bit complex to mock entirely without refactoring.
        # Instead, we'll run it for 1 epoch with a very small model and dataset.
        # We'll monkeypatch EPOCHS to 1 for speed.
        import src.train
        src.train.EPOCHS = 1
        
        run_training(mock_dataset, mock_dataset, test_dataset=mock_dataset, config=config, plot=True)
        
        expected_dir = os.path.join(PLOTS_DIR, 'single_run')
        if os.path.exists(expected_dir):
            files = os.listdir(expected_dir)
            expected_files = ['loss_curve.png', 'val_ap_curve.png', 'confusion_matrix.png', 'roc_curve.png', 'pr_curve.png']
            missing_files = [f for f in expected_files if f not in files]
            
            if not missing_files:
                print(f"SUCCESS: All plots created in {expected_dir}")
                print(f"Files: {files}")
            else:
                print(f"FAILURE: Missing files in {expected_dir}: {missing_files}")
        else:
            print(f"FAILURE: Directory not found: {expected_dir}")
            
    except Exception as e:
        print(f"Error in Test Case 1: {e}")

    # Test Case 2: Fold Run (fold_idx=0)
    print("\nTest Case 2: Fold Run")
    try:
        run_training(mock_dataset, mock_dataset, test_dataset=mock_dataset, config=config, fold_idx=0, plot=True)
        
        expected_dir = os.path.join(PLOTS_DIR, 'fold_0')
        if os.path.exists(expected_dir):
            files = os.listdir(expected_dir)
            expected_files = ['loss_curve.png', 'val_ap_curve.png', 'confusion_matrix.png', 'roc_curve.png', 'pr_curve.png']
            missing_files = [f for f in expected_files if f not in files]
            
            if not missing_files:
                print(f"SUCCESS: All plots created in {expected_dir}")
                print(f"Files: {files}")
            else:
                print(f"FAILURE: Missing files in {expected_dir}: {missing_files}")
        else:
            print(f"FAILURE: Directory not found: {expected_dir}")
            
    except Exception as e:
        print(f"Error in Test Case 2: {e}")

    # Cleanup (Optional, maybe we want to inspect)
    # shutil.rmtree(PLOTS_DIR)

if __name__ == "__main__":
    test_plot_organization()
