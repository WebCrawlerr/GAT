import sys
import os
import torch
import numpy as np
import shutil

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import plot_roc_curve, plot_pr_curve, plot_loss_curve, plot_val_ap_curve

def test_plotting_functions():
    print("Testing plotting functions...")
    
    # Create a dummy output directory
    output_dir = "test_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate dummy data
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred_logits = np.random.randn(10)
    
    # Test ROC Curve
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plot_roc_curve(y_true, y_pred_logits, roc_path)
    assert os.path.exists(roc_path)
    print("ROC Curve plot generated successfully.")
    
    # Test PR Curve
    pr_path = os.path.join(output_dir, "pr_curve.png")
    plot_pr_curve(y_true, y_pred_logits, pr_path)
    assert os.path.exists(pr_path)
    print("PR Curve plot generated successfully.")

    # Test Loss Curve
    loss_path = os.path.join(output_dir, "loss_curve.png")
    plot_loss_curve([0.5, 0.4, 0.3], loss_path)
    assert os.path.exists(loss_path)
    print("Loss Curve plot generated successfully.")

    # Test Val AP Curve
    ap_path = os.path.join(output_dir, "val_ap_curve.png")
    plot_val_ap_curve([0.5, 0.6, 0.7], ap_path)
    assert os.path.exists(ap_path)
    print("Val AP Curve plot generated successfully.")
    
    # Cleanup
    shutil.rmtree(output_dir)
    print("Cleanup completed.")

if __name__ == "__main__":
    try:
        test_plotting_functions()
        print("All plotting tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
