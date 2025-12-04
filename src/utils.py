import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, roc_curve
import seaborn as sns
import os

def calculate_metrics(y_true, y_pred_logits):
    """
    Calculate evaluation metrics.
    y_true: true labels (0 or 1)
    y_pred_logits: raw logits from the model
    """
    y_probs = torch.sigmoid(torch.tensor(y_pred_logits)).numpy()
    y_pred = (y_probs > 0.5).astype(int)
    
    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = 0.0 # Handle case with only one class
        
    ap = average_precision_score(y_true, y_probs)
    f1 = f1_score(y_true, y_pred)
    
    return {
        'AUC': auc,
        'AP': ap,
        'F1': f1
    }

def plot_training_curves(train_losses, val_aps, save_path):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_aps, label='Val AP')
    plt.xlabel('Epoch')
    plt.ylabel('Average Precision')
    plt.title('Validation AP')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred_logits, save_path):
    y_probs = torch.sigmoid(torch.tensor(y_pred_logits)).numpy()
    y_pred = (y_probs > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_pred_logits, save_path):
    y_probs = torch.sigmoid(torch.tensor(y_pred_logits)).numpy()
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_pr_curve(y_true, y_pred_logits, save_path):
    from sklearn.metrics import precision_recall_curve
    y_probs = torch.sigmoid(torch.tensor(y_pred_logits)).numpy()
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ap_score = average_precision_score(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AP = {ap_score:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
