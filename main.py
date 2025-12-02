import os
import pandas as pd
import torch
from src.config import *
from src.data_processing import load_data, filter_brd4, clean_and_label_data
from src.dataset import BRD4Dataset, scaffold_split
from src.train import run_training

def main():
    print("Starting GAT BRD4 Binding Prediction Pipeline...")
    
    # 1. Data Acquisition & Processing
    raw_path = os.path.join(DATA_RAW_DIR, BINDINGDB_FILENAME)
    processed_path = os.path.join(DATA_PROCESSED_DIR, 'data.pt')
    
    if os.path.exists(processed_path):
        print(f"Found processed data at {processed_path}. Loading...")
        dataset = BRD4Dataset(root=DATA_PROCESSED_DIR)
    else:
        if not os.path.exists(raw_path):
            print(f"ERROR: Raw data file not found at {raw_path}")
            print("Please download 'BindingDB_All.tsv' (or zip) and place it in data/raw/")
            return
            
        print("Loading raw data...")
        df = load_data(raw_path)
        if df is None:
            return
            
        print("Filtering for BRD4...")
        df = filter_brd4(df)
        print(f"Found {len(df)} records for BRD4.")
        
        print("Cleaning and Labeling...")
        df = clean_and_label_data(df)
        print(f"Final dataset size: {len(df)}")
        
        print("Creating Graph Dataset (this may take a while)...")
        dataset = BRD4Dataset(root=DATA_PROCESSED_DIR, df=df)
        
    # 2. Split Data
    print("Splitting data (Scaffold Split)...")
    train_dataset, val_dataset, test_dataset = scaffold_split(dataset)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # 3. Training
    print("Starting Training...")
    run_training(train_dataset, val_dataset, test_dataset)
    
    print("Pipeline Completed.")

if __name__ == "__main__":
    main()
