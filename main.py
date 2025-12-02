import os
import argparse
import pandas as pd
import torch
from src.config import *
from src.data_processing import load_and_filter_data, clean_and_label_data
from src.dataset import BRD4Dataset, scaffold_split
from src.train import run_training

def main():
    parser = argparse.ArgumentParser(description="GAT BRD4 Binding Prediction Pipeline")
    parser.add_argument('--raw_file', type=str, default=os.path.join(DATA_RAW_DIR, BINDINGDB_FILENAME),
                        help='Path to the raw BindingDB TSV file')
    parser.add_argument('--processed_dir', type=str, default=DATA_PROCESSED_DIR,
                        help='Directory to save/load processed data')
    args = parser.parse_args()

    print("Starting GAT BRD4 Binding Prediction Pipeline...")
    
    # 1. Data Acquisition & Processing
    raw_path = args.raw_file
    processed_dir = args.processed_dir
    processed_path = os.path.join(processed_dir, 'data.pt')
    
    # Ensure processed directory exists if using a custom one
    os.makedirs(processed_dir, exist_ok=True)
    
    if os.path.exists(processed_path):
        print(f"Found processed data at {processed_path}. Loading...")
        dataset = BRD4Dataset(root=processed_dir)
    else:
        if not os.path.exists(raw_path):
            print(f"ERROR: Raw data file not found at {raw_path}")
            print("Please download 'BindingDB_All.tsv' (or zip) and place it in data/raw/ or specify path with --raw_file")
            return
            
        # Load and filter in one go to save memory
        df = load_and_filter_data(raw_path)
        if df is None or df.empty:
            print("No data found or error loading data.")
            return
            
        print(f"Found {len(df)} records for BRD4.")
        
        print("Cleaning and Labeling...")
        df = clean_and_label_data(df)
        print(f"Final dataset size: {len(df)}")
        
        print(f"Creating Graph Dataset in {processed_dir} (this may take a while)...")
        dataset = BRD4Dataset(root=processed_dir, df=df)
        
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
