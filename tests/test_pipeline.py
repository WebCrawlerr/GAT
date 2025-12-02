import sys
import os
import pandas as pd
import shutil
from src.config import *
from src.data_processing import load_data, filter_brd4, clean_and_label_data
from src.dataset import BRD4Dataset, scaffold_split
from src.train import run_training

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_dummy_data():
    data = {
        'Target Name': ['BRD4', 'BRD4', 'Other', 'Bromodomain-containing protein 4'],
        'Ligand SMILES': ['CCO', 'c1ccccc1', 'CC', 'CN'],
        'Ki (nM)': ['100', '>2000', '50', '500'],
        'IC50 (nM)': [None, None, None, None],
        'Kd (nM)': [None, None, None, None]
    }
    df = pd.DataFrame(data)
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_RAW_DIR, 'test_bindingdb.tsv'), sep='\t', index=False)
    return os.path.join(DATA_RAW_DIR, 'test_bindingdb.tsv')

def test_pipeline():
    print("Testing Full Pipeline...")
    
    # 1. Create Dummy Data
    dummy_path = create_dummy_data()
    
    # 2. Load & Process
    df = load_data(dummy_path)
    df = filter_brd4(df)
    assert len(df) == 3 # Should keep 3 (BRD4, BRD4, Bromodomain...)
    
    df = clean_and_label_data(df)
    # Check labels
    # 100 nM -> 1
    # >2000 nM -> 0
    # 500 nM -> 1
    assert len(df) == 3
    assert df.iloc[0]['Label'] == 1
    assert df.iloc[1]['Label'] == 0
    assert df.iloc[2]['Label'] == 1
    
    # 3. Create Dataset
    # Clean up processed dir first to force reprocessing
    if os.path.exists(DATA_PROCESSED_DIR):
        shutil.rmtree(DATA_PROCESSED_DIR)
    os.makedirs(DATA_PROCESSED_DIR)
    
    dataset = BRD4Dataset(root=DATA_PROCESSED_DIR, df=df)
    assert len(dataset) == 3
    
    # 4. Split
    train, val, test = scaffold_split(dataset, frac_train=0.4, frac_val=0.3, frac_test=0.3)
    # With 3 items, split might be tricky due to rounding, but let's just check they run
    print(f"Split sizes: {len(train)}, {len(val)}, {len(test)}")
    
    # 5. Train (Short run)
    # Mock config for speed
    config = {
        'epochs': 2,
        'batch_size': 2,
        'hidden_dim': 8,
        'heads': 1,
        'layers': 2
    }
    
    # We need to monkeypatch config in train.py or pass it. 
    # I updated run_training to accept config.
    
    try:
        metrics = run_training(train, val, test, config=config)
        print("Training ran successfully.")
    except Exception as e:
        print(f"Training failed: {e}")
        raise e
        
    print("Pipeline Test Passed.")

if __name__ == "__main__":
    test_pipeline()
