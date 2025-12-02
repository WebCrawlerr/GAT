import pandas as pd
import numpy as np
from rdkit import Chem
from src.config import THRESHOLD_NM, TARGET_NAMES

def load_data(filepath):
    """
    Loads BindingDB data from TSV file.
    """
    # BindingDB TSV can be messy, use error_bad_lines=False or on_bad_lines='skip'
    try:
        df = pd.read_csv(filepath, sep='\t', on_bad_lines='skip', low_memory=False)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    return df

def filter_brd4(df):
    """
    Filters the dataframe for BRD4 target.
    """
    # Target Name is usually the column name in BindingDB
    # We look for 'Target Name' column
    target_col = 'Target Name'
    if target_col not in df.columns:
        # Try to find a similar column
        candidates = [c for c in df.columns if 'Target Name' in c]
        if candidates:
            target_col = candidates[0]
        else:
            print("Could not find Target Name column.")
            return pd.DataFrame()
            
    # Filter
    mask = df[target_col].apply(lambda x: any(name in str(x) for name in TARGET_NAMES))
    return df[mask].copy()

def clean_and_label_data(df):
    """
    Cleans data, prioritizes Ki > Kd > IC50, and creates binary labels.
    """
    # Columns of interest (BindingDB names are specific)
    # Usually: 'Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'Ligand SMILES'
    
    # Helper to parse numeric
    def parse_activity(val):
        try:
            # Remove > or < signs
            val = str(val).replace('>', '').replace('<', '').strip()
            return float(val)
        except:
            return np.nan

    # Create a consolidated 'Activity_nM' column
    # Priority: Ki > Kd > IC50
    # Note: BindingDB column names might vary slightly, checking common ones
    
    ki_col = 'Ki (nM)'
    kd_col = 'Kd (nM)'
    ic50_col = 'IC50 (nM)'
    smiles_col = 'Ligand SMILES'
    
    # Ensure columns exist
    for col in [ki_col, kd_col, ic50_col]:
        if col not in df.columns:
            df[col] = np.nan
            
    df['Ki_val'] = df[ki_col].apply(parse_activity)
    df['Kd_val'] = df[kd_col].apply(parse_activity)
    df['IC50_val'] = df[ic50_col].apply(parse_activity)
    
    # Coalesce
    df['Activity_nM'] = df['Ki_val'].fillna(df['Kd_val']).fillna(df['IC50_val'])
    
    # Drop rows with no activity
    df = df.dropna(subset=['Activity_nM'])
    
    # Drop rows with no SMILES
    if smiles_col in df.columns:
        df = df.dropna(subset=[smiles_col])
    else:
        print("SMILES column not found!")
        return pd.DataFrame()
        
    # Binary Labeling
    # Active (1) if Activity <= THRESHOLD (1000 nM)
    # Inactive (0) if Activity > THRESHOLD
    
    df['Label'] = (df['Activity_nM'] <= THRESHOLD_NM).astype(int)
    
    # Optional: Remove grey zone? User mentioned it as optional.
    # Let's keep it simple for now, or maybe add a flag.
    
    return df[[smiles_col, 'Activity_nM', 'Label']]

def sanitize_smiles(smiles):
    """
    Sanitizes a SMILES string using RDKit.
    Returns canonical SMILES or None if invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Remove salts/solvents (simple version: keep largest fragment)
            # For now just canonicalize
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return None
