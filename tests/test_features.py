import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import smiles_to_graph

def test_smiles_to_graph():
    print("Testing SMILES to Graph conversion...")
    
    # Test case 1: Simple molecule (Ethanol)
    smiles = "CCO"
    data = smiles_to_graph(smiles, label=1)
    
    assert data is not None
    assert data.x.shape[0] == 3 # 2 C + 1 O (Hydrogens are implicit in features usually, but let's check implementation)
    # RDKit MolFromSmiles("CCO") has 3 atoms (C, C, O).
    
    # Check features dimension
    # Atom features length: 44 (symbol) + 11 (degree) + 1 (charge) + 6 (hybrid) + 1 (aromatic) + 5 (Hs) = 68
    assert data.x.shape[1] == 68
    
    # Check edge index
    # C-C, C-O. 2 bonds. Undirected -> 4 edges.
    assert data.edge_index.shape[1] == 4
    
    # Check edge attr
    # Bond features length: 4 (type) + 1 (conj) + 1 (ring) = 6
    assert data.edge_attr.shape[1] == 6
    
    print("Test 1 (Ethanol) Passed.")
    
    # Test case 2: Benzene (Aromatic)
    smiles = "c1ccccc1"
    data = smiles_to_graph(smiles, label=0)
    
    assert data.x.shape[0] == 6
    assert data.edge_index.shape[1] == 12 # 6 bonds * 2
    
    print("Test 2 (Benzene) Passed.")

if __name__ == "__main__":
    try:
        test_smiles_to_graph()
        print("All feature tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
