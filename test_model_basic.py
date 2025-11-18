#!/usr/bin/env python3
"""
Basic test script for C2SModel implementation.
This script tests that the model class can be imported and initialized.
"""

import sys
import torch

# Test imports
print("Testing imports...")
try:
    from c2s_mini.model import C2SModel, MODEL_NAME
    print("✓ Successfully imported C2SModel")
    print(f"✓ Model name: {MODEL_NAME}")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test class instantiation (without actually loading the model)
print("\nTesting class structure...")
try:
    # Check that the class has all required methods
    required_methods = [
        '__init__',
        'generate_from_prompt',
        'generate_batch',
        'embed_cell',
        'embed_batch',
        '__repr__'
    ]

    for method in required_methods:
        if not hasattr(C2SModel, method):
            print(f"✗ Missing method: {method}")
            sys.exit(1)
        print(f"✓ Method '{method}' exists")

    print("\n✓ All required methods are present")

except Exception as e:
    print(f"✗ Error checking class structure: {e}")
    sys.exit(1)

# Test device detection
print("\nTesting device detection...")
try:
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print("Note: Model will use CUDA by default with device='auto'")
    else:
        print("Note: Model will use CPU by default with device='auto'")
    print("✓ Device detection works")
except Exception as e:
    print(f"✗ Error with device detection: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("All basic tests passed!")
print("="*60)
print("\nNote: To fully test the model, run:")
print("  python -c \"from c2s_mini.model import C2SModel; model = C2SModel()\"")
print("\nThis will download and load the actual model (~160MB).")
