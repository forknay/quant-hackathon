#!/usr/bin/env python3
"""
Quick test script for the main pipeline
"""

import subprocess
import sys
from pathlib import Path

def test_main_pipeline():
    """Test the main pipeline with healthcare sector."""
    
    print("🧪 Testing Main Pipeline")
    print("=" * 50)
    
    # Test with healthcare sector, using recent algo results
    cmd = [
        sys.executable, "main_pipeline.py",
        "--sector", "healthcare",
        "--year", "2024", 
        "--month", "6",
        "--top-n", "10",
        "--bottom-m", "5",
        "--skip-algo",  # Skip algo for now since we tested it works
        "--output", "test_results/test_portfolio.json"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Test PASSED!")
        else:
            print(f"❌ Test FAILED (exit code: {result.returncode})")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Test FAILED with exception: {e}")
        return False

if __name__ == "__main__":
    success = test_main_pipeline()
    sys.exit(0 if success else 1)