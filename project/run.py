#!/usr/bin/env python3
"""
Entry point for running hallucination detection experiments.
Run from project directory: python run.py [args]
Or: python -m src.run_experiments [args]
"""
import sys
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.run_experiments import main

if __name__ == "__main__":
    main()
