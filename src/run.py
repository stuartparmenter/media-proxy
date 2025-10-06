# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT
#!/usr/bin/env python3
"""
Simple launcher script for running the media proxy server from the src directory.
Usage: python run.py [options]
"""

import sys
from pathlib import Path


# Add parent directory to path so we can import src as a package
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

if __name__ == "__main__":
    # Import and run the main function from the package
    from src.main import run  # type: ignore[import-untyped]

    run()
