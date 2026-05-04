"""
Pytest configuration — centralizes import path setup.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
