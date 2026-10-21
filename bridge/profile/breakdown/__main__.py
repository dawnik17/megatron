import importlib
import sys
from pathlib import Path

# run as python bridge/profile/breakdown: import the package by its folder's name, and keep the folder itself off the
# path, where torch.py would shadow PyTorch
here = Path(__file__).resolve().parent
sys.path[0] = str(here.parent)
breakdown = importlib.import_module(here.name)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(breakdown.__doc__)
    breakdown.main(sys.argv[1])
