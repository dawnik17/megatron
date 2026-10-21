import importlib
import sys
from pathlib import Path

# run as python bridge/validate/<this folder>: import the package by its folder's name, so its relative imports work
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent))
main = importlib.import_module(here.name).main

if __name__ == "__main__":
    main()
