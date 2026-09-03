from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared.reproduce import main

if __name__ == "__main__":
    main(Path(__file__).with_name("spec.json"))
