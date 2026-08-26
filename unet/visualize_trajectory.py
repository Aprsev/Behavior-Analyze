"""Backward-compatible launcher for tools.visualize_trajectory."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools.visualize_trajectory import main  # noqa: E402

if __name__ == "__main__":
    main()
