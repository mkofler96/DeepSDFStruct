import subprocess
import sys
from pathlib import Path


def main():
    script_path = Path(__file__).parent / "latent_viewer.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(script_path)])
