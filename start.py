#!/usr/bin/env python3
"""Gestra launcher — double-click or run `python start.py` to play."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
GAME_DIR = ROOT / "game_base" / "Street-Pyter"
REQUIREMENTS = ROOT / "requirements.txt"


def setup_venv():
    if VENV_PYTHON.exists():
        return
    print("First run: creating virtual environment...")
    subprocess.check_call([sys.executable, "-m", "venv", str(ROOT / ".venv")])
    subprocess.check_call([str(VENV_PYTHON), "-m", "pip", "install", "-q",
                           "-r", str(REQUIREMENTS)])
    print("Setup complete.\n")


def main():
    os.chdir(GAME_DIR)
    os.environ["GESTRA_WEBCAM"] = "1"
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), "main.py"])


if __name__ == "__main__":
    setup_venv()
    main()
