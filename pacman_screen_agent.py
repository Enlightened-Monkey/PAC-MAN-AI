#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    target = root / "scripts" / "pacman_screen_agent.py"

    venv_python = root / ".venv" / "bin" / "python"
    current_python = Path(sys.executable)

    # If launched by system python, re-exec with project venv to ensure deps (cv2, torch, etc.).
    if venv_python.exists() and current_python != venv_python:
        os.execv(str(venv_python), [str(venv_python), str(target), *sys.argv[1:]])

    os.execv(sys.executable, [sys.executable, str(target), *sys.argv[1:]])
