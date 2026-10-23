import subprocess
import sys
from pathlib import Path

import uvicorn

SRC = Path(__file__).resolve().parent / "src"


def main() -> None:
    # Streamlit in a child process; the API owns the main thread (a blocking subprocess.run here starved it before).
    ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(SRC / "app" / "ui_streamlit.py")])
    print("UI at http://localhost:8501 | API docs at http://localhost:8000/docs")
    try:
        uvicorn.run("api:app", host="0.0.0.0", port=8000, app_dir=str(SRC))
    finally:
        ui.terminate()


if __name__ == "__main__":
    main()
