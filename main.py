from pathlib import Path
import subprocess
import sys


def main() -> None:
    app = Path(__file__).resolve().parent / "src" / "serving" / "ui_streamlit.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)], check=True)


if __name__ == "__main__":
    main()
