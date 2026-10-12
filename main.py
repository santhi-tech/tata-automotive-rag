import asyncio
import sys
import subprocess
from pathlib import Path
import uvicorn


async def run_api():
    """Run FastAPI app asynchronously."""
    src = str(Path(__file__).resolve().parent / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    # Start uvicorn server
    config = uvicorn.Config("api:v1", host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


def run_streamlit():
    """Run Streamlit app normally (blocking)."""
    app = Path(__file__).resolve().parent / "src" / "app" / "ui_streamlit.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)], check=True)


async def main():
    # Run FastAPI in background
    api_task = asyncio.create_task(run_api())
    print("API docs at http://localhost:8000/docs")

    # Run Streamlit in foreground (blocking)
    run_streamlit()

    # Ensure API task finishes gracefully when Streamlit exits
    await api_task


if __name__ == "__main__":
    asyncio.run(main())
