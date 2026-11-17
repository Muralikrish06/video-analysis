import subprocess
import time

from config import settings

# Start FastAPI server using uvicorn in background
api_process = subprocess.Popen([
    "uvicorn", "api:app",
    "--host", settings.api.host,
    "--port", str(settings.api.port),
    "--workers", str(settings.api.workers),
])
print(f"FastAPI started (PID: {api_process.pid})")

# Wait for API to be ready
time.sleep(5)  # increased to 5s to make sure uvicorn is fully up

# Start Langflow
langflow_process = subprocess.run(["langflow", "run"])

# When langflow exits, kill the API too
api_process.terminate()
print("FastAPI stopped.")