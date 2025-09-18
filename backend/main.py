import uvicorn
import os
from dotenv import load_dotenv

#updated during deployment
import sys
from pathlib import Path



# Adding the project root to the Python path ---
# This ensures that imports like 'from backend.app...' work correctly.
sys.path.append(str(Path(__file__).resolve().parent.parent))


# Load environment variables
load_dotenv()

# Set default values for host and port
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
RELOAD = os.getenv("RELOAD", "False").lower() in ("true", "1", "t")
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")


if __name__ == "__main__":
    # Run the application with uvicorn
    uvicorn.run(
        "backend.app.api.app:app",
        host=HOST,
        port=PORT,
        reload=RELOAD,
        log_level=LOG_LEVEL,
    )