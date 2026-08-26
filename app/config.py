import os

PORT = int(os.environ.get("PORT", "4000"))
CORS_ORIGIN = os.environ.get("CORS_ORIGIN", "http://localhost:5173")
