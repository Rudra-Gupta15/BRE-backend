"""GST underwriting subsystem — self-contained.

    app/gst/
      dataset.csv   — bundled training data (6000 GST profiles)
      schema.py     — canonical field names / feature groups / targets
      parser.py     — parse_gst(bytes, filename) -> list[record dict]
      model.py      — train / predict / status  (HistGB regressor + classifier)
      service.py    — high-level helpers used by the Model Hub pipeline
      router.py     — FastAPI router, mounted at /api/gst

Artifacts are written to Backend/models/gst/ (versioned, SHA-256 + HMAC signed).
"""

from app.gst.router import router as api_router

__all__ = ["api_router"]
