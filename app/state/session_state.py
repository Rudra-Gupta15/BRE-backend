# In-memory session state. This is a single-tenant demo backend (no DB,
# no multi-user auth) so state lives process-wide and resets on restart.

from app.data.data_sources import DATA_SOURCES


def _default_pipeline():
    return {
        "status": "idle",  # idle | done
        "currentStage": 0,
        "noisePercent": 60,
        "llmActive": True,
        "processedTable": None,
        "lastRunAt": None,
    }


class SessionState:
    def __init__(self):
        self.selected_ids: list[str] = []
        self.custom_sources: list[dict] = []
        self.uploaded_files: dict[str, dict] = {}
        self.parsed_statements: dict[str, dict] = {}
        self.pipeline: dict = _default_pipeline()
        self.inference_history: list[dict] = []

    def all_data_sources(self) -> list[dict]:
        return [*DATA_SOURCES, *self.custom_sources]

    def reset(self):
        self.selected_ids = []
        self.custom_sources = []
        self.uploaded_files = {}
        self.parsed_statements = {}
        self.pipeline = _default_pipeline()
        self.inference_history = []


session_state = SessionState()
