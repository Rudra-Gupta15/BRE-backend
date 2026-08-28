from app import config


class AiArchitectureState:
    def __init__(self):
        self.selected_llm = "gemma"
        self.data_extracted = False
        self.cleanliness_percent = 60
        # Reflects the real statement-parsing engine (Backend/.env). The
        # endpoint / model are read live from config; status comes from an
        # actual connectivity check (see /ai-architecture/vllm/test).
        self.vllm = {
            "enabled": True,
            "endpoint": config.OLLAMA_HOST,
            "modelName": config.STATEMENT_LLM_MODEL,
            "runtime": "Ollama Cloud" if config.STATEMENT_LLM_MODEL.endswith("-cloud") else "Ollama (local)",
            "timeoutSec": config.STATEMENT_LLM_TIMEOUT,
            "renderDpi": config.STATEMENT_LLM_DPI,
            "status": "unknown",  # unknown | connected | model-missing | disconnected
        }


ai_architecture_state = AiArchitectureState()
