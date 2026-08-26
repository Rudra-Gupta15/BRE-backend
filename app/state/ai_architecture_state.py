class AiArchitectureState:
    def __init__(self):
        self.selected_llm = "gemma"
        self.data_extracted = False
        self.cleanliness_percent = 60
        self.vllm = {
            "enabled": True,
            "endpoint": "http://localhost:8000/v1",
            "modelName": "gemma-2-9b-it",
            "gpuCount": "2",
            "gpuMemoryUtil": "0.90",
            "status": "connected",  # connected | testing | disconnected
        }


ai_architecture_state = AiArchitectureState()
