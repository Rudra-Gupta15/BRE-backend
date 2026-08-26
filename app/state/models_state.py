from app.data.model_catalog import MODEL_TEMPLATES


def _default_version_map():
    return {"risk_model": "v3.4", "cashflow_model": "v3.4", "fraud_model": "v3.4", "money_balance_model": "v3.4"}


def _default_deployed_map():
    return {"risk_model": "Deployed", "cashflow_model": "Deployed", "fraud_model": "Ready", "money_balance_model": "Ready"}


class ModelsState:
    def __init__(self):
        self.trained_models: list[dict] = []
        self.selected_version_map: dict[str, str] = _default_version_map()
        self.deployed_status_map: dict[str, str] = _default_deployed_map()
        self.last_training_run: dict | None = None
        # Fitted sklearn model objects keyed by model_id — used for live inference
        self.trained_sklearn_map: dict = {}
        # Real features extracted from the last uploaded statement
        self.real_features: dict = {}

    def reset(self):
        self.trained_models = []
        self.selected_version_map = _default_version_map()
        self.deployed_status_map = _default_deployed_map()
        self.last_training_run = None
        self.trained_sklearn_map = {}
        self.real_features = {}


models_state = ModelsState()


def known_model_ids() -> list[str]:
    return [m["id"] for m in MODEL_TEMPLATES]
