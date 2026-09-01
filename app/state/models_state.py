from app.data.model_catalog import MODEL_TEMPLATES


def _default_version_map():
    return {"risk_model": "v3.4", "cashflow_model": "v3.4", "fraud_model": "v3.4", "money_balance_model": "v3.4"}


def _default_deployed_map():
    # Every model ships deployed by default. A Revoke pulls one from the live
    # registry (and from Model Testing's "Select Models"); a browser reload
    # hits /reset and restores all four to Deployed.
    return {"risk_model": "Deployed", "cashflow_model": "Deployed", "fraud_model": "Deployed", "money_balance_model": "Deployed"}


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
        # Real 5-fold cross-validation results per model_id, computed at train
        # time and served by the Model Evaluation tab. Empty until first train.
        self.evaluation_cache: dict = {}

    def reset(self):
        self.trained_models = []
        self.selected_version_map = _default_version_map()
        self.deployed_status_map = _default_deployed_map()
        self.last_training_run = None
        self.trained_sklearn_map = {}
        self.real_features = {}
        self.evaluation_cache = {}


models_state = ModelsState()


def known_model_ids() -> list[str]:
    return [m["id"] for m in MODEL_TEMPLATES]
