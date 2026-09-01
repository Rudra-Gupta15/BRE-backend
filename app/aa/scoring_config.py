# Credit-score scorecard weights + decision thresholds.
#
# These are underwriting-policy numbers, not code constants — they belong to
# whoever owns credit policy. Defaults live here; they're editable at runtime
# via  GET/PUT /api/settings/scoring  and persisted (DB or session cache).

from dataclasses import asdict, dataclass, fields


@dataclass
class ScoringConfig:
    # ── Scorecard ──────────────────────────────────────────────────────────
    baseline: float = 740.0

    surplus_weight: float = 220.0          # points per unit of monthly surplus ratio
    surplus_floor: float = -180.0
    surplus_cap: float = 55.0
    overspend_trigger: float = -0.15       # surplus ratio below this = sustained overspend
    overspend_extra_penalty: float = 45.0

    income_stability_weight: float = 130.0
    income_stability_pivot: float = 0.60   # stability at/above pivot = neutral-to-positive

    bounce_first: float = 110.0            # first cheque/NACH/ECS return
    bounce_each_after: float = 80.0
    bounce_cap: float = 260.0

    overdraft_penalty: float = 170.0       # min balance < 0
    low_balance_2k_penalty: float = 70.0   # min balance < ₹2,000
    low_balance_10k_penalty: float = 25.0  # min balance < ₹10,000
    high_balance_threshold: float = 100000.0
    high_balance_bonus: float = 25.0

    volatility_weight: float = 70.0        # penalty per unit of balance CV above pivot
    volatility_pivot: float = 0.40

    cash_withdrawal_weight: float = 250.0  # penalty per unit of cash-withdrawal ratio above pivot
    cash_withdrawal_pivot: float = 0.12

    dscr_bonus_threshold: float = 1.30
    dscr_bonus_weight: float = 40.0
    dscr_bonus_cap: float = 35.0

    # ── Grade bands (300-900) ─────────────────────────────────────────────
    grade_low_min: int = 700
    grade_medium_min: int = 550

    # ── Credit Score Gate (hard approve/reject cutoff) ────────────────────
    gate_threshold: int = 650

    # ── Probability of default ────────────────────────────────────────────
    pd_scale_max: float = 18.0
    pd_scale_denom: float = 850.0
    pd_floor: float = 0.3

    # ── Blend with the trained sklearn risk model ─────────────────────────
    # 0.0 = pure scorecard, 1.0 = pure model. Only applied when a model has
    # been trained on the Model Hub page.
    ml_blend_weight: float = 0.45

    def to_dict(self) -> dict:
        return asdict(self)

    def apply(self, patch: dict) -> list[str]:
        """Update numeric fields from a dict; returns the names that changed."""
        valid = {f.name: f.type for f in fields(self)}
        changed = []
        for k, v in (patch or {}).items():
            if k not in valid or v is None:
                continue
            try:
                cast = int(v) if valid[k] == "int" else float(v)
            except (TypeError, ValueError):
                continue
            if getattr(self, k) != cast:
                setattr(self, k, cast)
                changed.append(k)
        return changed


DEFAULT_SCORING = ScoringConfig()
