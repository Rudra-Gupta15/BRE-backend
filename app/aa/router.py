"""AA HTTP layer — one APIRouter per URL area, aggregated into `router`.

Route bodies still hold their logic (Phase 5 will thin them); the split from the
old flat app/routers/ is by ownership: /pipeline, /models, /inference,
/bre-products and /settings are all bank-statement underwriting. GST branches
inside these call app.gst.* lazily.
"""
from fastapi import APIRouter

from app.aa.routes import bre_products, inference, models, pipeline, settings

router = APIRouter()
for _m in (pipeline, models, inference, bre_products, settings):
    router.include_router(_m.router)
