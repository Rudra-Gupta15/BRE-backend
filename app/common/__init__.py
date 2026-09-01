"""Shared infrastructure used by every data-source package (aa, gst, ...).

Config, the optional PostgreSQL layer, the persistence facade, ML-security
guardrails, the session-state singleton, the 11-source catalogue, per-source
rule catalogues + toggle state, and rule explainer text. This package imports
nothing from aa/ or gst/ — the dependency arrow only points inward.
"""
