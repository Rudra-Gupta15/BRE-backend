"""ML-security layer for the BRE portal.

Seven concerns, each in its own module:

  guardrails.py        - bounds/validation on every untrusted input & output
  outliers.py          - feature-vector outlier detection vs. the training corpus
  pattern_baseline.py  - fraud/anomaly pattern deviation vs. the training baseline
  poisoning.py         - training-data validation + candidate-model promotion guard
  drift.py             - concept-drift monitoring (PSI) from stored inference runs
  lineage.py           - provenance helpers (file hashes, parser/model tagging)
  serialization.py     - hashed + signed model artifacts, safe load

Every function degrades safely - if the DB is off or data is missing, the
security check logs and returns a permissive/empty result rather than blocking
the pipeline, EXCEPT the hard guardrails which raise on a real violation.
"""

from app.common.security import (  # noqa: F401
    drift,
    guardrails,
    lineage,
    outliers,
    pattern_baseline,
    poisoning,
    serialization,
)
