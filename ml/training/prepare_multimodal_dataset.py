"""
PhishGuard – Multimodal Dataset Preparation Module Alias

Re-exports preparation and audit functions from datasets.scripts.prepare_multimodal_dataset.
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.scripts.prepare_multimodal_dataset import (
    DEFAULT_AUDIT_JSON_PATH,
    DEFAULT_INPUT_CSV,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_REPORT_MD_PATH,
    DEFAULT_TEST_CSV,
    DEFAULT_TRAIN_CSV,
    METADATA_COLUMNS,
    MULTIMODAL_FIELDNAMES,
    PROVENANCE_LIMITATION_NOTICE,
    audit_multimodal_compatibility,
    build_multimodal_record,
    prepare_multimodal_dataset,
    run_pipeline,
)

__all__ = [
    "DEFAULT_AUDIT_JSON_PATH",
    "DEFAULT_INPUT_CSV",
    "DEFAULT_OUTPUT_CSV",
    "DEFAULT_REPORT_MD_PATH",
    "DEFAULT_TEST_CSV",
    "DEFAULT_TRAIN_CSV",
    "METADATA_COLUMNS",
    "MULTIMODAL_FIELDNAMES",
    "PROVENANCE_LIMITATION_NOTICE",
    "audit_multimodal_compatibility",
    "build_multimodal_record",
    "prepare_multimodal_dataset",
    "run_pipeline",
]

if __name__ == "__main__":
    from datasets.scripts.prepare_multimodal_dataset import main
    main()

