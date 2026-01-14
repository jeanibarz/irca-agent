"""
Argilla Integration

Utilities for converting traces to Argilla records for human annotation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.domain import Trace

logger = logging.getLogger(__name__)


def trace_to_argilla_record(
    available_functions: str,
    user_query: str,
    trace: Trace,
) -> Any:
    """
    Convert a trace to an Argilla FeedbackRecord.

    Args:
        available_functions: JSON string of available functions
        user_query: The user's query
        trace: The generated trace

    Returns:
        Argilla FeedbackRecord ready for annotation
    """
    try:
        import argilla as rg

        logger.info("Creating Argilla record for generated trace...")
        record = rg.FeedbackRecord(
            fields={
                "available_functions": available_functions,
                "user_query": user_query,
                "agent_trace": trace.to_string(),
            },
        )
        return record
    except ImportError:
        logger.error("Argilla not installed. Install with: pip install argilla")
        raise
