"""Application layer - DTOs and use cases.

The application layer orchestrates domain objects and infrastructure services
to fulfill user requests. It contains Data Transfer Objects (DTOs) for
request/response and use cases that coordinate the workflow.
"""

from .dto import (
    OutputFormat,
    SynthesisRequest,
    SynthesisResult,
)
from .synthesis_use_case import SynthesisUseCase

__all__ = [
    'OutputFormat',
    'SynthesisRequest',
    'SynthesisResult',
    'SynthesisUseCase',
]
