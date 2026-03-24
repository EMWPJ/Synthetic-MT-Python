"""Presentation layer - GUI and CLI interfaces.

This layer is responsible for presenting information to users and
capturing user commands. It includes GUI components, CLI interfaces,
and any presentation-specific logic.
"""

try:
    from .gui import SyntheticMTGui
except ImportError:
    SyntheticMTGui = None

__all__ = ['SyntheticMTGui']
