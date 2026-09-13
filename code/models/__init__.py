"""Public model interface for DCR-GRN."""

from .model import ATFGRN


class DCRGRN(ATFGRN):
    """Public DCR-GRN model class.

    Inheriting without changing the module hierarchy preserves all state-dict
    keys used by the formal experiment checkpoints.
    """

__all__ = ["DCRGRN", "ATFGRN"]
