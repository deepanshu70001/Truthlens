"""TruthLens synthetic demonstration data service.

College demonstration data access layer for illustrative content indicators,
spread trajectories, and neutral regional exposure context.
"""

from backend.app.services.synthetic.data_access import (
    SyntheticDataAccess,
    get_synthetic_data_access,
)

__all__ = [
    "SyntheticDataAccess",
    "get_synthetic_data_access",
]
