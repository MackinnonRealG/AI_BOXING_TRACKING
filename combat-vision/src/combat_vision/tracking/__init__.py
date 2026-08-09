"""Multi-person identity assignment across frames.

Two implementations behind one protocol: the built-in greedy centroid
tracker and a ByteTrack-based tracker from the ``supervision`` library,
selected via ``tracking.backend`` in config and hot-swappable at runtime
through :class:`SwitchableTracker`.
"""

from combat_vision.tracking.base import Tracker
from combat_vision.tracking.supervision_tracker import SupervisionTracker
from combat_vision.tracking.switchable import SwitchableTracker
from combat_vision.tracking.tracker import FighterTracker

__all__ = ["FighterTracker", "SupervisionTracker", "SwitchableTracker", "Tracker"]
