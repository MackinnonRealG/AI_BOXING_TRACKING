"""The SportProfile abstraction.

A profile *declares* what is active for a sport — strike classes, striking
limbs, scoring body zones — and the pipeline consumes those declarations.
The pipeline itself never branches on the sport name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from combat_vision.events.types import Limb, StrikeType


@dataclass(frozen=True, slots=True)
class BodyZone:
    """A scoring target zone, expressed as keypoint-relative regions.

    v1 keeps zones symbolic (head/body/legs); geometric zone resolution
    against a defender's pose lands with the accuracy/landed detection work.
    """

    name: str
    description: str


class SportProfile(ABC):
    """Everything sport-specific the pipeline needs, in one object."""

    name: str

    @property
    @abstractmethod
    def strike_types(self) -> frozenset[StrikeType]:
        """Strike classes the classifier may emit in this sport."""

    @property
    @abstractmethod
    def striking_limbs(self) -> frozenset[Limb]:
        """Limbs monitored for strike candidates (hands only in boxing)."""

    @property
    @abstractmethod
    def target_zones(self) -> tuple[BodyZone, ...]:
        """Legal scoring zones for this sport."""

    def allows(self, strike_type: StrikeType) -> bool:
        """True if this sport recognises ``strike_type``."""
        return strike_type in self.strike_types
