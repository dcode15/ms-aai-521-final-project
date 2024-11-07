from dataclasses import dataclass
from typing import Optional, Literal

TeamColor = Literal['white', 'black']


@dataclass
class BoundingBox:
    frame_idx: int
    track_id: str
    label: str
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    occluded: bool
    team: Optional[TeamColor] = None
