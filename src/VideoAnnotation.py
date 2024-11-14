from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.BoundingBox import BoundingBox


@dataclass
class HockeyClip:
    video_id: str
    frame_count: int
    width: int
    height: int
    tracks: Dict[str, List[BoundingBox]]
    frames: list[np.ndarray]
