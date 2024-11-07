from dataclasses import dataclass
from typing import Dict, List

from src.BoundingBox import BoundingBox


@dataclass
class VideoAnnotations:
    video_id: str
    frame_count: int
    width: int
    height: int
    tracks: Dict[str, List[BoundingBox]]
