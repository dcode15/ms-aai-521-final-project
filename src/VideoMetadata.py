from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoMetadata:
    original_path: Path
    processed_path: Path
    original_fps: float
    original_resolution: tuple[int, int]
    frame_count: int
    processed_frames: int
