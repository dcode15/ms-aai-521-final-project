from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from BoundingBox import BoundingBox


def draw_detections(
        frame: np.ndarray,
        boxes: List[BoundingBox],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes on a frame.

    Args:
        frame: RGB frame to draw on
        boxes: List of BoundingBox objects to draw
        color: BGR color tuple for the boxes
        thickness: Line thickness for the boxes

    Returns:
        Frame with drawn boxes
    """
    frame = frame.copy()

    for box in boxes:
        p1 = (int(box.xtl), int(box.ytl))
        p2 = (int(box.xbr), int(box.ybr))
        cv2.rectangle(frame, p1, p2, color, thickness)

        label = f"{box.label}"
        if box.team:
            label += f" ({box.team})"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        cv2.rectangle(
            frame,
            (p1[0], p1[1] - text_height - baseline),
            (p1[0] + text_width, p1[1]),
            color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (p1[0], p1[1] - baseline),
            font,
            font_scale,
            (0, 0, 0),
            thickness
        )

    return frame


def create_detection_video(
        frames: List[np.ndarray],
        detections: List[List[BoundingBox]],
        output_path: str,
        fps: int = 30
) -> None:
    """
    Create a video with visualized detections.

    Args:
        frames: List of RGB frames
        detections: List of detection lists for each frame
        output_path: Path to save the output video
        fps: Frames per second for the output video
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    try:
        for frame, frame_dets in zip(frames, detections):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_viz = draw_detections(frame_bgr, frame_dets)
            out.write(frame_viz)

    finally:
        out.release()
