from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

from BoundingBox import BoundingBox


def draw_detections(
        frame: np.ndarray,
        boxes: List[BoundingBox],
        color_map: Optional[Dict[str, Tuple[int, int, int]]] = None,
        thickness: int = 2
) -> np.ndarray:
    frame = frame.copy()
    color_map = color_map or {}

    for box in boxes:
        if box.track_id not in color_map:
            track_num = hash(box.track_id) % 100000
            color_map[box.track_id] = (
                (track_num * 123) % 255,
                (track_num * 456) % 255,
                (track_num * 789) % 255
            )

        color = color_map[box.track_id]
        p1 = (int(box.xtl), int(box.ytl))
        p2 = (int(box.xbr), int(box.ybr))
        cv2.rectangle(frame, p1, p2, color, thickness)

        label_parts = []
        if box.track_id:
            label_parts.append(f"ID: {box.track_id}")
        if box.team:
            label_parts.append(f"Team: {box.team}")
        label = " | ".join(label_parts) if label_parts else box.label

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
        fps: int = 30,
        codec: str = 'mp4v'
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames[0].shape[:2]

    color_map = {}
    for frame_dets in detections:
        for det in frame_dets:
            if det.track_id and det.track_id not in color_map:
                track_num = hash(det.track_id) % 100000
                color_map[det.track_id] = (
                    (track_num * 123) % 255,
                    (track_num * 456) % 255,
                    (track_num * 789) % 255
                )

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    try:
        for frame, frame_dets in zip(frames, detections):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_viz = draw_detections(frame_bgr, frame_dets, color_map)
            out.write(frame_viz)

    finally:
        out.release()