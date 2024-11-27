from pathlib import Path
from typing import List, Tuple, Dict, Optional, Literal

import cv2
import numpy as np

from BoundingBox import BoundingBox


def draw_boxes(
        frame: np.ndarray,
        pred_boxes: List[BoundingBox],
        gt_boxes: Optional[List[BoundingBox]] = None,
        color_map: Optional[Dict[str, Tuple[int, int, int]]] = None,
        thickness: int = 2
) -> np.ndarray:
    frame = frame.copy()
    color_map = color_map or {}

    def draw_box(
            box: BoundingBox,
            box_type: Literal['pred', 'gt']
    ) -> None:
        if box.track_id not in color_map:
            track_num = hash(box.track_id) % 100000
            color_map[box.track_id] = (
                (track_num * 123) % 255,
                (track_num * 456) % 255,
                (track_num * 789) % 255
            )

        color = color_map[box.track_id]

        if box_type == 'gt':
            dash_length = 10
            x1, y1 = int(box.xtl), int(box.ytl)
            x2, y2 = int(box.xbr), int(box.ybr)

            for x in range(x1, x2, dash_length * 2):
                x_end = min(x + dash_length, x2)
                cv2.line(frame, (x, y1), (x_end, y1), color, thickness)
                cv2.line(frame, (x, y2), (x_end, y2), color, thickness)

            # Draw vertical dashed lines
            for y in range(y1, y2, dash_length * 2):
                y_end = min(y + dash_length, y2)
                cv2.line(frame, (x1, y), (x1, y_end), color, thickness)
                cv2.line(frame, (x2, y), (x2, y_end), color, thickness)
        else:
            p1 = (int(box.xtl), int(box.ytl))
            p2 = (int(box.xbr), int(box.ybr))
            cv2.rectangle(frame, p1, p2, color, thickness)

        label_parts = []
        if box.track_id:
            prefix = "GT" if box_type == 'gt' else "P"
            label_parts.append(f"{prefix}-{box.track_id}")
        if box.team:
            label_parts.append(f"Team: {box.team}")
        label = " | ".join(label_parts) if label_parts else box.label

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        text_x = int(box.xtl)
        text_y = int(box.ytl) - baseline

        cv2.rectangle(
            frame,
            (text_x, text_y - text_height - baseline),
            (text_x + text_width, text_y + baseline),
            color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness
        )

    if gt_boxes:
        for box in gt_boxes:
            draw_box(box, 'gt')

    for box in pred_boxes:
        draw_box(box, 'pred')

    return frame


def create_detection_video(
        frames: List[np.ndarray],
        pred_detections: List[List[BoundingBox]],
        output_path: str,
        gt_detections: Optional[List[List[BoundingBox]]] = None,
        fps: int = 30,
        codec: str = 'mp4v'
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames[0].shape[:2]

    color_map = {}
    for frame_dets in pred_detections:
        for det in frame_dets:
            if det.track_id and det.track_id not in color_map:
                track_num = hash(det.track_id) % 100000
                color_map[det.track_id] = (
                    (track_num * 123) % 255,
                    (track_num * 456) % 255,
                    (track_num * 789) % 255
                )

    if gt_detections:
        for frame_dets in gt_detections:
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
        for i, (frame, frame_pred_dets) in enumerate(zip(frames, pred_detections)):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_gt_dets = gt_detections[i] if gt_detections else None

            frame_viz = draw_boxes(
                frame_bgr,
                frame_pred_dets,
                frame_gt_dets,
                color_map
            )
            out.write(frame_viz)

    finally:
        out.release()
