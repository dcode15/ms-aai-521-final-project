from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from src.BoundingBox import BoundingBox


class YOLODetector:

    def __init__(
            self,
            model_name: str = "yolov8m.pt",
            conf_threshold: float = 0.25,
            device: Optional[str] = None
    ):
        """
        Initialize YOLO detector.

        Args:
            model_name: Name/path of the YOLO model to use
            conf_threshold: Confidence threshold for detections
            device: Device to run inference on ('cuda' or 'cpu'). If None, use CUDA if available.
        """
        self.conf_threshold = conf_threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = YOLO(model_name)
        self.person_class = 0

    def detect_frame(
            self,
            frame: np.ndarray,
            frame_idx: int
    ) -> list[BoundingBox]:
        """
        Detect players in a single frame.

        Args:
            frame: RGB image as numpy array (H, W, C)
            frame_idx: Index of the frame in the video

        Returns:
            List of BoundingBox objects for detected players
        """
        results = self.model(frame, verbose=False)[0]

        boxes = []
        for i, det in tqdm(enumerate(results.boxes), desc="Extracting detections"):
            if int(det.cls) != self.person_class or float(det.conf) < self.conf_threshold:
                continue

            x1, y1, x2, y2 = map(float, det.xyxy[0])

            bbox = BoundingBox(
                frame_idx=frame_idx,
                track_id=f"det_{frame_idx}_{i}",
                label="player",
                xtl=x1,
                ytl=y1,
                xbr=x2,
                ybr=y2,
                occluded=False,
                team=None
            )
            boxes.append(bbox)

        return boxes

    def detect_video(
            self,
            frames: list[np.ndarray],
            batch_size: int = 16
    ) -> list[list[BoundingBox]]:
        """
        Detect players in all frames of a video.

        Args:
            frames: List of RGB frames as numpy arrays
            batch_size: Number of frames to process at once

        Returns:
            List of BoundingBox lists for each frame
        """
        all_detections = []

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_results = self.model(batch_frames, verbose=False)

            for frame_offset, results in enumerate(batch_results):
                frame_idx = i + frame_offset
                frame_boxes = []

                for j, det in enumerate(results.boxes):
                    if int(det.cls) != self.person_class or float(det.conf) < self.conf_threshold:
                        continue

                    x1, y1, x2, y2 = map(float, det.xyxy[0])

                    bbox = BoundingBox(
                        frame_idx=frame_idx,
                        track_id=f"det_{frame_idx}_{j}",
                        label="player",
                        xtl=x1,
                        ytl=y1,
                        xbr=x2,
                        ybr=y2,
                        occluded=False,
                        team=None
                    )
                    frame_boxes.append(bbox)

                all_detections.append(frame_boxes)

        return all_detections
