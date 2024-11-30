import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

from BoundingBox import BoundingBox
from ObjectDetector import ObjectDetector
from Preprocessor import Preprocessor
from VideoAnnotation import HockeyClip


@dataclass
class ProcessedFrameData:
    """Holds processed frame data and detections."""
    frames: List[np.ndarray]
    frame_indices: List[int]
    pred_detections: List[List[BoundingBox]]
    gt_detections: List[List[BoundingBox]]


class ModelEvaluator:
    """Class for evaluating model performance on hockey tracking data."""

    def __init__(
            self,
            detector: ObjectDetector,
            preprocessor: Preprocessor,
            output_dir: Path,
            batch_size: int = 16,
    ):
        self.logger = logging.getLogger(__name__)
        self.detector = detector
        self.preprocessor = preprocessor
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size

    def evaluate_clips(
            self,
            clips: List[HockeyClip]
    ) -> Dict[HockeyClip, Tuple[List[np.ndarray], List[List[BoundingBox]], List[List[BoundingBox]]]]:
        """
        Evaluate model performance on a list of clips.

        Returns:
            Dict mapping clips to tuples of (frames, predictions, ground_truth)
        """
        self.logger.info("Starting model evaluation")
        results = {}

        for clip in clips:
            self.logger.info(f"Processing clip {clip.video_id}")
            processed_data = self._process_clip_frames(clip)
            results[clip] = (
                processed_data.frames,
                processed_data.pred_detections,
                processed_data.gt_detections
            )

        return results

    def _process_clip_frames(self, clip: HockeyClip) -> ProcessedFrameData:
        """Process frames from a clip and collect predictions."""
        frames: List[np.ndarray] = []
        frame_indices: List[int] = []
        pred_detections: List[List[BoundingBox]] = []
        gt_detections: List[List[BoundingBox]] = []

        for batch_data in self._process_frame_batches(clip):
            batch_indices, batch_frames, batch_preds, batch_gt = batch_data

            frames.extend(batch_frames)
            frame_indices.extend(batch_indices)
            pred_detections.extend(batch_preds)
            gt_detections.extend(batch_gt)

        return ProcessedFrameData(
            frames=frames,
            frame_indices=frame_indices,
            pred_detections=pred_detections,
            gt_detections=gt_detections
        )

    def _process_frame_batches(
            self,
            clip: HockeyClip
    ) -> List[Tuple[List[int], List[np.ndarray], List[List[BoundingBox]], List[List[BoundingBox]]]]:
        """Process clip frames in batches."""
        batches = []

        for batch_indices, batch_frames in self.preprocessor.process_clip_frames(
                clip,
                lambda x: x,
                batch_size=self.batch_size
        ):
            batch_pred_detections = self._get_batch_predictions(batch_frames)
            batch_gt_detections = self._get_batch_ground_truth(clip, batch_indices)

            batches.append((
                batch_indices,
                batch_frames,
                batch_pred_detections,
                batch_gt_detections
            ))

        return batches

    def _get_batch_predictions(
            self,
            batch_frames: List[np.ndarray]
    ) -> List[List[BoundingBox]]:
        """Get model predictions for a batch of frames."""
        return self.detector.detect_video(
            batch_frames,
            batch_size=self.batch_size
        )

    def _get_batch_ground_truth(
            self,
            clip: HockeyClip,
            batch_indices: List[int]
    ) -> List[List[BoundingBox]]:
        """Get ground truth detections for a batch of frames."""
        return [
            self._get_frame_ground_truth(clip, idx)
            for idx in batch_indices
        ]

    def _get_frame_ground_truth(
            self,
            clip: HockeyClip,
            frame_idx: int
    ) -> List[BoundingBox]:
        """Get ground truth detections for a specific frame."""
        frame_boxes = []
        for track in clip.tracks.values():
            frame_boxes.extend([
                box for box in track
                if box.frame_idx == frame_idx and box.label in ['player', 'keeper']
            ])
        return frame_boxes
