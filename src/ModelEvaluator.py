import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from BoundingBox import BoundingBox
from Preprocessor import Preprocessor
from VideoAnnotation import HockeyClip
from Visualizer import Visualizer
from YOLODetector import YOLODetector


@dataclass
class ClipEvaluationContext:
    """Holds data and settings for evaluating a single clip."""
    clip: HockeyClip
    output_dir: Path
    fps: int
    save_visualizations: bool


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
            detector: YOLODetector,
            preprocessor: Preprocessor,
            output_dir: Path,
            batch_size: int = 16,
    ):
        self.logger = logging.getLogger(__name__)
        self.detector = detector
        self.preprocessor = preprocessor
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.visualizer = Visualizer()

    def evaluate_clips(
            self,
            clips: List[HockeyClip],
            save_visualizations: bool = True
    ) -> None:
        """Evaluate model performance on a list of clips."""
        self.logger.info("Starting model evaluation")

        for clip in clips:
            self.logger.info(f"Processing clip {clip.video_id}")
            context = self._create_evaluation_context(clip, save_visualizations)
            self._evaluate_single_clip(context)

    def _create_evaluation_context(
            self,
            clip: HockeyClip,
            save_visualizations: bool
    ) -> ClipEvaluationContext:
        """Create evaluation context for a single clip."""
        output_dir = self._prepare_output_directory()
        metadata = self.preprocessor.get_clip_metadata(clip)

        return ClipEvaluationContext(
            clip=clip,
            output_dir=output_dir,
            fps=metadata['fps'],
            save_visualizations=save_visualizations
        )

    def _prepare_output_directory(self) -> Path:
        """Prepare the output directory for evaluation results."""
        output_dir = self.output_dir / 'detections' / 'test'
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _evaluate_single_clip(self, context: ClipEvaluationContext) -> None:
        """Evaluate model performance on a single clip."""
        processed_data = self._process_clip_frames(context.clip)

        if context.save_visualizations:
            self._create_visualizations(context, processed_data)

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

    def _create_visualizations(
            self,
            context: ClipEvaluationContext,
            data: ProcessedFrameData
    ) -> None:
        """Create visualization videos for predictions and ground truth."""
        self._create_prediction_visualization(context, data)
        self._create_combined_visualization(context, data)

    def _create_prediction_visualization(
            self,
            context: ClipEvaluationContext,
            data: ProcessedFrameData
    ) -> None:
        """Create visualization video with predictions only."""
        output_path = context.output_dir / f'{context.clip.video_id}_pred_only.mp4'
        self.logger.info(f"Creating prediction-only visualization: {output_path}")

        self.visualizer.create_detection_video(
            frames=data.frames,
            pred_detections=data.pred_detections,
            output_path=str(output_path),
            fps=context.fps
        )

    def _create_combined_visualization(
            self,
            context: ClipEvaluationContext,
            data: ProcessedFrameData
    ) -> None:
        """Create visualization video with both predictions and ground truth."""
        output_path = context.output_dir / f'{context.clip.video_id}_pred_and_gt.mp4'
        self.logger.info(f"Creating combined visualization: {output_path}")

        self.visualizer.create_detection_video(
            frames=data.frames,
            pred_detections=data.pred_detections,
            gt_detections=data.gt_detections,
            output_path=str(output_path),
            fps=context.fps
        )
