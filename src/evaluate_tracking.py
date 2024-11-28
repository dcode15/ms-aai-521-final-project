import logging
from pathlib import Path
from typing import List

from Preprocessor import Preprocessor
from TrackingEvaluator import MOTEvaluator
from VideoAnnotation import HockeyClip
from YOLODetector import YOLODetector


def evaluate_tracking(
        clips: List[HockeyClip],
        detector: YOLODetector,
        preprocessor: Preprocessor,
        output_dir: Path,
        batch_size: int = 16,
        iou_threshold: float = 0.5
) -> None:
    logger = logging.getLogger(__name__)
    evaluator = MOTEvaluator(iou_threshold=iou_threshold)

    all_pred_detections = []

    for clip in clips:
        logger.info(f"Processing clip {clip.video_id}")

        frames = []
        frame_indices = []
        pred_detections = []

        for batch_indices, batch_frames in preprocessor.process_clip_frames(
                clip,
                lambda x: x,
                batch_size=batch_size
        ):
            frames.extend(batch_frames)
            frame_indices.extend(batch_indices)

            batch_pred_detections = detector.detect_video(
                batch_frames,
                batch_size=batch_size
            )
            pred_detections.extend(batch_pred_detections)

        all_pred_detections.append(pred_detections)

    logger.info("Computing tracking metrics")
    results = evaluator.evaluate_dataset(
        clips,
        all_pred_detections,
        output_dir=output_dir
    )

    logger.info("\nTracking Metrics Summary:")
    for clip_id, metrics in results.items():
        logger.info(f"\nClip: {clip_id}")
        logger.info(f"MOTA: {metrics.mota:.3f}")
        logger.info(f"MOTP: {metrics.motp:.3f}")
        logger.info(f"ID Switches: {metrics.num_switches}")
        logger.info(f"Fragmentations: {metrics.num_fragmentations}")
        logger.info(f"Precision: {metrics.precision:.3f}")
        logger.info(f"Recall: {metrics.recall:.3f}")
