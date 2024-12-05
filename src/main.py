import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from ModelEvaluator import ModelEvaluator
from ModelTrainer import ModelTrainer
from ObjectDetector import ObjectDetector
from Preprocessor import Preprocessor
from VideoAnnotation import HockeyClip
from Visualizer import Visualizer
from config import (
    CLIPS_DIR,
    CVAT_DIR,
    OUTPUT_DIR,
    YOLO_MODEL,
    YOLO_CONFIDENCE_THRESHOLD,
    TRACK_BUFFER,
    MATCH_THRESH,
    TRAIN_EPOCHS,
    TRAIN_BATCH_SIZE,
    TRAIN_LEARNING_RATE,
    EVAL_BATCH_SIZE,
    VIZ_FPS,
    VIZ_CODEC,
    VIZ_BOX_THICKNESS
)


def process_clips_for_evaluation(
        clips: List[HockeyClip],
        preprocessor: Preprocessor,
        detector: ObjectDetector,
        batch_size: int
) -> Dict[str, Tuple[List[np.ndarray], List, List]]:
    """
    Process clips to get frames, predictions and ground truth for evaluation.

    Returns:
        Dict mapping clip ID to tuple of (frames, predictions, ground_truth)
    """
    results = {}

    for clip in tqdm(clips, desc="Processing clips"):
        frames = []
        pred_detections = []
        gt_detections = []

        for frame_indices, batch_frames in preprocessor.process_clip_frames(clip, lambda x: x, batch_size):
            frames.extend(batch_frames)
            batch_detections = detector.detect_video(batch_frames, batch_size)
            pred_detections.extend(batch_detections)

            for frame_idx in frame_indices:
                frame_gt = []
                for track in clip.tracks.values():
                    frame_gt.extend([
                        box for box in track
                        if box.frame_idx == frame_idx
                    ])
                gt_detections.append(frame_gt)

        results[clip.video_id] = (frames, pred_detections, gt_detections)

    return results


def main():
    parser = argparse.ArgumentParser(description='Train and run YOLO on hockey videos')
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip dataset preprocessing')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training and use existing model')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    random.seed(1)

    preprocessor = Preprocessor(CLIPS_DIR, CVAT_DIR)
    train_clips, val_clips, test_clips = preprocessor.split_dataset()

    if not args.skip_preprocessing:
        preprocessor.prepare_dataset(OUTPUT_DIR)

    if not args.skip_training:
        logger.info("Starting YOLO fine-tuning")
        trainer = ModelTrainer(
            model_path=YOLO_MODEL,
            output_dir=OUTPUT_DIR,
        )

        trainer.train(
            epochs=TRAIN_EPOCHS,
            batch_size=TRAIN_BATCH_SIZE,
            learning_rate=TRAIN_LEARNING_RATE,
        )
        trainer.export_model()
    else:
        logger.info("Skipping training due to --skip-training flag.")

    logger.info("Loading fine-tuned model for detection")
    detector = ObjectDetector(
        model_name=str(Path(OUTPUT_DIR) / 'finetune' / 'weights' / 'best.pt'),
        conf_threshold=YOLO_CONFIDENCE_THRESHOLD,
        track_buffer=TRACK_BUFFER,
        match_thresh=MATCH_THRESH,
    )

    logger.info("Starting model evaluation")
    evaluator = ModelEvaluator()

    predictions = process_clips_for_evaluation(
        test_clips,
        preprocessor,
        detector,
        EVAL_BATCH_SIZE
    )

    eval_output_dir = Path(OUTPUT_DIR) / 'evaluation'
    eval_results = evaluator.evaluate_dataset(
        test_clips,
        [pred[1] for pred in predictions.values()],
        output_dir=eval_output_dir,
        include_keepers=True
    )

    # Log evaluation results
    logger.info("Evaluation Results:")
    for clip_id, metrics in eval_results.items():
        logger.info(f"\nClip: {clip_id}")
        logger.info(f"MOTA: {metrics.mota:.4f}")
        logger.info(f"MOTP: {metrics.motp:.4f}")
        logger.info(f"Precision: {metrics.precision:.4f}")
        logger.info(f"Recall: {metrics.recall:.4f}")
        logger.info(f"Number of track switches: {metrics.num_switches}")
        logger.info(f"Number of fragmentations: {metrics.num_fragmentations}")

    logger.info("Creating visualizations")
    visualizer = Visualizer(default_thickness=VIZ_BOX_THICKNESS)
    output_dir = Path(OUTPUT_DIR) / 'detections' / 'test'
    output_dir.mkdir(parents=True, exist_ok=True)

    for clip in tqdm(test_clips, desc="Creating visualizations"):
        frames, pred_detections, gt_detections = predictions[clip.video_id]

        pred_output_path = output_dir / f'{clip.video_id}_pred_only.mp4'
        visualizer.create_detection_video(
            frames=frames,
            pred_detections=pred_detections,
            output_path=str(pred_output_path),
            fps=VIZ_FPS,
            codec=VIZ_CODEC
        )

        combined_output_path = output_dir / f'{clip.video_id}_pred_and_gt.mp4'
        visualizer.create_detection_video(
            frames=frames,
            pred_detections=pred_detections,
            gt_detections=gt_detections,
            output_path=str(combined_output_path),
            fps=VIZ_FPS,
            codec=VIZ_CODEC
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()
