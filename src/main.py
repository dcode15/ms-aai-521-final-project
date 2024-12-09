import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from HyperparameterTuner import TuningConfig, HyperparameterTuner
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
    TRAIN_EPOCHS,
    TRAIN_BATCH_SIZE,
    TRAIN_PATIENCE,
    EVAL_BATCH_SIZE,
    VIZ_FPS,
    VIZ_CODEC,
    VIZ_BOX_THICKNESS,
    TUNING_TRIALS,
    TRAIN_LEARNING_RATE,
    TRAIN_WEIGHT_DECAY,
    TRAIN_DROPOUT,
    CONFIDENCE_THRESHOLD,
    IOU,
    TRACK_BUFFER,
    MATCH_THRESHOLD,
    TRACK_LOW_THRESHOLD,
    TRACK_HIGH_THRESHOLD,
    NEW_TRACK_THRESHOLD
)


def main():
    parser = argparse.ArgumentParser(description='Train and run YOLO on hockey videos')
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip dataset preprocessing')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training and use existing model')
    parser.add_argument('--tune-hyperparameters', action='store_true',
                        help='Perform hyperparameter tuning using Optuna')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    random.seed(1)

    preprocessor = Preprocessor(CLIPS_DIR, CVAT_DIR)
    train_clips, val_clips, test_clips = preprocessor.split_dataset()

    if not args.skip_preprocessing:
        preprocessor.prepare_dataset(OUTPUT_DIR)

    if args.tune_hyperparameters:
        logger.info("Starting hyperparameter tuning")
        tuning_config = TuningConfig(
            n_trials=TUNING_TRIALS,
            study_name="hockey_tracking_optimization"
        )

        tuner = HyperparameterTuner(
            base_model_path=YOLO_MODEL,
            output_dir=OUTPUT_DIR,
            train_clips=train_clips,
            val_clips=val_clips,
            preprocessor=preprocessor,
            config=tuning_config
        )

        hyperparameters = tuner.tune()
        logger.info(f"Best hyperparameters found: {hyperparameters}")
    else:
        hyperparameters = {
            'training': {
                'lr0': TRAIN_LEARNING_RATE,
                'weight_decay': TRAIN_WEIGHT_DECAY,
                'dropout': TRAIN_DROPOUT,
                'epochs': TRAIN_EPOCHS,
                'batch': TRAIN_BATCH_SIZE,
                'patience': TRAIN_PATIENCE,
            },
            'detection': {
                'conf': CONFIDENCE_THRESHOLD,
                'iou': IOU,
                'track_buffer': TRACK_BUFFER,
                'match_thresh': MATCH_THRESHOLD,
                'track_low_thresh': TRACK_LOW_THRESHOLD,
                'track_high_thresh': TRACK_HIGH_THRESHOLD,
                'new_track_thresh': NEW_TRACK_THRESHOLD,
            }
        }

        ObjectDetector.write_tracking_params(hyperparameters['detection'], OUTPUT_DIR)

    if not args.skip_training:
        logger.info("Starting YOLO fine-tuning")
        trainer = ModelTrainer(
            model_path=YOLO_MODEL,
            output_dir=OUTPUT_DIR,
        )

        trainer.train(hyperparameters['training'])
        trainer.export_model('torchscript')
    else:
        logger.info("Skipping training due to --skip-training flag.")

    logger.info("Loading fine-tuned model for detection")

    detector = ObjectDetector(model_name=str(Path(OUTPUT_DIR) / 'finetune' / 'weights' / 'best.pt'))

    logger.info("Starting model evaluation")
    evaluator = ModelEvaluator()

    evaluator.plot_training_metrics(
        results_path=Path(OUTPUT_DIR) / 'finetune' / 'results.csv',
        output_dir=Path(OUTPUT_DIR) / 'evaluation',
    )

    predictions = ModelEvaluator.process_clips_for_evaluation(
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

    logger.info("Evaluation Results:")
    for clip_id, metrics in eval_results.items():
        logger.info(f"\nClip: {clip_id}")
        logger.info(f"MOTA: {metrics.mota:.4f}")
        logger.info(f"MOTP: {metrics.motp:.4f}")
        logger.info(f"Precision: {metrics.precision:.4f}")
        logger.info(f"Recall: {metrics.recall:.4f}")
        f1_score = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall + 1e-10)
        logger.info(f"F1 Score: {f1_score:.4f}")
        logger.info(f"Number of track switches: {metrics.num_switches}")
        logger.info(f"Number of fragmentations: {metrics.num_fragmentations}")

        combined_metric = 0.4 * metrics.mota + 0.3 * metrics.motp + 0.3 * f1_score
        logger.info(f"Combined Metric: {combined_metric:.4f}")

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
