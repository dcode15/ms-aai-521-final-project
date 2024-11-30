import argparse
import logging
import random
from pathlib import Path

from ModelEvaluator import ModelEvaluator
from ModelTrainer import ModelTrainer
from ObjectDetector import ObjectDetector
from Preprocessor import Preprocessor
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

    if not args.skip_training:
        logger.info("Starting YOLO fine-tuning")
        trainer = ModelTrainer(
            model_path=YOLO_MODEL,
            preprocessor=preprocessor,
            output_dir=OUTPUT_DIR
        )

        trainer.train(
            epochs=TRAIN_EPOCHS,
            batch_size=TRAIN_BATCH_SIZE,
            learning_rate=TRAIN_LEARNING_RATE,
            force_prepare=not args.skip_preprocessing
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
    evaluator = ModelEvaluator(
        detector=detector,
        preprocessor=preprocessor,
        output_dir=Path(OUTPUT_DIR),
        batch_size=EVAL_BATCH_SIZE
    )

    predictions = evaluator.evaluate_clips(test_clips)

    logger.info("Creating visualizations")
    visualizer = Visualizer(default_thickness=VIZ_BOX_THICKNESS)
    output_dir = Path(OUTPUT_DIR) / 'detections' / 'test'
    output_dir.mkdir(parents=True, exist_ok=True)

    for clip, (frames, pred_detections, gt_detections) in predictions.items():
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
