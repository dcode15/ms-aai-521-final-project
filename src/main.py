import argparse
import logging
import random
from pathlib import Path

from ModelEvaluator import ModelEvaluator
from Preprocessor import Preprocessor
from YOLODetector import YOLODetector
from YOLOTrainer import YOLOTrainer
from config import (
    CLIPS_DIR,
    CVAT_DIR,
    OUTPUT_DIR,
    YOLO_MODEL,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_BATCH_SIZE,
    TRACK_BUFFER,
    MATCH_THRESH,
)


def main():
    parser = argparse.ArgumentParser(description='Train and run YOLO on hockey videos')
    parser.add_argument('--force-prepare', action='store_true',
                        help='Force dataset preparation')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training and use existing model')
    parser.add_argument('--no-visualizations', action='store_true',
                        help='Skip creating visualization videos')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    random.seed(1)

    preprocessor = Preprocessor(CLIPS_DIR, CVAT_DIR)
    train_clips, val_clips, test_clips = preprocessor.split_dataset()

    if not args.skip_training:
        logger.info("Starting YOLO fine-tuning")
        trainer = YOLOTrainer(
            model_path=YOLO_MODEL,
            preprocessor=preprocessor,
            output_dir=OUTPUT_DIR
        )

        trainer.train(force_prepare=args.force_prepare)
        trainer.export_model()
    else:
        logger.info("Skipping training due to --skip-training flag.")

    logger.info("Loading fine-tuned model for detection")
    detector = YOLODetector(
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
        batch_size=YOLO_BATCH_SIZE
    )

    evaluator.evaluate_clips(
        clips=test_clips,
        save_visualizations=not args.no_visualizations
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
