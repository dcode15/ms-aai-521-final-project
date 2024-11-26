import logging
from pathlib import Path
import argparse

from tqdm import tqdm

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
)
from src.visualize import create_detection_video


def main():
    parser = argparse.ArgumentParser(description='Train and run YOLO on hockey videos')
    parser.add_argument('--force-prepare', action='store_true',
                        help='Force dataset preparation even if already exists')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training and use existing model')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if not args.skip_training:
        logger.info("Starting YOLO fine-tuning")
        trainer = YOLOTrainer(
            model_path=YOLO_MODEL,
            clips_dir=CLIPS_DIR,
            cvat_dir=CVAT_DIR,
            output_dir=OUTPUT_DIR,
            force_data_preparation=args.force_prepare
        )

        try:
            trainer.train()
            trainer.export_model()
        finally:
            trainer.cleanup()
    else:
        logger.info("Skipping training due to --skip-training flag.")

    logger.info("Loading fine-tuned model for detection")
    detector = YOLODetector(
        model_name=str(Path(OUTPUT_DIR) / 'finetune' / 'best.pt'),
        conf_threshold=YOLO_CONFIDENCE_THRESHOLD
    )

    logger.info("Processing clips")
    preprocessor = Preprocessor(CLIPS_DIR, CVAT_DIR)

    for clip in preprocessor.get_clips():
        logger.info(f"Processing clip {clip.video_id}")

        output_dir = Path(OUTPUT_DIR) / 'detections' / clip.video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = preprocessor.get_clip_metadata(clip)
        frame_count = metadata['frame_count']
        fps = metadata['fps']

        frames = []
        detections = []

        with tqdm(total=frame_count, desc="Processing frames") as pbar:
            for frame_indices, batch_frames in preprocessor.process_clip_frames(
                    clip,
                    lambda x: x,
                    batch_size=YOLO_BATCH_SIZE
            ):
                batch_detections = detector.detect_video(
                    batch_frames,
                    batch_size=YOLO_BATCH_SIZE
                )

                frames.extend(batch_frames)
                detections.extend(batch_detections)

                batch_output_path = output_dir / f'batch_{frame_indices[0]:06d}.mp4'
                create_detection_video(
                    frames=batch_frames,
                    detections=batch_detections,
                    output_path=str(batch_output_path),
                    fps=fps
                )

                pbar.update(len(frame_indices))

                frames = []
                detections = []

        logger.info(f"Finished processing clip {clip.video_id}")

    logger.info("Done!")


if __name__ == "__main__":
    main()