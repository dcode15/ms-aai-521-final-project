import argparse
import logging
import random
from pathlib import Path

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
    TRACK_BUFFER,
    MATCH_THRESH,
    TRAIN_PROPORTION,
    TEST_PROPORTION,
)
from visualize import create_detection_video


def get_frame_detections(clip, frame_idx):
    frame_boxes = []
    for track in clip.tracks.values():
        frame_boxes.extend([
            box for box in track
            if box.frame_idx == frame_idx and box.label in ['player', 'keeper']
        ])
    return frame_boxes


def main():
    parser = argparse.ArgumentParser(description='Train and run YOLO on hockey videos')
    parser.add_argument('--force-prepare', action='store_true',
                        help='Force dataset preparation even if already exists')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training and use existing model')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    random.seed(1)

    preprocessor = Preprocessor(CLIPS_DIR, CVAT_DIR)
    all_clips = list(preprocessor.get_clips())
    random.shuffle(all_clips)

    n_clips = len(all_clips)
    n_test = max(1, int(n_clips * TEST_PROPORTION))
    n_train = max(1, int(n_clips * TRAIN_PROPORTION))

    test_clips = all_clips[-n_test:]
    train_clips = all_clips[:n_train]
    val_clips = all_clips[n_train:-n_test]

    logger.info(f"Dataset split: {len(train_clips)} train, {len(val_clips)} val, {len(test_clips)} test clips")

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
        model_name=str(Path(OUTPUT_DIR) / 'finetune' / 'weights' / 'best.pt'),
        conf_threshold=YOLO_CONFIDENCE_THRESHOLD,
        track_buffer=TRACK_BUFFER,
        match_thresh=MATCH_THRESH,
    )

    logger.info("Processing test clips")
    for clip in test_clips:
        logger.info(f"Processing test clip {clip.video_id}")

        output_dir = Path(OUTPUT_DIR) / 'detections' / 'test'
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = preprocessor.get_clip_metadata(clip)
        frame_count = metadata['frame_count']
        fps = metadata['fps']

        frames = []
        frame_indices = []
        pred_detections = []
        gt_detections = []

        with tqdm(total=frame_count, desc="Processing frames") as pbar:
            for batch_indices, batch_frames in preprocessor.process_clip_frames(
                    clip,
                    lambda x: x,
                    batch_size=YOLO_BATCH_SIZE
            ):
                frames.extend(batch_frames)
                frame_indices.extend(batch_indices)

                batch_pred_detections = detector.detect_video(
                    batch_frames,
                    batch_size=YOLO_BATCH_SIZE
                )
                pred_detections.extend(batch_pred_detections)

                batch_gt_detections = [
                    get_frame_detections(clip, idx)
                    for idx in batch_indices
                ]
                gt_detections.extend(batch_gt_detections)

                pbar.update(len(batch_indices))

        pred_output_path = output_dir / f'{clip.video_id}_pred_only.mp4'
        logger.info(f"Creating prediction-only visualization: {pred_output_path}")

        create_detection_video(
            frames=frames,
            pred_detections=pred_detections,
            output_path=str(pred_output_path),
            fps=fps
        )

        combined_output_path = output_dir / f'{clip.video_id}_pred_and_gt.mp4'
        logger.info(f"Creating combined visualization: {combined_output_path}")

        create_detection_video(
            frames=frames,
            pred_detections=pred_detections,
            gt_detections=gt_detections,
            output_path=str(combined_output_path),
            fps=fps
        )

        logger.info(f"Finished processing clip {clip.video_id}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
