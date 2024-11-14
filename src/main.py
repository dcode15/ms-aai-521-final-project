import logging
from pathlib import Path

from tqdm import tqdm

from src.Preprocessor import Preprocessor
from src.YOLODetector import YOLODetector
from src.config import OUTPUT_DIR, YOLO_MODEL, YOLO_CONFIDENCE_THRESHOLD, YOLO_BATCH_SIZE
from src.visualize import create_detection_video

logger = logging.getLogger(__name__)

detector = YOLODetector(
    model_name=YOLO_MODEL,
    conf_threshold=YOLO_CONFIDENCE_THRESHOLD
)

logger.info("Loading dataset")
preprocessor = Preprocessor()
clips = preprocessor.prepare_dataset()

for clip in tqdm(clips):
    logger.info(f"Processing clip {clip.video_id}")

    detections = detector.detect_video(
        clip.frames,
        batch_size=YOLO_BATCH_SIZE
    )

    output_path = Path(OUTPUT_DIR) / 'detections' / f'{clip.video_id}_detections.mp4'
    logger.info(f"Creating visualization video: {output_path}")

    create_detection_video(
        frames=clip.frames,
        detections=detections,
        output_path=str(output_path)
    )

logger.info("Done!")
