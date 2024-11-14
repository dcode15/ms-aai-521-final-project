import logging
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

from src.AnnotationParser import AnnotationParser
from src.VideoAnnotation import HockeyClip
from src.config import CLIPS_DIR, CVAT_DIR


class Preprocessor:

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def prepare_dataset(
            self
    ) -> list[HockeyClip]:
        logger = logging.getLogger(__name__)

        logger.info("Parsing annotations...")
        annotation_parser = AnnotationParser()
        clips: list[HockeyClip] = []

        for xml_path in tqdm(glob(f"{CVAT_DIR}/**/*.xml", recursive=True)):
            annotation = annotation_parser.parse_cvat_xml(xml_path)
            annotation.frames = self._load_video_frames(f"{CLIPS_DIR}/{annotation.video_id}.mp4")
            clips.append(annotation_parser.parse_cvat_xml(xml_path))

        return clips

    def _load_video_frames(self, video_path: str) -> list[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        return frames
