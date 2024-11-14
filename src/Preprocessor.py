import logging
import os
import pickle
from glob import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from src.AnnotationParser import AnnotationParser
from src.VideoAnnotation import HockeyClip
from src.config import CLIPS_DIR, CVAT_DIR, OUTPUT_DIR


class Preprocessor:

    def __init__(self, cache_file: str = "processed_dataset.pkl"):
        """
        Initialize the Preprocessor.

        Args:
            cache_file: Name of the file to save/load processed data
        """
        self.logger = logging.getLogger(__name__)
        self.cache_path = Path(OUTPUT_DIR) / cache_file

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def prepare_dataset(
            self,
            force_preprocess: bool = False
    ) -> list[HockeyClip]:
        """
        Prepare the dataset by either loading from cache or processing from scratch.

        Args:
            force_preprocess: If True, reprocess the data even if cache exists

        Returns:
            List of processed HockeyClip objects
        """
        if not force_preprocess:
            clips = self._load_from_cache()
            if clips is not None:
                self.logger.info("Successfully loaded dataset from cache")
                return clips

        self.logger.info("Processing dataset from scratch...")
        clips = self._process_dataset()

        self._save_to_cache(clips)
        return clips

    def _process_dataset(self) -> list[HockeyClip]:
        """
        Process the dataset from raw files.

        Returns:
            List of processed HockeyClip objects
        """
        annotation_parser = AnnotationParser()
        clips: list[HockeyClip] = []

        self.logger.info("Parsing annotations and loading video frames...")
        for xml_path in tqdm(glob(f"{CVAT_DIR}/**/*.xml", recursive=True), desc="Loading clips"):
            annotation = annotation_parser.parse_cvat_xml(xml_path)

            video_path = Path(CLIPS_DIR) / f"{annotation.video_id}.mp4"
            if not video_path.exists():
                self.logger.warning(f"Video file not found: {video_path}")
                continue

            frames = self._load_video_frames(str(video_path))
            if not frames:
                self.logger.warning(f"No frames loaded for video: {video_path}")
                continue

            annotation.frames = frames
            clips.append(annotation)
            break

        return clips

    def _load_video_frames(self, video_path: str) -> list[np.ndarray]:
        """
        Load frames from a video file.

        Args:
            video_path: Path to the video file

        Returns:
            List of video frames as numpy arrays (RGB format)
        """
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        finally:
            cap.release()

        return frames

    def _save_to_cache(self, clips: list[HockeyClip]) -> None:
        """
        Save processed data to cache file using pickle.

        Args:
            clips: List of HockeyClip objects to save
        """
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(clips, f)
            self.logger.info(f"Successfully saved processed dataset to {self.cache_path}")
        except Exception as e:
            self.logger.error(f"Failed to save dataset to cache: {str(e)}")

    def _load_from_cache(self) -> Optional[list[HockeyClip]]:
        """
        Try to load processed data from cache file using pickle.

        Returns:
            List of HockeyClip objects if successful, None otherwise
        """
        if not self.cache_path.exists():
            self.logger.info("No cache file found")
            return None

        try:
            with open(self.cache_path, 'rb') as f:
                clips = pickle.load(f)
            return clips
        except Exception as e:
            self.logger.error(f"Failed to load dataset from cache: {str(e)}")
            return None
