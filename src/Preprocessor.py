import logging
from glob import glob
from pathlib import Path
from typing import Iterator, List

import cv2
import numpy as np

from src.AnnotationParser import AnnotationParser
from src.VideoAnnotation import HockeyClip


class Preprocessor:

    def __init__(self, clips_dir: str, cvat_dir: str):
        """
        Initialize the Preprocessor.

        Args:
            clips_dir: Directory containing video clips
            cvat_dir: Directory containing CVAT annotations
        """
        self.logger = logging.getLogger(__name__)
        self.clips_dir = Path(clips_dir)
        self.cvat_dir = Path(cvat_dir)

    def get_clips(self) -> Iterator[HockeyClip]:
        """
        Get an iterator over all clips without loading frames into memory.

        Yields:
            HockeyClip objects with metadata but no frames loaded
        """
        annotation_parser = AnnotationParser()

        for xml_path in glob(f"{self.cvat_dir}/**/*.xml", recursive=True):
            annotation = annotation_parser.parse_cvat_xml(xml_path)

            video_path = self.clips_dir / f"{annotation.video_id}.mp4"
            if not video_path.exists():
                self.logger.warning(f"Video file not found: {video_path}")
                continue

            yield annotation

    def frame_iterator(self, clip: HockeyClip) -> Iterator[tuple[int, np.ndarray]]:
        """
        Get an iterator over frames in a clip.

        Args:
            clip: HockeyClip object to get frames from

        Yields:
            Tuples of (frame_index, frame_data)
        """
        video_path = self.clips_dir / f"{clip.video_id}.mp4"
        cap = cv2.VideoCapture(str(video_path))

        try:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame_idx, frame
                frame_idx += 1

        finally:
            cap.release()

    def process_clip_frames(
            self,
            clip: HockeyClip,
            frame_processor: callable,
            batch_size: int = 32
    ) -> Iterator[List]:
        """
        Process frames in batches using a provided function.

        Args:
            clip: HockeyClip object to process
            frame_processor: Function that takes a list of frames and returns processed results
            batch_size: Number of frames to process at once

        Yields:
            Processed results for each batch of frames
        """
        current_batch = []
        current_indices = []

        for frame_idx, frame in self.frame_iterator(clip):
            current_batch.append(frame)
            current_indices.append(frame_idx)

            if len(current_batch) >= batch_size:
                results = frame_processor(current_batch)
                yield current_indices, results
                current_batch = []
                current_indices = []

        if current_batch:
            results = frame_processor(current_batch)
            yield current_indices, results

    def get_frame_count(self, clip: HockeyClip) -> int:
        video_path = self.clips_dir / f"{clip.video_id}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        try:
            return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()

    def get_clip_metadata(self, clip: HockeyClip) -> dict:
        """Get video metadata without loading frames."""
        video_path = self.clips_dir / f"{clip.video_id}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        try:
            metadata = {
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': int(cap.get(cv2.CAP_PROP_FPS)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            }
            return metadata
        finally:
            cap.release()
