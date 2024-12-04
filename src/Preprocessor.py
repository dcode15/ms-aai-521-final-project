import logging
from glob import glob
from pathlib import Path
from typing import Iterator, List, Tuple, Dict

import cv2
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from AnnotationParser import AnnotationParser
from BoundingBox import BoundingBox
from VideoAnnotation import HockeyClip
from config import VALIDATION_PROPORTION, TEST_PROPORTION


class Preprocessor:
    def __init__(self, clips_dir: str, cvat_dir: str):
        """Initialize the Preprocessor."""
        self.logger = logging.getLogger(__name__)
        self.clips_dir = Path(clips_dir)
        self.cvat_dir = Path(cvat_dir)

    def get_clips(self) -> Iterator[HockeyClip]:
        """Get an iterator over all clips without loading frames."""
        annotation_parser = AnnotationParser()
        for xml_path in self._find_annotation_files():
            annotation = annotation_parser.parse_cvat_xml(xml_path)
            if self._validate_video_exists(annotation):
                yield annotation

    def _find_annotation_files(self) -> List[str]:
        """Find all CVAT annotation XML files."""
        return glob(f"{self.cvat_dir}/**/*.xml", recursive=True)

    def _validate_video_exists(self, clip: HockeyClip) -> bool:
        """Check if video file exists for the given clip."""
        video_path = self.clips_dir / f"{clip.video_id}.mp4"
        if not video_path.exists():
            self.logger.warning(f"Video file not found: {video_path}")
            return False
        return True

    def split_dataset(self, seed: int = 1) -> Tuple[List[HockeyClip], List[HockeyClip], List[HockeyClip]]:
        """Split the dataset into train, validation, and test sets."""
        all_clips = list(self.get_clips())
        train_clips, val_clips, test_clips = self._perform_dataset_split(all_clips, seed)
        self._log_split_info(train_clips, val_clips, test_clips)
        return train_clips, val_clips, test_clips

    def _perform_dataset_split(
            self,
            clips: List[HockeyClip],
            seed: int
    ) -> Tuple[List[HockeyClip], List[HockeyClip], List[HockeyClip]]:
        """Perform the actual dataset splitting."""
        train_val_clips, test_clips = train_test_split(
            clips,
            test_size=TEST_PROPORTION,
            random_state=seed
        )

        val_proportion_adjusted = VALIDATION_PROPORTION / (1 - TEST_PROPORTION)
        train_clips, val_clips = train_test_split(
            train_val_clips,
            test_size=val_proportion_adjusted,
            random_state=seed
        )

        return train_clips, val_clips, test_clips

    def _log_split_info(
            self,
            train_clips: List[HockeyClip],
            val_clips: List[HockeyClip],
            test_clips: List[HockeyClip]
    ) -> None:
        """Log information about the dataset split."""
        self.logger.info(
            f"Dataset split: {len(train_clips)} train, "
            f"{len(val_clips)} validation, {len(test_clips)} test clips"
        )

    def prepare_dataset(
            self,
            output_dir: str
    ) -> None:
        """Prepare the dataset for YOLO training."""
        dataset_paths = self._setup_dataset_directories(output_dir)

        self.logger.info("Preparing dataset...")
        train_clips, val_clips, test_clips = self.split_dataset()
        splits = {'train': train_clips, 'val': val_clips, 'test': test_clips}

        self._process_splits(splits, dataset_paths)
        self._create_dataset_config(dataset_paths)

    def _setup_dataset_directories(self, output_dir: str) -> Dict[str, Path]:
        """Create and return dataset directory structure."""
        dataset_dir = Path(output_dir) / 'dataset'
        images_dir = dataset_dir / 'images'
        labels_dir = dataset_dir / 'labels'

        for split in ['train', 'val', 'test']:
            (images_dir / split).mkdir(parents=True, exist_ok=True)
            (labels_dir / split).mkdir(parents=True, exist_ok=True)

        return {
            'dataset': dataset_dir,
            'images': images_dir,
            'labels': labels_dir
        }

    def _process_splits(
            self,
            splits: Dict[str, List[HockeyClip]],
            paths: Dict[str, Path]
    ) -> None:
        """Process each split of the dataset."""
        for split, clips in splits.items():
            self.logger.info(f"Processing {split} split ({len(clips)} clips)...")
            for clip in tqdm(clips, desc=f"Processing {split} data"):
                self._process_clip(clip, split, paths)

    def _process_clip(
            self,
            clip: HockeyClip,
            split: str,
            paths: Dict[str, Path]
    ) -> None:
        """Process a single clip for YOLO training."""
        video_path = self.clips_dir / f"{clip.video_id}.mp4"
        if not video_path.exists():
            self.logger.warning(f"Video file not found: {video_path}")
            return

        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_boxes = self._get_frame_boxes(clip, frame_idx)
            if frame_boxes:
                self._save_frame_and_labels(
                    frame, frame_boxes, clip, frame_idx, split, paths
                )

            frame_idx += 1

        cap.release()

    def _get_frame_boxes(self, clip: HockeyClip, frame_idx: int) -> List:
        """Get bounding boxes for a specific frame."""
        frame_boxes = []
        for track in clip.tracks.values():
            frame_boxes.extend([
                box for box in track
                if box.frame_idx == frame_idx and box.label in ['player', 'keeper']
            ])
        return frame_boxes

    def _save_frame_and_labels(
            self,
            frame: np.ndarray,
            boxes: List[BoundingBox],
            clip: HockeyClip,
            frame_idx: int,
            split: str,
            paths: Dict[str, Path]
    ) -> None:
        """Save frame image and corresponding YOLO labels."""
        img_path = paths['images'] / split / f"{clip.video_id}_{frame_idx:06d}.jpg"
        cv2.imwrite(str(img_path), frame)

        label_path = paths['labels'] / split / f"{clip.video_id}_{frame_idx:06d}.txt"
        with open(label_path, 'w') as f:
            for box in boxes:
                if box.category is None:
                    continue

                yolo_coords = self._convert_box_to_yolo(box, clip.width, clip.height)
                class_idx = box.category.to_class_idx()
                f.write(f"{class_idx} {yolo_coords[0]:.6f} {yolo_coords[1]:.6f} "
                        f"{yolo_coords[2]:.6f} {yolo_coords[3]:.6f}\n")

    def _create_dataset_config(self, paths: Dict[str, Path]) -> None:
        """Create YOLO dataset configuration file."""
        dataset_yaml = {
            'path': str(paths['dataset']),
            'train': str(paths['images'] / 'train'),
            'val': str(paths['images'] / 'val'),
            'test': str(paths['images'] / 'test'),
            'nc': 4,
            'names': [
                'white_keeper',
                'white_player',
                'black_keeper',
                'black_player'
            ]
        }

        yaml_path = paths['dataset'] / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(dataset_yaml, f)

        self._log_dataset_stats(paths)

    def _log_dataset_stats(self, paths: Dict[str, Path]) -> None:
        """Log statistics about the prepared dataset."""
        train_images = list((paths['images'] / 'train').glob('*.jpg'))
        val_images = list((paths['images'] / 'val').glob('*.jpg'))
        test_images = list((paths['images'] / 'test').glob('*.jpg'))

        self.logger.info(
            f"Dataset prepared with {len(train_images)} training images, "
            f"{len(val_images)} validation images, and {len(test_images)} test images"
        )

    def _convert_box_to_yolo(
            self,
            box,
            img_width: int,
            img_height: int
    ) -> Tuple[float, float, float, float]:
        """Convert bounding box coordinates to YOLO format."""
        x_center = ((box.xtl + box.xbr) / 2) / img_width
        y_center = ((box.ytl + box.ybr) / 2) / img_height
        width = (box.xbr - box.xtl) / img_width
        height = (box.ybr - box.ytl) / img_height
        return x_center, y_center, width, height

    def frame_iterator(self, clip: HockeyClip) -> Iterator[Tuple[int, np.ndarray]]:
        """Get an iterator over frames in a clip."""
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
    ) -> Iterator[Tuple[List[int], List]]:
        """Process frames in batches using a provided function."""
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
