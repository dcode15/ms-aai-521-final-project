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
    """
    Handles data preprocessing for hockey player detection dataset.
    """

    def __init__(self, clips_dir: str, cvat_dir: str):
        """
        Initializes preprocessor with video and annotation directories.

        Args:
            clips_dir: Directory containing video clips
            cvat_dir: Directory containing CVAT XML annotations
        """
        self.logger = logging.getLogger(__name__)
        self.clips_dir = Path(clips_dir)
        self.cvat_dir = Path(cvat_dir)

    def get_clips(self) -> Iterator[HockeyClip]:
        """
        Iterates over all valid annotated clips in the dataset.

        Returns:
            Iterator yielding HockeyClip objects for valid clips
        """
        annotation_parser = AnnotationParser()
        for xml_path in self._find_annotation_files():
            annotation = annotation_parser.parse_cvat_xml(xml_path)
            if self._validate_video_exists(annotation):
                yield annotation

    def _find_annotation_files(self) -> List[str]:
        """
        Finds all CVAT annotation XML files.

        Returns:
            List of paths to XML annotation files
        """
        return glob(f"{self.cvat_dir}/**/*.xml", recursive=True)

    def _validate_video_exists(self, clip: HockeyClip) -> bool:
        """
        Checks if corresponding video file exists for a clip.

        Args:
            clip: HockeyClip object to validate

        Returns:
            True if video file exists, False otherwise
        """
        video_path = self.clips_dir / f"{clip.video_id}.mp4"
        if not video_path.exists():
            self.logger.warning(f"Video file not found: {video_path}")
            return False
        return True

    def split_dataset(self, seed: int = 1) -> Tuple[List[HockeyClip], List[HockeyClip], List[HockeyClip]]:
        """
        Splits dataset into train, validation and test sets.

        Args:
            seed: Random seed for reproducible splitting

        Returns:
            Tuple of (train_clips, validation_clips, test_clips) lists
        """
        all_clips = list(self.get_clips())
        train_clips, val_clips, test_clips = self._perform_dataset_split(all_clips, seed)
        self._log_split_info(train_clips, val_clips, test_clips)
        return train_clips, val_clips, test_clips

    def _perform_dataset_split(
            self,
            clips: List[HockeyClip],
            seed: int
    ) -> Tuple[List[HockeyClip], List[HockeyClip], List[HockeyClip]]:
        """
        Performs dataset splitting using sklearn's train_test_split.

        Args:
            clips: List of all clips to split
            seed: Random seed for reproducible splitting

        Returns:
            Tuple of (train_clips, validation_clips, test_clips) lists
        """
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
        """
        Logs information about the dataset split sizes.

        Args:
            train_clips: List of training clips
            val_clips: List of validation clips
            test_clips: List of test clips
        """
        self.logger.info(
            f"Dataset split: {len(train_clips)} train, "
            f"{len(val_clips)} validation, {len(test_clips)} test clips"
        )

    def prepare_dataset(
            self,
            output_dir: str
    ) -> None:
        """
        Prepares the full dataset for YOLO training.
        Creates directory structure, processes all splits, and generates config.

        Args:
            output_dir: Base directory for processed dataset
        """
        dataset_paths = self._setup_dataset_directories(output_dir)

        self.logger.info("Preparing dataset...")
        train_clips, val_clips, test_clips = self.split_dataset()
        splits = {'train': train_clips, 'val': val_clips, 'test': test_clips}

        self._process_splits(splits, dataset_paths)
        self._create_dataset_config(dataset_paths)

    def _setup_dataset_directories(self, output_dir: str) -> Dict[str, Path]:
        """
        Creates directory structure for YOLO dataset.
        Sets up separate directories for images and labels in each split.

        Args:
            output_dir: Base directory for dataset

        Returns:
            Dictionary mapping directory types to their paths
        """
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
        """
        Processes all dataset splits (train/val/test).
        Extracts frames and generates YOLO labels for each clip.

        Args:
            splits: Dictionary mapping split names to clip lists
            paths: Dictionary of dataset directory paths
        """
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
        """
        Processes a single clip into YOLO training format.
        Extracts frames and generates corresponding label files.

        Args:
            clip: HockeyClip to process
            split: Dataset split name (train/val/test)
            paths: Dictionary of dataset directory paths
        """
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
        """
        Gets all bounding boxes for a specific frame.
        Filters for player and keeper annotations only.

        Args:
            clip: HockeyClip containing annotations
            frame_idx: Index of frame to get boxes for

        Returns:
            List of BoundingBox objects for the frame
        """
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
        """
        Saves a video frame and its corresponding YOLO label file.
        Converts bounding boxes to YOLO format.

        Args:
            frame: Video frame as numpy array
            boxes: List of BoundingBox objects for the frame
            clip: Parent HockeyClip object
            frame_idx: Index of the frame
            split: Dataset split name
            paths: Dictionary of dataset directory paths
        """
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
        """
        Creates YOLO dataset configuration file.
        Specifies paths, classes, and class weights for training.

        Args:
            paths: Dictionary of dataset directory paths
        """
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
            ],
            'weights': [
                33.33,
                2.55,
                25.73,
                2.43
            ]
        }

        yaml_path = paths['dataset'] / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(dataset_yaml, f)

        self._log_dataset_stats(paths)

    def _log_dataset_stats(self, paths: Dict[str, Path]) -> None:
        """
        Logs statistics about the prepared dataset.
        Counts number of images in each split.

        Args:
            paths: Dictionary of dataset directory paths
        """
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
        """
        Converts bounding box coordinates to YOLO format.
        Converts from absolute coordinates to normalized center/width/height format.

        Args:
            box: BoundingBox object to convert
            img_width: Width of the image
            img_height: Height of the image

        Returns:
            Tuple of (center_x, center_y, width, height) in normalized coordinates
        """
        x_center = ((box.xtl + box.xbr) / 2) / img_width
        y_center = ((box.ytl + box.ybr) / 2) / img_height
        width = (box.xbr - box.xtl) / img_width
        height = (box.ybr - box.ytl) / img_height
        return x_center, y_center, width, height

    def frame_iterator(self, clip: HockeyClip) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Iterates over frames in a video clip.
        Converts frames from BGR to RGB.

        Args:
            clip: HockeyClip to read frames from

        Returns:
            Iterator yielding tuples of (frame_index, frame_data)
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
    ) -> Iterator[Tuple[List[int], List]]:
        """
        Processes frames from a clip in batches.

        Args:
            clip: HockeyClip to process frames from
            frame_processor: Function to apply to each batch of frames
            batch_size: Number of frames to process at once

        Returns:
            Iterator yielding tuples of (frame_indices, processed_frames)
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
