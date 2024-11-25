import logging
import shutil
from pathlib import Path
from typing import Optional

import cv2
import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO

from src.AnnotationParser import AnnotationParser


class YOLOTrainer:
    """Class for fine-tuning YOLO models on hockey data."""

    def __init__(
            self,
            model_path: str,
            clips_dir: str,
            cvat_dir: str,
            output_dir: str,
            device: Optional[str] = None,
            force_data_preparation: bool = False
    ):
        self.logger = logging.getLogger(__name__)
        self.clips_dir = Path(clips_dir).resolve()
        self.cvat_dir = Path(cvat_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.force_data_preparation = force_data_preparation

        self.model = YOLO(model_path)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_dir = (self.output_dir / 'dataset').resolve()
        self.images_dir = (self.dataset_dir / 'images').resolve()
        self.labels_dir = (self.dataset_dir / 'labels').resolve()

    def _convert_box_to_yolo(self, box, img_width: int, img_height: int) -> tuple[float, float, float, float]:
        """Convert box coordinates to YOLO format (normalized xcenter, ycenter, width, height)."""
        x_center = ((box.xtl + box.xbr) / 2) / img_width
        y_center = ((box.ytl + box.ybr) / 2) / img_height
        width = (box.xbr - box.xtl) / img_width
        height = (box.ybr - box.ytl) / img_height
        return x_center, y_center, width, height

    def is_dataset_prepared(self) -> bool:
        """Check if dataset is already prepared."""
        if not self.dataset_dir.exists():
            return False

        yaml_path = self.dataset_dir / 'dataset.yaml'
        if not yaml_path.exists():
            return False

        try:
            with open(yaml_path) as f:
                dataset_config = yaml.safe_load(f)

            train_path = Path(dataset_config['train'])
            val_path = Path(dataset_config['val'])

            train_images = list(train_path.glob('*.jpg'))
            val_images = list(val_path.glob('*.jpg'))

            if not train_images or not val_images:
                return False

            for img_path in train_images + val_images:
                label_path = self.labels_dir / img_path.parent.name / f"{img_path.stem}.txt"
                if not label_path.exists():
                    return False

            return True

        except (yaml.YAMLError, KeyError):
            return False

    def prepare_data(self) -> None:
        """Prepare training data in YOLO format."""
        if not self.force_data_preparation and self.is_dataset_prepared():
            self.logger.info("Dataset already prepared, skipping preparation")
            return

        self.logger.info("Preparing dataset...")

        if self.dataset_dir.exists():
            shutil.rmtree(self.dataset_dir)

        for split in ['train', 'val']:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)

        annotation_files = list(Path(self.cvat_dir).glob('**/*.xml'))

        if not annotation_files:
            raise ValueError(f"No annotation files found in {self.cvat_dir}")

        train_size = int(0.8 * len(annotation_files))
        splits = {
            'train': annotation_files[:train_size],
            'val': annotation_files[train_size:]
        }

        annotation_parser = AnnotationParser()

        for split, files in splits.items():
            self.logger.info(f"Processing {split} split ({len(files)} files)...")

            for xml_path in tqdm(files, desc=f"Processing {split} data"):
                annotation = annotation_parser.parse_cvat_xml(xml_path)
                video_path = self.clips_dir / f"{annotation.video_id}.mp4"

                if not video_path.exists():
                    self.logger.warning(f"Video file not found: {video_path}")
                    continue

                cap = cv2.VideoCapture(str(video_path))
                frame_idx = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_boxes = []
                    for track in annotation.tracks.values():
                        frame_boxes.extend([
                            box for box in track
                            if box.frame_idx == frame_idx and box.label in ['player', 'keeper']
                        ])

                    if frame_boxes:
                        img_path = self.images_dir / split / f"{annotation.video_id}_{frame_idx:06d}.jpg"
                        cv2.imwrite(str(img_path), frame)

                        label_path = self.labels_dir / split / f"{annotation.video_id}_{frame_idx:06d}.txt"
                        with open(label_path, 'w') as f:
                            for box in frame_boxes:
                                x_center, y_center, width, height = self._convert_box_to_yolo(
                                    box,
                                    annotation.width,
                                    annotation.height
                                )

                                class_idx = 0

                                f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                    frame_idx += 1

                cap.release()

        train_images = list((self.images_dir / 'train').glob('*.jpg'))
        val_images = list((self.images_dir / 'val').glob('*.jpg'))

        self.logger.info(
            f"Dataset prepared with {len(train_images)} training images and {len(val_images)} validation images")

        if not train_images or not val_images:
            raise ValueError("No images were generated for training/validation")

        dataset_yaml = {
            'path': str(self.dataset_dir),
            'train': str(self.images_dir / 'train'),
            'val': str(self.images_dir / 'val'),
            'nc': 1,
            'names': ['player']
        }

        yaml_path = self.dataset_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(dataset_yaml, f)

        self.logger.info(f"Created dataset configuration at {yaml_path}")

    def train(
            self,
            epochs: int = 100,
            batch_size: int = 16,
            learning_rate: float = 0.001
    ) -> None:
        """Fine-tune the YOLO model."""
        self.prepare_data()

        self.model.train(
            data=str(self.dataset_dir / 'dataset.yaml'),
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            device=self.device,
            project=str(self.output_dir),
            name='finetune',
            lr0=learning_rate,
            lrf=learning_rate / 10,
            warmup_epochs=3,
            save=True,
            save_period=10,
            val=True,
            resume=False
        )

    def export_model(self, format: str = 'torchscript') -> None:
        """Export the fine-tuned model."""
        self.model.export(format=format)

    def cleanup(self) -> None:
        """Clean up temporary dataset files."""
        if self.dataset_dir.exists():
            shutil.rmtree(self.dataset_dir)
