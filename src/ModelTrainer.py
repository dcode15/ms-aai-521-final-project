import logging
from pathlib import Path
from typing import Optional

import torch
from ultralytics import YOLO

from Preprocessor import Preprocessor


class ModelTrainer:
    """Class for fine-tuning YOLO models on hockey data."""

    def __init__(
            self,
            model_path: str,
            preprocessor: Preprocessor,
            output_dir: str,
            device: Optional[str] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.preprocessor = preprocessor
        self.output_dir = Path(output_dir).resolve()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = YOLO(model_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir = (self.output_dir / 'dataset').resolve()

    def train(
            self,
            epochs: int = 3,
            batch_size: int = 16,
            learning_rate: float = 0.001,
            force_prepare: bool = False
    ) -> None:
        """
        Train the YOLO model on the dataset.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            force_prepare: Whether to force dataset preparation
        """
        self.preprocessor.prepare_yolo_dataset(
            self.output_dir,
            force_prepare=force_prepare
        )

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
        """Export the trained model to the specified format."""
        self.model.export(format=format)
