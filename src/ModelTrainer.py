import logging
from pathlib import Path
from typing import Optional

import torch
from ultralytics import YOLO


class ModelTrainer:
    """Handles training and export of YOLO models for hockey player detection and tracking."""

    def __init__(
            self,
            model_path: str,
            output_dir: str,
            device: Optional[str] = None
    ):
        """
        Initializes the trainer with model and output settings.

        Args:
            model_path: Path to the base YOLO model
            output_dir: Directory for saving training outputs
            device: Optional device specification
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir).resolve()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = YOLO(model_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir = (self.output_dir / 'dataset').resolve()

    def train(
            self,
            training_params: dict
    ) -> None:
        """
        Fine-tunes the YOLO model on hockey player detection data.

        Args:
            training_params: Dictionary of training hyperparameters
        """
        self.model.train(
            data=str(self.dataset_dir / 'dataset.yaml'),
            seed=1,
            device=self.device,
            project=str(self.output_dir),
            name='finetune',
            save=True,
            val=True,
            exist_ok=True,
            plots=True,
            cache=True,
            rect=True,
            tracker='bytetrack.yaml',
            **training_params
        )

    def export_model(self, export_format: str) -> None:
        """
        Exports the trained model to a specified format for deployment.

        Args:
            export_format: Target format for model export
        """
        self.model.export(format=export_format)
