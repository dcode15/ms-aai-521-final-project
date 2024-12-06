import logging
from pathlib import Path
from typing import Optional

import torch
from ultralytics import YOLO


class ModelTrainer:
    """Class for fine-tuning YOLO models on hockey data."""

    def __init__(
            self,
            model_path: str,
            output_dir: str,
            device: Optional[str] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir).resolve()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = YOLO(model_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir = (self.output_dir / 'dataset').resolve()

    def train(
            self,
            epochs: int = 10,
            batch_size: int = 16,
            learning_rate: float = 0.001,
            warmup_epochs: int = 3,
            weight_decay: float = 0.0005,
            dropout: float = 0.0,
            box_loss_weight: float = 7.5,
            cls_loss_weight: float = 0.5,
            patience: int = 2,
            image_size: int = 640,
    ) -> None:
        """
        Train the YOLO model on the dataset with specified hyperparameters.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            warmup_epochs: Number of warmup epochs
            weight_decay: Weight decay for regularization
            dropout: Dropout rate
            box_loss_weight: Weight for box loss component
            cls_loss_weight: Weight for classification loss component
            patience: Number of epochs to wait for improvement before early stopping
            image_size: Input image size
        """
        self.model.train(
            data=str(self.dataset_dir / 'dataset.yaml'),
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            device=self.device,
            project=str(self.output_dir),
            name='finetune',
            lr0=learning_rate,
            warmup_epochs=warmup_epochs,
            weight_decay=weight_decay,
            dropout=dropout,
            box=box_loss_weight,
            cls=cls_loss_weight,
            patience=patience,
            save=True,
            save_period=10,
            val=True,
            resume=False,
            exist_ok=True,
            plots=True,
            cos_lr=True
        )

    def export_model(self, format: str = 'torchscript') -> None:
        """Export the trained model to the specified format."""
        self.model.export(format=format)
