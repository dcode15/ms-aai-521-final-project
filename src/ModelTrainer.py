import logging
from pathlib import Path
from typing import Optional

import torch
from ultralytics import YOLO


class ModelTrainer:

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
            training_params: dict
    ) -> None:
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
        """Export the trained model to the specified format."""
        self.model.export(format=export_format)
