from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import optuna
from optuna.trial import Trial
import numpy as np

from ModelTrainer import ModelTrainer
from ObjectDetector import ObjectDetector
from Preprocessor import Preprocessor
from ModelEvaluator import ModelEvaluator
from VideoAnnotation import HockeyClip


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""
    n_trials: int = 100
    study_name: str = "hockey_tracking_optimization"


class HyperparameterTuner:
    """Class for tuning hyperparameters using Optuna."""

    def __init__(
            self,
            base_model_path: str,
            output_dir: str,
            train_clips: list[HockeyClip],
            val_clips: list[HockeyClip],
            preprocessor: Preprocessor,
            config: TuningConfig,
            device: Optional[str] = None
    ):
        """Initialize the tuner with dataset and configuration."""
        self.base_model_path = base_model_path
        self.output_dir = Path(output_dir)
        self.train_clips = train_clips
        self.val_clips = val_clips
        self.preprocessor = preprocessor
        self.config = config
        self.device = device

        self.evaluator = ModelEvaluator()

    def _suggest_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial."""
        params = {
            'learning_rate': trial.suggest_categorical('learning_rate', [0.0098]),
            'warmup_epochs': trial.suggest_categorical('warmup_epochs', [2]),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3),
            'dropout': trial.suggest_float('dropout', 0.0, 0.75),

            'box_loss_weight': trial.suggest_categorical('box_loss_weight', [8.91]),
            'cls_loss_weight': trial.suggest_categorical('cls_loss_weight', [0.63]),

            'conf_threshold': trial.suggest_float('conf_threshold', 0.2, 0.8),
            'iou_threshold': trial.suggest_float('iou_threshold', 0.2, 0.8),
            'track_buffer': trial.suggest_int('track_buffer', 5, 50),
            'match_thresh': trial.suggest_float('match_thresh', 0.2, 0.8)
        }
        return params

    def _evaluate_model(
            self,
            model: ObjectDetector,
            clips: list[HockeyClip],
            batch_size: int = 16
    ) -> float:
        """Evaluate model performance on a set of clips."""
        all_predictions = []

        for clip in clips:
            frame_predictions = []
            for frame_indices, batch_frames in self.preprocessor.process_clip_frames(clip, lambda x: x, batch_size):
                batch_detections = model.detect_video(batch_frames, batch_size)
                frame_predictions.extend(batch_detections)
            all_predictions.append(frame_predictions)

        metrics = []
        for clip, predictions in zip(clips, all_predictions):
            clip_metrics = self.evaluator.evaluate_clip(clip, predictions)
            f1_score = 2 * (clip_metrics.precision * clip_metrics.recall) / (
                    clip_metrics.precision + clip_metrics.recall + 1e-10)

            metric = (0.4 * clip_metrics.mota +
                      0.3 * clip_metrics.motp +
                      0.3 * f1_score)
            metrics.append(metric)

        return np.mean(metrics)

    def _objective(self, trial: Trial) -> float:
        """Objective function for Optuna optimization."""
        params = self._suggest_hyperparameters(trial)

        trainer = ModelTrainer(
            model_path=self.base_model_path,
            output_dir=str(self.output_dir),
            device=self.device
        )

        try:
            trainer.train(
                epochs=15,
                batch_size=8,
                learning_rate=params['learning_rate'],
                warmup_epochs=params['warmup_epochs'],
                weight_decay=params['weight_decay'],
                dropout=params['dropout'],
                box_loss_weight=params['box_loss_weight'],
                cls_loss_weight=params['cls_loss_weight']
            )

            detector = ObjectDetector(
                model_name=str(Path(self.output_dir) / 'finetune' / 'weights' / 'best.pt'),
                conf_threshold=params['conf_threshold'],
                device=self.device,
                track_buffer=params['track_buffer'],
                match_thresh=params['match_thresh']
            )

            metric = self._evaluate_model(detector, self.val_clips)

            return metric

        except Exception as e:
            print(f"Trial {trial.number} failed with error: {str(e)}")
            return float('-inf')

    def tune(self) -> Dict[str, Any]:
        """Run hyperparameter tuning and return best parameters."""
        study = optuna.create_study(
            study_name=self.config.study_name,
            direction="maximize",
            load_if_exists=True
        )

        study.optimize(
            self._objective,
            n_trials=self.config.n_trials,
        )

        return study.best_params
