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
        # track_range = trial.suggest_float('track_range', 0.01, 0.5)
        params = {
            'training': {
                'lr0': trial.suggest_float('learning_rate', 1e-5, 1e-2),
                'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
                'dropout': trial.suggest_float('dropout', 0.0, 0.75),
                'rect': trial.suggest_categorical('rect', [True, False])
            },
            'detection': {
                # 'conf': trial.suggest_float('conf', 0.1, 0.9),
                # 'iou': trial.suggest_float('iou', 0.1, 0.9),
                # 'track_buffer': trial.suggest_int('track_buffer', 1, 120),
                # 'match_thresh': trial.suggest_float('match_thresh', 0.1, 0.9),
                # 'track_low_thresh': trial.suggest_float('track_low_thresh', 0.01, 0.5),
                # 'new_track_thresh': trial.suggest_float('new_track_thresh', 0.01, 0.9),
            }
        }
        # params['detection']['track_high_thresh'] = params['detection']['track_low_thresh'] + track_range

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
            # harmonic_mean = 4 / (
            #             1 / clip_metrics.mota + 1 / clip_metrics.motp + 1 / clip_metrics.precision + 1 / clip_metrics.recall)
            harmonic_mean = 2 / (
                        1 / clip_metrics.precision + 1 / clip_metrics.recall)
            metrics.append(harmonic_mean)

        return np.mean(metrics)

    def _objective(self, trial: Trial) -> float:
        """Objective function for Optuna optimization."""
        params = self._suggest_hyperparameters(trial)

        trainer = ModelTrainer(
            model_path=self.base_model_path,
            output_dir=str(self.output_dir),
            device=self.device
        )

        params['training'].update({
            'epochs': 15,
            'batch': 8
        })
        try:
            trainer.train(
                training_params=params['training'],
            )

            detector = ObjectDetector(
                model_name=str(Path(self.output_dir) / 'finetune' / 'weights' / 'best.pt'),
                tracking_params=params['detection']
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
