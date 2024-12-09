import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import motmetrics as mm
import numpy as np
import pandas as pd
from tqdm import tqdm

from BoundingBox import BoundingBox
from VideoAnnotation import HockeyClip
from Preprocessor import Preprocessor
from ObjectDetector import ObjectDetector


@dataclass
class TrackingMetrics:
    mota: float
    motp: float
    num_switches: int
    num_fragmentations: int
    num_frames: int
    precision: float
    recall: float


class ModelEvaluator:

    def __init__(self, iou_threshold: float = 0.5):
        self.logger = logging.getLogger(__name__)
        self.iou_threshold = iou_threshold

    def _convert_to_mot_format(
            self,
            boxes: List[BoundingBox],
            frame_width: int,
            frame_height: int
    ) -> np.ndarray:
        mot_boxes = []
        for box in boxes:
            x1 = max(0, min(box.xtl, frame_width - 1))
            y1 = max(0, min(box.ytl, frame_height - 1))
            x2 = max(0, min(box.xbr, frame_width - 1))
            y2 = max(0, min(box.ybr, frame_height - 1))

            w = x2 - x1
            h = y2 - y1

            mot_boxes.append([x1, y1, w, h])

        return np.array(mot_boxes)

    def _get_frame_detections(
            self,
            clip: HockeyClip,
            frame_idx: int,
            include_keepers: bool = True
    ) -> List[BoundingBox]:
        frame_boxes = []
        valid_labels = {'player'} | ({'keeper'} if include_keepers else set())

        for track in clip.tracks.values():
            frame_boxes.extend([
                box for box in track
                if box.frame_idx == frame_idx and box.label in valid_labels
            ])
        return frame_boxes

    def evaluate_clip(
            self,
            clip: HockeyClip,
            pred_detections: List[List[BoundingBox]],
            include_keepers: bool = True
    ) -> TrackingMetrics:
        acc = mm.MOTAccumulator(auto_id=True)

        num_frames = len(pred_detections)
        frame_indices = range(num_frames)

        for frame_idx in tqdm(frame_indices, desc="Computing tracking metrics"):
            gt_boxes = self._get_frame_detections(
                clip,
                frame_idx,
                include_keepers
            )
            gt_mot = self._convert_to_mot_format(
                gt_boxes,
                clip.width,
                clip.height
            )

            pred_boxes = pred_detections[frame_idx]
            pred_mot = self._convert_to_mot_format(
                pred_boxes,
                clip.width,
                clip.height
            )

            gt_ids = [box.track_id for box in gt_boxes]
            pred_ids = [box.track_id for box in pred_boxes]

            if len(gt_mot) == 0 and len(pred_mot) == 0:
                continue

            distances = mm.distances.iou_matrix(
                gt_mot,
                pred_mot,
                max_iou=1 - self.iou_threshold
            )

            acc.update(
                gt_ids,
                pred_ids,
                distances
            )

        mh = mm.metrics.create()
        summary = mh.compute(
            acc,
            metrics=[
                'mota', 'motp', 'num_switches',
                'num_fragmentations', 'precision', 'recall'
            ],
            name='acc'
        )

        return TrackingMetrics(
            mota=summary['mota']['acc'],
            motp=summary['motp']['acc'],
            num_switches=summary['num_switches']['acc'],
            num_fragmentations=summary['num_fragmentations']['acc'],
            num_frames=num_frames,
            precision=summary['precision']['acc'],
            recall=summary['recall']['acc']
        )

    def evaluate_dataset(
            self,
            clips: List[HockeyClip],
            all_pred_detections: List[List[List[BoundingBox]]],
            output_dir: Optional[Path] = None,
            include_keepers: bool = True
    ) -> Dict[str, TrackingMetrics]:
        results = {}

        for clip, pred_detections in zip(clips, all_pred_detections):
            self.logger.info(f"Evaluating clip {clip.video_id}")

            metrics = self.evaluate_clip(
                clip,
                pred_detections,
                include_keepers
            )
            results[clip.video_id] = metrics

            if output_dir:
                self._save_clip_results(
                    clip.video_id,
                    metrics,
                    output_dir
                )

        if output_dir:
            self._save_dataset_results(results, output_dir)

        return results

    def _save_clip_results(
            self,
            clip_id: str,
            metrics: TrackingMetrics,
            output_dir: Path
    ) -> None:
        results_dir = output_dir / 'tracking_metrics'
        results_dir.mkdir(parents=True, exist_ok=True)

        results = {
            'mota': metrics.mota,
            'motp': metrics.motp,
            'num_switches': metrics.num_switches,
            'num_fragmentations': metrics.num_fragmentations,
            'num_frames': metrics.num_frames,
            'precision': metrics.precision,
            'recall': metrics.recall
        }

        df = pd.DataFrame([results])
        df.to_csv(results_dir / f'{clip_id}_metrics.csv', index=False)

    def _save_dataset_results(
            self,
            results: Dict[str, TrackingMetrics],
            output_dir: Path
    ) -> None:
        results_dir = output_dir / 'tracking_metrics'
        results_dir.mkdir(parents=True, exist_ok=True)

        total_frames = sum(m.num_frames for m in results.values())

        weighted_metrics = {
            'mota': sum(m.mota * m.num_frames for m in results.values()) / total_frames,
            'motp': sum(m.motp * m.num_frames for m in results.values()) / total_frames,
            'num_switches': sum(m.num_switches for m in results.values()),
            'num_fragmentations': sum(m.num_fragmentations for m in results.values()),
            'total_frames': total_frames,
            'precision': sum(m.precision * m.num_frames for m in results.values()) / total_frames,
            'recall': sum(m.recall * m.num_frames for m in results.values()) / total_frames
        }

        df = pd.DataFrame([weighted_metrics])
        df.to_csv(results_dir / 'dataset_metrics.csv', index=False)

    def plot_training_metrics(self, results_path: Path, output_dir: Optional[Path] = None) -> None:
        """
        Create a 2x2 plot grid showing key training metrics.

        Args:
            results_path: Path to the results.csv file
            output_dir: Optional directory to save the plot. If None, display instead.
        """
        df = pd.read_csv(results_path)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics', fontsize=16)

        ax1.plot(df['epoch'], df['train/box_loss'], label='Train')
        ax1.plot(df['epoch'], df['val/box_loss'], label='Validation')
        ax1.set_title('Box Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(df['epoch'], df['train/cls_loss'], label='Train')
        ax2.plot(df['epoch'], df['val/cls_loss'], label='Validation')
        ax2.set_title('Classification Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        ax3.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
        ax3.set_title('mAP50-95')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('mAP')
        ax3.legend()
        ax3.grid(True)

        ax4.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
        ax4.set_title('Precision')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Precision')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    @staticmethod
    def process_clips_for_evaluation(
            clips: List[HockeyClip],
            preprocessor: Preprocessor,
            detector: ObjectDetector,
            batch_size: int
    ) -> Dict[str, Tuple[List[np.ndarray], List, List]]:
        results = {}

        for clip in tqdm(clips, desc="Processing clips"):
            frames = []
            pred_detections = []
            gt_detections = []

            for frame_indices, batch_frames in preprocessor.process_clip_frames(clip, lambda x: x, batch_size):
                frames.extend(batch_frames)
                batch_detections = detector.detect_video(batch_frames, batch_size)
                pred_detections.extend(batch_detections)

                for frame_idx in frame_indices:
                    frame_gt = []
                    for track in clip.tracks.values():
                        frame_gt.extend([
                            box for box in track
                            if box.frame_idx == frame_idx
                        ])
                    gt_detections.append(frame_gt)

            results[clip.video_id] = (frames, pred_detections, gt_detections)

        return results
