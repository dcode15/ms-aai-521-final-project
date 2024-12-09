from pathlib import Path
from typing import Dict, List, Any, Union

import numpy as np
import torch
import yaml
from ultralytics import YOLO

from BoundingBox import BoundingBox


class ObjectDetector:
    def __init__(
            self,
            model_name: str,
            tracking_params: dict
    ):
        self.tracking_params = tracking_params
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_name)
        self.class_names = [
            'white_keeper',
            'white_player',
            'black_keeper',
            'black_player'
        ]

    def detect_video(
            self,
            frames: List[np.ndarray],
            batch_size: int,
    ) -> List[List[BoundingBox]]:
        all_detections = []

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_results = self.model.track(
                source=batch_frames,
                verbose=False,
                tracker='D:\\Repos\\ms-aai-521-final-project\\out\\tracker-config.yaml'
            )

            for frame_offset, results in enumerate(batch_results):
                frame_idx = i + frame_offset
                frame_boxes = []

                if results.boxes is not None:
                    for det in results.boxes:
                        class_idx = int(det.cls)
                        x1, y1, x2, y2 = map(float, det.xyxy[0])
                        track_id = int(det.id) if det.id is not None else -1

                        bbox = BoundingBox.create_from_detection(
                            frame_idx=frame_idx,
                            track_id=str(track_id),
                            xtl=x1,
                            ytl=y1,
                            xbr=x2,
                            ybr=y2,
                            class_idx=class_idx,
                            occluded=False
                        )
                        frame_boxes.append(bbox)

                all_detections.append(frame_boxes)

        return all_detections

    @staticmethod
    def write_tracking_params(params: Dict[str, Any], output_dir: Union[str, Path]) -> None:
        output_dir = Path(output_dir)
        tracking_config = {
            "tracker_type": "bytetrack",
            "track_high_thresh": params["track_high_thresh"],
            "track_low_thresh": params["track_low_thresh"],
            "new_track_thresh": params["new_track_thresh"],
            "track_buffer": params["track_buffer"],
            "match_thresh": params["match_thresh"],
            "fuse_score": True
        }

        yaml_content = yaml.safe_dump(tracking_config, sort_keys=False)

        tracking_yaml_path = output_dir / 'tracker-config.yaml'
        with open(tracking_yaml_path, 'w') as f:
            f.write(yaml_content)
