from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Literal, NamedTuple

import cv2
import numpy as np

from BoundingBox import BoundingBox


@dataclass
class VisualizationContext:
    """Settings and parameters for visualization."""
    thickness: int
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.5
    dash_length: int = 10


@dataclass
class VideoWriterParams:
    """Parameters for video writer configuration."""
    fps: int
    codec: str
    width: int
    height: int


class BoxDrawingParams(NamedTuple):
    """Parameters for drawing a single box."""
    box: BoundingBox
    box_type: Literal['pred', 'gt']
    color: Tuple[int, int, int]
    context: VisualizationContext


class Visualizer:
    """Class for creating visualizations of detection and tracking results."""

    def __init__(self, default_thickness: int = 2):
        """Initialize the Visualizer."""
        self.default_context = VisualizationContext(thickness=default_thickness)

    def create_detection_video(
            self,
            frames: List[np.ndarray],
            pred_detections: List[List[BoundingBox]],
            output_path: str,
            gt_detections: Optional[List[List[BoundingBox]]] = None,
            fps: int = 30,
            codec: str = 'mp4v'
    ) -> None:
        """Create a video with visualized detections."""
        output_path = Path(output_path)
        self._ensure_output_directory(output_path)

        video_params = self._create_video_params(frames[0], fps, codec)
        color_map = self._initialize_color_map(pred_detections, gt_detections)

        self._write_detection_video(
            frames=frames,
            pred_detections=pred_detections,
            gt_detections=gt_detections,
            output_path=output_path,
            video_params=video_params,
            color_map=color_map
        )

    def _ensure_output_directory(self, output_path: Path) -> None:
        """Ensure the output directory exists."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

    def _create_video_params(
            self,
            first_frame: np.ndarray,
            fps: int,
            codec: str
    ) -> VideoWriterParams:
        """Create video writer parameters from first frame."""
        height, width = first_frame.shape[:2]
        return VideoWriterParams(fps=fps, codec=codec, width=width, height=height)

    def _write_detection_video(
            self,
            frames: List[np.ndarray],
            pred_detections: List[List[BoundingBox]],
            gt_detections: Optional[List[List[BoundingBox]]],
            output_path: Path,
            video_params: VideoWriterParams,
            color_map: Dict[str, Tuple[int, int, int]]
    ) -> None:
        """Write the detection video to disk."""
        fourcc = cv2.VideoWriter_fourcc(*video_params.codec)
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            video_params.fps,
            (video_params.width, video_params.height)
        )

        try:
            for i, (frame, frame_pred_dets) in enumerate(zip(frames, pred_detections)):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_gt_dets = gt_detections[i] if gt_detections else None

                frame_viz = self.draw_boxes(
                    frame_bgr,
                    frame_pred_dets,
                    frame_gt_dets,
                    color_map
                )
                out.write(frame_viz)
        finally:
            out.release()

    def draw_boxes(
            self,
            frame: np.ndarray,
            pred_boxes: List[BoundingBox],
            gt_boxes: Optional[List[BoundingBox]] = None,
            color_map: Optional[Dict[str, Tuple[int, int, int]]] = None,
            thickness: Optional[int] = None
    ) -> np.ndarray:
        """Draw bounding boxes on a frame."""
        frame = frame.copy()
        color_map = color_map or {}
        context = VisualizationContext(
            thickness=thickness or self.default_context.thickness
        )

        if gt_boxes:
            for box in gt_boxes:
                color = self._get_box_color(box, color_map)
                params = BoxDrawingParams(box, 'gt', color, context)
                self._draw_single_box(frame, params)

        for box in pred_boxes:
            color = self._get_box_color(box, color_map)
            params = BoxDrawingParams(box, 'pred', color, context)
            self._draw_single_box(frame, params)

        return frame

    def _get_box_color(
            self,
            box: BoundingBox,
            color_map: Dict[str, Tuple[int, int, int]]
    ) -> Tuple[int, int, int]:
        """Get or generate color for a box."""
        if box.track_id not in color_map:
            base_color = self._get_category_base_color(box)
            color_map[box.track_id] = self._generate_track_color(box.track_id, base_color)
        return color_map[box.track_id]

    def _generate_track_color(
            self,
            track_id: str,
            base_color: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        """Generate a consistent color for a track ID based on category."""
        track_num = hash(track_id) % 100000
        r = (base_color[0] + track_num * 123) % 255
        g = (base_color[1] + track_num * 456) % 255
        b = (base_color[2] + track_num * 789) % 255
        return (r, g, b)

    def _draw_single_box(
            self,
            frame: np.ndarray,
            params: BoxDrawingParams
    ) -> None:
        """Draw a single bounding box with its label."""
        self._draw_box_outline(frame, params)
        self._draw_box_label(frame, params)

    def _draw_box_outline(
            self,
            frame: np.ndarray,
            params: BoxDrawingParams
    ) -> None:
        """Draw the box outline (solid or dashed)."""
        if params.box_type == 'gt':
            self._draw_dashed_box(frame, params)
        else:
            self._draw_solid_box(frame, params)

    def _draw_solid_box(
            self,
            frame: np.ndarray,
            params: BoxDrawingParams
    ) -> None:
        """Draw a solid rectangle for predicted boxes."""
        p1 = (int(params.box.xtl), int(params.box.ytl))
        p2 = (int(params.box.xbr), int(params.box.ybr))
        cv2.rectangle(
            frame,
            p1,
            p2,
            params.color,
            params.context.thickness
        )

    def _draw_dashed_box(
            self,
            frame: np.ndarray,
            params: BoxDrawingParams
    ) -> None:
        """Draw a dashed rectangle for ground truth boxes."""
        x1, y1 = int(params.box.xtl), int(params.box.ytl)
        x2, y2 = int(params.box.xbr), int(params.box.ybr)
        dash_length = params.context.dash_length

        for x in range(x1, x2, dash_length * 2):
            x_end = min(x + dash_length, x2)
            cv2.line(
                frame,
                (x, y1),
                (x_end, y1),
                params.color,
                params.context.thickness
            )
            cv2.line(
                frame,
                (x, y2),
                (x_end, y2),
                params.color,
                params.context.thickness
            )

        for y in range(y1, y2, dash_length * 2):
            y_end = min(y + dash_length, y2)
            cv2.line(
                frame,
                (x1, y),
                (x1, y_end),
                params.color,
                params.context.thickness
            )
            cv2.line(
                frame,
                (x2, y),
                (x2, y_end),
                params.color,
                params.context.thickness
            )

    def _draw_box_label(
            self,
            frame: np.ndarray,
            params: BoxDrawingParams
    ) -> None:
        """Draw label text for a bounding box."""
        label = self._create_box_label(params.box, params.box_type)
        text_size = cv2.getTextSize(
            label,
            params.context.font,
            params.context.font_scale,
            params.context.thickness
        )
        (text_width, text_height), baseline = text_size

        text_x = int(params.box.xtl)
        text_y = int(params.box.ytl) - baseline

        cv2.rectangle(
            frame,
            (text_x, text_y - text_height - baseline),
            (text_x + text_width, text_y + baseline),
            params.color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            params.context.font,
            params.context.font_scale,
            (0, 0, 0),
            params.context.thickness
        )

    def _create_box_label(
            self,
            box: BoundingBox,
            box_type: Literal['pred', 'gt']
    ) -> str:
        """Create the label text for a box."""
        label_type = "Actual" if box_type == 'gt' else "Pred"
        position_label = "Skater" if box.player_type == 'player' else "Goalie"
        team_label = "Home" if box.team == 'black' else "Away"

        return f"{label_type} | {team_label} {position_label} - Track {box.track_id}"

    def _initialize_color_map(
            self,
            pred_detections: List[List[BoundingBox]],
            gt_detections: Optional[List[List[BoundingBox]]]
    ) -> Dict[str, Tuple[int, int, int]]:
        """Initialize color mapping for track IDs."""
        color_map: Dict[str, Tuple[int, int, int]] = {}

        for frame_dets in pred_detections:
            for det in frame_dets:
                if det.track_id and det.track_id not in color_map:
                    base_color = self._get_category_base_color(det)
                    color_map[det.track_id] = self._generate_track_color(det.track_id, base_color)

        if gt_detections:
            for frame_dets in gt_detections:
                for det in frame_dets:
                    if det.track_id and det.track_id not in color_map:
                        base_color = self._get_category_base_color(det)
                        color_map[det.track_id] = self._generate_track_color(det.track_id, base_color)

        return color_map

    def _get_category_base_color(
            self,
            box: BoundingBox
    ) -> Tuple[int, int, int]:
        """Get base color for player category."""
        if box.team == 'white':
            return (200, 200, 200) if box.player_type == 'player' else (255, 255, 255)
        elif box.team == 'black':
            return (50, 50, 50) if box.player_type == 'player' else (0, 0, 0)
        return 128, 128, 128
