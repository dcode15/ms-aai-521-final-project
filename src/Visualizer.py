from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Literal, NamedTuple

import cv2
import numpy as np

from BoundingBox import BoundingBox


@dataclass
class VisualizationContext:
    thickness: int
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.5
    dash_length: int = 10


@dataclass
class VideoWriterParams:
    fps: int
    codec: str
    width: int
    height: int


class BoxDrawingParams(NamedTuple):
    box: BoundingBox
    box_type: Literal['pred', 'gt']
    color: Tuple[int, int, int]
    context: VisualizationContext


class Visualizer:
    """
    Creates visualizations of object detection and tracking results.
    """

    def __init__(self, default_thickness: int = 2):
        """
        Initializes visualizer with default drawing parameters.

        Args:
            default_thickness: Default line thickness for drawing
        """
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
        """
        Creates video visualization of detection and tracking results.

        Args:
            frames: List of video frames as numpy arrays
            pred_detections: Predicted bounding boxes for each frame
            output_path: Path to save output video
            gt_detections: Optional ground truth boxes for comparison
            fps: Frame rate for output video
            codec: Video codec for compression
        """
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
        """
        Creates output directory if it doesn't exist.

        Args:
            output_path: Path where video will be saved
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

    def _create_video_params(
            self,
            first_frame: np.ndarray,
            fps: int,
            codec: str
    ) -> VideoWriterParams:
        """
        Creates video writer parameters based on first frame dimensions.

        Args:
            first_frame: First frame of video to determine dimensions
            fps: Desired frames per second
            codec: Video codec string

        Returns:
            VideoWriterParams configured for video output
        """
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
        """
        Writes visualization video to disk with detections overlaid.

        Args:
            frames: Video frames to process
            pred_detections: Predicted boxes for each frame
            gt_detections: Optional ground truth boxes for each frame
            output_path: Path to save video file
            video_params: Video writer configuration
            color_map: Mapping of track IDs to colors
        """
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
        """
        Draws all bounding boxes on a frame with labels.

        Args:
            frame: Frame to draw on
            pred_boxes: Predicted bounding boxes
            gt_boxes: Optional ground truth boxes
            color_map: Mapping of track IDs to colors
            thickness: Optional line thickness override

        Returns:
            Frame with boxes and labels drawn
        """
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
        """
        Gets or generates a consistent color for a tracked object.

        Args:
            box: Bounding box to get color for
            color_map: Existing color mappings

        Returns:
            RGB color tuple for the box
        """
        if box.track_id not in color_map:
            base_color = self._get_category_base_color(box)
            color_map[box.track_id] = self._generate_track_color(box.track_id, base_color)
        return color_map[box.track_id]

    def _generate_track_color(
            self,
            track_id: str,
            base_color: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        """
        Generates a unique color for a track based on its ID.

        Args:
            track_id: Identifier for the track
            base_color: Base color to modify

        Returns:
            RGB color tuple unique to the track
        """
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
        """
        Draws a single bounding box and its label on the frame.

        Args:
            frame: Frame to draw on
            params: Parameters for drawing the box
        """
        self._draw_box_outline(frame, params)
        self._draw_box_label(frame, params)

    def _draw_box_outline(
            self,
            frame: np.ndarray,
            params: BoxDrawingParams
    ) -> None:
        """
        Draws box outline, either solid or dashed based on type.

        Args:
            frame: Frame to draw on
            params: Parameters for drawing the box
        """
        if params.box_type == 'gt':
            self._draw_dashed_box(frame, params)
        else:
            self._draw_solid_box(frame, params)

    def _draw_solid_box(
            self,
            frame: np.ndarray,
            params: BoxDrawingParams
    ) -> None:
        """
        Draws solid rectangle for predicted boxes.

        Args:
            frame: Frame to draw on
            params: Parameters for drawing the box
        """
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
        """
        Draws dashed rectangle for ground truth boxes.

        Args:
            frame: Frame to draw on
            params: Parameters for drawing the box
        """
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
        """
        Draws label text with background for a bounding box.

        Args:
            frame: Frame to draw on
            params: Parameters for drawing the label
        """
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
        """
        Creates text label for a bounding box.

        Args:
            box: Bounding box to create label for
            box_type: Whether box is prediction or ground truth

        Returns:
            Formatted label string
        """
        label_type = "Actual" if box_type == 'gt' else "Pred"
        position_label = "Skater" if box.player_type == 'player' else "Goalie"
        team_label = "Home" if box.team == 'black' else "Away"

        return f"{label_type} | {team_label} {position_label} - Track {box.track_id}"

    def _initialize_color_map(
            self,
            pred_detections: List[List[BoundingBox]],
            gt_detections: Optional[List[List[BoundingBox]]]
    ) -> Dict[str, Tuple[int, int, int]]:
        """
        Initializes consistent colors for all tracked objects.

        Args:
            pred_detections: All predicted detections
            gt_detections: Optional ground truth detections

        Returns:
            Dictionary mapping track IDs to RGB colors
        """
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
        """
        Gets base color based on player category.

        Args:
            box: Bounding box containing player category

        Returns:
            Base RGB color tuple for the category
        """
        if box.team == 'white':
            return (200, 200, 200) if box.player_type == 'player' else (255, 255, 255)
        elif box.team == 'black':
            return (50, 50, 50) if box.player_type == 'player' else (0, 0, 0)
        return 128, 128, 128
