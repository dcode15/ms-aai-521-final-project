import logging
import xml.etree.ElementTree as ET
from collections import defaultdict

from BoundingBox import BoundingBox
from VideoAnnotation import HockeyClip


class AnnotationParser:

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def parse_cvat_xml(xml_path: str) -> HockeyClip:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        meta = root.find('meta')
        task = meta.find('task')
        video_id = task.find('name').text

        size = task.find('original_size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        tracks: dict[str, list[BoundingBox]] = defaultdict(list)
        frame_indices: set[int] = set()

        for track in root.findall('track'):
            track_id = track.get('id')
            label = track.get('label')

            for box in track.findall('box'):
                frame_idx = int(box.get('frame'))
                frame_indices.add(frame_idx)

                team = None
                player_type = None

                if label in ['player', 'keeper']:
                    team_attr = box.find(".//attribute[@name='team']")
                    if team_attr is not None:
                        team = team_attr.text
                    player_type = 'keeper' if label == 'keeper' else 'player'

                bbox = BoundingBox(
                    frame_idx=frame_idx,
                    track_id=track_id,
                    label=label,
                    xtl=float(box.get('xtl')),
                    ytl=float(box.get('ytl')),
                    xbr=float(box.get('xbr')),
                    ybr=float(box.get('ybr')),
                    occluded=box.get('occluded') == '1',
                    team=team,
                    player_type=player_type
                )
                tracks[track_id].append(bbox)

        frame_count = max(frame_indices) + 1
        return HockeyClip(video_id, frame_count, width, height, tracks, [])
