from dataclasses import dataclass
from typing import Optional, Literal

TeamColor = Literal['white', 'black']
PlayerType = Literal['keeper', 'player']


@dataclass
class PlayerCategory:
    """Represents a player's team and position."""
    team: TeamColor
    player_type: PlayerType

    def to_class_idx(self) -> int:
        """Convert player category to class index for YOLO."""
        # Class mapping:
        # 0: white keeper
        # 1: white player
        # 2: black keeper
        # 3: black player
        base_idx = 0 if self.team == 'white' else 2
        return base_idx + (0 if self.player_type == 'keeper' else 1)

    @staticmethod
    def from_class_idx(idx: int) -> 'PlayerCategory':
        """Create PlayerCategory from class index."""
        team = 'white' if idx < 2 else 'black'
        player_type = 'keeper' if idx % 2 == 0 else 'player'
        return PlayerCategory(team=team, player_type=player_type)


@dataclass
class BoundingBox:
    frame_idx: int
    track_id: str
    label: str
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    occluded: bool
    team: Optional[TeamColor] = None
    player_type: Optional[PlayerType] = None

    @property
    def category(self) -> Optional[PlayerCategory]:
        """Get the player category if both team and player_type are set."""
        if self.team is not None and self.player_type is not None:
            return PlayerCategory(team=self.team, player_type=self.player_type)
        return None

    @staticmethod
    def create_from_detection(
            frame_idx: int,
            track_id: str,
            xtl: float,
            ytl: float,
            xbr: float,
            ybr: float,
            class_idx: int,
            occluded: bool = False
    ) -> 'BoundingBox':
        """Create a BoundingBox from detection results."""
        category = PlayerCategory.from_class_idx(class_idx)
        return BoundingBox(
            frame_idx=frame_idx,
            track_id=track_id,
            label=category.player_type,
            xtl=xtl,
            ytl=ytl,
            xbr=xbr,
            ybr=ybr,
            occluded=occluded,
            team=category.team,
            player_type=category.player_type
        )
