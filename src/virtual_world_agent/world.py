from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

Position = tuple[int, int]
Direction = str

DEFAULT_LAYOUT = (
    "###########",
    "#S..#....G#",
    "#.#.#.###.#",
    "#.#...#...#",
    "#.###.#.#.#",
    "#...K.D.#.#",
    "###########",
)

CELL_SIZE_M = 2.0
EYE_HEIGHT_M = 1.45
WALL_HEIGHT_M = 3.2
SENSOR_RADIUS_CELLS = 2
VISIBLE_ENTITY_RANGE_M = 7.0
FOV_DEGREES = 105

DIRECTIONS: tuple[Direction, ...] = ("north", "east", "south", "west")
VECTORS: dict[Direction, Position] = {
    "north": (0, -1),
    "east": (1, 0),
    "south": (0, 1),
    "west": (-1, 0),
}
YAW_DEGREES: dict[Direction, int] = {
    "north": 0,
    "east": 90,
    "south": 180,
    "west": 270,
}
ARROWS: dict[Direction, str] = {
    "north": "^",
    "east": ">",
    "south": "v",
    "west": "<",
}

ACTION_DESCRIPTIONS: dict[str, str] = {
    "turn_left": "Rotate 90 degrees counter-clockwise in the 3D scene.",
    "turn_right": "Rotate 90 degrees clockwise in the 3D scene.",
    "move_forward": "Move one floor cell forward in the direction of the camera yaw.",
    "look": "Refresh the local 3D scan without changing pose.",
    "pick_up": "Pick up an interactable artifact at the agent's current position.",
    "unlock": "Unlock the blue force door directly in front of the agent using the brass key.",
    "open": "Open an unlocked blue force door directly in front of the agent.",
    "finish": "Declare the mission complete while standing inside the emerald beacon.",
}


@dataclass
class DoorState:
    locked: bool = True
    open: bool = False


@dataclass
class AgentState:
    x: int
    z: int
    facing: Direction = "east"
    inventory: list[str] = field(default_factory=list)
    steps: int = 0
    done: bool = False
    success: bool = False

    @property
    def position(self) -> Position:
        return (self.x, self.z)


class AetherVaultWorld:
    """A deterministic 3D scene with an LLM-friendly observation/action harness."""

    mission = (
        "Explore the Aether Vault, recover the brass key from its pedestal, "
        "unlock and open the blue force door, enter the emerald beacon, then finish."
    )

    def __init__(
        self,
        layout: tuple[str, ...] = DEFAULT_LAYOUT,
        scan_radius_cells: int = SENSOR_RADIUS_CELLS,
    ):
        if not layout:
            raise ValueError("layout must not be empty")
        widths = {len(row) for row in layout}
        if len(widths) != 1:
            raise ValueError("layout rows must have equal width")

        self.layout = layout
        self.depth = len(layout)
        self.width = len(layout[0])
        self.scan_radius_cells = scan_radius_cells

        self._walls: set[Position] = set()
        self._initial_items: dict[Position, str] = {}
        self._initial_doors: dict[Position, DoorState] = {}
        self._start: Position | None = None
        self._goal: Position | None = None
        self._parse_layout()

        if self._start is None:
            raise ValueError("layout must contain S")
        if self._goal is None:
            raise ValueError("layout must contain G")

        self.items: dict[Position, str] = {}
        self.doors: dict[Position, DoorState] = {}
        self.seen: set[Position] = set()
        self.state: AgentState
        self.last_event = ""
        self.reset()

    def _parse_layout(self) -> None:
        for z, row in enumerate(self.layout):
            for x, char in enumerate(row):
                pos = (x, z)
                if char == "#":
                    self._walls.add(pos)
                elif char == "S":
                    self._start = pos
                elif char == "K":
                    self._initial_items[pos] = "brass_key"
                elif char == "D":
                    self._initial_doors[pos] = DoorState(locked=True, open=False)
                elif char == "G":
                    self._goal = pos
                elif char == ".":
                    continue
                else:
                    raise ValueError(f"unsupported layout character {char!r} at {pos}")

    def reset(self) -> dict[str, Any]:
        start_x, start_z = self._start or (0, 0)
        self.items = dict(self._initial_items)
        self.doors = {
            pos: DoorState(locked=door.locked, open=door.open)
            for pos, door in self._initial_doors.items()
        }
        self.seen = set()
        self.state = AgentState(x=start_x, z=start_z)
        self.last_event = "Spawned in the Aether Vault atrium."
        return self.observe()

    def observe(self) -> dict[str, Any]:
        self._update_seen()
        front_position = self.front_position()
        return {
            "mission": self.mission,
            "world": self.world_metadata(),
            "scene_state": self.scene_state(),
            "step": self.state.steps,
            "agent": {
                "grid_position": {"x": self.state.x, "z": self.state.z},
                "pose": self.agent_pose(),
                "facing": self.state.facing,
                "inventory": list(self.state.inventory),
            },
            "front_cell": self.describe_cell(front_position),
            "visible_cells": self.visible_cells(),
            "visible_entities": self.visible_entities(),
            "known_map": self.render(known_only=True),
            "available_actions": self.available_actions(),
            "last_event": self.last_event,
            "status": {
                "done": self.state.done,
                "success": self.state.success,
            },
        }

    def world_metadata(self) -> dict[str, Any]:
        return {
            "name": "Aether Vault",
            "type": "3d_scene",
            "dimensions": {
                "width_cells": self.width,
                "depth_cells": self.depth,
                "cell_size_m": CELL_SIZE_M,
                "wall_height_m": WALL_HEIGHT_M,
            },
            "coordinate_frame": {
                "x": "east-west",
                "y": "vertical height",
                "z": "north-south depth",
                "yaw_degrees": "0=north, 90=east, 180=south, 270=west",
            },
            "sensors": {
                "local_depth_scan_radius_cells": self.scan_radius_cells,
                "visible_entity_range_m": VISIBLE_ENTITY_RANGE_M,
                "field_of_view_degrees": FOV_DEGREES,
            },
            "symbols": {
                "#": "wall volume",
                ".": "seen floor",
                "?": "unscanned space",
                "K": "brass key pedestal",
                "D": "locked blue force door",
                "d": "closed unlocked force door",
                "/": "open force door",
                "G": "emerald beacon goal",
                "^>v<": "agent yaw",
            },
        }

    def scene_snapshot(self) -> dict[str, Any]:
        return {
            "name": "Aether Vault",
            "layout": list(self.layout),
            "cell_size_m": CELL_SIZE_M,
            "wall_height_m": WALL_HEIGHT_M,
            "eye_height_m": EYE_HEIGHT_M,
            "start": self._position_payload(self._start or (0, 0)),
            "goal": self._position_payload(self._goal or (0, 0)),
            "decorations": self._decorations(),
            "palette": {
                "floor": "#171b24",
                "wall": "#222938",
                "wall_trim": "#7f8cff",
                "door_locked": "#236bff",
                "door_open": "#67e8f9",
                "key": "#f5c542",
                "goal": "#20f5a6",
                "agent": "#ff6b35",
            },
        }

    def scene_state(self) -> dict[str, Any]:
        return {
            "doors": [
                {
                    "id": self._entity_id("door", pos),
                    "grid_position": {"x": pos[0], "z": pos[1]},
                    "center_m": self.grid_to_world(pos, y=1.5),
                    "locked": door.locked,
                    "open": door.open,
                }
                for pos, door in sorted(self.doors.items())
            ],
            "items": [
                {
                    "id": self._entity_id(item, pos),
                    "type": item,
                    "grid_position": {"x": pos[0], "z": pos[1]},
                    "center_m": self.grid_to_world(pos, y=0.85),
                }
                for pos, item in sorted(self.items.items())
            ],
        }

    def step(self, action: Mapping[str, Any] | str) -> dict[str, Any]:
        if self.state.done:
            self.last_event = "Episode is already done."
            return self.observe()

        action_type = action if isinstance(action, str) else action.get("type")
        if not isinstance(action_type, str):
            self.state.steps += 1
            self.last_event = "Invalid action: expected a string type."
            return self.observe()

        available = {item["type"] for item in self.available_actions()}
        if action_type not in ACTION_DESCRIPTIONS:
            self.state.steps += 1
            self.last_event = f"Invalid action: unknown action {action_type!r}."
            return self.observe()
        if action_type not in available:
            self.state.steps += 1
            self.last_event = (
                f"Invalid action: {action_type!r} is not currently available. "
                f"Available actions are {sorted(available)}."
            )
            return self.observe()

        self.state.steps += 1
        if action_type == "turn_left":
            self.state.facing = self._turned(-1)
            self.last_event = f"Rotated left; camera yaw is now {self.agent_pose()['yaw_degrees']} degrees."
        elif action_type == "turn_right":
            self.state.facing = self._turned(1)
            self.last_event = f"Rotated right; camera yaw is now {self.agent_pose()['yaw_degrees']} degrees."
        elif action_type == "move_forward":
            next_x, next_z = self.front_position()
            self.state.x = next_x
            self.state.z = next_z
            self.last_event = f"Moved to floor cell ({next_x},{next_z})."
        elif action_type == "look":
            self.last_event = "Swept the local 3D depth scan."
        elif action_type == "pick_up":
            item = self.items.pop(self.state.position)
            self.state.inventory.append(item)
            self.last_event = f"Lifted {item} from the brass pedestal."
        elif action_type == "unlock":
            door = self.doors[self.front_position()]
            door.locked = False
            self.last_event = "The brass key dissolved the blue force lock."
        elif action_type == "open":
            door = self.doors[self.front_position()]
            door.open = True
            self.last_event = "The blue force door opened into a light bridge."
        elif action_type == "finish":
            self.state.done = True
            self.state.success = True
            self.last_event = "Mission complete: the agent entered the emerald beacon."

        return self.observe()

    def available_actions(self) -> list[dict[str, str]]:
        action_types = ["turn_left", "turn_right", "look"]
        current = self.state.position
        front = self.front_position()
        front_door = self.doors.get(front)

        if self.is_passable(front):
            action_types.append("move_forward")
        if current in self.items:
            action_types.append("pick_up")
        if front_door and front_door.locked and "brass_key" in self.state.inventory:
            action_types.append("unlock")
        if front_door and not front_door.locked and not front_door.open:
            action_types.append("open")
        if current == self._goal:
            action_types.append("finish")

        return [
            {"type": action_type, "description": ACTION_DESCRIPTIONS[action_type]}
            for action_type in action_types
        ]

    def front_position(self) -> Position:
        dx, dz = VECTORS[self.state.facing]
        return (self.state.x + dx, self.state.z + dz)

    def is_passable(self, pos: Position) -> bool:
        if not self.in_bounds(pos) or pos in self._walls:
            return False
        door = self.doors.get(pos)
        if door is not None:
            return door.open
        return True

    def describe_cell(self, pos: Position) -> dict[str, Any]:
        x, z = pos
        terrain = "stone_floor"
        passable = True
        objects: list[str] = []
        entity_id: str | None = None

        if not self.in_bounds(pos):
            terrain = "void"
            passable = False
        elif pos in self._walls:
            terrain = "wall_volume"
            passable = False
            entity_id = self._entity_id("wall", pos)
        elif pos in self.doors:
            door = self.doors[pos]
            entity_id = self._entity_id("door", pos)
            if door.open:
                terrain = "open_force_door"
                passable = True
            elif door.locked:
                terrain = "sealed_force_door"
                passable = False
            else:
                terrain = "closed_force_door"
                passable = False
        elif pos == self._goal:
            terrain = "emerald_beacon"
            objects.append("emerald_beacon")
            entity_id = self._entity_id("goal", pos)

        item = self.items.get(pos)
        if item:
            objects.append(item)
            entity_id = self._entity_id(item, pos)

        return {
            "grid_position": {"x": x, "z": z},
            "center_m": self.grid_to_world(pos, y=0),
            "terrain": terrain,
            "objects": objects,
            "passable": passable,
            "entity_id": entity_id,
            "is_agent": pos == self.state.position,
        }

    def visible_cells(self) -> list[dict[str, Any]]:
        cells: list[dict[str, Any]] = []
        x0, z0 = self.state.position
        for z in range(z0 - self.scan_radius_cells, z0 + self.scan_radius_cells + 1):
            for x in range(x0 - self.scan_radius_cells, x0 + self.scan_radius_cells + 1):
                pos = (x, z)
                if self.in_bounds(pos):
                    cell = self.describe_cell(pos)
                    cell["distance_m"] = round(self.distance_m(self.state.position, pos), 2)
                    cells.append(cell)
        return cells

    def visible_entities(self) -> list[dict[str, Any]]:
        entities: list[dict[str, Any]] = []
        for cell in self.visible_cells():
            terrain = cell["terrain"]
            objects = cell["objects"]
            if terrain == "stone_floor" and not objects:
                continue
            pos = (cell["grid_position"]["x"], cell["grid_position"]["z"])
            distance = self.distance_m(self.state.position, pos)
            if distance > VISIBLE_ENTITY_RANGE_M:
                continue
            bearing = self.bearing_degrees(pos)
            if abs(bearing) > FOV_DEGREES / 2 and distance > CELL_SIZE_M:
                continue
            entities.append(
                {
                    "id": cell["entity_id"] or self._entity_id(terrain, pos),
                    "label": self._entity_label(cell),
                    "kind": self._entity_kind(cell),
                    "grid_position": cell["grid_position"],
                    "center_m": cell["center_m"],
                    "distance_m": round(distance, 2),
                    "bearing_degrees": round(bearing, 1),
                    "blocks_movement": not cell["passable"],
                    "state": self._entity_state(pos, cell),
                }
            )
        entities.sort(key=lambda entity: (entity["distance_m"], entity["id"]))
        return entities

    def render(self, known_only: bool = False) -> str:
        rows: list[str] = []
        for z in range(self.depth):
            chars: list[str] = []
            for x in range(self.width):
                pos = (x, z)
                if known_only and pos not in self.seen and pos != self.state.position:
                    chars.append("?")
                elif pos == self.state.position:
                    chars.append(ARROWS[self.state.facing])
                else:
                    chars.append(self._glyph(pos))
            rows.append("".join(chars))
        return "\n".join(rows)

    def agent_pose(self) -> dict[str, Any]:
        return {
            "position_m": self.grid_to_world(self.state.position, y=EYE_HEIGHT_M),
            "yaw_degrees": YAW_DEGREES[self.state.facing],
            "facing": self.state.facing,
        }

    def grid_to_world(self, pos: Position, y: float) -> dict[str, float]:
        x, z = pos
        return {
            "x": round((x - (self.width - 1) / 2) * CELL_SIZE_M, 3),
            "y": round(y, 3),
            "z": round((z - (self.depth - 1) / 2) * CELL_SIZE_M, 3),
        }

    def distance_m(self, left: Position, right: Position) -> float:
        return math.dist(
            (left[0] * CELL_SIZE_M, left[1] * CELL_SIZE_M),
            (right[0] * CELL_SIZE_M, right[1] * CELL_SIZE_M),
        )

    def bearing_degrees(self, pos: Position) -> float:
        dx = pos[0] - self.state.x
        dz = pos[1] - self.state.z
        absolute = math.degrees(math.atan2(dx, -dz)) % 360
        relative = (absolute - YAW_DEGREES[self.state.facing] + 540) % 360 - 180
        return relative

    def in_bounds(self, pos: Position) -> bool:
        x, z = pos
        return 0 <= x < self.width and 0 <= z < self.depth

    def _position_payload(self, pos: Position) -> dict[str, Any]:
        return {
            "grid_position": {"x": pos[0], "z": pos[1]},
            "center_m": self.grid_to_world(pos, y=0),
        }

    def _glyph(self, pos: Position) -> str:
        if pos in self._walls:
            return "#"
        door = self.doors.get(pos)
        if door:
            if door.open:
                return "/"
            return "D" if door.locked else "d"
        if pos in self.items:
            return "K"
        if pos == self._goal:
            return "G"
        return "."

    def _turned(self, offset: int) -> Direction:
        index = DIRECTIONS.index(self.state.facing)
        return DIRECTIONS[(index + offset) % len(DIRECTIONS)]

    def _update_seen(self) -> None:
        x0, z0 = self.state.position
        for z in range(z0 - self.scan_radius_cells, z0 + self.scan_radius_cells + 1):
            for x in range(x0 - self.scan_radius_cells, x0 + self.scan_radius_cells + 1):
                pos = (x, z)
                if self.in_bounds(pos):
                    self.seen.add(pos)

    def _entity_label(self, cell: dict[str, Any]) -> str:
        terrain = cell["terrain"]
        objects = cell["objects"]
        if "brass_key" in objects:
            return "brass key hovering above a pedestal"
        if "emerald_beacon" in objects:
            return "emerald beacon goal volume"
        if terrain == "wall_volume":
            return "obsidian wall volume"
        if terrain == "sealed_force_door":
            return "locked blue force door"
        if terrain == "closed_force_door":
            return "unlocked blue force door"
        if terrain == "open_force_door":
            return "open blue force doorway"
        return terrain.replace("_", " ")

    def _entity_kind(self, cell: dict[str, Any]) -> str:
        if cell["objects"]:
            return "artifact"
        if "door" in cell["terrain"]:
            return "door"
        if cell["terrain"] == "wall_volume":
            return "architecture"
        return "landmark"

    def _entity_state(self, pos: Position, cell: dict[str, Any]) -> dict[str, Any]:
        if pos in self.doors:
            door = self.doors[pos]
            return {"locked": door.locked, "open": door.open}
        return {"terrain": cell["terrain"], "objects": list(cell["objects"])}

    def _entity_id(self, prefix: str, pos: Position) -> str:
        return f"{prefix}-{pos[0]}-{pos[1]}"

    def _decorations(self) -> list[dict[str, Any]]:
        decorations: list[dict[str, Any]] = []
        for index, pos in enumerate(((1, 3), (5, 1), (7, 3), (9, 5))):
            if self.in_bounds(pos) and pos not in self._walls:
                decorations.append(
                    {
                        "id": f"light-pillar-{index}",
                        "type": "light_pillar",
                        "grid_position": {"x": pos[0], "z": pos[1]},
                        "center_m": self.grid_to_world(pos, y=0),
                    }
                )
        return decorations


# Backwards-compatible name for older imports and tests.
GridWorld = AetherVaultWorld
