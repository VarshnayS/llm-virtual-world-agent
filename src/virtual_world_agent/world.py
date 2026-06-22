from __future__ import annotations

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

DIRECTIONS: tuple[Direction, ...] = ("north", "east", "south", "west")
VECTORS: dict[Direction, Position] = {
    "north": (0, -1),
    "east": (1, 0),
    "south": (0, 1),
    "west": (-1, 0),
}
ARROWS: dict[Direction, str] = {
    "north": "^",
    "east": ">",
    "south": "v",
    "west": "<",
}

ACTION_DESCRIPTIONS: dict[str, str] = {
    "turn_left": "Rotate 90 degrees counter-clockwise.",
    "turn_right": "Rotate 90 degrees clockwise.",
    "move_forward": "Move one cell in the facing direction.",
    "look": "Refresh the local observation without changing position.",
    "pick_up": "Pick up an object on the current cell.",
    "unlock": "Unlock the door directly in front of the agent using the brass key.",
    "open": "Open an unlocked door directly in front of the agent.",
    "finish": "Declare the task complete while standing on the goal tile.",
}


@dataclass
class DoorState:
    locked: bool = True
    open: bool = False


@dataclass
class AgentState:
    x: int
    y: int
    facing: Direction = "east"
    inventory: list[str] = field(default_factory=list)
    steps: int = 0
    done: bool = False
    success: bool = False

    @property
    def position(self) -> Position:
        return (self.x, self.y)


class GridWorld:
    """A small partially observable key-door grid world."""

    mission = (
        "Find the brass key, unlock and open the blue door, reach the emerald "
        "goal tile, then finish."
    )

    def __init__(self, layout: tuple[str, ...] = DEFAULT_LAYOUT, view_radius: int = 2):
        if not layout:
            raise ValueError("layout must not be empty")
        widths = {len(row) for row in layout}
        if len(widths) != 1:
            raise ValueError("layout rows must have equal width")

        self.layout = layout
        self.height = len(layout)
        self.width = len(layout[0])
        self.view_radius = view_radius

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
        for y, row in enumerate(self.layout):
            for x, char in enumerate(row):
                pos = (x, y)
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
        start_x, start_y = self._start or (0, 0)
        self.items = dict(self._initial_items)
        self.doors = {
            pos: DoorState(locked=door.locked, open=door.open)
            for pos, door in self._initial_doors.items()
        }
        self.seen = set()
        self.state = AgentState(x=start_x, y=start_y)
        self.last_event = "Entered the grid world."
        return self.observe()

    def observe(self) -> dict[str, Any]:
        self._update_seen()
        front_position = self.front_position()
        return {
            "mission": self.mission,
            "world": {
                "width": self.width,
                "height": self.height,
                "view_radius": self.view_radius,
                "symbols": {
                    "#": "wall",
                    ".": "seen empty floor",
                    "?": "unseen cell",
                    "K": "brass key",
                    "D": "locked door",
                    "d": "closed unlocked door",
                    "/": "open door",
                    "G": "emerald goal",
                    "^>v<": "agent facing direction",
                },
            },
            "step": self.state.steps,
            "agent": {
                "position": {"x": self.state.x, "y": self.state.y},
                "facing": self.state.facing,
                "inventory": list(self.state.inventory),
            },
            "front_cell": self.describe_cell(front_position),
            "known_map": self.render(known_only=True),
            "visible_cells": self.visible_cells(),
            "available_actions": self.available_actions(),
            "last_event": self.last_event,
            "status": {
                "done": self.state.done,
                "success": self.state.success,
            },
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
            self.last_event = f"Turned left; now facing {self.state.facing}."
        elif action_type == "turn_right":
            self.state.facing = self._turned(1)
            self.last_event = f"Turned right; now facing {self.state.facing}."
        elif action_type == "move_forward":
            next_x, next_y = self.front_position()
            self.state.x = next_x
            self.state.y = next_y
            self.last_event = f"Moved to ({next_x},{next_y})."
        elif action_type == "look":
            self.last_event = "Looked around."
        elif action_type == "pick_up":
            item = self.items.pop(self.state.position)
            self.state.inventory.append(item)
            self.last_event = f"Picked up {item}."
        elif action_type == "unlock":
            door = self.doors[self.front_position()]
            door.locked = False
            self.last_event = "Unlocked the blue door with the brass key."
        elif action_type == "open":
            door = self.doors[self.front_position()]
            door.open = True
            self.last_event = "Opened the blue door."
        elif action_type == "finish":
            self.state.done = True
            self.state.success = True
            self.last_event = "Goal completed: key used, door passed, goal reached."

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
        dx, dy = VECTORS[self.state.facing]
        return (self.state.x + dx, self.state.y + dy)

    def is_passable(self, pos: Position) -> bool:
        if not self.in_bounds(pos) or pos in self._walls:
            return False
        door = self.doors.get(pos)
        if door is not None:
            return door.open
        return True

    def describe_cell(self, pos: Position) -> dict[str, Any]:
        x, y = pos
        terrain = "floor"
        passable = True
        objects: list[str] = []

        if not self.in_bounds(pos):
            terrain = "void"
            passable = False
        elif pos in self._walls:
            terrain = "wall"
            passable = False
        elif pos in self.doors:
            door = self.doors[pos]
            if door.open:
                terrain = "open_door"
                passable = True
            elif door.locked:
                terrain = "locked_door"
                passable = False
            else:
                terrain = "closed_door"
                passable = False
        elif pos == self._goal:
            terrain = "goal"
            objects.append("emerald_goal")
        else:
            terrain = "floor"

        item = self.items.get(pos)
        if item:
            objects.append(item)

        return {
            "position": {"x": x, "y": y},
            "terrain": terrain,
            "objects": objects,
            "passable": passable,
            "is_agent": pos == self.state.position,
        }

    def visible_cells(self) -> list[dict[str, Any]]:
        cells: list[dict[str, Any]] = []
        x0, y0 = self.state.position
        for y in range(y0 - self.view_radius, y0 + self.view_radius + 1):
            for x in range(x0 - self.view_radius, x0 + self.view_radius + 1):
                pos = (x, y)
                if self.in_bounds(pos):
                    cells.append(self.describe_cell(pos))
        return cells

    def render(self, known_only: bool = False) -> str:
        rows: list[str] = []
        for y in range(self.height):
            chars: list[str] = []
            for x in range(self.width):
                pos = (x, y)
                if known_only and pos not in self.seen and pos != self.state.position:
                    chars.append("?")
                elif pos == self.state.position:
                    chars.append(ARROWS[self.state.facing])
                else:
                    chars.append(self._glyph(pos))
            rows.append("".join(chars))
        return "\n".join(rows)

    def in_bounds(self, pos: Position) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

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
        x0, y0 = self.state.position
        for y in range(y0 - self.view_radius, y0 + self.view_radius + 1):
            for x in range(x0 - self.view_radius, x0 + self.view_radius + 1):
                pos = (x, y)
                if self.in_bounds(pos):
                    self.seen.add(pos)
