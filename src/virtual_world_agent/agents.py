from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

from .world import DIRECTIONS, VECTORS, Direction, Position


class Policy(Protocol):
    def choose_action(self, observation: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
        ...


class HeuristicPolicy:
    """A deterministic policy that uses only the public observation stream."""

    def __init__(self) -> None:
        self.known: dict[Position, dict[str, Any]] = {}

    def choose_action(self, observation: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
        self._update_memory(observation)
        available = {item["type"] for item in observation["available_actions"]}
        inventory = set(observation["agent"]["inventory"])

        for direct_action, thought in (
            ("finish", "standing on the goal tile"),
            ("pick_up", "collecting the key before approaching the door"),
            ("unlock", "using the key on the locked door"),
            ("open", "opening the unlocked door"),
        ):
            if direct_action in available:
                return {"type": direct_action}, {"thought": thought}

        current = self._agent_position(observation)
        facing = observation["agent"]["facing"]

        if "brass_key" not in inventory:
            key_pos = self._find_object("brass_key")
            if key_pos:
                action = self._move_toward(observation, key_pos)
                return action, {"thought": "moving toward the known brass key"}
        else:
            door_target = self._target_closed_door(current)
            if door_target:
                target_pos, target_facing = door_target
                if current == target_pos:
                    action = self._turn_toward(facing, target_facing)
                else:
                    action = self._move_toward(observation, target_pos)
                return action, {"thought": "positioning to operate the blue door"}

            goal_pos = self._find_object("emerald_goal")
            if goal_pos:
                action = self._move_toward(observation, goal_pos)
                return action, {"thought": "moving toward the known emerald goal"}

        frontier = self._nearest_frontier(current)
        if frontier:
            if frontier == current:
                action = self._probe_unknown_neighbor(observation)
            else:
                action = self._move_toward(observation, frontier)
            return action, {"thought": "moving toward nearest useful frontier"}

        if "move_forward" in available:
            return {"type": "move_forward"}, {"thought": "probing the only apparent route"}
        return {"type": "turn_right"}, {"thought": "rotating to search for a new route"}

    def _update_memory(self, observation: dict[str, Any]) -> None:
        for cell in observation["visible_cells"]:
            position = cell["position"]
            self.known[(position["x"], position["y"])] = cell

    def _find_object(self, object_name: str) -> Position | None:
        for pos, cell in self.known.items():
            if object_name in cell["objects"]:
                return pos
        return None

    def _target_closed_door(self, current: Position) -> tuple[Position, Direction] | None:
        candidates: list[tuple[int, Position, Direction]] = []
        for door_pos, cell in self.known.items():
            if cell["terrain"] not in {"locked_door", "closed_door"}:
                continue
            for direction, vector in VECTORS.items():
                dx, dy = vector
                stand_pos = (door_pos[0] - dx, door_pos[1] - dy)
                if self._is_known_passable(stand_pos):
                    distance = self._distance(current, stand_pos)
                    candidates.append((distance, stand_pos, direction))
        if not candidates:
            return None
        _, stand_pos, direction = min(candidates, key=lambda item: item[0])
        return stand_pos, direction

    def _nearest_frontier(self, current: Position) -> Position | None:
        reachable = self._reachable_from(current)
        frontiers = [
            pos
            for pos in reachable
            if any(neighbor not in self.known for neighbor in self._neighbors(pos))
        ]
        if not frontiers:
            return None
        return min(frontiers, key=lambda pos: self._distance(current, pos))

    def _move_toward(self, observation: dict[str, Any], target: Position) -> dict[str, str]:
        current = self._agent_position(observation)
        path = self._path(current, target)
        if not path or len(path) < 2:
            return self._probe_unknown_neighbor(observation)

        next_pos = path[1]
        desired_facing = self._direction_between(current, next_pos)
        if desired_facing is None:
            return {"type": "turn_right"}

        if observation["agent"]["facing"] == desired_facing:
            available = {item["type"] for item in observation["available_actions"]}
            if "move_forward" in available:
                return {"type": "move_forward"}
            return {"type": "turn_right"}
        return self._turn_toward(observation["agent"]["facing"], desired_facing)

    def _probe_unknown_neighbor(self, observation: dict[str, Any]) -> dict[str, str]:
        current = self._agent_position(observation)
        facing = observation["agent"]["facing"]
        available = {item["type"] for item in observation["available_actions"]}

        front = self._offset(current, facing)
        if front not in self.known and "move_forward" in available:
            return {"type": "move_forward"}

        for direction in DIRECTIONS:
            if self._offset(current, direction) not in self.known:
                return self._turn_toward(facing, direction)

        return {"type": "move_forward"} if "move_forward" in available else {"type": "turn_right"}

    def _turn_toward(self, current: Direction, desired: Direction) -> dict[str, str]:
        if current == desired:
            return {"type": "look"}
        current_index = DIRECTIONS.index(current)
        desired_index = DIRECTIONS.index(desired)
        clockwise = (desired_index - current_index) % len(DIRECTIONS)
        if clockwise == 1:
            return {"type": "turn_right"}
        if clockwise == 3:
            return {"type": "turn_left"}
        return {"type": "turn_right"}

    def _path(self, start: Position, goal: Position) -> list[Position] | None:
        if start == goal:
            return [start]
        queue: deque[Position] = deque([start])
        previous: dict[Position, Position | None] = {start: None}
        while queue:
            pos = queue.popleft()
            for neighbor in self._neighbors(pos):
                if neighbor in previous or not self._is_known_passable(neighbor):
                    continue
                previous[neighbor] = pos
                if neighbor == goal:
                    return self._reconstruct(previous, goal)
                queue.append(neighbor)
        return None

    def _reachable_from(self, start: Position) -> set[Position]:
        if not self._is_known_passable(start):
            return {start}
        seen = {start}
        queue: deque[Position] = deque([start])
        while queue:
            pos = queue.popleft()
            for neighbor in self._neighbors(pos):
                if neighbor in seen or not self._is_known_passable(neighbor):
                    continue
                seen.add(neighbor)
                queue.append(neighbor)
        return seen

    def _is_known_passable(self, pos: Position) -> bool:
        cell = self.known.get(pos)
        if not cell:
            return False
        return bool(cell["passable"])

    def _neighbors(self, pos: Position) -> list[Position]:
        return [self._offset(pos, direction) for direction in DIRECTIONS]

    def _offset(self, pos: Position, direction: Direction) -> Position:
        dx, dy = VECTORS[direction]
        return (pos[0] + dx, pos[1] + dy)

    def _distance(self, left: Position, right: Position) -> int:
        return abs(left[0] - right[0]) + abs(left[1] - right[1])

    def _direction_between(self, start: Position, end: Position) -> Direction | None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        for direction, vector in VECTORS.items():
            if vector == (dx, dy):
                return direction
        return None

    def _agent_position(self, observation: dict[str, Any]) -> Position:
        position = observation["agent"]["position"]
        return (position["x"], position["y"])

    def _reconstruct(
        self, previous: dict[Position, Position | None], goal: Position
    ) -> list[Position]:
        path = [goal]
        while previous[path[-1]] is not None:
            parent = previous[path[-1]]
            if parent is None:
                break
            path.append(parent)
        path.reverse()
        return path


@dataclass
class OpenAICompatibleChatClient:
    model: str
    api_key: str
    api_base: str = "https://api.openai.com/v1"
    temperature: float = 0.1
    max_tokens: int = 240
    timeout: float = 60.0
    json_mode: bool = True

    @classmethod
    def from_env(
        cls,
        model: str | None = None,
        api_base: str | None = None,
        json_mode: bool = True,
    ) -> "OpenAICompatibleChatClient":
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM_API_KEY or OPENAI_API_KEY is required when using --agent llm."
            )
        return cls(
            model=model or os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
            api_key=api_key,
            api_base=api_base or os.getenv("LLM_API_BASE", "https://api.openai.com/v1"),
            json_mode=json_mode,
        )

    def complete(self, messages: list[dict[str, str]]) -> str:
        url = f"{self.api_base.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM response: {data!r}") from exc


class LLMPolicy:
    def __init__(self, client: OpenAICompatibleChatClient):
        self.client = client

    def choose_action(self, observation: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": "Observation JSON:\n" + json.dumps(observation, indent=2),
            },
        ]
        raw = self.client.complete(messages)
        parsed = self._parse_response(raw)
        action = parsed.get("action", parsed)
        if not isinstance(action, dict) or not isinstance(action.get("type"), str):
            action = {"type": "look"}
        return {"type": action["type"]}, {"thought": parsed.get("thought", ""), "raw": raw}

    def _system_prompt(self) -> str:
        return (
            "You control an agent in a partially observable grid world. "
            "The mission is to find the brass key, unlock and open the blue door, "
            "reach the emerald goal tile, and finish. The observation contains a "
            "known_map, visible_cells, front_cell, inventory, and available_actions. "
            "Choose exactly one action whose type appears in available_actions. "
            "Do not invent actions or parameters. Respond only with JSON in this "
            'shape: {"thought": "brief reason", "action": {"type": "move_forward"}}.'
        )

    def _parse_response(self, raw: str) -> dict[str, Any]:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                return {"thought": "model returned non-JSON", "action": {"type": "look"}}
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                return {"thought": "model returned malformed JSON", "action": {"type": "look"}}

        if isinstance(parsed, dict):
            return parsed
        return {"thought": "model returned a non-object JSON value", "action": {"type": "look"}}
