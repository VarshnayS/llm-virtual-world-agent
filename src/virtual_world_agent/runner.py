from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .agents import Policy
from .world import GridWorld


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    transcript: str


def run_episode(
    world: GridWorld,
    policy: Policy,
    max_steps: int = 80,
    map_mode: str = "known",
    show_observation: bool = False,
) -> EpisodeResult:
    observation = world.reset()
    lines = [
        "Mission: " + observation["mission"],
        "Initial known map:",
        observation["known_map"],
        "",
    ]

    for _ in range(max_steps):
        action, metadata = policy.choose_action(observation)
        observation = world.step(action)
        lines.extend(_format_step(observation, action, metadata, world, map_mode, show_observation))
        if observation["status"]["done"]:
            break

    result = "success" if observation["status"]["success"] else "failure"
    lines.append(f"RESULT: {result} in {observation['step']} steps")
    return EpisodeResult(
        success=bool(observation["status"]["success"]),
        steps=int(observation["step"]),
        transcript="\n".join(lines) + "\n",
    )


def _format_step(
    observation: dict[str, Any],
    action: dict[str, str],
    metadata: dict[str, Any],
    world: GridWorld,
    map_mode: str,
    show_observation: bool,
) -> list[str]:
    agent = observation["agent"]
    position = agent["position"]
    inventory = ",".join(agent["inventory"]) if agent["inventory"] else "[]"
    lines = [
        (
            f"Step {observation['step']:02d} | pos=({position['x']},{position['y']}) "
            f"facing={agent['facing']} inventory={inventory}"
        )
    ]
    thought = metadata.get("thought")
    if thought:
        lines.append(f"thought: {thought}")
    lines.append(f"action: {action.get('type')}")
    lines.append(f"event: {observation['last_event']}")
    if map_mode == "known":
        lines.append("known map:")
        lines.append(observation["known_map"])
    elif map_mode == "full":
        lines.append("full map:")
        lines.append(world.render(known_only=False))
    if show_observation:
        lines.append("available actions: " + ", ".join(a["type"] for a in observation["available_actions"]))
        front = observation["front_cell"]
        lines.append(
            "front cell: "
            f"{front['terrain']} at ({front['position']['x']},{front['position']['y']})"
        )
    lines.append("")
    return lines
