from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .agents import Policy
from .world import AetherVaultWorld


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    transcript: str
    replay: dict[str, Any]


def run_episode(
    world: AetherVaultWorld,
    policy: Policy,
    max_steps: int = 80,
    map_mode: str = "known",
    show_observation: bool = False,
) -> EpisodeResult:
    observation = world.reset()
    frames = [_replay_frame(observation, action=None, metadata={})]
    lines = [
        "Mission: " + observation["mission"],
        f"Scene: {observation['world']['name']} ({observation['world']['type']})",
        "Initial known map:",
        observation["known_map"],
        "",
    ]

    for _ in range(max_steps):
        action, metadata = policy.choose_action(observation)
        observation = world.step(action)
        frames.append(_replay_frame(observation, action=action, metadata=metadata))
        lines.extend(_format_step(observation, action, metadata, world, map_mode, show_observation))
        if observation["status"]["done"]:
            break

    result = "success" if observation["status"]["success"] else "failure"
    lines.append(f"RESULT: {result} in {observation['step']} steps")
    replay = {
        "mission": observation["mission"],
        "scene": world.scene_snapshot(),
        "frames": frames,
        "result": {
            "success": bool(observation["status"]["success"]),
            "steps": int(observation["step"]),
        },
    }
    return EpisodeResult(
        success=bool(observation["status"]["success"]),
        steps=int(observation["step"]),
        transcript="\n".join(lines) + "\n",
        replay=replay,
    )


def _replay_frame(
    observation: dict[str, Any],
    action: dict[str, str] | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "step": observation["step"],
        "agent": observation["agent"],
        "action": action,
        "thought": metadata.get("thought", ""),
        "event": observation["last_event"],
        "known_map": observation["known_map"],
        "visible_entities": observation["visible_entities"],
        "available_actions": [item["type"] for item in observation["available_actions"]],
        "scene_state": observation["scene_state"],
        "status": observation["status"],
    }


def _format_step(
    observation: dict[str, Any],
    action: dict[str, str],
    metadata: dict[str, Any],
    world: AetherVaultWorld,
    map_mode: str,
    show_observation: bool,
) -> list[str]:
    agent = observation["agent"]
    grid_position = agent["grid_position"]
    pose = agent["pose"]
    metric_position = pose["position_m"]
    inventory = ",".join(agent["inventory"]) if agent["inventory"] else "[]"
    lines = [
        (
            f"Step {observation['step']:02d} | cell=({grid_position['x']},{grid_position['z']}) "
            f"pose=({metric_position['x']},{metric_position['y']},{metric_position['z']})m "
            f"yaw={pose['yaw_degrees']} inventory={inventory}"
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
        lines.append(
            "visible entities: "
            + ", ".join(entity["label"] for entity in observation["visible_entities"][:6])
        )
        front = observation["front_cell"]
        front_position = front["grid_position"]
        lines.append(
            "front cell: "
            f"{front['terrain']} at ({front_position['x']},{front_position['z']})"
        )
    lines.append("")
    return lines
