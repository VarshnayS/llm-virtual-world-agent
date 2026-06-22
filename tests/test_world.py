from __future__ import annotations

import unittest

from virtual_world_agent.agents import HeuristicPolicy
from virtual_world_agent.runner import run_episode
from virtual_world_agent.world import GridWorld


class GridWorldTests(unittest.TestCase):
    def test_door_requires_key_before_unlocking(self) -> None:
        world = GridWorld()
        observation = world.reset()

        actions = [
            "turn_right",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_left",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
        ]
        for action in actions:
            observation = world.step({"type": action})

        self.assertEqual(observation["agent"]["position"], {"x": 5, "y": 5})
        self.assertEqual(observation["front_cell"]["terrain"], "locked_door")
        self.assertNotIn("unlock", {item["type"] for item in observation["available_actions"]})

    def test_heuristic_policy_completes_world(self) -> None:
        result = run_episode(GridWorld(), HeuristicPolicy(), max_steps=80, map_mode="none")

        self.assertTrue(result.success)
        self.assertIn("RESULT: success", result.transcript)


if __name__ == "__main__":
    unittest.main()
