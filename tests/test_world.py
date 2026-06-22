from __future__ import annotations

import unittest

from virtual_world_agent.agents import HeuristicPolicy
from virtual_world_agent.runner import run_episode
from virtual_world_agent.world import AetherVaultWorld


class AetherVaultWorldTests(unittest.TestCase):
    def test_observation_exposes_3d_scene_contract(self) -> None:
        observation = AetherVaultWorld().reset()

        self.assertEqual(observation["world"]["type"], "3d_scene")
        self.assertIn("pose", observation["agent"])
        self.assertIn("position_m", observation["agent"]["pose"])
        self.assertIn("yaw_degrees", observation["agent"]["pose"])
        self.assertIn("visible_entities", observation)
        self.assertIn("scene_state", observation)

    def test_door_requires_key_before_unlocking(self) -> None:
        world = AetherVaultWorld()
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

        self.assertEqual(observation["agent"]["grid_position"], {"x": 5, "z": 5})
        self.assertEqual(observation["front_cell"]["terrain"], "sealed_force_door")
        self.assertNotIn("unlock", {item["type"] for item in observation["available_actions"]})

    def test_heuristic_policy_completes_world_and_exports_replay(self) -> None:
        result = run_episode(AetherVaultWorld(), HeuristicPolicy(), max_steps=80, map_mode="none")

        self.assertTrue(result.success)
        self.assertIn("RESULT: success", result.transcript)
        self.assertEqual(result.replay["scene"]["name"], "Aether Vault")
        self.assertGreater(len(result.replay["frames"]), 1)


if __name__ == "__main__":
    unittest.main()
