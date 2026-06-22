# LLM Agent in a Virtual World

This project is a compact agent harness for a virtual grid world. The agent receives structured observations, chooses validated actions, and completes a goal: find the brass key, unlock/open the blue door, reach the emerald goal tile, and finish.

The system includes:

- A partially observable 2D grid world with walls, a key, a locked door, and a goal.
- A JSON observation format with current state, visible cells, a remembered map, front-cell details, inventory, and valid actions.
- A small action space: `turn_left`, `turn_right`, `move_forward`, `look`, `pick_up`, `unlock`, `open`, and `finish`.
- An OpenAI-compatible LLM policy that asks the model to return a single JSON action.
- A deterministic observation-only heuristic policy for offline testing and reproducible example logs.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

virtual-world-agent --agent heuristic --log-file examples/latest_run.log
```

Expected result: the run ends with `RESULT: success`.

You can also run the tests:

```bash
python -m unittest discover -s tests
```

## Run With An LLM

The LLM client uses a generic OpenAI-compatible chat completions API. For OpenAI:

```bash
export OPENAI_API_KEY="sk-..."
export LLM_MODEL="gpt-4o-mini"

virtual-world-agent --agent llm --max-steps 80 --log-file examples/llm_run.log
```

For a local or alternative OpenAI-compatible server:

```bash
export LLM_API_BASE="http://localhost:1234/v1"
export LLM_API_KEY="local-key"
export LLM_MODEL="your-model"

virtual-world-agent --agent llm
```

If your endpoint does not support JSON response mode, disable it:

```bash
virtual-world-agent --agent llm --no-llm-json-mode
```

## Example Output

An example deterministic run is committed at [examples/demo_heuristic.log](examples/demo_heuristic.log). A typical step looks like this:

```text
Step 05 | pos=(3,1) facing=south inventory=[]
thought: moving toward nearest useful frontier
action: move_forward
event: Moved to (3,2).
known map:
???????
?#..#??
?#.@#??
?#.#???
???????
```

## Observation Format

Each turn, the policy receives a dictionary like:

```json
{
  "mission": "Find the brass key, unlock and open the blue door, reach the emerald goal tile, then finish.",
  "step": 6,
  "agent": {
    "position": { "x": 3, "y": 2 },
    "facing": "south",
    "inventory": []
  },
  "front_cell": {
    "position": { "x": 3, "y": 3 },
    "terrain": "floor",
    "objects": [],
    "passable": true
  },
  "known_map": "???????\n?#..#??\n?#.v#??\n?#.#???\n???????",
  "visible_cells": [
    {
      "position": { "x": 3, "y": 3 },
      "terrain": "floor",
      "objects": [],
      "passable": true
    }
  ],
  "available_actions": [
    { "type": "turn_left", "description": "Rotate 90 degrees counter-clockwise." },
    { "type": "turn_right", "description": "Rotate 90 degrees clockwise." },
    { "type": "move_forward", "description": "Move one cell in the facing direction." }
  ],
  "last_event": "Moved to (3,2)."
}
```

The LLM is not allowed to invent actions. The harness parses the model response, validates it against `available_actions`, and returns an invalid-action event if the choice cannot be executed.

The expected LLM response is:

```json
{
  "thought": "I need the key before the door, so I will continue exploring the reachable corridor.",
  "action": { "type": "move_forward" }
}
```

## Design Choices

The main design goal is to make the environment-agent boundary explicit. The world never exposes hidden state in the observation, only a local view and a remembered map. The observation includes dynamic valid actions because this reduces malformed model output without removing the need for planning.

The action space is intentionally small and physical: the agent turns, moves forward, picks up objects, unlocks, opens, and finishes. This keeps the LLM's decisions grounded in state transitions that the environment can verify.

The LLM harness is stateless per turn except for the environment's remembered map. This keeps prompts short and makes failure modes easier to inspect in logs. The deterministic heuristic agent is included to prove the environment is solvable without requiring an API key; the LLM policy uses the same observation and action contract.

What worked well: structured JSON observations and validated action output make the loop reliable and easy to debug. What remains imperfect: partial observability means weaker LLMs can still loop or over-explore, so production use would benefit from retry prompting, trajectory memory, and a richer planner.
