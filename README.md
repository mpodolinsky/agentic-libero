# agentic-libero

An agentic manipulation harness on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)-10: a Gemini model
controls the robot arm by watching short videos of what just happened and choosing the next actions. It is built on
[WAM-eval](https://github.com/mpodolinsky/WAM-eval), so the Cosmos3 and VLA-JEPA policies can be evaluated here too.

## How it works

`agent.py` runs one episode as one chat with `gemini-4-argon`:

1. The first turn shows the starting scene and the task.
2. The model answers with JSON: what it sees, a plan, and the next **10 actions** (0.5 s), each the 7-D LIBERO command
   `[dx, dy, dz, droll, dpitch, dyaw, gripper]` in [-1, 1]. These are executed open loop.
3. The next turn adds a short video of those 10 steps (agentview and wrist side by side, every second step), and the
   model answers again. Its own earlier answers stay in the conversation, so it remembers what it commanded.

The system prompt explains the action in image terms (which way each axis moves in the agentview, about 1 cm per step
at 1.0, +1 closes the gripper); those numbers were measured in the simulator. Invalid replies are retried once, then
the robot holds still. Everything is written as the run goes, so killing it loses nothing: `outputs/agent/<run>/` has
`calls.jsonl` (every raw reply of the model, with latency and token usage), `clips/` (the video sent on each turn),
`episode.mp4` (the whole rollout, both cameras), `config.json` and `summary.json`.

## Setup

Linux x86_64, one NVIDIA GPU for the simulator renderer, and `git`, `uv`, `cmake`, `g++`, `curl`.

```bash
git clone --recurse-submodules https://github.com/mpodolinsky/agentic-libero && cd agentic-libero
./setup.sh
```

A Gemini API key is read from `GEMINI_API_KEY`, else from `~/.config/gemini/api_key`. Details of the environment
(pinned versions, the MuJoCo 2.3.7 build the simulator relies on): [docs/implementation-notes.md](docs/implementation-notes.md).

## Run

```bash
./eval.sh agent --task-id 0                           # one episode of task 0 (of 0 to 9)
./eval.sh agent --task-id 0 --max-steps 60            # a short check: 6 calls
./eval.sh agent --chunk 20 --clip-stride 1 --model gemini-4-carbon
./eval.sh cosmos3                                     # the policies of WAM-eval are still here
./eval.sh vla-jepa
```

An episode of 520 steps is about 52 calls, 10 to 15 minutes, because a call takes 7 to 25 s.

## The action space

The first version asks the model for the raw 7-D action, as a baseline. Directions to try next:

| action space | what Gemini outputs | needs |
|---|---|---|
| raw 7-D chunk (now) | `[dx, dy, dz, droll, dpitch, dyaw, gripper]` x 10 | nothing |
| end-effector waypoints | a target position and gripper state, turned into deltas by a controller | a small controller |
| skills with targets | `move_to`, `grasp`, `release`, `lift` with a pixel the model points at | pointing, depth from the simulator |
| policy as a tool | a language subgoal for Cosmos3 or VLA-JEPA to execute, and a check of the result | one of the policies in the loop |
| selector or corrector | choose among chunks the policies propose, or override the gripper | the policies in the loop, optionally Cosmos3's predicted video |

## Results

One task and one seed, so these are observations, not results. On task 0 ("put both the alphabet soup and the tomato
sauce in the basket") an episode stopped at step 240 (25 calls, about 8 minutes) had nothing in the basket.

- **What works.** The replies are valid JSON, the model names the objects correctly, uses the axis conventions of the
  prompt consistently, relates the wrist camera to the gripper, remembers its earlier actions, and diagnoses its own
  failures ("the can slipped out during the lift because I moved before the gripper had fully closed").
- **What fails.** It plans descend, close and lift inside one 10-step chunk, so the gripper closes mid-motion and the
  lift starts before the grip is firm. The can was lost in three grasp-lift cycles, and repeating the diagnosis did not
  change the next plan. Later the can was released away from the basket and knocked over.
- **Cost of the conversation.** The prompt grows by about 900 tokens per call (the video and the model's own long
  replies), from 1k to 22k tokens after 25 calls, and latency from 10 s to 30 s or more. Gemini caches part of the
  prefix automatically; explicit caching is not available for `gemini-4-argon`.

## Licence

The code is Apache-2.0. Model weights keep their own licences and are not part of this repository; the Gemini API
has its own terms. LeRobot (Apache-2.0) and LIBERO (MIT) keep theirs.
