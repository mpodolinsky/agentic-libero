# agentic-libero

An agentic manipulation harness on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)-10: a Gemini model
controls the robot arm by watching short videos of what just happened and choosing the next actions. It is built on
[WAM-eval](https://github.com/mpodolinsky/WAM-eval), so the Cosmos3 and VLA-JEPA policies can be evaluated here too.

## How it works

`agent.py` runs one episode as one chat with a Gemini model (`gemini-4-argon` by default; `gemini-robotics-er-2-preview`
works too). On every turn:

1. The model gets the last **2 seconds** as a video (agentview and wrist side by side, 10 fps, 480 px cameras) and the
   current image of each camera at full resolution.
2. It answers with JSON: what it sees, a plan, and the next **10 actions** (0.5 s), each the 7-D LIBERO command
   `[dx, dy, dz, droll, dpitch, dyaw, gripper]` in [-1, 1]. These are executed open loop.
3. The chat keeps its earlier answers as text, so it remembers what it did, but the videos and images of earlier turns
   are dropped, and answers older than 3 turns shrink to a one-line summary (net movement and gripper state). The
   prompt therefore stays near 10k tokens (about 95 more per call) instead of growing with every video.

The system prompt explains the action in image terms (which way each axis moves in the agentview, about 1 cm per step
at 1.0, +1 closes the gripper); those numbers were measured in the simulator. Invalid replies are retried once, then
the robot holds still. Everything is written as the run goes, so killing it loses nothing: `outputs/agent/<run>/` has
`calls.jsonl` (every raw reply of the model, with latency and token usage), `clips/` (the media sent on each turn;
deleted when an episode finishes, unless `--keep-clips`), `episode.mp4` (the whole rollout, both cameras), `config.json`
and `summary.json`.

## Setup

Linux x86_64, one NVIDIA GPU for the simulator renderer, and `git`, `uv`, `cmake`, `g++`, `curl`.

```bash
git clone --recurse-submodules https://github.com/mpodolinsky/agentic-libero && cd agentic-libero
./setup.sh
```

Gemini access is keyless by default: the agent calls Vertex AI with your Google Cloud application default credentials
(`gcloud auth application-default login`; the project comes from `--project`, `GOOGLE_CLOUD_PROJECT` or the gcloud
config, and `gcloud` is found on the PATH or in `~/google-cloud-sdk/bin`). `--auth key` uses a Gemini API key instead
(`GEMINI_API_KEY`, else `~/.config/gemini/api_key`). Details of the environment
(pinned versions, the MuJoCo 2.3.7 build the simulator relies on): [docs/implementation-notes.md](docs/implementation-notes.md).

## Run

```bash
./eval.sh agent --task-id 0                           # one episode of task 0 (of 0 to 9)
./eval.sh agent --task-id 0 --max-steps 60            # a short check: 6 calls
./eval.sh agent --model gemini-robotics-er-2-preview --window 3 --media-resolution default
./eval.sh cosmos3                                     # the policies of WAM-eval are still here
./eval.sh vla-jepa
```

An episode of 520 steps is about 52 calls, 10 to 20 minutes, because a call takes 6 to 30 s.

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

All 10 tasks, one episode each, with `gemini-4-argon`: 1 of 10 succeeded; see [results/](results/README.md) for the table
and a GIF of every episode.

Earlier, on task 0 ("put both the alphabet soup and the tomato sauce in the basket"), one seed, one episode per model: these are
observations, not results. Neither model completed the task.

| model | steps run | calls | prompt tokens at the end | seconds per call | what happened |
|---|---|---|---|---|---|
| `gemini-4-argon` | 440 (stopped: API credits ran out) | 44 | 14k | 8 to 26 | grasp, lift, lose the tomato sauce can, six times; once carried it over the basket and released it, but it fell outside |
| `gemini-robotics-er-2-preview` | 520 (step limit) | 52 | 14k | 5 to 11 | hovered above the alphabet soup can and tried to grasp it about 40 times without getting the fingers around it |

- **The models fail differently.** Argon commands full-size movements (about 9 cm net per 10-step chunk) and gets to the
  can, but it closes the gripper in the middle of a chunk and lifts in the next one, before the grip is firm (the
  gripper needs 10 to 15 steps to close), so the can slips out. Robotics-ER commands about 4 times smaller movements
  (1.7 cm per chunk, actions of 0.1 to 0.5) even though the prompt asks for large steps, so it never descends far enough
  and closes on air, then reopens.
- **What works.** Valid JSON replies, correct object names, consistent use of the axis conventions, correct use of the
  wrist camera, memory of its earlier actions, and accurate diagnoses of its own failures. Robotics-ER also announced
  "task done" while the environment reported no success, so success has to come from the simulator.
- **The conversation stays small.** The prompt grows by about 95 tokens per call once the 2 s video window is full, and
  ends near 14k tokens. An earlier version that kept every video in the history reached 22k to 30k after 25 to 30 calls.
- **Cost.** An episode uses about 0.5 to 0.6 million prompt tokens, because the 2 s video and the two images are resent
  on every call.

## Licence

The code is Apache-2.0. Model weights keep their own licences and are not part of this repository; the Gemini API
has its own terms. LeRobot (Apache-2.0) and LIBERO (MIT) keep theirs.
