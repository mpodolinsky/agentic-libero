"""An agentic harness: Gemini controls the robot arm in LIBERO-10 from short videos of what just happened.

    ./eval.sh agent --task-id 0

Every call is one turn of a single chat, so the model remembers its own earlier actions. Each turn adds a short video
of the last `--chunk` steps (agentview and wrist side by side) and the model answers with the next `--chunk` actions,
which are executed open loop. Everything is written as the run goes, so killing it loses nothing:
`outputs/agent/<run>/` holds config.json, calls.jsonl (every raw reply), clips/ (the video sent each turn),
episode.mp4 (the whole rollout) and summary.json.

The 7-D action is the LIBERO command `[dx, dy, dz, droll, dpitch, dyaw, gripper]` in [-1, 1].
"""

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(REPO / ".libero"))

import imageio_ffmpeg  # noqa: E402  (after the environment variables LIBERO and MuJoCo read at import)
from lerobot.envs import make_env  # noqa: E402
from lerobot.envs.configs import LiberoEnv  # noqa: E402
from lerobot.envs.utils import NEW_ROLLOUT_OPTION  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
FPS = 20  # control rate of the simulator

# What an action of 1.0 does, measured in the simulator (about 1 cm and 3 to 5 degrees per step, from rest).
SYSTEM = """You control a Franka Panda robot arm in a tabletop simulator (LIBERO) and must complete this task:
"{task}"

You act in rounds. In every round you see a short video of what happened since your last action chunk, and you
answer with the next {chunk} actions, which are executed one after the other at {fps} Hz ({seconds:.1f} s in total)
without you seeing the result. Your earlier answers are in this conversation. The episode ends after {max_steps} steps.

THE VIDEO. Left: the agentview camera, a fixed camera in front of the robot, looking at the table. Right: the wrist
camera, mounted on the gripper (the gripper fingers show at its bottom). The number in the corner is the simulator step.

THE ACTION. Seven numbers in [-1, 1]: [dx, dy, dz, droll, dpitch, dyaw, gripper].
- dx, dy, dz move the gripper. 1.0 moves it about 1 cm per step (about 10 cm over a chunk of 10 steps).
  In the agentview image: +dx moves the gripper toward the camera (down in the image), -dx away from it (up);
  +dy moves it to the left in the image, -dy to the right; +dz moves it up, -dz down toward the table.
- droll, dpitch, dyaw rotate the gripper (1.0 is about 4 degrees per step). Keep them at 0 unless you must turn.
- gripper: -1 opens the fingers, +1 closes them. Keep the same value while you carry an object, or it falls.
Actions are smoothed by a controller, so motion lags a little and keeps going for a step or two after you stop.

HOW TO WORK. Say what you see and the plan for the next chunk, then give the actions. Move in steps that fit the
distance (several steps of about 0.5 to 1.0 per axis), slow down near objects, grasp with the object between the
fingers (check the wrist camera), and lift before moving sideways. If the last chunk did not do what you intended,
correct it.

Answer with JSON: "observation" (what you see now, including where the gripper is relative to the objects),
"plan" (what the next chunk should achieve), and "actions" (a list of {chunk} lists of 7 numbers)."""


def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        key = Path("~/.config/gemini/api_key").expanduser().read_text()
    return key.strip()


def schema(chunk: int) -> dict:
    action = {"type": "ARRAY", "items": {"type": "NUMBER"}, "minItems": 7, "maxItems": 7}
    return {
        "type": "OBJECT",
        "properties": {
            "observation": {"type": "STRING"},
            "plan": {"type": "STRING"},
            "actions": {"type": "ARRAY", "items": action, "minItems": chunk, "maxItems": chunk},
        },
        "required": ["observation", "plan", "actions"],
        "propertyOrdering": ["observation", "plan", "actions"],  # reason first, then act
    }


def generate(body: dict, model: str, key: str, retries: int = 4) -> dict:
    """One generateContent call; backs off on rate limits and server errors."""
    request = urllib.request.Request(
        API.format(model=model), data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )  # fmt: skip
    for attempt in range(retries + 1):
        try:
            return json.load(urllib.request.urlopen(request, timeout=300))
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise RuntimeError(f"Gemini API error {e.code}: {e.read().decode()[:300]}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
        time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def upright(frame: np.ndarray) -> np.ndarray:
    """LIBERO renders upside down; the policies' preprocessing rotates the images by 180 degrees, so do we."""
    return frame[::-1, ::-1]


def tile(obs: dict, step: int) -> np.ndarray:
    """Agentview and wrist side by side, with the step number in the corner of each."""
    out = []
    for key in ("image", "image2"):
        image = Image.fromarray(np.ascontiguousarray(upright(obs["pixels"][key][0])))
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, 40, 12], fill=(0, 0, 0))
        draw.text((3, 0), str(step), fill=(255, 255, 255))
        out.append(np.array(image))
    return np.concatenate(out, axis=1)


def encode(frames: list[np.ndarray], fps: float) -> bytes:
    """H.264 mp4 of RGB frames."""
    h, w = frames[0].shape[:2]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "clip.mp4"
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)]  # fmt: skip
        subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), check=True)
        return out.read_bytes()


def video_part(data: bytes, fps: float) -> dict:
    """A video part. Gemini samples video at about 1 fps unless `fps` is given."""
    return {"inline_data": {"mime_type": "video/mp4", "data": base64.b64encode(data).decode()},
            "video_metadata": {"fps": fps}}  # fmt: skip


def stream_video(path: Path, frame: np.ndarray) -> subprocess.Popen:
    """An ffmpeg process writing the rollout to `path` as frames are piped in. The mp4 is fragmented, so it plays
    even if the run is killed; ffmpeg finishes the file when its input closes."""
    h, w = frame.shape[:2]
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
           "-pix_fmt", "yuv420p", "-g", str(FPS), "-movflags", "frag_keyframe+empty_moov", str(path)]  # fmt: skip
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def append(path: Path, record: dict) -> None:
    """One JSON line, flushed to disk at once."""
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def parse_actions(response: dict, chunk: int) -> tuple[np.ndarray | None, dict, str]:
    """The (clipped) action chunk of a reply, the reply as a dict, and an error message if it is unusable."""
    texts = [p["text"] for p in response["candidates"][0]["content"]["parts"] if "text" in p and not p.get("thought")]
    try:
        reply = json.loads("".join(texts))
        actions = np.asarray(reply["actions"], dtype=float)
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
        return None, {}, f"not valid JSON with an 'actions' list ({type(e).__name__})"
    if actions.shape != (chunk, 7) or not np.isfinite(actions).all():
        return None, reply, f"'actions' must be {chunk} lists of 7 finite numbers, got shape {actions.shape}"
    return np.clip(actions, -1.0, 1.0), reply, ""


def run(args: argparse.Namespace) -> dict:
    run_dir = args.output_dir or REPO / "outputs" / "agent" / time.strftime(f"task{args.task_id}_%Y%m%d_%H%M%S")
    (run_dir / "clips").mkdir(parents=True, exist_ok=True)
    cfg = LiberoEnv(task="libero_10", task_ids=[args.task_id], obs_type="pixels", observation_height=256,
                    observation_width=256, episode_length=args.max_steps)  # fmt: skip
    env = next(iter(next(iter(make_env(cfg, n_envs=1, use_async_envs=False).values())).values()))
    obs, _ = env.reset(seed=[args.seed], options={NEW_ROLLOUT_OPTION: True})
    task = list(env.call("task_description"))[0]
    key = api_key()

    system = SYSTEM.format(task=task, chunk=args.chunk, fps=FPS, seconds=args.chunk / FPS, max_steps=args.max_steps)
    config = {"systemInstruction": {"parts": [{"text": system}]},
              "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema(args.chunk)}}  # fmt: skip
    if args.thinking_budget is not None:
        config["generationConfig"]["thinkingConfig"] = {"thinkingBudget": args.thinking_budget}
    (run_dir / "config.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "task": task,
                                                     "request_config": config}, indent=1))  # fmt: skip

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # `kill` ends the run through the finally block below
    frame = tile(obs, 0)
    video = stream_video(run_dir / "episode.mp4", frame)
    video.stdin.write(frame.tobytes())
    recent, n_new = [frame], 0  # the frames of the last chunk, and the steps in it
    step, turns, calls, tokens, success, note = 0, [], 0, {"prompt": 0, "output": 0}, False, ""
    gripper = -1.0  # the last gripper command, held if a reply is unusable
    started, status = time.time(), "running"
    print(f"task {args.task_id}: {task}", flush=True)
    try:
        while step < args.max_steps and not success:
            # The first turn shows the starting scene; later turns the video of the last chunk.
            if step == 0:
                data, fps, text = encode([recent[0]] * 2, 1), 1, "The episode starts. This is the scene."
            else:
                shown = recent[:: args.clip_stride]  # from the scene before the chunk, every n-th step
                data, fps = encode(shown, FPS / args.clip_stride), FPS / args.clip_stride
                text = f"{note}Steps {step - n_new} to {step} just ran. This is what happened."
            clip = f"clips/turn_{len(turns) // 2:03d}.mp4"
            (run_dir / clip).write_bytes(data)
            turn = {"role": "user", "parts": [{"text": text}, video_part(data, fps),
                                              {"text": f"Give the next {args.chunk} actions."}]}  # fmt: skip

            actions, reply, error = None, {}, ""
            for attempt in range(2):  # one retry with the same request if the reply is unusable
                t0 = time.time()
                response = generate({**config, "contents": [*turns, turn]}, args.model, key)
                latency = time.time() - t0
                usage = response.get("usageMetadata", {})
                tokens["prompt"] += usage.get("promptTokenCount", 0)
                tokens["output"] += usage.get("candidatesTokenCount", 0)
                calls += 1
                actions, reply, error = parse_actions(response, args.chunk)
                append(run_dir / "calls.jsonl", {  # everything the model sent back, as received
                    "call": calls, "attempt": attempt, "step": step, "clip": clip, "user_text": [text],
                    "latency_s": round(latency, 1), "usage": usage, "error": error, "response": response})  # fmt: skip
                print(f"  call {calls} @ step {step}: {latency:.0f}s, {usage.get('promptTokenCount')} prompt tokens"
                      f"{', ' + error if error else ''} | {reply.get('plan', '')[:110]}", flush=True)
                if actions is not None:
                    break
            if actions is None:  # hold still, keep the gripper where it was
                actions = np.array([[0, 0, 0, 0, 0, 0, gripper]] * args.chunk, dtype=float)
                note = "Your last reply could not be used, so the robot held still. "
            else:
                note, gripper = "", float(actions[-1, 6])
                turns += [turn, response["candidates"][0]["content"]]  # the model's own turn goes back verbatim

            recent, n_new = recent[-1:], 0
            for a in actions:
                obs, _, terminated, truncated, info = env.step(a[None].astype(np.float32))
                step, n_new = step + 1, n_new + 1
                recent.append(tile(obs, step))
                video.stdin.write(recent[-1].tobytes())
                success = bool(np.any(info.get("is_success", False)))
                if success or bool(np.any(terminated)) or bool(np.any(truncated)) or step >= args.max_steps:
                    break
            video.stdin.flush()
            if success or step >= args.max_steps:
                break
        status = "success" if success else "step limit"
    except BaseException as e:  # a kill, Ctrl-C or an API error: keep what was recorded
        status = f"interrupted: {type(e).__name__}"
        raise
    finally:
        video.stdin.close()
        video.wait()
        summary = {"task_id": args.task_id, "task": task, "status": status, "success": success, "steps": step,
                   "calls": calls, "tokens": tokens, "wall_s": round(time.time() - started), "model": args.model,
                   "chunk": args.chunk, "clip_stride": args.clip_stride, "seed": args.seed}  # fmt: skip
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task-id", type=int, default=0, help="LIBERO-10 task (0-9)")
    ap.add_argument("--chunk", type=int, default=10, help="actions per call")
    ap.add_argument("--clip-stride", type=int, default=2, help="show every n-th step of the last chunk in the video")
    ap.add_argument("--max-steps", type=int, default=520, help="episode length (the LIBERO-10 limit)")
    ap.add_argument("--model", default="gemini-4-argon")
    ap.add_argument("--thinking-budget", type=int, default=None, help="thinking tokens (default: the model's choice)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=None)
    run(ap.parse_args())
