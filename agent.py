"""An agentic harness: Gemini controls the robot arm in LIBERO-10 from short videos of what just happened.

    ./eval.sh agent --task-id 0

Every call is one turn of a single chat. Each turn shows the model the last `--window` seconds as a video (agentview
and wrist side by side) plus the current image of each camera, and the model answers with the next `--chunk` actions,
which are executed open loop. The chat keeps the model's own answers as text; the videos and images of earlier turns are
dropped, so the prompt grows slowly. Everything is written as the run goes, so killing it loses nothing:
`outputs/agent/<run>/` holds config.json, calls.jsonl (every raw reply), clips/ (the media sent each turn, deleted
when an episode finishes unless --keep-clips), episode.mp4 (the whole rollout) and summary.json.

The 7-D action is the LIBERO command `[dx, dy, dz, droll, dpitch, dyaw, gripper]` in [-1, 1].
"""

import argparse
import base64
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
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

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"  # with an API key
VERTEX = "https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/publishers/google/models/{model}:generateContent"
FPS = 20  # control rate of the simulator

# What an action of 1.0 does, measured in the simulator (about 1 cm and 3 to 5 degrees per step, from rest).
SYSTEM = """You control a Franka Panda robot arm in a tabletop simulator (LIBERO) and must complete this task:
"{task}"

You act in rounds. In every round you get the last {window:.1f} seconds as a video (agentview and wrist side by side,
the newest frame last, the simulator step in the corner of each view), then the current agentview and wrist images at
full resolution. You answer with the next {chunk} actions, which are executed one after the other at {fps} Hz
({seconds:.1f} s in total) without you seeing the result. Your earlier answers stay in this conversation as text, but
their videos and images do not: rely on your own answers and the current images. The episode ends after {max_steps} steps.

THE VIEWS. Agentview: a fixed camera in front of the robot, looking at the table. Wrist: a camera mounted on the
gripper; the gripper fingers show at the bottom of its image.

THE ACTION. Seven numbers in [-1, 1]: [dx, dy, dz, droll, dpitch, dyaw, gripper].
- dx, dy, dz move the gripper. 1.0 moves it about 1 cm per step (about 10 cm over a chunk of 10 steps).
  In the agentview image: +dx moves the gripper toward the camera (down in the image), -dx away from it (up);
  +dy moves it to the left in the image, -dy to the right; +dz moves it up, -dz down toward the table.
- droll, dpitch, dyaw rotate the gripper (1.0 is about 4 degrees per step). Keep them at 0 unless you must turn.
- gripper: -1 opens the fingers, +1 closes them. Keep the same value while you carry an object, or it falls.
Actions are smoothed by a controller, so motion lags a little and keeps going for a step or two after you stop.

HOW TO WORK. Far from an object, use large steps (0.8 to 1.0 on the axes you need to move); slow down only for the last
few centimetres. Check the wrist camera before you close the gripper: the object must be between the fingers. If the
last chunk did not do what you intended, correct it.

Answer with JSON and keep it short: "observation" (one sentence, at most 25 words: where the gripper is relative to the
target), "plan" (one sentence, at most 25 words) and "actions" ({chunk} lists of 7 numbers, one decimal each)."""


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
            "observation": {"type": "STRING", "description": "one sentence, at most 25 words"},
            "plan": {"type": "STRING", "description": "one sentence, at most 25 words"},
            "actions": {"type": "ARRAY", "items": action, "minItems": chunk, "maxItems": chunk},
        },
        "required": ["observation", "plan", "actions"],
        "propertyOrdering": ["observation", "plan", "actions"],  # reason first, then act
    }


def gcloud() -> str:
    return shutil.which("gcloud") or str(Path("~/google-cloud-sdk/bin/gcloud").expanduser())


def endpoint(args: argparse.Namespace) -> tuple[str, Callable[[], dict]]:
    """The generateContent URL and a function that gives the headers of a request. By default Vertex AI with the
    application default credentials of `gcloud` (no key); with `--auth key`, the Gemini API with an API key."""
    if args.auth == "key":
        key = api_key()
        return API.format(model=args.model), lambda: {"x-goog-api-key": key}
    project = args.project or os.environ.get("GOOGLE_CLOUD_PROJECT") or subprocess.run(
        [gcloud(), "config", "get-value", "project"], capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
    token = {"value": "", "time": 0.0}

    def headers() -> dict:
        if time.time() - token["time"] > 1800:  # access tokens last an hour
            token["value"] = subprocess.run([gcloud(), "auth", "application-default", "print-access-token"],
                                            capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
            token["time"] = time.time()
        return {"Authorization": f"Bearer {token['value']}", "x-goog-user-project": project}

    return VERTEX.format(project=project, model=args.model), headers


def generate(body: dict, url: str, headers: Callable[[], dict], retries: int = 4) -> dict:
    """One generateContent call; backs off on rate limits and server errors."""
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", **headers()})  # fmt: skip
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


def views(obs: dict) -> list[np.ndarray]:
    """Agentview and wrist, upright."""
    return [np.ascontiguousarray(upright(obs["pixels"][key][0])) for key in ("image", "image2")]


def tile(vs: list[np.ndarray], step: int) -> np.ndarray:
    """The two views side by side, with the step number in the corner of each."""
    out = []
    for v in vs:
        image = Image.fromarray(v)
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, 56, 18], fill=(0, 0, 0))
        draw.text((4, 3), str(step), fill=(255, 255, 255))
        out.append(np.array(image))
    return np.concatenate(out, axis=1)


def jpeg(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, "JPEG", quality=92)
    return buffer.getvalue()


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


MEDIA_RESOLUTION = {"default": None, "low": "MEDIA_RESOLUTION_LOW", "high": "MEDIA_RESOLUTION_HIGH"}


def summarise(actions: np.ndarray) -> str:
    """One line for an old chunk: the net movement in cm (an action of 1.0 is about 1 cm) and the gripper."""
    x, y, z = actions[:, :3].sum(axis=0)
    closed = actions[:, 6] > 0
    gripper = "closed" if closed.all() else "open" if not closed.any() else (
        f"{'closed' if closed[-1] else 'opened'} at action {int(np.argmax(closed != closed[0])) + 1}")
    return f"net move x {x:+.0f} cm, y {y:+.0f} cm, z {z:+.0f} cm; gripper {gripper}"


def history_contents(entries: list[tuple], full: int) -> list[dict]:
    """The chat so far as text: the last `full` turns with the model's reply in full, older ones with the actions
    replaced by a one-line summary. Each entry is (user text, reply, clipped actions)."""
    contents = []
    for i, (text, reply, actions) in enumerate(entries):
        if i >= len(entries) - full:
            answer = {**reply, "actions": np.round(actions, 1).tolist()}
        else:
            answer = {"observation": reply.get("observation", ""), "plan": reply.get("plan", ""),
                      "executed": summarise(actions)}  # fmt: skip
        contents += [{"role": "user", "parts": [{"text": text}]},
                     {"role": "model", "parts": [{"text": json.dumps(answer, separators=(",", ":"))}]}]  # fmt: skip
    return contents


def run(args: argparse.Namespace) -> dict:
    run_dir = args.output_dir or REPO / "outputs" / "agent" / time.strftime(f"task{args.task_id}_%Y%m%d_%H%M%S")
    (run_dir / "clips").mkdir(parents=True, exist_ok=True)
    cfg = LiberoEnv(task="libero_10", task_ids=[args.task_id], obs_type="pixels", observation_height=args.image_size,
                    observation_width=args.image_size, episode_length=args.max_steps)  # fmt: skip
    env = next(iter(next(iter(make_env(cfg, n_envs=1, use_async_envs=False).values())).values()))
    obs, _ = env.reset(seed=[args.seed], options={NEW_ROLLOUT_OPTION: True})
    task = list(env.call("task_description"))[0]
    url, headers = endpoint(args)

    window = max(1, round(args.window * FPS))  # env steps of the video
    system = SYSTEM.format(task=task, window=args.window, chunk=args.chunk, fps=FPS, seconds=args.chunk / FPS,
                           max_steps=args.max_steps)  # fmt: skip
    generation = {"responseMimeType": "application/json", "responseSchema": schema(args.chunk)}
    if MEDIA_RESOLUTION[args.media_resolution]:
        generation["mediaResolution"] = MEDIA_RESOLUTION[args.media_resolution]
    if args.thinking_budget is not None:
        generation["thinkingConfig"] = {"thinkingBudget": args.thinking_budget}
    config = {"systemInstruction": {"parts": [{"text": system}]}, "generationConfig": generation}
    (run_dir / "config.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "task": task,
                                                     "request_config": config}, indent=1))  # fmt: skip

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # `kill` ends the run through the finally block below
    current = views(obs)
    frame = tile(current, 0)
    video = stream_video(run_dir / "episode.mp4", frame)
    video.stdin.write(frame.tobytes())
    recent, n_new = [frame], 0  # the last `window + 1` tiles, and the steps of the last chunk
    history, calls, tokens, success, note, step = [], 0, {"prompt": 0, "output": 0}, False, "", 0  # history: entries
    gripper = -1.0  # the last gripper command, held if a reply is unusable
    started, status = time.time(), "running"
    print(f"task {args.task_id}: {task}", flush=True)
    try:
        while step < args.max_steps and not success:
            turn_id, files = len(history), []
            parts = []
            if step == 0:
                parts.append({"text": "The episode starts. These are the two cameras."})
            else:
                parts.append({"text": f"{note}Steps {step - n_new} to {step} just ran. Here are the last "
                                      f"{(len(recent) - 1) / FPS:.1f} s, then the current images."})  # fmt: skip
                shown = recent[:: -args.clip_stride][::-1]  # every n-th step, ending with the current one
                data = encode(shown, FPS / args.clip_stride)
                files.append(f"clips/turn_{turn_id:03d}.mp4")
                (run_dir / files[-1]).write_bytes(data)
                parts.append(video_part(data, FPS / args.clip_stride))
            text = parts[0]["text"]
            if args.stills:
                for name, image in zip(("agentview", "wrist"), current):
                    data = jpeg(image)
                    files.append(f"clips/turn_{turn_id:03d}_{name}.jpg")
                    (run_dir / files[-1]).write_bytes(data)
                    parts += [{"text": f"Current {name}:"},
                              {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(data).decode()}}]  # fmt: skip
            parts.append({"text": f"Give the next {args.chunk} actions."})
            turn = {"role": "user", "parts": parts}

            actions, reply, error = None, {}, ""
            for attempt in range(2):  # one retry with the same request if the reply is unusable
                t0 = time.time()
                response = generate({**config, "contents": [*history_contents(history, args.full_turns), turn]}, url, headers)
                latency = time.time() - t0
                usage = response.get("usageMetadata", {})
                tokens["prompt"] += usage.get("promptTokenCount", 0)
                tokens["output"] += usage.get("candidatesTokenCount", 0)
                calls += 1
                actions, reply, error = parse_actions(response, args.chunk)
                append(run_dir / "calls.jsonl", {  # everything the model sent back, as received
                    "call": calls, "attempt": attempt, "step": step, "media": files, "user_text": text,
                    "latency_s": round(latency, 1), "usage": usage, "error": error, "response": response})  # fmt: skip
                print(f"  call {calls} @ step {step}: {latency:.0f}s, {usage.get('promptTokenCount')} prompt tokens, "
                      f"{usage.get('candidatesTokenCount')} out{', ' + error if error else ''} | "
                      f"{reply.get('plan', '')[:100]}", flush=True)  # fmt: skip
                if actions is not None:
                    break
            if actions is None:  # hold still, keep the gripper where it was
                actions = np.array([[0, 0, 0, 0, 0, 0, gripper]] * args.chunk, dtype=float)
                note = "Your last reply could not be used, so the robot held still. "
            else:
                note, gripper = "", float(actions[-1, 6])
                # The history keeps the text of each turn and the model's own reply, without the media and without the
                # thought signature (it carries hidden thinking, charged again when replayed). calls.jsonl keeps the raw reply.
                history.append((text, reply, actions))

            n_new = 0
            for a in actions:
                obs, _, terminated, truncated, info = env.step(a[None].astype(np.float32))
                step, n_new = step + 1, n_new + 1
                current = views(obs)
                recent = [*recent[-window:], tile(current, step)]
                video.stdin.write(recent[-1].tobytes())
                success = bool(np.any(info.get("is_success", False)))
                if success or bool(np.any(terminated)) or bool(np.any(truncated)) or step >= args.max_steps:
                    break
            video.stdin.flush()
        status = "success" if success else "step limit"
    except BaseException as e:  # a kill, Ctrl-C or an API error: keep what was recorded
        status = f"interrupted: {type(e).__name__}"
        raise
    finally:
        video.stdin.close()
        video.wait()
        summary = {"task_id": args.task_id, "task": task, "status": status, "success": success, "steps": step,
                   "calls": calls, "tokens": tokens, "wall_s": round(time.time() - started), "model": args.model,
                   "chunk": args.chunk, "window_s": args.window, "clip_stride": args.clip_stride,
                   "stills": args.stills, "image_size": args.image_size,
                   "media_resolution": args.media_resolution, "seed": args.seed}  # fmt: skip
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        if status in ("success", "step limit") and not args.keep_clips:  # an interrupted run keeps its clips
            shutil.rmtree(run_dir / "clips", ignore_errors=True)
        print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task-id", type=int, default=0, help="LIBERO-10 task (0-9)")
    ap.add_argument("--chunk", type=int, default=10, help="actions per call")
    ap.add_argument("--window", type=float, default=2.0, help="seconds of video shown on every call")
    ap.add_argument("--clip-stride", type=int, default=2, help="show every n-th step of the window in the video")
    ap.add_argument("--full-turns", type=int, default=3, help="recent turns kept with all 10 actions in the history")
    ap.add_argument("--keep-clips", action="store_true", help="keep clips/ after a finished episode (default: delete)")
    ap.add_argument("--no-stills", dest="stills", action="store_false", help="do not send the current camera images")
    ap.add_argument("--image-size", type=int, default=480, help="camera resolution (the policies use 256)")
    ap.add_argument("--media-resolution", choices=list(MEDIA_RESOLUTION), default="high",
                    help="Gemini tokens per image or video frame (high: 262 per video frame, default: 64)")
    ap.add_argument("--max-steps", type=int, default=520, help="episode length (the LIBERO-10 limit)")
    ap.add_argument("--model", default="gemini-4-argon")
    ap.add_argument("--auth", choices=["adc", "key"], default="adc",
                    help="adc: Vertex AI with the gcloud application default credentials (no key); key: Gemini API key")
    ap.add_argument("--project", default=None, help="Google Cloud project (default: GOOGLE_CLOUD_PROJECT, else gcloud's)")
    ap.add_argument("--thinking-budget", type=int, default=None, help="thinking tokens (default: the model's choice)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=None)
    run(ap.parse_args())
