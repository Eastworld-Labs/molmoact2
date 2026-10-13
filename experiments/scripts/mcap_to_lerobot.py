"""Convert ROS2 rosbag2 MCAP bimanual YAM episodes into one LeRobot v3.0 dataset.

Same conventions as abc_minimal/export_ros2_mcap.py: a fixed 30 Hz causal clock (each tick takes the
latest message received at or before it), state = measured joints + gripper state, action = commanded
joints + gripper command, left arm first (6 joints, gripper, 6 joints, gripper = 14-D).
Camera packets are h264 and start mid-GOP, so each camera starts at its first keyframe.

    .venv/bin/python scripts/mcap_to_lerobot.py \
        --root <dir with ep_*/ep_*.mcap> --out /path/unfold_linen_20261009 --task-id unfold_linen

Also writes <out>_split.json in launch_scripts/yam_data_collection_split.json's schema (episode indices
= conversion order, whole-episode 95/5 split stratified by towel type); single prompt "unfold linen".
"""

import argparse
import gc
import json
import shutil
import tempfile
from pathlib import Path

import av
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

TICK_NS = 33_333_333
CAMERAS = (
    ("head_cam", "/cameras/head_cam/image"),
    ("wrist_left", "/cameras/wrist_left/image"),
    ("wrist_right", "/cameras/wrist_right/image"),
)
STATE_TOPICS = ("/arms/left/joint_state/measured", "/arms/left/gripper/state",
                "/arms/right/joint_state/measured", "/arms/right/gripper/state")
ACTION_TOPICS = ("/arms/left/joint_state/commanded", "/arms/left/gripper/command",
                 "/arms/right/joint_state/commanded", "/arms/right/gripper/command")
DIM_NAMES = [f"{a}_{j}" for a in ("left", "right") for j in ("j1", "j2", "j3", "j4", "j5", "j6", "gripper")]


def floor_indices(source_ts, target_ts):
    return np.clip(np.searchsorted(source_ts, target_ts, side="right") - 1, 0, len(source_ts) - 1)


def read_bag(path):
    scalar_topics = set(STATE_TOPICS + ACTION_TOPICS)
    cam_topics = {t for _, t in CAMERAS}
    scalars, cams, flags = {}, {}, {}
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, ch, msg, dec in reader.iter_decoded_messages(topics=sorted(scalar_topics | cam_topics)):
            if ch.topic in scalar_topics:
                scalars.setdefault(ch.topic, []).append((msg.log_time, np.array(dec.position, dtype=np.float64)))
            else:
                cams.setdefault(ch.topic, []).append((msg.log_time, bytes(dec.data)))
                flags.setdefault(ch.topic, []).append(int(dec.flags))
    for m in (*scalars.values(), *cams.values()):
        m.sort(key=lambda x: x[0])
    return scalars, cams, flags


def frame_stream(h264_path, frame_idx):
    """Yield the RGB frame for each tick (frame_idx is non-decreasing), decoding sequentially."""
    with av.open(h264_path, format="h264") as container:
        decoder = container.decode(video=0)
        cur, img = -1, None
        for want in frame_idx:
            while cur < want:
                img = next(decoder).to_ndarray(format="rgb24")
                cur += 1
            yield img


def load_episode(ep_dir, work):
    ep = ep_dir.name
    scalars, cams, flags = read_bag(str(ep_dir / f"{ep}.mcap"))
    for t in STATE_TOPICS + ACTION_TOPICS:
        if t not in scalars:
            raise RuntimeError(f"{ep}: no {t}")
    for _, t in CAMERAS:
        if t not in cams:
            raise RuntimeError(f"{ep}: no {t}")
        key = [i for i, fl in enumerate(flags[t]) if fl & 1]
        if not key:
            raise RuntimeError(f"{ep}: {t} has no keyframe")
        cams[t] = cams[t][key[0]:]
    streams = [cams[t] for _, t in CAMERAS] + [scalars[t] for t in STATE_TOPICS + ACTION_TOPICS]
    t0, t_end = max(s[0][0] for s in streams), min(s[-1][0] for s in streams)
    ticks = np.arange(t0 + TICK_NS, t_end + 1, TICK_NS, dtype=np.int64)
    if len(ticks) < 10:
        raise RuntimeError(f"{ep}: too short")

    def stack(topics):
        parts = []
        for t in topics:
            ts = np.array([x for x, _ in scalars[t]], dtype=np.int64)
            parts.append(np.stack([v for _, v in scalars[t]])[floor_indices(ts, ticks)])
        return np.concatenate(parts, axis=-1).astype(np.float32)

    state, action = stack(STATE_TOPICS), stack(ACTION_TOPICS)
    if state.shape[1] != 14 or not (np.isfinite(state).all() and np.isfinite(action).all()):
        raise RuntimeError(f"{ep}: bad state/action {state.shape}")
    paths, idx = {}, {}
    for name, t in CAMERAS:
        p = Path(work) / f"{name}.h264"
        p.write_bytes(b"".join(d for _, d in cams[t]))
        paths[name] = str(p)
        idx[name] = floor_indices(np.array([x for x, _ in cams[t]], dtype=np.int64), ticks)
    return state, action, paths, idx


PROMPT = "unfold linen"


def convert_one(job):
    """Convert one episode into its own single-episode LeRobot dataset (shard) so episodes run in parallel."""
    d, shard, vcodec = job
    info = Path(shard) / "meta" / "info.json"
    if info.exists():  # resume: reuse a shard that finished (finalize writes episodes + frame count)
        meta = json.loads(info.read_text())
        if meta.get("total_episodes") == 1 and meta.get("total_frames", 0) > 0 and (Path(shard) / "videos").exists():
            print(f"[SKIP] {Path(d).name} already converted", flush=True)
            return meta["total_frames"]
    shutil.rmtree(shard, ignore_errors=True)
    try:
        with tempfile.TemporaryDirectory() as work:
            state, action, paths, idx = load_episode(Path(d), work)
            streams = [frame_stream(paths[c], idx[c]) for c, _ in CAMERAS]
            first = [next(s) for s in streams]
            feats = {"observation.state": {"dtype": "float32", "shape": (14,), "names": DIM_NAMES},
                     "action": {"dtype": "float32", "shape": (14,), "names": DIM_NAMES}}
            for (c, _), im in zip(CAMERAS, first):
                feats[f"observation.images.{c}"] = {"dtype": "video", "shape": im.shape,
                                                    "names": ["height", "width", "channels"]}
            ds = LeRobotDataset.create(repo_id=Path(shard).name, fps=30, root=shard,
                                       robot_type="yam_bimanual_follower", features=feats,
                                       use_videos=True, vcodec=vcodec,
                                       streaming_encoding=True, encoder_threads=2, encoder_queue_maxsize=8)
            for i in range(len(state)):
                frame = {"observation.state": state[i], "action": action[i], "task": PROMPT}
                for (c, _), im in zip(CAMERAS, first):
                    frame[f"observation.images.{c}"] = im
                ds.add_frame(frame)
                if i + 1 < len(state):
                    first = [next(s) for s in streams]
            ds.save_episode()
            ds.finalize()
        print(f"[OK] {Path(d).name} {len(state)} frames", flush=True)
        return len(state)
    except Exception as exc:
        print(f"[FAIL] {Path(d).name}: {exc}", flush=True)
        shutil.rmtree(shard, ignore_errors=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--task-id", default="unfold_linen")
    ap.add_argument("--max-episodes", type=int)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--vcodec", default="h264")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    eps = []
    for d in sorted(args.root.glob("ep_*"), key=lambda p: tuple(p.name.split("_")[2:4])):
        lab = json.loads((d / "episode_labels.json").read_text())
        if lab.get("task_id") == args.task_id and d.name not in args.exclude:
            eps.append((d, (lab.get("task_tags") or {}).get("type")))
    if args.max_episodes:
        eps = eps[: args.max_episodes]
    print(f"{len(eps)} episodes, {args.workers} workers")
    shards_dir = args.out.parent / f"{args.out.name}_shards"
    if args.out.exists():
        shutil.rmtree(args.out)
    shards_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(d), str(shards_dir / f"s{n:04d}"), args.vcodec) for n, (d, _) in enumerate(eps)]
    with ProcessPoolExecutor(args.workers) as pool:
        frames = list(pool.map(convert_one, jobs))
    ok = [n for n, f in enumerate(frames) if f]
    manifest = [{"episode": eps[n][0].name, "type": eps[n][1], "frames": frames[n], "prompt": PROMPT} for n in ok]
    print(f"{len(ok)}/{len(eps)} converted; aggregating")
    aggregate_datasets(repo_ids=[f"s{n:04d}" for n in ok], aggr_repo_id=args.out.name,
                       roots=[shards_dir / f"s{n:04d}" for n in ok], aggr_root=args.out)
    shutil.rmtree(shards_dir)

    # whole-episode split stratified by towel type; indices = dataset episode_index (= manifest order)
    rng = np.random.default_rng(args.seed)
    train, val = [], []
    for typ in sorted({m["type"] or "" for m in manifest}):
        ids = [i for i, m in enumerate(manifest) if (m["type"] or "") == typ]
        perm = rng.permutation(ids).tolist()
        nv = max(1, round(len(ids) * args.val_fraction))
        val += perm[:nv]
        train += perm[nv:]
    split = {"seed": args.seed, "val_fraction": args.val_fraction,
             "repos": {args.out.name: {"total_episodes": len(manifest), "train": sorted(train), "val": sorted(val)}},
             "excluded": {}, "excluded_episodes": {}, "manifest": manifest}
    split_path = args.out.parent / f"{args.out.name}_split.json"
    split_path.write_text(json.dumps(split, indent=1))
    print(f"done: {len(manifest)} episodes, {len(train)} train / {len(val)} val -> {args.out}, {split_path}")


if __name__ == "__main__":
    main()
