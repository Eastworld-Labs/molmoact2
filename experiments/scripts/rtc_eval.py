"""Offline check of real-time chunking (RTC) against a recorded episode.

Plays robot_class's RemoteRTCBroker on recorded frames (open loop: the executed actions don't change the
observations). Needs `examples/yam/serve_yam_websocket.py --rtc` running; e/d come from its metadata.

Runs two chains over the same frames: RTC (submit after e actions, swap d steps later onto new[d:]) and
the plain chunked runner (execute h actions, then a fresh chunk from step 0). For each executed command
stream it reports the step at chunk swaps vs a normal step (max over arm joints, rad), mean |2nd
difference| (jerk) and error vs the recorded actions. A swap step far above a normal step is a lurch.

    PYTHONPATH="$PWD/lerobot/src:$PWD/scripts" .venv/bin/python scripts/rtc_eval.py --repo unfold_linen_20261009 \\
        --data-root /home/robo/Desktop/EW/dataset/lerobot --episode 68 --max-cycles 150
"""

import argparse
import time
from pathlib import Path

import msgpack
import numpy as np
import websockets.sync.client
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from yam_replay import CAMS, FrameStream

ARM = np.r_[0:6, 7:13]  # joint dims (grippers excluded from the jump stats)


def _pack(o):
    return {b"__ndarray__": True, b"data": o.tobytes(), b"dtype": o.dtype.str, b"shape": o.shape} if isinstance(o, np.ndarray) else o


def _unpack(o):
    return np.ndarray(buffer=o[b"data"], dtype=np.dtype(o[b"dtype"]), shape=o[b"shape"]) if b"__ndarray__" in o else o


class Client:
    def __init__(self, url):
        self.ws = websockets.sync.client.connect(url, compression=None, max_size=None, ping_interval=None)
        self.meta = msgpack.unpackb(self.ws.recv(), object_hook=_unpack)

    def infer(self, obs, leftover=None, seed=None):
        obs = dict(obs)
        if leftover is not None:
            obs["_rtc_prev_chunk_left_over"] = leftover
        if seed is not None:
            obs["_seed"] = seed
        t = time.perf_counter()
        self.ws.send(msgpack.packb(obs, default=_pack))
        reply = self.ws.recv()
        if isinstance(reply, str):
            raise RuntimeError(f"server error:\n{reply}")
        out = msgpack.unpackb(reply, object_hook=_unpack)
        out["latency_ms"] = (time.perf_counter() - t) * 1000
        return out


def reversals_per_s(seq, fps=30.0, eps=2e-3):
    """Mean over arm joints of commanded direction changes per second (moves under eps ignored).
    Wobble shows as far more reversals than the recorded demonstration has."""
    d = np.diff(seq, axis=0)
    sgn = np.sign(np.where(np.abs(d) > eps, d, 0))
    out = []
    for j in range(sgn.shape[1]):
        x = sgn[:, j][sgn[:, j] != 0]
        out.append((x[1:] != x[:-1]).sum())
    return float(np.mean(out)) / (len(seq) / fps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--server", default="ws://127.0.0.1:8203")
    p.add_argument("--instruction", default=None)
    p.add_argument("--max-cycles", type=int, default=150)
    p.add_argument("--seed", type=int, default=2000)
    p.add_argument("--no-seed", action="store_true",
                   help="send no per-request seed (like the robot client), so the server's own noise policy applies")
    p.add_argument("--chunked-horizon", type=int, default=15, help="baseline: actions executed per chunk without RTC")
    args = p.parse_args()

    client = Client(args.server)
    if not client.meta.get("rtc_enabled"):
        raise SystemExit("server is not running with --rtc")
    e, d = int(client.meta["execution_horizon"]), int(client.meta["inference_delay"])
    root = Path(args.data_root) / args.repo
    ds = LeRobotDataset(args.repo, root=root, episodes=[args.episode], video_backend="pyav")
    rows = ds.hf_dataset.with_format("numpy")
    gt = np.stack(rows["action"]).astype(np.float32)
    states = np.stack(rows["observation.state"]).astype(np.float32)
    ep_meta = ds.meta.episodes[args.episode]
    instruction = args.instruction or ep_meta["tasks"][0]
    streams = {cam: FrameStream(root / ds.meta.get_video_file_path(args.episode, key),
                                float(ep_meta[f"videos/{key}/from_timestamp"]), ds.meta.fps)
               for cam, key in CAMS.items()}

    def obs_at(k):
        o = {CAMS[cam]: np.ascontiguousarray(s.get(k)) for cam, s in streams.items()}
        o.update({"observation.state": states[k], "prompt": instruction})
        return o

    print(f"episode {args.episode}: {len(gt)} frames, RTC e={e} d={d} ({client.meta.get('rtc_method', 'guided')}), "
          f"chunked e={args.chunked_horizon}, prompt={instruction!r}")
    limit = min(len(gt), args.max_cycles * e + e + d)
    seed_for = (lambda i: None) if args.no_seed else (lambda i: args.seed + i)

    # RTC chain, as robot_class's RemoteRTCBroker runs it.
    first = client.infer(obs_at(0), seed=seed_for(0))
    old, old_raw, t_old = first["actions"], first["raw_actions"], 0
    rtc_cmd, rtc_swaps, lat_rtc = {k: old[k] for k in range(e + d)}, [], []
    for cycle in range(args.max_cycles):
        t = t_old + e  # broker submits after e executed actions
        if t + e + d >= limit:
            break
        leftover = np.zeros_like(old_raw)
        leftover[: len(old_raw) - e] = old_raw[e:]
        new = client.infer(obs_at(t), leftover=leftover, seed=seed_for(1 + cycle))
        lat_rtc.append(new["latency_ms"])
        for i in range(d, e + d):  # the new chunk runs [t+d, t+e+d)
            rtc_cmd[t + i] = new["actions"][i]
        rtc_swaps.append(t + d)
        old, old_raw, t_old = new["actions"], new["raw_actions"], t

    # Chunked chain, as the plain runner: execute h actions, then a fresh chunk from step 0.
    h = args.chunked_horizon
    chk_cmd, chk_swaps, lat_chk = {}, [], []
    for cycle, t in enumerate(range(0, limit - h, h)):
        out = client.infer(obs_at(t), seed=seed_for(1 + cycle))
        lat_chk.append(out["latency_ms"])
        for i in range(h):
            chk_cmd[t + i] = out["actions"][i]
        if t:
            chk_swaps.append(t)
    for s_ in streams.values():
        s_.close()

    def stats(cmd, swaps):
        ks = np.array(sorted(cmd))
        seq = np.stack([cmd[k] for k in ks])[:, ARM]
        steps = np.abs(np.diff(seq, axis=0))  # step k-1 -> k
        at_swap = np.isin(ks[1:], swaps)
        swap, normal = steps[at_swap], steps[~at_swap]
        jerk = np.abs(np.diff(seq, 2, axis=0)).mean()
        err = np.abs(seq - gt[ks][:, ARM]).mean()
        return swap.max(1).mean(), swap.max(1).max(), np.median(normal.max(1)), jerk, err, reversals_per_s(seq)

    print(f"{'':10s} {'swap step (max joint)':>24s} {'normal step':>12s} {'swap/normal':>11s} {'jerk':>8s} "
          f"{'err vs GT':>10s} {'reversals/s':>12s}")
    for name, cmd, swaps in (("RTC", rtc_cmd, rtc_swaps), ("chunked", chk_cmd, chk_swaps)):
        sm, sx, nm, jerk, err, rev = stats(cmd, swaps)
        print(f"{name:10s} mean {sm:.4f} max {sx:.4f} {nm:12.4f} {sm / max(nm, 1e-9):10.1f}x {jerk:8.4f} {err:10.4f} {rev:12.2f}")
    print(f"{'recorded':10s} {'':>24s} {'':>12s} {'':>11s} {'':>8s} {'':>10s} {reversals_per_s(gt[:limit][:, ARM]):12.2f}")
    print(f"latency ms median: RTC {np.median(lat_rtc):.0f} (p90 {np.percentile(lat_rtc, 90):.0f}, budget d={d} steps = "
          f"{d / ds.meta.fps * 1000:.0f} ms) | chunked {np.median(lat_chk):.0f}")

if __name__ == "__main__":
    main()
