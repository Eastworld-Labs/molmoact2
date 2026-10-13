# Changes in this fork (bimanual YAM fine-tuning, serving and evaluation)

Summary of what was added or changed relative to upstream MolmoAct2. Deployment-specific values
(hosts, keys, buckets) are placeholders: `<...>`.

## Serving (`examples/yam/`)

- **`host_server_yam.py`** (HTTP `/act`, default port 8202)
  - `--repo-id` accepts a local converted checkpoint directory; `--norm-tag` selects the tag in that
    checkpoint's `norm_stats.json` (e.g. `eastworlds_yam_data_collection` for our fine-tunes).
  - Opt-in attention capture: send `"return_attention": true` in the `/act` payload. The reply then
    also carries `attention` (per-layer, per-camera maps of how the action expert's cross-attention
    reads the image tokens) and `token_norm` (per-cell image-token norms, used to mask ViT
    high-norm artifact cells). Off by default, so normal latency is unchanged.
  - Optional `"seed"` in the payload seeds the flow-matching noise, so a request can be replayed
    exactly; `--deterministic` makes seeded requests bit-for-bit reproducible (slower). The health
    reply includes `server_identity` (checkpoint, norm tag, dtype, GPU, torch version).
- **`serve_yam_websocket.py`** (new, default port 8203): serves the same `Policy` over the openpi
  websocket protocol (msgpack frames, metadata sent on connect, `GET /healthz`), so an openpi-style
  YAM robot runner (`robot_class`) can drive MolmoAct2 unchanged. Needs `msgpack`. It does **not**
  return attention maps; use the HTTP server for those. Options:
  - `--cuda-graph --num-steps 5`: the speed settings. On an RTX 5080 they take a query from
    ~317 ms to ~180 ms; 5 flow steps matched 10 on held-out replays (error difference ≤ 0.0003 rad).
  - `--smooth-sigma S`: zero-phase Gaussian low-pass along each chunk's arm joints (grippers
    untouched), S in control steps. The model's raw chunks reverse direction ~15× more often than
    the human demonstrations; `--smooth-sigma 1.5` cut reversals ~6× and jerk ~6× offline with no
    change in error vs the recording. No lag, unlike an EMA, because the whole chunk is known.
  - `--fixed-seed N`: same flow noise for every request (unless the client sends `_seed`), so
    consecutive chunks are less likely to switch between modes.
  - `--rtc`: real-time chunking, so the arm keeps moving while the next chunk is computed. Metadata
    then advertises `rtc_enabled`, `execution_horizon` (e), `inference_delay` (d); replies carry
    `raw_actions`/`raw_state`; a request with `_rtc_prev_chunk_left_over` is sampled toward that
    leftover. Implemented in **`molmo_rtc.py`** (new), which patches the loaded model instance,
    not the cached modeling file.
    - `--rtc-method inpaint` (default): forward-only. Each flow step pins the chunk's overlap with
      the previous chunk to its leftover, and the executed overlap is blended from the old plan
      to the new one. Costs about the same as a plain request (~225 ms with the speed settings).
    - `--rtc-method guided`: the RTC paper's VJP guidance, ported from evo. About 2× slower; with
      MolmoAct2's 30-step chunk that forces d≈14, which left the executed actions unconstrained
      and made the arm lurch at every swap on the robot. Kept for reference only.
    - Use e=10, d=9 with inpaint. e+d must be ≤ 30, and d must cover the request latency at 30 Hz
      (d=9 = 300 ms). e=20 brought the lurch back offline.
  ```bash
  uv run python examples/yam/serve_yam_websocket.py --host 0.0.0.0 --port 8203 \
    --repo-id <checkpoint dir or HF repo> --norm-tag <norm tag> \
    --cuda-graph --num-steps 5 --smooth-sigma 1.5 \
    --rtc --rtc-method inpaint --execution-horizon 10 --inference-delay 9 --fixed-seed 0
  ```
  Drop the `--rtc ... --fixed-seed 0` line for chunked execution. The client must match: start it
  with `--rtc` only if the server has `--rtc`.

## Training (`experiments/`)

- **`finetune.sh`** (new): one launcher for `smoke`, `probe`, `lora`, `full` and `depth` modes.
  Handles the data sync (rclone to local NVMe), the environment workarounds, run naming, GPU pinning
  (`GPUS=0,1,2,3`) and resume (`RESUME=<run>`). Header comments document every mode.
  Reads `WANDB_API_KEY` / `WANDB_ENTITY` from the environment or from the gitignored
  `experiments/.wandb_api_key` and `experiments/.wandb_entity`; `.hf_access_token` is optional.
  Force-added past the `*.sh` ignore rule.
  - `MIXTURE=<name>` picks the mixture for any mode; `MIXTURE=yam_unfold_linen` skips the COS sync and
    reads a local dataset (`UNFOLD_DATA_DIR`, default `/opt/dlami/nvme/yam_lerobot`).
  - `lora` mode reads `STEPS` (default 25000), `DBS` (per-GPU batch, default 2) and
    `SAVE_INTERVAL` (default 5000).
  - Fix: `GPUS=<single id>` used to exit silently (`grep -o ','` returns 1 under `set -e`).
    `smoke` is a 1-GPU mode (global batch 1).
- **`launch_scripts/data_mixtures.py`**: new mixture `yam_data_collection` (multi-task, one tag) and
  `MOLMOACT2_LEROBOT_VAL_DATASETS` for held-out validation. Episodes come from
  `launch_scripts/yam_data_collection_split.json`: a per-task 80/20 split by whole episode,
  plus `excluded_episodes` with reasons. `*_depth` companion folders are excluded because
  `olmo/data/lerobot_wrapper.py` can't resolve their tag metadata.
  - New single-task mixture `yam_unfold_linen` (tag `eastworlds_yam_unfold_linen`), split in
    `launch_scripts/yam_unfold_linen_split.json`: 103 linen episodes (hand towel, foot towel,
    pillow case), 95/5 by whole episode stratified by towel type (98 train / 5 val), one prompt
    `unfold linen`. Shares `_build_yam_mixture` with `yam_data_collection`.
- **`scripts/mcap_to_lerobot.py`** (new): converts ROS2 rosbag2 `.mcap` YAM recordings to one
  LeRobot v3.0 dataset plus a split file. 30 Hz causal clock, state = measured joints + gripper,
  action = commanded joints + gripper, left arm first (14-D); cameras `head_cam`/`wrist_left`/
  `wrist_right`. One episode per worker into a shard, then `aggregate_datasets`; resumable
  (finished shards are reused). Uses LeRobot's streaming video encoder: the default PNG-then-encode
  path was ~10× slower and needed 100+ GB of temporary files. Needs `mcap` and `mcap-ros2-support`.
  ```bash
  PYTHONPATH="$PWD/lerobot/src" .venv/bin/python scripts/mcap_to_lerobot.py \
    --root <dir with ep_*/ep_*.mcap> --out <out dir>/unfold_linen_<date> --task-id unfold_linen --workers 8
  ```
- **`lerobot/src/lerobot/datasets/video_utils.py`**: `int(video_stream.base_rate)` fails on PyAV 19
  (`AVRational`); now `int(rate.numerator / rate.denominator)`, which also works on older PyAV.
- **`launch_scripts/train_lerobot.py`**: `--action_val_interval` and
  `--action_val_examples_per_dataset` flags that wire in the validator below.
- **`olmo/eval/action_validator.py`** (new): held-out action validation. Before training and every N
  steps it samples action chunks on val episodes and logs per-task `val/<task>/mse`,
  `mse_vs_hold` (model MSE ÷ MSE of holding the first action; < 1 beats standing still) and
  `accuracy@{tau}`, plus means over tasks. Open-loop only.
- **`olmo/train/{trainer,trainer_config,run_trainer}.py`**: run the validator in the training loop;
  wandb 0.30 compatibility (`wandb.run.get_url()` → `wandb.run.url`, `wandb.finish` without `quiet`).
- **`.gitignore` / `experiments/.gitignore`**: ignore per-user secret files and the local
  `tencent_cos.local.yaml`.

## Offline evaluation tools (`experiments/scripts/`)

- **`yam_replay.py`**: replays a held-out episode open-loop through a server (`--policy molmo` or
  `--policy openpi`), compares predicted chunks with the recording, and logs everything to Rerun
  (`.rrd`) plus per-joint plots. `--compare` overlays earlier runs of other models, `--attn` adds
  attention heatmaps and rings, `--video` burns the overlay into an mp4.
  `--split <json>` picks the train/val split file (default: the 15-task collection split);
  `--num-steps` and `--seed` (query i uses seed+i) let two molmo runs see the same noise.
- **`rtc_eval.py`** (new): offline check of execution settings against a recorded episode. Plays
  `robot_class`'s RTC broker and the plain chunked runner on the same frames through
  `serve_yam_websocket.py --rtc`, and reports per chain: command step at chunk swaps vs a normal
  step (a lurch shows as many times larger), jerk, direction reversals per second vs the recorded
  demonstration, error vs the recording, and request latency vs the `d` budget. `--no-seed` lets the
  server's own noise policy apply, as with the real client. Open loop: chunked swap steps are
  inflated, because the next chunk starts from the recorded pose rather than where the commands
  went.
  ```bash
  PYTHONPATH="$PWD/lerobot/src:$PWD/scripts" .venv/bin/python scripts/rtc_eval.py \
    --repo unfold_linen_<date> --data-root <dir> --episode 68 --max-cycles 60 --no-seed
  ```
- **`yam_calib.py`**: one-off head-camera pose estimate per camera placement (click gripper tips,
  then PnP against forward kinematics) so predicted paths can be projected onto the head cam.
- **`yam_compare_video.py`**: builds a comparison grid video from saved replay runs, no server needed.
- **`evo_attn_server.py`**: wraps a pi0.5 (openpi) policy server and adds opt-in attention capture,
  so both models can be compared with the same viewer.

## Workflow

1. Fine-tune: `cd experiments && ./finetune.sh full 8` (checkpoints in `experiments/checkpoints/<run>/`).
2. Convert a checkpoint to HF format (`python -m olmo.hf_model.convert_molmoact2_to_hf <step dir> <out dir>`).
3. Check it offline with `yam_replay.py` before robot trials (needs the HTTP server from step 4a).
4. Serve it:
   ```bash
   # a) HTTP, for replay / attention maps (port 8202)
   uv run python examples/yam/host_server_yam.py --host 127.0.0.1 --port 8202 \
     --repo-id <checkpoint dir> --norm-tag <norm tag>
   # b) websocket, for the robot runner (port 8203); see the options under Serving
   uv run python examples/yam/serve_yam_websocket.py --host 0.0.0.0 --port 8203 \
     --repo-id <checkpoint dir> --norm-tag <norm tag> --cuda-graph --num-steps 5 --smooth-sigma 1.5
   curl http://<server>:8203/healthz        # -> OK
   ```
   Serve from a desktop GPU over wired Ethernet. On an RTX 3080 Ti Laptop GPU a query took ~1.6 s at
   default settings, longer than a whole 1 s chunk. Over Wi-Fi, uploading the three 480x640 images
   (~2.8 MB per request) took 0.4-1.8 s and tripped the runner's 2 s `--policy-timeout`, which
   pauses the run and holds the arms. Over Ethernet the upload is a few ms.
5. Run it on the robot with the `robot_class` client (separate repo, on the machine wired to the arms):
   ```bash
   git clone git@github.com:Eastworld-Labs/robot_class.git && cd robot_class
   source setup_env.sh --yam                 # creates/activates .venv, installs the i2rt SDK, brings up CAN
   python -m inference.yam.run \
     --left-channel <can_follower_l> --right-channel <can_follower_r> \
     --remote-host <server> --remote-port 8203 \
     --prompt "<task prompt>"                # see the list below
   ```
   Add `--execution-horizon 15` for chunked execution, or `--rtc` if the server was started with
   `--rtc` (the runner then takes e and d from the server; a mismatch is refused). Add
   `--enable-motion` to move the arms. The runner starts paused (`--interactive`); unpause from its
   dashboard. Camera serials come from `configs/yam_cameras.yaml`. Ctrl+C once ramps both arms to
   the home pose unless `--no-home-ramp` is set; it interpolates joints, so clear the workspace first.

## Task prompts

Use the exact training string. The model was fine-tuned on these, and a slug like `stack_cup` or a
paraphrase may behave worse. Dataset folders are under `data-collection/yam/`.

| Prompt | Dataset folder |
|---|---|
| `Build Cup Pyramid` | `build_cup_pyramid_20260928` |
| `Fold Handkerchief` | `fold_handkerchief_20260918` |
| `Peg In Hole` | `peg_in_hole_20260924` |
| `Pick Up Bottle` | `pick_up_bottle_20260917` |
| `Sort Block` | `sort_block_20260922` |
| `Sort Column Toy` | `sort_column_toy_20260922` |
| `Sort Wall Plug` | `sort_wall_plug_20260922` |
| `Stack Cup` | `stack_cup_20260911`, `yam_stack_cup_20260928` |
| `T - Pack Bolt M12` | `t_pack_bolt_m12_20260924` |
| `T - Pack Nut M12` | `t_pack_nut_m12_20260923` |
| `T - Pack Washer M16` | `yam_t_pack_washer_m16_20260923` |
| `T - Packing` | `yam_t_packing_20260924` |
| `Collect Pen` | `yam_collect_pen_20260904` |
| `Handover` | `yam_handover_20260904` |

The single-task unfold-linen model (`yam_unfold_linen` mixture) was trained on one prompt only,
`unfold linen` (lowercase), from `unfold_linen_<date>` converted by `scripts/mcap_to_lerobot.py`.
