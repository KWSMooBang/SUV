# SUV → WorldLens adapter

Evaluates SUV's generated future front-camera video with
[WorldLens](https://github.com/worldbench/worldlens) (pinned `fa6c5ff`) Generation metrics.
WorldLens targets 6-view nuScenes generators; SUV predicts one front camera
(NAVSIM `cam_f0`, 384×640, 2 Hz, 8 future frames) plus depth/seg/instance streams and a trajectory.
For which of the 24 WorldLens dimensions apply, see the research note `wl-integrate-dims.md`.

Nothing here modifies `src/suv` or `experiments/navsim*`; `suv_rollout.py` subclasses
`SUVNavsimV1Agent` and calls `infer_action_slot(..., return_slot_videos=True)`.

## Layout

| file | env | role |
|---|---|---|
| `layout.py` | both | raw dump format (`<token>/streams.npz` + `meta.json`) and WorldLens tree helpers |
| `suv_rollout.py` | SUV (py3.10, torch 2.7, navsim v1) | NAVSIM scenes → SUV rollout → raw dumps (RGB/depth/seg/instance + GT cam_f0 frames + pred/GT trajectory) |
| `export_worldlens.py` | either | raw dumps → `generated_results/{<method>,gt}/video_submission/<token>_gen0/CAM_FRONT.mp4` |
| `run_metrics.py` | WorldLens (py3.9, torch 2.1+cu118) | FVD (NAVSIM GT), Temporal Consistency (CLIP), Subject Consistency (DINO) |
| `make_dummy.py` | either | synthetic dumps for a checkpoint-free smoke |
| `scripts/run_navtest_a.sh` | — | protocol (a) end to end: `rollout export metrics` |
| `scripts/smoke_dummy.sh` | — | dummy pipeline smoke |
| `env/install_worldlens_env.sh` | — | WorldLens env: `base` (conda + nvcc 11.8), `gen` (Generation only, no mmcv), `mm` (mmcv ops etc.) |

## Run

```bash
# Generation-only env (enough for everything in this directory)
bash experiments/worldlens/env/install_worldlens_env.sh base
bash experiments/worldlens/env/install_worldlens_env.sh gen

# protocol (a) on a few navtest scenes (needs SUV ckpt + v1 text embeddings, see run_pdm_score.sh)
MAX_SCENES=8 bash experiments/worldlens/scripts/run_navtest_a.sh
# subsample the full split
STRIDE=10 MAX_SCENES=0 bash experiments/worldlens/scripts/run_navtest_a.sh
```

Paths default to `/data-work/SUV/{SUV_ckpt,navsim,text_embeds/v1,worldlens/pretrained_models}`;
override with `DATA_ROOT`, `CKPT_LOCAL_DIR`, `TEXT_EMB`, `OPENSCENE_DATA_ROOT`, `WL_PRETRAINED`, `OUT`.

## Deviations from stock WorldLens (deliberate)

- **FVD**: the GT set is the NAVSIM future clips exported next to the predictions instead of the nuScenes
  12 Hz info pkl; clips shorter than 16 frames are nearest-neighbour upsampled to 16 frames, because the I3D
  needs at least 16. Same resize/crop as stock (225×400 → 224×400). With fewer than about 400 clips the
  covariance is rank-deficient, so treat FVD as a smoke number only.
- **Subject Consistency**: DINO weights are loaded from `pretrained_models/dino/` (stock code ignores
  `local_save_path` and downloads from the internet).
- Temporal and Subject Consistency are otherwise the stock classes, including the stock
  `gt_tji` bug (`S_tji ≡ 1`), so numbers stay comparable with the stock code.
- `--clip history_future` prepends the 4 real history frames to both sides (12-frame clips).

## Sanity checks

- `pred == GT` on real NAVSIM clips gives FVD ≈ 0 and TS = ACM (S_acm = MRS = 1).
- The GT history frames produced by `gt_front_frames` are bit-identical to the model's input features.
