# G1–box carry SMP data (`smp-reward`)

This branch adds a portable `mimickit_smp` package and an object-aware dataset
adapter for the existing TinyMDM trainer. Runtime data/features require NumPy and
PyTorch, with no FAR-Holosoma or simulator imports. MuJoCo and SciPy are optional
offline-preparation dependencies. The initial class is `carry` (`class_id=0`);
the data comes from human mocap retargeted to G1, so the prior models **G1–box**
interaction rather than SMPL-X poses.

## Window choice

Use **16 frames at 50 Hz**, oldest to newest. The first-to-last span is
`(16 - 1) / 50 = 0.30 s`, the same as `(10 - 1) / 30` for the paper's 10-frame
window and the upstream 30 Hz configuration. Policy frequency does not require
the prior to use the same rate; matching 50 Hz avoids runtime history resampling.
The longer token sequence increases prior computation relative to 10 frames.

The diffusion model receives a noisy version of a motion window and predicts
noise of the same shape. This is not a past-window/future-window predictor or an
action-chunk policy. The reward will score recently **executed** motion.

References: [SMP §6](https://arxiv.org/html/2512.03028v3#S6),
[upstream configuration](../tools/diffusion_model/config/tinymdm_multi_clip.yaml).

## Existing object support and this extension

MimicKit's engines support rigid objects, with examples in
[`task_dodgeball_env.py`](../mimickit/envs/task_dodgeball_env.py) and
[`static_objects_env.py`](../mimickit/envs/static_objects_env.py). Its original
SMP dataset/AMP features contain character state only; dodgeball projectiles do
not become channels of the learned motion prior. There is no ready-made carry
HOI motion loader in the checked-out base revision `2ed1e6c`.

This extension adds box state to the learned motion distribution. It does not
add a simulated carry task, contact-force labels, an online FAR-Holosoma reward
adapter, or object-aware generative state initialization. The stock SMP agent's
character-only history cannot directly consume this checkpoint.

## Data contract

| Stage | Input | Output |
| --- | --- | --- |
| Prepare | Selected UMR manifest, native timestamps, WXYZ root/box poses, 29 named joints, source scene | Portable 50 Hz float32 state arrays and a versioned index |
| Sample | One actor split and a quality policy | 16-state windows entirely within one lifted segment |
| Encode | Named robot state, four hand/foot body origins, box center state/dimensions | `[B, 16, 219]` features; flatten to `[B, 3504]` for TinyMDM |
| Normalize | Training windows only | Frozen per-feature mean/std with schema and dataset hashes |
| Denoise | Noisy normalized features and diffusion timestep | Predicted noise with the same shape |

Preparation linearly interpolates positions and scalar joint angles and uses
shortest-arc SLERP for root/object orientations. It does not smooth trajectories.
World linear and angular velocities use backward differences at 50 Hz; the
first resampled state is dropped so all derivatives use positive-phase samples.
There is no padding, wrapping, extrapolation, or interpolation across segments.
The original native-rate files remain unchanged.

The box pose is recentered at its object-local visual-mesh AABB center, and its
dimensions are **full extents in meters**, after the recorded object scaling.
The original local center offset is recorded. Cached rotations and extents use
the source **mesh axes**, which differ across HiPHI boxes (mesh Y is often up).
They must not be interpreted as an already-standardized Z-up box frame.
Mass, friction and contact forces
are absent because they are not measured by these kinematic trajectories.
Box dimensions are channels of the joint motion distribution in v1, rather than
an external conditioning input to a shape-conditioned denoiser.

**Feature schema v2 canonicalizes the box axes inside `encode_motion`.** In the
final frame's root-heading coordinates, select the signed mesh axis most aligned
with world-up as box Z, then the remaining signed axis most aligned with root
forward as box X. Set Y by `Z × X`. Select once from the final frame and apply
that same signed permutation to every frame of the window; permute the full
extents into matching X/Y/Z order. Geometric ties use physical directions, with
a `1e-6` tolerance, rather than source axis labels. This preserves the cuboid's
geometry, continuous tilt within a window, and world-frame angular velocity.
All 24 proper signed axis relabelings encode identically, including cube faces.
The representation is piecewise smooth: canonical-sector boundaries can still
occur between successive windows. This convention assumes the input cuboid
extents are aligned with its supplied local axes.

The existing v1 **state files and index stay unchanged**. `CarryWindowDataset`
validates their recorded `cache_schema` and exposes the current v2 feature
`schema`. New checkpoints and normalizers carry the v2 hash; v1 checkpoint or
normalizer hashes are rejected by the v2 training/data path. Refit normalization
and retrain a prior when changing this representation.

The shared tensor encoder uses Z-up meters/radians, WXYZ input quaternions, and
6D rotations formed by the rotated X and Z axes (MimicKit's tangent/normal
convention). Heading is fixed to the final root yaw across the window.

| Channels | Count | Frame convention |
| --- | ---: | --- |
| Root position, orientation | 9 | Final root XY origin; world height retained; final root heading |
| 29 hinge-joint rotations | 174 | Local axes recorded in the schema, 6D per joint |
| Four hand/foot body positions | 12 | Each frame's root position; final root heading |
| Root linear/angular velocity | 6 | World spatial velocity rotated into final root heading |
| Box position, orientation | 9 | Same position/heading convention as root; orientation axes canonicalized as above |
| Box linear/angular velocity | 6 | Same velocity convention as root |
| Box dimensions | 3 | Full extents in canonical box X/Y/Z order |
| **Total** | **219** | |

The feature schema records exact joint and body names, local joint axes, robot
kinematics hash, coordinate conventions, frequency and window length. An online
adapter must map simulator tensors by name, convert XYZW if necessary, and use
box-center velocities rather than a displaced center-of-mass velocity. It then
calls the same `encode_motion`. Reset history at episode boundaries; start
scoring only after a complete valid window. Simulator velocities are instantaneous;
source velocities are finite-difference estimates and should be compared during
integration. Shape equality alone is insufficient validation.

## Current prepared dataset

Source: the October 8 HiPHI UMR carry selection, **780 segments, 332 recordings**.
The prepared cache is approximately **214 MiB**:

`/data/sky/projects/MimicKit/datasets/hiphi_umr_carry_50hz_v1/index.json`

| Split | Actors | Segments | Valid windows |
| --- | ---: | ---: | ---: |
| Train | 28 | 699 | 621,725 |
| Validation | 3 | 39 | 111,695 |
| Test | 3 | 42 | 36,182 |
| **Total** | **34** | **780** | **769,602** |

Windows overlap, so these counts do not represent independent examples. Sampling
is uniform over valid window starts, giving longer segments proportionally more
weight. All segments from the same actor stay in the same split. The cache and
index can be copied together; loading them needs no original `/data` source paths
or mesh assets. Absolute source paths are provenance only.

The TinyMDM adapter loads its selected split into one packed state tensor
(about 169 MiB for training, plus 5 MiB of valid-start offsets), then gathers
whole batches without per-sample Python indexing. Sampling order, distribution,
segment boundaries and quality filtering are unchanged. `data_device: cpu`
controls packed storage; feature encoding runs on the model device, including
CUDA when training there. The general metadata-rich DataLoader remains useful
for inspection and worker-process loading.

The user's selected set is included in full using `quality=all`. Original
quality metadata remains: 727 await visual review and 53 have explicit review
flags. Preparation does **not** turn them into approved physical demonstrations.
`quality=unflagged` excludes explicit flags; `quality=approved` requires an
explicit `training_approval=approved` entry and currently yields no data.
Preserved annotations in `source_manifest.json` support later class refinement.

## Use and reproduce

From this checkout, source `scripts/source_smp_reward.sh` to keep caches and
generated outputs under `/data/sky/projects/MimicKit`. Use an environment with
the appropriate dependencies; a partner can install just the portable package
with `pip install -e .`, or `pip install -e '.[prepare]'` for conversion.
MimicKit's existing diffusion requirements are additionally needed for training.

```bash
source scripts/source_smp_reward.sh
python -m mimickit_smp.prepare \
  --manifest /path/to/selected/smp_manifest.json \
  --output "$PROJECT_STORAGE_ROOT/datasets/carry_50hz_new" --workers 8

python -m mimickit_smp.audit \
  --index "$PROJECT_STORAGE_ROOT/datasets/carry_50hz_new/index.json" \
  --output "$PROJECT_STORAGE_ROOT/runs/carry_data_audit_new"
```

Both commands require fresh output directories. The audit verifies every cache
hash, finite state, unit quaternion, phase/time bounds and split; it encodes
first/middle/last windows from every segment and exercises worker-process
collation. Normalization uses 20,000 randomly sampled **training** windows.

```python
from mimickit_smp import CarryWindowDataset, make_dataloader

dataset = CarryWindowDataset("/path/to/index.json", split="train", quality="all")
loader = make_dataloader(dataset, batch_size=256, num_workers=4)
batch = next(iter(loader))
motion = batch["motion"]  # [256, 16, 219], raw feature values
labels = batch["class_id"]  # carry = 0
# Metadata includes trial/actor IDs, original source time and quality status.
```

Train with the existing TinyMDM implementation:

```bash
export SMP_CARRY_DATASET="$PROJECT_STORAGE_ROOT/datasets/hiphi_umr_carry_50hz_v1/index.json"
python tools/diffusion_model/train_tinymdm.py \
  --cfg_path tools/diffusion_model/config/tinymdm_carry_hoi.yaml \
  --out_dir "$PROJECT_STORAGE_ROOT/runs/carry_prior_new" --device cuda
```

The carry configuration sets `normalizer_std_clip: 0.03`, replacing 0.2 so small
hand/foot position deviations are not suppressed as strongly. This is a shared
numeric floor over feature units, not a 3 cm threshold on every channel. The
standalone audit uses the same floor by default.

The adapter samples only training data and validates frequency, window size and
saved schema hashes. The trainer saves resolved dimensions, the feature schema
and dataset provenance alongside the model; its normalizer is fitted on the
training split and stored in the checkpoint. **At reward time use the normalizer
in `model.pt`.** The audit's independently sampled `normalization.json` is for
data diagnostics; matching schema/floor does not make its statistics identical
to a trained model's. The trainer saves generated **feature
windows**, without applying the original character-only inverse conversion.

The first prior is a single-class DiT. Adding class IDs to the existing CondDiT
training/reward calls, a standalone frozen reward bundle, and the live simulator
adapter remain separate work. Data preparation, an actual denoising
forward/backward pass, and a three-iteration CPU training smoke have completed;
full prior training has not been launched.

Local validation artifacts:

- `/data/sky/projects/MimicKit/runs/carry_loader_validation_20261008/validation.json`
- `/data/sky/projects/MimicKit/runs/carry_loader_validation_20261008/normalization.json`
- `/data/sky/projects/MimicKit/runs/carry_tinymdm_smoke_20261008/output/`

The above artifacts record the initial v1 implementation. The v2 review-fix
audit, normalization diagnostics, loading benchmark, and training smoke are
under `/data/sky/projects/MimicKit/runs/carry_review_fixes_20261008/`.
The one-thread CPU benchmark measured 512-window median loading/encoding at
138 ms for the original scalar path and 37 ms for `fetch_obs_demo` with packed
gathering (about 3.7× faster), with exact packed/scalar v2 feature agreement.
The source cache's 780 files still match their original hashes.
All 20 regression tests pass in both `hsmujoco` and `ei_world_model`, including
nonzero box-center resampling offsets in float32/float64. The v2 CPU smoke also
checks checkpoint reload and the saved model normalizer's 0.03 floor.

Tests: `python -m pytest tests/smp_reward` (set its cache directory under `/data`).
