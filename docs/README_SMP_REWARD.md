# G1–box carry SMP data (`smp-reward`)

This branch adds a portable `mimickit_smp` package and an object-aware dataset
adapter for the existing TinyMDM trainer. Runtime data/features require NumPy and
PyTorch, with no FAR-Holosoma or simulator imports. MuJoCo and SciPy are optional
offline-preparation dependencies. Classes are skills: `carry` (`class_id=0`),
`push` (1) and `slide` (2), see [Skills](#skills-carry-push-and-slide-2026-10-11);
the data comes from human mocap retargeted to G1, so the prior models **G1–box**
interaction rather than SMPL-X poses.

Current contract (feature schema **v3**): the prior denoises robot and box motion
together and receives the canonical **box size as a clean condition**. A
**segment** cache adds each lift's pickup and set-down context to the same carry
class, so the prior also covers lifting a box off the floor and putting it down.

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
| Sample | One actor split, a quality policy and a stage policy | 16-state windows entirely within one segment |
| Encode | Named robot state, four hand/foot body origins, box center state/dimensions | `[B, 16, 216]` motion (flatten to `[B, 3456]` for TinyMDM) and a `[B, 3]` box-size condition |
| Normalize | Training windows only | Frozen per-feature and per-condition mean/std with schema and dataset hashes |
| Denoise | Noisy normalized motion, diffusion timestep, clean normalized box size | Predicted noise with the motion's shape |

Preparation linearly interpolates positions and scalar joint angles and uses
shortest-arc SLERP for root/object orientations. It does not smooth trajectories.
World linear and angular velocities use backward differences at 50 Hz; the
first resampled state is dropped so all derivatives use samples inside the
selected interval. There is no padding, wrapping, extrapolation, or
interpolation across segments. The original native-rate files remain unchanged.

`--window-policy` selects that interval. `lifted` (the default, and the v1 cache)
keeps only the lifted phase. `segment` keeps the whole exported segment: pickup
context, lift and set-down context, which the UMR export bounds by stable rest or
by 1.25 s before and 0.5 s after the lift. Every window stays class `carry`. The
index records each segment's `lift_index_bounds` in 50 Hz frames, and each window
is labeled `pickup`, `lifted` or `setdown` by whether it reaches either context.
`CarryWindowDataset(..., stages="lifted")` restores lifted-only windows from a
segment cache. Context frames can include standing still beside the box; they are
retained as the source recorded them.

### Quality checks and stage weights

Preparation also runs `mimickit_smp.quality` on every native frame of each
segment, using the segment's replay scene, and stores one bit-flag byte per 50 Hz
state (`quality/<trial>.npy`). A resampled state carries the flags of both
native frames it interpolates and of the previous state, whose difference gives
its velocity. The index records the thresholds and per-segment flag counts.
Training then drops **every window touching a flagged state** for the checks it
selects, so a recording keeps its clean parts:

| Check | A native frame is flagged when |
| --- | --- |
| `physical` | robot–box or robot–floor penetration > 3 cm, box > 2 cm below the floor, any joint > 25 rad/s, or box > 3 m/s (tracking glitch) |
| `chatter` | any joint's residual from a 78 ms quadratic fit has a 0.3 s RMS > 1° (frame-alternating solver wobble; fast smooth motion fits such a short window) |
| `foot_slide` | during a robot foot contact (a sole sphere < 3 mm from the floor for ≥ 0.11 s) its least-moving sole sphere travels > 5 cm farther than the person's least-moving toe/ankle joint; heel and toe roll count as planted on both sides |
| `support` | the box is > 2 cm above the floor but no robot geom is within 3 cm (e.g. a box resting on a raised surface absent from the replay scene, which would otherwise look like a floating box) |

Grip type is deliberately not checked; support from any body part counts.
The floor threshold is 2 cm because a box resting in the source mocap reads
−0.2 cm deep (median, p5–p95 −1.3 to +1.1 cm); at 1 cm the check dropped about
12% of pickup and set-down context for tracking noise, at 2 cm under 1%.
On the original 780 segments the four checks drop 2.2%, 2.9%, 1.8% and 0.9% of
lifted windows (7.3% together).

`stage_weights` (e.g. `{pickup: 0.2, lifted: 0.6, setdown: 0.2}`) draws each
training window's stage first and then a uniform window of that stage; without
weights, pickup and set-down are about 15% of draws. `null` keeps the uniform
sampler bit-for-bit. Context windows are almost stationary (median root speed
0.07 m/s, against 0.28 m/s while lifted): HiPHI has essentially no walking up to
the box, so the carry prior does not learn an approach.

The box pose is recentered at its object-local visual-mesh AABB center, and its
dimensions are **full extents in meters**, after the recorded object scaling.
The original local center offset is recorded. Cached rotations and extents use
the source **mesh axes**, which differ across HiPHI boxes (mesh Y is often up).
They must not be interpreted as an already-standardized Z-up box frame.
Mass, friction and contact forces
are absent because they are not measured by these kinematic trajectories.
In v1 and v2, box dimensions are denoised channels of the joint motion
distribution. **In v3 they are a clean condition** (below).

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

**Feature schema v3 conditions on box size instead of denoising it.**
`encode_window` returns the 216 motion channels and the canonical X/Y/Z full
extents at the window's last frame (X most forward, Z most up, as above). The
trial fixes the box size; the policy does not produce it. As a denoised channel,
an unusual size would lower the score of every motion with that box. As a
condition, the prior scores how natural the motion is *for a box of this size*.
TinyMDM embeds the normalized size with a small MLP whose output layer starts at
zero and adds it to the diffusion timestep (and class) embedding; the condition
is never noised. The model stores the condition's training mean/std in its
checkpoint, next to the motion normalizer. Conditioning interpolates between
sizes seen in training, so box-size augmentation determines its useful range.

Cached **state files and indexes stay unchanged** across schema versions.
`CarryWindowDataset` validates the recorded `cache_schema` and exposes the
current v3 feature `schema`; `encode_motion` remains for motion-only callers.
New checkpoints and normalizers carry the v3 hash; v1/v2 checkpoint or
normalizer hashes are rejected by the v3 training/data path. Refit normalization
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
| **Total motion** | **216** | |
| Box size (condition, not noised) | 3 | Full extents in canonical box X/Y/Z order at the last frame |

The feature schema records exact joint and body names, local joint axes, robot
kinematics hash, coordinate conventions, frequency and window length. An online
adapter must map simulator tensors by name, convert XYZW if necessary, and use
box-center velocities rather than a displaced center-of-mass velocity. It then
calls the same `encode_window` and passes its box size as the prior's `cond`.
Reset history at episode boundaries; start
scoring only after a complete valid window. Simulator velocities are instantaneous;
source velocities are finite-difference estimates and should be compared during
integration. Shape equality alone is insufficient validation.

## Current prepared datasets

Source: the October 8 HiPHI UMR carry selection, **780 segments, 332 recordings**
(the 402 original Bringing/box recordings, 7.2 h, of which 332 contain a segment
passing the lift/transport filter; HiPHI's 617.5 h includes mirrored copies and
non-box data).

**Box-size augmented segment cache (current, 1.8 GiB, with quality flags):**
`/data/sky/projects/MimicKit/datasets/hiphi_umr_carry_aug7_50hz_v1/index.json`.
Each recording was re-retargeted seven times with UMR: once at its original box
size and six times with random per-axis box factors in 0.7–1.3, all with the
same recipe (box-contact weight 15 on the hands, a per-frame lift keeping a
resized box out of the floor, `temporal_smooth_cost` 5 against arm chatter).
Run and scripts: `/data/sky/projects/FAR-Holosoma/runs/hiphi_umr_size_aug_20261008_v1`
and FAR-Holosoma `projectpivot/experiments/hiphi_umr_size_aug/`. Two of 2,324
solves failed (Clarabel `InsufficientProgress`), leaving 5,458 segments. Every
variant keeps its actor's split.

| Split | Segments | Windows | Kept by all four quality checks |
| --- | ---: | ---: | ---: |
| Train | 4,891 | 5,144,727 (489,246 pickup, 4,352,254 lifted, 303,227 set-down) | 4,411,592 (92.7% / 84.4% / 94.4% of each stage) |
| Validation | 273 | 847,805 | 808,302 |
| Test | 294 | 301,462 | 292,348 |

On training windows the checks alone drop 9.6% (physical, nearly all robot–box
penetration > 3 cm), 4.8% (chatter), 1.8% (foot slide) and 1.1% (support).
Penetration concentrates in shoulder carries (`Bringing-hump`, 18.6% of states),
the large scanned boxes A_1/A_2 (~12%) and taller boxes (10.2% above 1.1×
height). The augmentation adds box geometry, not new human motion: the source
is still 5.07 h of segments. Audit: `/data/sky/projects/MimicKit/runs/carry_data_audit_aug7_20261008/`.

**Segment cache (original boxes, 249 MiB, no quality flags):**
`/data/sky/projects/MimicKit/datasets/hiphi_umr_carry_50hz_v2/index.json`,
prepared with `--window-policy segment`. It holds 899,938 windows. Training has
735,757: 622,502 lifted, 69,918 pickup and 43,337 set-down (15.4% context);
validation 121,115 and test 43,066. Splits, actors and quality flags match the
lifted cache below.

**Lifted cache (v1, 214 MiB):**
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
  --output "$PROJECT_STORAGE_ROOT/datasets/carry_50hz_new" --workers 8 --window-policy segment

python -m mimickit_smp.audit \
  --index "$PROJECT_STORAGE_ROOT/datasets/carry_50hz_new/index.json" \
  --output "$PROJECT_STORAGE_ROOT/runs/carry_data_audit_new"
```

Both commands require fresh output directories. Preparation needs each
segment's replay scene (`scene_xml`) for the quality checks. The audit verifies
every cache and quality-flag hash, finite state, unit quaternion, interval/lift
bounds and split; it encodes first/middle/last windows from every segment,
checks each box-size condition against the segment's extents, counts training
windows by stage, reports the windows each quality check keeps per split and
stage, and exercises worker-process collation. Normalization uses 20,000
randomly sampled **training** windows (unfiltered).

```python
from mimickit_smp import CarryWindowDataset, make_dataloader

dataset = CarryWindowDataset(
    "/path/to/index.json",
    split="train",
    quality="all",
    stages="all",
    quality_checks=["physical", "chatter", "foot_slide", "support"],
    stage_weights={"pickup": 0.2, "lifted": 0.6, "setdown": 0.2},
)
indices = dataset.sample_indices(512)  # stage-weighted draw used by the TinyMDM adapter
loader = make_dataloader(dataset, batch_size=256, num_workers=4)  # uniform over the kept windows
batch = next(iter(loader))
motion = batch["motion"]  # [256, 16, 216], raw feature values
box_size = batch["box_size"]  # [256, 3], canonical X/Y/Z extents (the condition)
labels = batch["class_id"]  # carry = 0 for pickup, lifted and set-down windows
stages = batch["stage"]  # "pickup", "lifted" or "setdown"
# Metadata also includes trial/actor IDs, original source time and quality status.
```

Train with the existing TinyMDM implementation:

```bash
export SMP_CARRY_DATASET="$PROJECT_STORAGE_ROOT/datasets/hiphi_umr_carry_aug7_50hz_v1/index.json"
python tools/diffusion_model/train_tinymdm.py \
  --cfg_path tools/diffusion_model/config/tinymdm_carry_hoi.yaml \
  --out_dir "$PROJECT_STORAGE_ROOT/runs/carry_prior_new" --device cuda
```

The carry configuration sets `normalizer_std_clip: 0.03`, replacing 0.2 so small
hand/foot position deviations are not suppressed as strongly. This is a shared
numeric floor over feature units, not a 3 cm threshold on every channel. The
standalone audit uses the same floor by default.

The adapter samples only training data and validates frequency, window size and
saved schema hashes. It sets `cond_dim: 3` for v3 data; the trainer fits the
condition normalizer from the same training draw as the motion normalizer and
passes each batch's box sizes to the model. Reward and sampling calls must pass
the raw box size as `cond=` (shape `[B, 3]`); the model normalizes it, and a
conditional checkpoint rejects calls without it. `window_stages` selects `all`
or `lifted` windows; `quality_checks` and `stage_weights` (above) filter and
weight them, and also draw the box sizes used for generated samples. The
trainer saves resolved dimensions, the feature schema
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

The v3 segment-cache audit and conditional training smoke are under
`/data/sky/projects/MimicKit/runs/carry_data_audit_v3_segment_20261008/` and
`/data/sky/projects/MimicKit/runs/carry_tinymdm_v3_smoke_20261008/`. The smoke
trains three iterations with EMA, reloads the checkpoint in test mode, and
generates windows with their sampled box sizes. The quality-filter and
stage-weight smoke (same three iterations, all four checks, weights 0.2/0.6/0.2)
is under `/data/sky/projects/MimicKit/runs/carry_tinymdm_v3_quality_smoke_20261008/`.
Of 39 tests, `ei_world_model` passes 38 and skips the MuJoCo scene test (no
`mujoco`); `hsmujoco` passes 36 and skips the three TinyMDM tests, which need
`diffusers`.

### First trained priors (2026-10-08)

Two priors trained 200k iterations on the augmented cache, with all four quality
checks and stage weights 0.2/0.6/0.2: 16 states (0.30 s) and 38 states (0.74 s,
`tinymdm_carry_hoi_w38.yaml` on `hiphi_umr_carry_aug7_50hz_w38_v1`). Runs:
`/data/sky/projects/MimicKit/runs/carry_prior_aug7_w{16,38}_20261008/`. The
trainer logs fixed-noise validation loss per stage (`validation_split`,
`eval_windows`) and keeps a checkpoint every `checkpoint_every` iterations.
TinyMDM's positional table now covers windows longer than 32 frames.

`tools/diffusion_model/eval_carry_prior.py` scores held-out windows with the RL
reward (diffusion steps 22/15/8, `exp(-6 * mean normalized SDS)`) against
corrupted copies and against real windows the quality filter dropped.
`sample_carry_prior.py` and `check_carry_samples.py` generate windows and check
them kinematically (MuJoCo FK of decoded joints; `mimickit_smp.decode` inverts
v3 features); `view_prior_samples.py` loops them next to real windows in Viser.

| Test-split AUROC (real vs …) | 0.30 s | 0.74 s |
| --- | ---: | ---: |
| splice / root jitter | 1.00 / 1.00 | 1.00 / 1.00 |
| skate (1 m/s sideways) / 2× speed | 0.96 / 0.87 | 0.98 / 0.97 |
| dropped: penetration / unsupported / chatter / foot slide | 0.77 / 0.92 / 0.77 / 0.46 | 0.88 / 0.96 / 0.75 / 0.53 |
| frozen box / box +15 cm / swapped box size | 0.59 / 0.53 / 0.55 | 0.64 / 0.53 / 0.55 |
| 0.5× speed / static (below 0.5: preferred over real) | 0.34 / 0.29 | 0.22 / 0.13 |

Both priors learn body kinematics and timing but only weakly the hand–box
relation and the size condition. They prefer slow and static windows (a static
window earns 3.1× / 5.9× the reward of a real one) and pickup/set-down context
(2.9× / 3.9× for pickup), a bias RL can exploit. Generated windows are
kinematically consistent (FK p95 1.5 / 2.1 cm) but hands leave a lifted box in
34% / 25% of frames (real 2%). With `T: 50` and `squaredcos_cap_v2`,
`alphas_cumprod[49]` is about 1e-6: unclipped 50-step DDPM/DDIM sampling diverges in the
first step, so sample with 10-step DDIM. SDS rewards at steps 22/15/8 are
unaffected.

### Mirroring, a larger model and sample prediction (2026-10-08 sweep)

`python -m mimickit_smp.mirror --index <cache>/index.json --output <new cache>`
adds a left-right mirror image of every segment (reflection through world
`y = 0`). Positions and linear velocities negate y, rotations become `M R M`
(WXYZ `(w, -x, y, -z)`), angular velocities become `(-wx, wy, -wz)`, and left and
right joints and hand/foot origins swap. Joint signs come from the skeleton by
forward kinematics: every roll and yaw joint flips, pitch, knee and elbow do
not. The G1 sole spheres 1–4 are numbered in the same lateral order on both
feet, so they are excluded from that derivation as a relabeling. Mirrored
origins match FK of the mirrored joints to 10 µm. Quality flags, lift bounds,
box size and actor split carry over; trial IDs gain `__mirror`. Mirrored
caches: `hiphi_umr_carry_aug7_mirror_50hz_v1` and `..._mirror_50hz_w38_v1`.

The large prior is 8 layers × 512 wide (36M parameters, against 2.9M) at
`lr: 0.0001`; `allow_tf32: true` makes it about 2.5× faster with fp32 weights.
`estimate_mode: sample` predicts the clean window instead of the noise. The
winning setups are `tinymdm_carry_hoi_large_x0.yaml` and
`tinymdm_carry_hoi_large_x0_w38.yaml`. Runs:
`/data/sky/projects/MimicKit/runs/carry_prior_mirror_{small,large}_{eps,x0}_w{16,38}_20261008`;
`tools/diffusion_model/evaluate_carry_prior_run.sh` evaluates a run and
`compare_carry_priors.py` tabulates runs
(`/data/sky/projects/MimicKit/runs/compare_w16_20261008.md`, `..._w38_...md`).
All are evaluated on the unmirrored test split.

At 0.30 s:

| | baseline | mirror | mirror + large | mirror + x0 | mirror + large + x0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AUROC frozen box / swapped size | 0.59 / 0.55 | 0.59 / 0.53 | 0.70 / 0.64 | 0.65 / 0.59 | 0.68 / 0.64 |
| static window reward vs real | 3.1× | 2.9× | 3.3× | 2.7× | 2.0× |
| real pickup reward vs real | 2.9× | 2.9× | 3.6× | 3.7× | 1.9× |
| AUROC dropped penetration / unsupported | 0.77 / 0.92 | 0.77 / 0.90 | 0.65 / 0.84 | 0.73 / 0.98 | 0.80 / 0.78 |
| samples: hands off lifted box | 34% | 32% | 15% | 9% | 8% |
| samples: FK p95 / planted-foot p50 | 1.5 cm / 0.16 m/s | 1.6 / 0.16 | 0.3 / 0.03 | 0.9 / 0.08 | 0.5 / 0.05 |

Samples are 10-step DDIM for epsilon priors and 50-step DDPM for sample (x0)
priors. Predicting x0 avoids dividing by `sqrt(alphas_cumprod)` at `t = 49`, so
50-step DDPM no longer diverges and beats 10-step DDIM. Mirroring alone halves
the train–validation loss gap but barely moves the reward benchmark. The larger
model learns the hand–box relation and kinematic consistency best. x0
prediction most reduces the preference for static, slow and pickup windows.
Combined, large x0 is the most balanced reward candidate; its weak spot is the
unsupported-box check (physics prevents that case in RL). Its benchmark improves
steadily to the final checkpoint although its loss gap widens.

At 0.74 s the pattern repeats and the large x0 prior also removes the stage bias:

| | baseline | mirror | mirror + large | mirror + x0 | mirror + large + x0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AUROC frozen box / box +15 cm / swapped size | 0.64 / 0.53 / 0.55 | 0.61 / 0.51 / 0.52 | 0.82 / 0.62 / 0.68 | 0.75 / 0.55 / 0.59 | 0.78 / 0.56 / 0.64 |
| static window reward vs real | 5.9× | 6.5× | 6.7× | 4.1× | 2.8× |
| real pickup / lifted reward vs real | 3.9× / 0.66× | 4.0× / 0.65× | 3.3× / 0.71× | 3.9× / 0.65× | 1.1× / 1.0× |
| AUROC dropped penetration / unsupported | 0.88 / 0.96 | 0.86 / 0.95 | 0.73 / 0.90 | 0.76 / 0.90 | 0.82 / 0.81 |
| samples: hands off lifted box | 25% | 23% | 16% | 12% | 7% |
| samples: FK p95 / planted-foot p50 | 2.1 cm / 0.14 m/s | 2.1 / 0.14 | 0.2 / 0.03 | 0.8 / 0.06 | 0.5 / 0.05 |

The large epsilon prior is the most box-aware and kinematically consistent but
keeps the strongest static preference. All priors still reward static windows
more than real carrying (2–3× for large x0), so RL needs a task reward that
outweighs stalling, or a speed-baselined SDS.

### Skills: carry, push and slide (2026-10-11)

Push and slide come from HiPHI's `Cause_motion` recordings, sorted by FrameNet
lexical unit (FAR-Holosoma `projectpivot/experiments/hiphi_umr_dataset/skills.py`):
**push** is push/shove/thrust/move, the hands moving a box along the floor;
**slide** is punt/hit with a foot named in the annotation, kicks. Dragging a lug
is not a push. `select_skill_segments.py` keeps stretches in which the box stays
on the floor and moves faster than 8 cm/s, joining pauses of up to 3 s (shoves and
kicks pause the box for 1.4–3.4 s), with at least 1 s of movement and 30 cm of
travel. Push needs a hand within 10 cm of the box in at least half the moving
frames; slide needs a foot there at least briefly and the hands mostly away. The
moving interval is stored as the lift interval, so for push and slide the stages
read **pickup = approach** (1.25 s), **lifted = box moving** and **setdown =
settle** (0.5 s). The selection holds 386 segments from 214 recordings: push 219
(1.39 h) and slide 167 (1.43 h). Run:
`/data/sky/projects/FAR-Holosoma/runs/hiphi_umr_pushslide_20261011_v1`.

Every actor has one split across skills, so no held-out actor's motion reaches
training through another skill. Push and slide come almost entirely from carry
training actors (A048 and A005 give 82% of slide), so `align_actor_splits.py
--validation-actors` moved seven small actors to validation in every skill: A002,
A003, A004, A026, A037, A043 and A127. The carry test set is unchanged; 8.6 min of
carry moved from training to validation, applied to the carry segments by
`prepare --actor-splits <push/slide manifest>`. Held out: push 11.3 min of
validation and 0.7 min of test (A040), slide 8.8 min of validation and no test.

The box-size augmentation repeats carry's seven UMR runs per recording (original
size and six random per-axis factors in 0.7–1.3, the floor lift,
`temporal_smooth_cost` 5) with a contact recipe per skill
(`hiphi_umr_size_aug/prepare_full.py`):

- push keeps UMR's box-contact weight 5 on the hands. At carry's 15 the hands sank
  more than 3 cm into the pushed face on 22–27% of frames (5: none), and at 5 they
  still touch resized boxes whenever the person's hands touch the box (six
  recordings, three sizes each: `hiphi_umr_push_handweight_test_20261011`);
- slide puts weight 15 on the feet (`ankle_roll`): the robot's foot is within 3 cm
  of the box on 86–100% of the frames in which the person's foot touches it,
  resized boxes included.

All 1,498 + 819 solves converged (runs `hiphi_umr_pushslide_size_aug_20261011_v1`,
whose push the hand-weight-5 run `hiphi_umr_push_size_aug_20261011_v2` replaces).

Caches `hiphi_umr_skills_aug7_50hz_w38_v2` (0.74 s) and `hiphi_umr_skills_aug7_50hz_v2`
(0.30 s), with `_mirror_` versions for training; `class_map` carry 0, push 1, slide 2:

| Skill | Segments (train / val / test) | Hours | Training windows (0.74 s) kept by the four checks |
| --- | ---: | ---: | ---: |
| carry | 4,821 / 343 / 294 | 35.5 | 4.04 M (83%) |
| push | 1,246 / 252 / 35 | 9.7 | 1.28 M (88%; 69% at hand weight 15) |
| slide | 1,036 / 133 / 0 | 10.0 | 1.53 M (97%) |

The skill prior (`tinymdm_skills_x0_w38.yaml`, 0.30 s: `tinymdm_skills_x0.yaml`) is a
`CondDiT` with the skill as its class label (AdaLN, like the timestep) and the box
size through the zero-initialized condition MLP. It is twice the large carry prior:
10 layers × 640 wide, 69.5M parameters, x0 prediction. Training draws a skill in
proportion to its kept windows (`class_weights: null`; the sweep below started with
equal draws, 1:1:1), then a stage within it (0.2 / 0.6 / 0.2), then a window.

Evaluation of skill-conditioned priors:

- `eval_carry_prior.py` reports each corruption per skill (`by_skill`), normalized
  by that skill's real windows as a single-skill policy's running mean would, and
  adds `skill_swap` (real motion scored under another skill's label) and stage
  ratios within each skill (`real_by_skill_stage`). Stage scores now use every
  window of a stage; corruptions still need 2W poses after the window start (2×
  speed). This moved the 0.74 s carry prior's set-down ratio from 0.150 to 0.217
  and nothing else;
- `evaluate_carry_prior_run.sh <run> <gpu> <index> [splits...]` evaluates several
  splits; `EVAL_DIR` keeps a re-evaluation apart (the carry-only priors' are in
  `eval_20261011/`);
- `compare_carry_priors.py --split validation --skill push` tabulates one skill;
  `--evals` names evaluation folders;
- carry comparisons score skill priors on the carry-only cache, the same windows
  the carry-only priors were scored on;
- `view_prior_samples.py` labels columns with skill and box size and has a Skill
  menu; several `--samples` files (one per prior, drawn from the same held-out
  windows) share one viewer through a Prior menu.

Results at 0.74 s (runs `/data/sky/projects/MimicKit/runs/skill_prior_v2_mirror_*_20261011`,
tables in `skill_compare_20261011/`). Carry is scored on the carry-only cache, the
windows the carry-only prior was scored on; push and slide on validation actors.
"Labeled X" scores a skill's real windows under skill X's label:

| | carry-only 36M | skill 36M | skill 2× | skill 4× | 36M prop. | 2× prop. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| carry test: AUROC frozen box / swapped size | 0.78 / 0.64 | 0.72 / 0.61 | 0.76 / 0.62 | 0.76 / 0.62 | 0.74 / 0.62 | 0.75 / 0.62 |
| carry test: reward × pickup / lifted / set-down | 1.1 / 1.0 / 0.22 | 4.1 / 0.65 / 2.2 | 3.8 / 0.68 / 2.0 | 2.5 / 0.85 / 1.0 | 2.9 / 0.78 / 1.4 | 2.6 / 0.82 / 0.63 |
| carry validation: reward × pickup / set-down | 0.06 / 0.017 | 1.05 / 0.23 | 0.58 / 0.11 | 0.29 / 0.056 | 0.43 / 0.081 | 0.20 / 0.047 |
| carry: reward × static, test / validation | 2.8 / 1.8 | 3.3 / 3.1 | 3.9 / 3.1 | 4.1 / 2.7 | 3.4 / 2.7 | 3.7 / 2.4 |
| carry test samples: hands off lifted box | 7.0% | 9.9% | 9.2% | 6.9% | 9.4% | 7.5% |
| push: AUROC frozen box (box moving) / swapped size | | 0.62 / 0.67 | 0.61 / 0.66 | 0.56 / 0.66 | 0.62 / 0.66 | 0.61 / 0.66 |
| slide: AUROC frozen box (box moving) / box +15 cm | | 0.49 / 0.70 | 0.51 / 0.66 | 0.50 / 0.64 | 0.48 / 0.66 | 0.49 / 0.66 |
| AUROC push labeled carry / slide labeled carry | | 0.50 / 0.35 | 0.50 / 0.40 | 0.47 / 0.34 | 0.78 / 0.63 | 0.78 / 0.73 |
| AUROC slide labeled push / push labeled slide | | 0.85 / 0.88 | 0.90 / 0.88 | 0.86 / 0.83 | 0.65 / 0.85 | 0.74 / 0.85 |

- The seed replicate of the 2× prior differs by up to ±0.04 in AUROC and ±0.2 in
  stage ratios. Push and slide numbers barely move with size (36M, 2×, 4×).
- With equal skill draws the carry label accepts push and slide motion (slide
  scores 2.2× higher labeled carry). Drawing skills in proportion to their windows
  (carry 59%) makes it specific, at some cost in telling slide from push; the 2×
  prior keeps more of that than the 36M one.
- Push and slide raise the reward of carry's floor phases (pickup, set-down)
  relative to the lift, by 3–20× against the carry-only prior. On the test actors
  this overshoots parity, on the validation actors it approaches it. Each split
  has only three carry actors, and the ratios differ more between the splits than
  between priors. Proportional draws and larger models move them back toward the
  carry-only prior.
- No prior notices a kicked box that fails to move (slide frozen box ≈ 0.5, also on
  windows whose box travels > 5 cm); push gets 0.6. Generated windows move the box
  less often than real ones (push 26–30% of frames against 36%, slide 10% against
  19%), while hand–box and foot–box distances match the real ones.
- Every skill prior rewards static windows more than the carry-only prior does.
- 400k iterations raise the validation loss (0.105 → 0.112) without improving the
  benchmark: 200k is enough.
- Label dropout (`cfg_dropout` 0.15) makes the conditional score nearly ignore the
  label (push labeled slide 0.88 → 0.56). Guided scores (`eval_carry_prior.py
  --cfg_scale`, uncond + s (cond − uncond)) recover part of it at s = 3 (0.75) but
  let the carry label accept push and slide more, and do not change box awareness.
- At 0.30 s the pattern repeats; proportional draws fix the carry label there too
  (push / slide labeled carry 0.53 / 0.41 → 0.78 / 0.68).

The current candidate is the 2× prior with proportional draws at 0.74 s,
`skill_prior_v2_mirror_2x_x0_w38_propskills_20261011` (the repository configs now
draw skills proportionally). A 4× prior with proportional draws was still training.

Tests: `python -m pytest tests/smp_reward` (set its cache directory under `/data`).
