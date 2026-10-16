"""Reward sanity benchmark for a trained carry prior: real held-out windows vs corrupted copies.

Scores windows with the SMP reward used in RL, exp(-sds_loss_scale * mean_t(SDS_t / mu_t)),
at the agent's diffusion steps. mu_t is the mean SDS of the real windows, so a real
window scores about exp(-6) on average and only ratios between conditions matter.
A useful prior separates real windows from implausible corruptions (AUROC near 1)
and does so in a single noise draw, which is what the policy sees each step.

Corruptions edit 50 Hz states and recompute velocities with the cache's backward
differences. Root/box edits also move the cached hand/foot positions rigidly, so
no forward kinematics is needed:

- reverse, fast (2x), slow (0.5x): time edits of the whole window;
- frozen_box: the box stays at its first pose while the robot carries on;
- box_offset: the box floats 15 cm above where the hands hold it;
- size_swap: the motion is real but conditioned on another window's box size;
- skate: the robot and box drift 1 m/s sideways together (feet slide);
- root_jitter: the whole robot shakes frame to frame (1.5 cm, 2 deg);
- splice: two windows joined at the middle, positions made continuous;
- static: every frame equals the first (a statue; plausible during short pauses);
- skill_swap (skill-conditioned priors): the real motion scored under another skill;
  skill_matrix scores each skill's real windows under every other label.

A skill-conditioned prior scores every window under its own skill label, draws the
same number of windows per skill and reports each corruption per skill too. Per-skill
numbers normalize by that skill's real windows, as a single-skill policy's running mean
would, so a skill's numbers match a prior trained on that skill alone. A skill without
windows in the split is listed as missing.

It also scores real windows the training filter dropped, one quality check at a
time (windows touching that check's flags and no other's), against clean ones:
retargeting chatter, foot slide, penetration and unsupported boxes are the kind
of artifacts a policy can produce, and the prior never saw them.

    python tools/diffusion_model/eval_carry_prior.py \
        --model_dir /data/.../runs/<run>/output --split test --out /data/.../eval.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation, Slerp

sys.path.append("mimickit")

from learning.tinymdm.tinymdm_model import TinyMDMModel  # noqa: E402

from mimickit_smp.dataset import WINDOW_STAGES, CarryWindowDataset, PackedWindowSampler  # noqa: E402
from mimickit_smp.features import encode_window  # noqa: E402

RATE = 50.0
CORRUPTIONS = (
    "reverse",
    "fast",
    "slow",
    "frozen_box",
    "box_offset",
    "size_swap",
    "skate",
    "root_jitter",
    "splice",
    "static",
    "skill_swap",
)


def quat(q):
    return Rotation.from_quat(q[..., [1, 2, 3, 0]])


def wxyz(rot):
    return rot.as_quat()[..., [3, 0, 1, 2]]


def with_velocities(seq):
    """seq: [W + 1, 70] poses (the first row only supplies the previous pose) -> [W, 70]."""
    out = seq[1:].copy()
    out[:, 48:51] = np.diff(seq[:, 0:3], axis=0) * RATE
    out[:, 61:64] = np.diff(seq[:, 54:57], axis=0) * RATE
    for q, w in ((slice(3, 7), slice(51, 54)), (slice(57, 61), slice(64, 67))):
        r = quat(seq[:, q])
        out[:, w] = (r[1:] * r[:-1].inv()).as_rotvec() * RATE
    return out


def rigid(seq, frames, shift=None, rotation=None, box=False):
    """Move the robot (root and cached body positions), and optionally the box, rigidly."""
    seq = seq.copy()
    for i in frames:
        root = seq[i, 0:3].copy()
        if rotation is not None:
            r = rotation[i]
            seq[i, 3:7] = wxyz(r * quat(seq[i, 3:7]))
            bodies = seq[i, 36:48].reshape(4, 3)
            seq[i, 36:48] = (r.apply(bodies - root) + root).reshape(-1)
            if box:
                seq[i, 54:57] = r.apply(seq[i, 54:57] - root) + root
                seq[i, 57:61] = wxyz(r * quat(seq[i, 57:61]))
        if shift is not None:
            seq[i, 0:3] += shift[i]
            seq[i, 36:48] += np.tile(shift[i], 4)
            if box:
                seq[i, 54:57] += shift[i]
    return seq


def resample(seq, times):
    """Interpolate poses at fractional frame indices (linear positions/joints, slerp rotations)."""
    k = np.arange(len(seq))
    out = np.empty((len(times), seq.shape[1]))
    for a, b in ((0, 3), (7, 57), (61, 70)):
        for j in range(a, b):
            out[:, j] = np.interp(times, k, seq[:, j])
    out[:, 3:7] = wxyz(Slerp(k, quat(seq[:, 3:7]))(times))
    out[:, 57:61] = wxyz(Slerp(k, quat(seq[:, 57:61]))(times))
    return out


def corrupt(name, ctx, s, width, rng, other=None):
    """ctx: poses [s - 2, s + 2W) of one segment around a window starting at s (index 2 in ctx)."""
    w0 = 2  # The window's first pose in ctx; ctx[1] is its previous pose.
    window = ctx[w0 - 1 : w0 + width]
    if name == "real":
        return with_velocities(window)
    if name == "reverse":
        return with_velocities(ctx[w0 : w0 + width + 1][::-1])
    if name == "fast":
        return with_velocities(ctx[w0 - 2 : w0 + 2 * width : 2][: width + 1])
    if name == "slow":
        return with_velocities(resample(ctx, w0 - 0.5 + 0.5 * np.arange(width + 1)))
    if name == "frozen_box":
        seq = window.copy()
        seq[:, 54:61] = seq[1, 54:61]
        return with_velocities(seq)
    if name == "box_offset":
        seq = window.copy()
        seq[:, 56] += 0.15
        return with_velocities(seq)
    if name == "size_swap":
        return with_velocities(window)  # The condition is swapped by the caller.
    if name == "skate":
        heading = quat(window[1, 3:7]).apply([0.0, 1.0, 0.0]) * np.array([1.0, 1.0, 0.0])
        heading /= max(np.linalg.norm(heading), 1e-6)
        shift = (np.arange(width + 1) / RATE)[:, None] * 1.0 * heading
        return with_velocities(rigid(window, range(width + 1), shift=shift, box=True))
    if name == "root_jitter":
        sign = (-1.0) ** np.arange(width + 1)
        shift = sign[:, None] * np.array([0.015, 0.0, 0.0])
        rotation = Rotation.from_rotvec(sign[:, None] * np.radians([2.0, 2.0, 2.0]) / np.sqrt(3))
        return with_velocities(rigid(window, range(width + 1), shift=shift, rotation=rotation))
    if name == "splice":
        half = (width + 1) // 2
        second = other[w0 - 1 + half : w0 + width].copy()
        offset = window[half - 1, 0:3] - second[0, 0:3]
        offset[2] = 0.0
        second = rigid(second, range(len(second)), shift=np.tile(offset, (len(second), 1)), box=True)
        return with_velocities(np.concatenate([window[:half], second]))
    if name == "static":
        return with_velocities(np.repeat(window[1:2], width + 1, axis=0))
    raise ValueError(name)


def draw(dataset, sampler, num, width, seed, stage=None, class_ids=None, span=None):
    """Window starts with two poses before and span poses (default 2W) from the start inside the segment.

    Corruptions need 2W poses (2x speed); real windows only W, so stage scores pass
    span=W and keep a segment's last windows (push and slide settle for only 0.5 s).
    class_ids: draw num windows from each of these skill classes (that has any).
    Returns contexts, segment indices, stage IDs and class IDs.
    """
    span = 2 * width if span is None else span
    frames = np.asarray([s["frames"] for s in dataset.segments])
    seg, start = dataset.window_segments, dataset.window_starts
    ok = (start >= 2) & (start + span <= frames[seg])
    if stage is not None:
        ok &= dataset.window_stage_ids == WINDOW_STAGES.index(stage)
    rng = np.random.default_rng(seed)
    groups = [ok] if class_ids is None else [ok & (dataset.window_class_ids == c) for c in class_ids]
    picks = [rng.choice(np.flatnonzero(g), size=min(num, int(g.sum())), replace=False) for g in groups if g.any()]
    if not picks:
        empty = np.zeros(0, np.int64)
        return np.zeros((0, span + 2, 70)), empty, empty, empty
    pick = np.concatenate(picks)
    offsets = np.concatenate([[0], np.cumsum(frames)[:-1]])
    states = sampler.states.cpu().numpy()
    ctx = [states[offsets[seg[i]] + start[i] - 2 : offsets[seg[i]] + start[i] + span] for i in pick]
    return (
        np.stack(ctx).astype(np.float64),
        seg[pick],
        dataset.window_stage_ids[pick],
        dataset.window_class_ids[pick],
    )


@torch.no_grad()
def flagged_windows(index, split, width, num, seed):
    """Per quality check: up to num windows touching that check's flags only (none touched by others)."""
    from mimickit_smp.quality import BITS

    everything = CarryWindowDataset(str(index), split=split, quality_checks=())
    touched = []
    for k, s in enumerate(everything.segments):
        flags = everything._flags(s)
        starts = everything.window_starts[everything.window_segments == k]
        bits = np.zeros(len(starts), np.uint8)
        for bit in BITS.values():
            c = np.concatenate([[0], np.cumsum((flags & bit) > 0)])
            bits[(c[starts + width] - c[starts]) > 0] |= bit
        touched.append(bits)
    touched = np.concatenate(touched)
    rng = np.random.default_rng(seed)
    sampler = PackedWindowSampler(everything)
    out = {}
    for name, bit in BITS.items():
        pool = np.flatnonzero(touched == bit)
        if len(pool) >= 32:
            pick = torch.from_numpy(rng.choice(pool, size=min(num, len(pool)), replace=False))
            out[name] = (sampler.gather(pick), torch.from_numpy(everything.window_class_ids[pick.numpy()]))
    return out


def sds(model, motion, cond, steps, draws, device, labels=None, cfg_scale=1.0):
    """cfg_scale != 1 scores with the guided prediction uncond + s (cond - uncond) of a prior
    trained with cfg_dropout; s = 0 ignores the skill label."""
    x = model.normalize(motion.to(device)).reshape(motion.shape[0], -1)
    kwargs = (
        {} if labels is None else {"class_labels": torch.as_tensor(labels, dtype=torch.long, device=device)}
    )
    if cfg_scale != 1.0:
        kwargs["cfg_scale"] = cfg_scale
    total = 0
    for _ in range(draws):
        total = total + model.ESM_SDS_loss(x, t_lst=steps, cond=cond.to(device), **kwargs)
    return (total / draws).cpu()


def auroc(real, fake):
    """P(a corrupted window scores worse than a real one); ties count half."""
    scores = np.concatenate([real, fake])
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):
        tie = scores == value
        ranks[tie] = ranks[tie].mean()
    return float((ranks[len(real) :].sum() - len(fake) * (len(fake) + 1) / 2) / (len(real) * len(fake)))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--model_file", default="model.pt")
    parser.add_argument("--index", type=Path, help="Defaults to the training config's dataset_index")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num", type=int, default=2048)
    parser.add_argument("--draws", type=int, default=8, help="Noise draws for the averaged score")
    parser.add_argument("--diffusion_steps", type=int, nargs="+", default=[22, 15, 8])
    parser.add_argument("--sds_loss_scale", type=float, default=6.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.0,
        help="Guided SDS for priors trained with cfg_dropout: uncond + s (cond - uncond); 0 ignores the skill",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    config = yaml.safe_load((args.model_dir / "diffusion_config.yaml").read_text())
    if args.cfg_scale != 1.0 and not config.get("cfg_dropout"):
        raise ValueError("Guided scores need a prior trained with cfg_dropout > 0 (it learns the null label)")
    config["env_config"] = str(args.model_dir / "env_config.yaml")
    model = TinyMDMModel(config, args.device)
    model.load_state_dict(torch.load(args.model_dir / args.model_file, map_location=args.device))
    model.to(args.device).eval()
    index = args.index or Path(os.path.expandvars(config["dataset_index"]))
    if "$" in str(index):
        raise ValueError("Pass --index or set the environment variable used in dataset_index")
    dataset = CarryWindowDataset(
        str(index), split=args.split, quality_checks=config.get("quality_checks") or (), stages="all"
    )
    if dataset.schema.sha256 != config["feature_schema_sha256"]:
        raise ValueError("The evaluation cache's feature schema differs from the prior's")
    width = dataset.schema.window_frames
    sampler = PackedWindowSampler(dataset)
    rng = np.random.default_rng(args.seed)
    skilled = config.get("arch_name") == "CondDiT" and config.get("num_class", 0) > 0
    class_names = dataset.class_names
    present = [c for c in sorted(class_names) if (dataset.window_class_ids == c).any()] if skilled else None
    per_class = args.num // len(present) if skilled else args.num

    def score(windows, cond=None, draws=args.draws, labels=None):
        states = torch.from_numpy(np.stack(windows)).float()
        motion, size = encode_window(states, dataset.schema)
        cond = size if cond is None else cond
        return sds(
            model, motion, cond, args.diffusion_steps, draws, args.device, labels if skilled else None, args.cfg_scale
        )

    ctx, segs, stages, labels = draw(dataset, sampler, per_class, width, args.seed, class_ids=present)
    others, other_segs, _, other_labels = draw(
        dataset, sampler, per_class, width, args.seed + 1, class_ids=present
    )
    lifted = stages == WINDOW_STAGES.index("lifted")
    real = [corrupt("real", c, 0, width, rng) for c in ctx]
    stored = ctx[:, 2 : 2 + width]
    # Windows whose real box travels > 5 cm: pushes and kicks pause the box, and freezing a
    # resting box changes nothing, so box corruptions are also scored on these alone.
    moving_box = np.linalg.norm(stored[:, -1, 54:57] - stored[:, 0, 54:57], axis=-1) > 0.05
    velocity_error = float(np.abs(np.stack(real)[..., 48:67] - stored[..., 48:67]).max())
    real_avg, real_one = score(real, labels=labels), score(real, draws=1, labels=labels)
    mu_avg, mu_one = real_avg.mean(0), real_one.mean(0)

    def reward(losses, mu):
        return torch.exp(-args.sds_loss_scale * (losses / mu).mean(-1)).numpy()

    def normalized(losses, mu):
        return (losses / mu).mean(-1).numpy()

    skill_mask = {c: labels == c for c in present} if skilled else {}
    skill_mu = {c: (real_avg[m].mean(0), real_one[m].mean(0)) for c, m in skill_mask.items()}

    def skill_metrics(c, losses, losses_one):
        m, (mu, mu1) = skill_mask[c], skill_mu[c]
        mm = m & moving_box
        return {
            "auroc": auroc(normalized(real_avg[m], mu), normalized(losses[m], mu)),
            "auroc_moving_box": auroc(normalized(real_avg[mm], mu), normalized(losses[mm], mu)) if mm.any() else None,
            "auroc_single_draw": auroc(normalized(real_one[m], mu1), normalized(losses_one[m], mu1)),
            "reward_ratio": float(reward(losses[m], mu).mean() / reward(real_avg[m], mu).mean()),
        }

    report = {
        "model": str(args.model_dir / args.model_file),
        "split": args.split,
        "windows": len(real),
        "window_frames": width,
        "window_seconds": (width - 1) / RATE,
        "diffusion_steps": args.diffusion_steps,
        "noise_draws_averaged": args.draws,
        "cfg_scale": args.cfg_scale,
        "velocity_recompute_max_abs_error": velocity_error,
        "real_mean_sds_per_step": real_avg.mean(0).tolist(),
        "corruptions": {},
        "real_by_stage": {},
        "skills": {class_names[c]: int((labels == c).sum()) for c in present} if skilled else None,
        "moving_box_windows": {class_names[c]: int(((labels == c) & moving_box).sum()) for c in present}
        if skilled
        else int(moving_box.sum()),
        "skills_missing_in_split": [n for c, n in class_names.items() if c not in present]
        if skilled
        else None,
    }
    real_norm, real_norm_one = normalized(real_avg, mu_avg), normalized(real_one, mu_one)
    for name in CORRUPTIONS:
        if name == "skill_swap" and not (skilled and len(present) > 1):
            continue
        windows, cond, fake_labels = [], None, labels
        for i, c in enumerate(ctx):
            other = others[i] if other_segs[i] != segs[i] else others[(i + 1) % len(others)]
            windows.append(corrupt(name if name != "skill_swap" else "real", c, 0, width, rng, other))
        if name == "size_swap":
            _, sizes = encode_window(torch.from_numpy(np.stack(real)).float(), dataset.schema)
            cond = sizes[torch.from_numpy(np.random.default_rng(args.seed + 2).permutation(len(real)))]
        if name == "skill_swap":
            # Each window under the next skill present in the split.
            nxt = {c: present[(k + 1) % len(present)] for k, c in enumerate(present)}
            fake_labels = np.asarray([nxt[c] for c in labels])
        losses = score(windows, cond, labels=fake_labels)
        losses_one = score(windows, cond, draws=1, labels=fake_labels)
        fake_norm = normalized(losses, mu_avg)
        report["corruptions"][name] = {
            "auroc": auroc(real_norm, fake_norm),
            "auroc_lifted_windows": auroc(real_norm[lifted], fake_norm[lifted]),
            "auroc_moving_box": auroc(real_norm[moving_box], fake_norm[moving_box]) if moving_box.any() else None,
            **{
                f"auroc_step_{t}": auroc(
                    (real_avg[:, k] / mu_avg[k]).numpy(), (losses[:, k] / mu_avg[k]).numpy()
                )
                for k, t in enumerate(args.diffusion_steps)
            },
            "auroc_single_draw": auroc(real_norm_one, normalized(losses_one, mu_one)),
            "normalized_sds_ratio": float(normalized(losses, mu_avg).mean() / real_norm.mean()),
            "reward_ratio": float(reward(losses, mu_avg).mean() / reward(real_avg, mu_avg).mean()),
        }
        print(
            f"{name:12s} "
            + " ".join(f"{k} {v:.3f}" for k, v in report["corruptions"][name].items() if v is not None),
            flush=True,
        )
        if skilled:
            report["corruptions"][name]["by_skill"] = {
                class_names[c]: skill_metrics(c, losses, losses_one) for c in present
            }
            for skill, values in report["corruptions"][name]["by_skill"].items():
                print(f"  {skill:6s} " + " ".join(f"{k} {v:.3f}" for k, v in values.items() if v is not None), flush=True)
    if skilled and len(present) > 1:
        # Every skill's real windows under every other label (skill_swap tries one other label).
        report["skill_matrix"] = {class_names[c]: {} for c in present}
        for j in present:
            under_j = score(real, labels=np.full(len(real), j))
            for c in present:
                if c != j:
                    m, mu = skill_mask[c], skill_mu[c][0]
                    ml = m & lifted  # Push and slide: the box is moving.
                    report["skill_matrix"][class_names[c]][class_names[j]] = {
                        "auroc": auroc(normalized(real_avg[m], mu), normalized(under_j[m], mu)),
                        "auroc_lifted_windows": auroc(normalized(real_avg[ml], mu), normalized(under_j[ml], mu)),
                        "reward_ratio": float(reward(under_j[m], mu).mean() / reward(real_avg[m], mu).mean()),
                    }
        print("skill_matrix (real skill -> label: auroc)", json.dumps(
            {i: {j: round(v["auroc"], 3) for j, v in row.items()} for i, row in report["skill_matrix"].items()}
        ), flush=True)
    report["filtered_out_real"] = {}
    for name, (states, flagged_labels) in flagged_windows(
        index, args.split, width, args.num, args.seed + 4
    ).items():
        motion, size = encode_window(states, dataset.schema)
        losses = sds(
            model,
            motion,
            size,
            args.diffusion_steps,
            args.draws,
            args.device,
            flagged_labels if skilled else None,
            args.cfg_scale,
        )
        fake_norm = normalized(losses, mu_avg)
        report["filtered_out_real"][name] = {
            "windows": len(states),
            "auroc": auroc(real_norm, fake_norm),
            "normalized_sds_ratio": float(fake_norm.mean() / real_norm.mean()),
        }
        if skilled:
            by_skill = {}
            for c in present:
                m = flagged_labels.numpy() == c
                if m.sum() >= 32:
                    mu = skill_mu[c][0]
                    by_skill[class_names[c]] = {
                        "windows": int(m.sum()),
                        "auroc": auroc(normalized(real_avg[skill_mask[c]], mu), normalized(losses[m], mu)),
                    }
        print(
            f"flagged {name:12s} "
            + " ".join(f"{k} {v:.3f}" for k, v in report["filtered_out_real"][name].items()),
            flush=True,
        )
        if skilled:
            report["filtered_out_real"][name]["by_skill"] = by_skill
    if skilled:
        report["real_by_skill"] = {
            class_names[c]: {
                "windows": int((labels == c).sum()),
                "reward_ratio_vs_all_real": float(
                    reward(real_avg[labels == c], mu_avg).mean() / reward(real_avg, mu_avg).mean()
                ),
            }
            for c in present
        }
        # Within each skill (for push and slide: pickup = approach, lifted = box moving, setdown = settle).
        report["real_by_skill_stage"] = {}
        for c in present:
            mu, entry = skill_mu[c][0], {}
            for stage in WINDOW_STAGES:
                stage_ctx, _, _, stage_labels = draw(
                    dataset, sampler, args.num // 2, width, args.seed + 3, stage, class_ids=[c], span=width
                )
                if not len(stage_ctx):
                    continue
                losses = score([corrupt("real", x, 0, width, rng) for x in stage_ctx], labels=stage_labels)
                entry[stage] = {
                    "windows": len(stage_ctx),
                    "reward_ratio_vs_skill_real": float(
                        reward(losses, mu).mean() / reward(real_avg[skill_mask[c]], mu).mean()
                    ),
                }
            report["real_by_skill_stage"][class_names[c]] = entry
    for stage in WINDOW_STAGES:
        stage_ctx, _, _, stage_labels = draw(
            dataset, sampler, args.num // 2, width, args.seed + 3, stage, span=width
        )
        if not len(stage_ctx):
            continue
        losses = score([corrupt("real", c, 0, width, rng) for c in stage_ctx], labels=stage_labels)
        report["real_by_stage"][stage] = {
            "windows": len(stage_ctx),
            "normalized_sds_mean": float(normalized(losses, mu_avg).mean()),
            "reward_ratio_vs_all_real": float(
                reward(losses, mu_avg).mean() / reward(real_avg, mu_avg).mean()
            ),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"velocity_recompute_max_abs_error": velocity_error, "real_by_stage": report["real_by_stage"]}
        )
    )


if __name__ == "__main__":
    main()
