"""Side-by-side table of trained priors: reward benchmark, generated-sample checks, losses.

Reads each run's eval/bench_<split>_final.json (eval_carry_prior.py),
eval/sample_checks_<split>_final*.json (check_carry_samples.py; test files may use the
older sample_checks_final* names) and output/log.txt. --evals names evaluation folders
directly (<run>/<folder>), e.g. a re-evaluation next to the original eval/.
--skill reads a skill-conditioned prior's per-skill numbers; single-skill priors have
only pooled numbers, which are their skill's. Validation losses are only comparable
between runs with the same estimate_mode.

    python tools/diffusion_model/compare_carry_priors.py --runs <run dir> [...] [--split test] [--skill carry]
"""

import argparse
import csv
import json
from pathlib import Path

import yaml

ROWS = [
    ("AUROC splice", "corruptions", "splice", "auroc"),
    ("AUROC root jitter", "corruptions", "root_jitter", "auroc"),
    ("AUROC skate", "corruptions", "skate", "auroc"),
    ("AUROC 2x speed", "corruptions", "fast", "auroc"),
    ("AUROC reverse", "corruptions", "reverse", "auroc"),
    ("AUROC frozen box", "corruptions", "frozen_box", "auroc"),
    ("AUROC frozen box, windows whose box moves > 5 cm", "corruptions", "frozen_box", "auroc_moving_box"),
    ("AUROC box +15 cm", "corruptions", "box_offset", "auroc"),
    ("AUROC swapped box size", "corruptions", "size_swap", "auroc"),
    ("AUROC 0.5x speed (<0.5 preferred)", "corruptions", "slow", "auroc"),
    ("AUROC static (<0.5 preferred)", "corruptions", "static", "auroc"),
    ("AUROC wrong skill label", "corruptions", "skill_swap", "auroc"),
    ("AUROC dropped: penetration", "filtered_out_real", "physical", "auroc"),
    ("AUROC dropped: chatter", "filtered_out_real", "chatter", "auroc"),
    ("AUROC dropped: foot slide", "filtered_out_real", "foot_slide", "auroc"),
    ("AUROC dropped: unsupported", "filtered_out_real", "support", "auroc"),
    ("reward x static vs real", "corruptions", "static", "reward_ratio"),
    ("reward x 0.5x speed vs real", "corruptions", "slow", "reward_ratio"),
    ("reward x real pickup (push/slide: approach) vs real", "real_by_stage", "pickup", "reward_ratio_vs_all_real"),
    ("reward x real lifted (push/slide: moving) vs real", "real_by_stage", "lifted", "reward_ratio_vs_all_real"),
    ("reward x real set-down (push/slide: settle) vs real", "real_by_stage", "setdown", "reward_ratio_vs_all_real"),
]
SAMPLE_ROWS = [
    ("samples: FK error p95 (cm)", "body_fk_error_cm_p50_p95", 1),
    ("samples: hands off lifted box", "hands_off_box_frames_when_lifted", None),
    ("samples: hand-box p50 (cm)", "hand_box_cm_p50_p95_when_lifted", 0),
    ("samples: planted foot p50 (m/s)", "planted_foot_speed_m_s_p50_p95", 0),
    ("samples: box velocity error p50", "box_velocity_error_m_s_p50_p95", 0),
    ("samples: box moving frames", "box_moving_frames", None),
    ("samples: hand-box p50 when moving (cm)", "hand_box_cm_p50_p95_when_moving", 0),
    ("samples: foot-box p50 when moving (cm)", "foot_box_cm_p50_p95_when_moving", 0),
    ("samples: moving windows without contact", "moving_windows_without_contact", None),
]


def value(bench, group, key, field, skill):
    """A benchmark number, per skill when the prior reports one (single-skill priors: pooled)."""
    entry = bench.get(group, {}).get(key, {})
    if skill is not None and group == "real_by_stage" and "real_by_skill_stage" in bench:
        entry = bench["real_by_skill_stage"].get(skill, {}).get(key, {})
        field = "reward_ratio_vs_skill_real"
    elif skill is not None and "by_skill" in entry:
        entry = entry["by_skill"].get(skill, {})
    v = entry.get(field)
    return float("nan") if v is None else v


def final_losses(log):
    rows = list(csv.reader(log.open(), delimiter=" ", skipinitialspace=True))
    head, last = rows[0], rows[-1]
    col = {k: i for i, k in enumerate(head)}
    get = lambda k: float(last[col[k]]) if k in col else float("nan")  # noqa: E731
    return {
        "iterations": int(float(last[col["Iteration"]])),
        "val": get("EvalEMA_val_all"),
        "gap": get("EvalEMA_val_all") - get("EvalEMA_train_all"),
        "val_setdown_gap": get("EvalEMA_val_setdown") - get("EvalEMA_train_setdown"),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=Path, nargs="*", default=[], help="Run dirs, read from <run>/eval")
    parser.add_argument("--evals", type=Path, nargs="*", default=[], help="Evaluation folders <run>/<folder>")
    parser.add_argument("--split", default="test")
    parser.add_argument("--skill", help="Per-skill numbers of skill-conditioned priors, e.g. push")
    parser.add_argument("--names", nargs="*")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    evals = [r / "eval" for r in args.runs] + list(args.evals)
    if not evals:
        parser.error("Give --runs or --evals")
    names = args.names or [
        e.parent.name.replace("carry_prior_", "").replace("_20261008", "") + ("" if e.name == "eval" else f" ({e.name})")
        for e in evals
    ]
    columns, reference = [], {}
    for folder in evals:
        run = folder.parent
        config = yaml.safe_load((run / "output/diffusion_config.yaml").read_text())
        bench = json.loads((folder / f"bench_{args.split}_final.json").read_text())
        sample = {}
        for sampler in ("ddim10", "ddpm50"):
            path = folder / f"sample_checks_{args.split}_final_{sampler}_x5.json"
            if not path.exists() and args.split == "test":
                path = folder / f"sample_checks_final_{sampler}_x5.json"
            if path.exists():
                checks = json.loads(path.read_text())
                key = f"_{args.skill}" if args.skill and f"generated_{args.skill}" in checks else ""
                sample[sampler] = checks["generated" + key]
                reference.setdefault(sampler, checks["real" + key])
        config["training_data"] = "mirrored" if "mirror" in run.name else "original"
        columns.append((config, bench, sample, final_losses(run / "output/log.txt")))
    names.append("real windows")
    lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]

    def row(label, values):
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    def fmt(m, key, k, digits=".3g"):
        if m is None or key not in m:
            return "-"
        return f"{m[key] if k is None else m[key][k]:{digits}}"

    row("split / skill", [f"{args.split} / {args.skill or 'all'}"] * len(columns) + [""])
    row("window (s)", [f"{(c['input_dim'] // c['input_channel'] - 1) / 50:.2f}" for c, *_ in columns] + [""])
    row(
        "layers x width",
        [f"{c['num_layers']} x {64 * c['num_attention_heads']}" for c, *_ in columns] + [""],
    )
    row("prediction", [c["estimate_mode"] for c, *_ in columns] + [""])
    row("skills", ["carry/push/slide" if c.get("arch_name") == "CondDiT" else "carry" for c, *_ in columns] + [""])
    row("training data", [c["training_data"] for c, *_ in columns] + [""])
    for label, group, key, field in ROWS:
        row(label, [f"{value(b, group, key, field, args.skill):.3f}" for _, b, *_ in columns] + [""])
    for other in ("carry", "push", "slide"):
        if args.skill and other != args.skill:
            cells = [b.get("skill_matrix", {}).get(args.skill, {}).get(other) for _, b, *_ in columns]
            row(f"AUROC {args.skill} motion labeled {other}", [f"{c['auroc']:.3f}" if c else "-" for c in cells] + [""])
            row(f"reward x {args.skill} motion labeled {other}", [f"{c['reward_ratio']:.3f}" if c else "-" for c in cells] + [""])
    for sampler in ("ddim10", "ddpm50"):
        for label, key, k in SAMPLE_ROWS:
            values = [fmt(s.get(sampler), key, k) for _, _, s, _ in columns]
            row(f"{label} [{sampler}]", values + [fmt(reference.get(sampler), key, k)])
    row("val loss (EMA)", [f"{loss['val']:.4f}" for *_, loss in columns] + [""])
    row("val - train loss", [f"{loss['gap']:.4f}" for *_, loss in columns] + [""])
    row("val - train loss, set-down", [f"{loss['val_setdown_gap']:.4f}" for *_, loss in columns] + [""])
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
