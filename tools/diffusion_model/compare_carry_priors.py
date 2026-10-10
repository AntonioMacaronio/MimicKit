"""Side-by-side table of trained carry priors: reward benchmark, generated-sample checks, losses.

Reads each run's eval/bench_test_final.json (eval_carry_prior.py),
eval/sample_checks_final*.json (check_carry_samples.py) and output/log.txt.
Validation losses are only comparable between runs with the same estimate_mode.

    python tools/diffusion_model/compare_carry_priors.py --runs <run dir> [<run dir> ...] [--out table.md]
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
    ("AUROC box +15 cm", "corruptions", "box_offset", "auroc"),
    ("AUROC swapped box size", "corruptions", "size_swap", "auroc"),
    ("AUROC 0.5x speed (<0.5 preferred)", "corruptions", "slow", "auroc"),
    ("AUROC static (<0.5 preferred)", "corruptions", "static", "auroc"),
    ("AUROC dropped: penetration", "filtered_out_real", "physical", "auroc"),
    ("AUROC dropped: chatter", "filtered_out_real", "chatter", "auroc"),
    ("AUROC dropped: foot slide", "filtered_out_real", "foot_slide", "auroc"),
    ("AUROC dropped: unsupported", "filtered_out_real", "support", "auroc"),
    ("reward x static vs real", "corruptions", "static", "reward_ratio"),
    ("reward x 0.5x speed vs real", "corruptions", "slow", "reward_ratio"),
    ("reward x real pickup vs real", "real_by_stage", "pickup", "reward_ratio_vs_all_real"),
    ("reward x real lifted vs real", "real_by_stage", "lifted", "reward_ratio_vs_all_real"),
]
SAMPLE_ROWS = [
    ("samples: FK error p95 (cm)", "body_fk_error_cm_p50_p95", 1),
    ("samples: hands off lifted box", "hands_off_box_frames_when_lifted", None),
    ("samples: hand-box p50 (cm)", "hand_box_cm_p50_p95_when_lifted", 0),
    ("samples: planted foot p50 (m/s)", "planted_foot_speed_m_s_p50_p95", 0),
    ("samples: box velocity error p50", "box_velocity_error_m_s_p50_p95", 0),
]


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
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--names", nargs="*")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    names = args.names or [r.name.replace("carry_prior_", "").replace("_20261008", "") for r in args.runs]
    columns = []
    for run in args.runs:
        config = yaml.safe_load((run / "output/diffusion_config.yaml").read_text())
        bench = json.loads((run / "eval/bench_test_final.json").read_text())
        sample = {}
        for sampler in ("ddim10", "ddpm50"):
            path = run / "eval" / f"sample_checks_final_{sampler}_x5.json"
            if path.exists():
                sample[sampler] = json.loads(path.read_text())["generated"]
        config["training_data"] = "mirrored" if "mirror" in run.name else "original"
        columns.append((config, bench, sample, final_losses(run / "output/log.txt")))
    lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]

    def row(label, values):
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    row("window (s)", [f"{(c['input_dim'] // c['input_channel'] - 1) / 50:.2f}" for c, *_ in columns])
    row("layers x width", [f"{c['num_layers']} x {64 * c['num_attention_heads']}" for c, *_ in columns])
    row("prediction", [c["estimate_mode"] for c, *_ in columns])
    row("training data", [c["training_data"] for c, *_ in columns])
    for label, group, key, field in ROWS:
        row(label, [f"{b[group].get(key, {}).get(field, float('nan')):.3f}" for _, b, *_ in columns])
    for sampler in ("ddim10", "ddpm50"):
        for label, key, k in SAMPLE_ROWS:
            values = []
            for _, _, s, _ in columns:
                m = s.get(sampler)
                values.append("-" if m is None else f"{m[key] if k is None else m[key][k]:.3g}")
            row(f"{label} [{sampler}]", values)
    row("val loss (EMA)", [f"{loss['val']:.4f}" for *_, loss in columns])
    row("val - train loss", [f"{loss['gap']:.4f}" for *_, loss in columns])
    row("val - train loss, set-down", [f"{loss['val_setdown_gap']:.4f}" for *_, loss in columns])
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
