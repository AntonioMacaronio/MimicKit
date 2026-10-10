"""Generate carry-prior windows next to real held-out windows with the same box sizes.

Saves unnormalized v3 features for both sets, for check_carry_samples.py (MuJoCo
forward kinematics) and replay. --samples_per_size draws several generated windows
per real window's box size, each from its own noise (generated_motion is then
[num, samples_per_size, frames, 216]). Generation uses the EMA denoiser, conditioned on
the box sizes of the drawn real windows. The default is 10-step DDIM: with T = 50
and the squaredcos_cap_v2 schedule the last step has alphas_cumprod ~ 1e-6, so the
unclipped first step of 50-step DDPM/DDIM injects huge clean-sample estimates.

    python tools/diffusion_model/sample_carry_prior.py --model_dir <run>/output --out <run>/samples_test.npz
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.append("mimickit")

from learning.tinymdm.tinymdm_model import TinyMDMModel  # noqa: E402

from mimickit_smp.dataset import CarryWindowDataset, PackedWindowSampler  # noqa: E402
from mimickit_smp.features import encode_window  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--model_file", default="model.pt")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--sampler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--steps", type=int, default=10, help="DDIM inference steps")
    parser.add_argument("--samples_per_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    config = yaml.safe_load((args.model_dir / "diffusion_config.yaml").read_text())
    config["env_config"] = str(args.model_dir / "env_config.yaml")
    model = TinyMDMModel(config, args.device)
    model.load_state_dict(torch.load(args.model_dir / args.model_file, map_location=args.device))
    model.to(args.device).eval()
    index = args.index or Path(os.path.expandvars(config["dataset_index"]))
    dataset = CarryWindowDataset(
        str(index), split=args.split, quality_checks=config.get("quality_checks") or ()
    )
    if dataset.schema.sha256 != config["feature_schema_sha256"]:
        raise ValueError("The cache's feature schema differs from the prior's")
    width = dataset.schema.window_frames
    pick = torch.from_numpy(np.random.default_rng(args.seed).choice(len(dataset), args.num, replace=False))
    states = PackedWindowSampler(dataset).gather(pick)
    real, cond = encode_window(states.double(), dataset.schema)
    generated = []
    for k in range(args.samples_per_size):
        # Sample k of every window has its own seed, so adding samples never changes earlier ones.
        torch.manual_seed(args.seed + 1000 * k)
        draws = []
        for i in range(0, args.num, args.batch):
            c = cond[i : i + args.batch].float().to(args.device)
            sample = model.sample_ema(
                shape=(width * 216,),
                batch_size=len(c),
                device=args.device,
                sampler=args.sampler,
                num_inference_steps=args.steps,
                cond=c,
            )
            draws.append(model.unnormalize(sample.reshape(len(c), width, -1)).cpu())
        generated.append(torch.cat(draws))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        generated_motion=torch.stack(generated, 1).numpy(),
        real_motion=real.float().numpy(),
        box_size=cond.float().numpy(),
        stage_ids=dataset.window_stage_ids[pick.numpy()],
        feature_schema_sha256=dataset.schema.sha256,
        index=str(index),
        model=str(args.model_dir / args.model_file),
        split=args.split,
        sampler=f"{args.sampler}_{args.steps if args.sampler == 'ddim' else 50}",
    )
    print(f"Saved {args.num} generated and real {width}-frame windows: {args.out}")


if __name__ == "__main__":
    main()
