import sys
sys.path.append("mimickit")

from argparse import ArgumentParser
import numpy as np
import os
import random
import shutil
import torch
import torch.optim as optim
import yaml

import anim.motion as motion
from learning.tinymdm.tinymdm_model import TinyMDMModel
import util.logger as logger

def build_dataset(config, device, split="train"):
    if config.get("dataset_type", "character") == "hoi":
        from mimickit_smp.tinymdm_data import HOIMotionPriorData
        return HOIMotionPriorData(config, device, split=split)
    if config.get("dataset_type", "character") != "character":
        raise ValueError("Unknown prior dataset_type")
    from motion_prior_dataset import MotionPriorData
    return MotionPriorData(config, device)

def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return

EVAL_STAGES = (None, "pickup", "lifted", "setdown")

def build_eval_sets(config, dataset_env, device):
    """Fixed windows (all stages, then per stage) from the training and validation splits."""
    num = config.get("eval_windows", 0)
    if not num or not hasattr(dataset_env, "fixed_windows"):
        return {}
    envs = {"train": dataset_env}
    if config.get("validation_split"):
        envs["val"] = build_dataset(config, device, split=config["validation_split"])
    sets = {}
    for name, env in envs.items():
        for k, stage in enumerate(EVAL_STAGES):
            motion, cond = env.fixed_windows(num, seed=10_000 + k, stage=stage)
            sets[f"{name}_{stage or 'all'}"] = (motion, cond)
    return sets

@torch.no_grad()
def eval_losses(model, sets, config, device, use_ema):
    """Denoising loss with fixed timesteps and noise, so values are comparable across iterations."""
    denoiser = model.ema_dmodel if use_ema else model.dmodel
    out = {}
    for key, (windows, cond) in sets.items():
        x = model.normalize(windows.to(device).reshape(windows.shape[0], -1, config["input_channel"])).reshape(windows.shape[0], -1)
        generator = torch.Generator(device=device).manual_seed(77)
        t = torch.randint(0, len(model.diffusion_scheduler.timesteps), (x.shape[0],), device=device, generator=generator)
        noise = torch.randn(x.shape, device=device, generator=generator)
        kwargs = model._with_condition({"cond": cond.to(device)} if cond is not None else {})
        total = 0.0
        for i in range(0, x.shape[0], 4096):
            noised = model.diffusion_scheduler.add_noise(x[i:i + 4096], noise[i:i + 4096], t[i:i + 4096])
            chunk = {k: v[i:i + 4096] for k, v in kwargs.items()}
            pred = denoiser(noised, t[i:i + 4096], **chunk)
            # Same target as training: noise (epsilon), the clean sample, or v.
            mode = model.diffusion_scheduler.config.prediction_type
            if mode == "epsilon":
                target = noise[i:i + 4096]
            elif mode == "sample":
                target = x[i:i + 4096]
            else:
                target = model.diffusion_scheduler.get_velocity(x[i:i + 4096], noise[i:i + 4096], t[i:i + 4096])
            total += torch.nn.functional.l1_loss(pred, target, reduction="sum").item()
        out[key] = total / x.numel()
    return out

def build_logger(log_file):
    log = logger.Logger()
    log.set_step_key("Iter")
    log.configure_output_file(log_file)
    return log

@torch.no_grad()
def generate(model, dataset, obs_space, config, device, out_motion_dir, enable_ema=False, num_samples=16):
    fps = dataset.control_freq
    num_frames = obs_space.shape[-1] // config["input_channel"]

    # Conditional priors generate for box sizes drawn from the training windows.
    kwargs = {}
    if config.get("cond_dim", 0):
        kwargs["cond"] = dataset.sample_conditions(num_samples).to(device)

    if enable_ema:
        gen_samples = model.sample_ema(shape=obs_space.shape, batch_size=num_samples, device=device, **kwargs)
    else:
        gen_samples = model.sample(shape=obs_space.shape, batch_size=num_samples, device=device, **kwargs)

    gen_samples = model.unnormalize(gen_samples.reshape([num_samples, num_frames, -1]))

    if hasattr(dataset, "save_generated_samples"):
        dataset.save_generated_samples(gen_samples, out_motion_dir, **kwargs)
        return

    for i, gen_sample in enumerate(gen_samples):
        frames = dataset.convert_sample_to_frames(gen_sample).detach().cpu().numpy()
        sample_motion = motion.Motion(loop_mode=motion.LoopMode.CLAMP, fps=fps, frames=frames)

        out_motion_file = os.path.join(out_motion_dir, f"motion_{i:03}.pkl")
        sample_motion.save(out_motion_file)

        pos_sample = dataset.calc_joint_position_from_frame(frames)[None,...]
        dataset.plot_jnt(jnt_pos=pos_sample, out_path=os.path.join(out_motion_dir, f"anim_{i:03}"))
    
    return
    
@torch.no_grad()
def test(cfg_path, model_file, out_dir=None, num_samples=16, device="cuda"):
    fixseed(0)
    assert(out_dir is not None and out_dir != ""), "Must specify --out_dir"
    assert(model_file != ""), "Must specify --model_file"

    with open(cfg_path, "r") as stream:
        config = yaml.safe_load(stream)

    with open(config["env_config"], "r") as stream:
        env_config = yaml.safe_load(stream)
    
    out_motion_dir = os.path.join(out_dir, "samples")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_motion_dir, exist_ok=True)

    dataset_env = build_dataset(config, device)
    obs_space = dataset_env.get_obs_space()
    num_obs_steps = env_config["num_disc_obs_steps"]
    config["input_dim"] = obs_space.shape[-1]
    config["input_channel"] = int(config["input_dim"] / num_obs_steps)
    config["cond_dim"] = getattr(dataset_env, "condition_dim", 0)

    priormodel = TinyMDMModel(config, device)
    prior_state_dict = torch.load(model_file, map_location=device)
    priormodel.load_state_dict(prior_state_dict)
    
    priormodel.eval()
    priormodel.to(device)
    
    generate(priormodel, dataset_env, obs_space, config, device, out_motion_dir=out_motion_dir,
            enable_ema=priormodel.model_ema, num_samples=num_samples)
    return

def train(cfg_path, out_dir=None, device="cuda"):
    assert out_dir is not None and out_dir != "", "Must specify --out_dir"
    
    with open(cfg_path, "r") as stream:
        config = yaml.safe_load(stream)
    if "seed" in config:
        fixseed(config["seed"])
    if config.get("allow_tf32", False):
        # TF32 matmuls: ~2.5x faster for large DiTs on Ampere/Hopper; weights stay fp32.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    env_file = config["env_config"]
    with open(env_file, "r") as stream:
        env_config = yaml.safe_load(stream)

    out_motion_dir = os.path.join(out_dir, "samples")
    out_model_file = os.path.join(out_dir, "model.pt")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_motion_dir, exist_ok=True)

    out_env_config_file = os.path.join(out_dir, "env_config.yaml")
    shutil.copy(env_file, out_env_config_file)
    
    config["env_config"] = out_env_config_file
    out_config_file = os.path.join(out_dir, "diffusion_config.yaml")
    with open(out_config_file, "w") as stream:
        yaml.dump(config, stream)

    out_log_file = os.path.join(out_dir, "log.txt")
    log = build_logger(out_log_file)
    
    batch_size = config["batch_size"]
    num_samples_stat = config.get("num_samples_stat", 10_000)
    output_iter = config.get("output_iter", 2_000)
    grad_clip_norm = config.get("grad_clip_norm", 1.0)

    dataset_env = build_dataset(config, device)
    obs_space = dataset_env.get_obs_space()
    num_obs_steps = env_config["num_disc_obs_steps"]
    
    config["input_dim"] = obs_space.shape[-1]
    config["input_channel"] = int(config["input_dim"] / num_obs_steps)
    print(f"Input_channel: {config['input_channel']}")
    # A continuous condition (HOI box size) is given to the denoiser without noise.
    cond_dim = getattr(dataset_env, "condition_dim", 0)
    config["cond_dim"] = cond_dim

    # Persist the resolved feature dimensions and HOI schema with the checkpoint.
    if hasattr(dataset_env, "schema"):
        config["feature_schema_sha256"] = dataset_env.schema.sha256
    with open(out_config_file, "w") as stream:
        yaml.safe_dump(config, stream)
    if hasattr(dataset_env, "save_metadata"):
        dataset_env.save_metadata(out_dir)

    if cond_dim:
        samples, cond = dataset_env.fetch_obs_demo(num_samples_stat, return_condition=True)
    else:
        samples = dataset_env.fetch_obs_demo(num_samples_stat)
    model = TinyMDMModel(config, device)
    model.update_normalizer(samples)
    if cond_dim:
        model.update_cond_normalizer(cond)
    eval_sets = build_eval_sets(config, dataset_env, device)
    checkpoint_every = config.get("checkpoint_every", 0)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'])
    model.to(device)
    
    model.train()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of model parameters: {num_params / 1_000_000:.2f} M")

    num_iters = config['num_iterations']
    curr_iters = 0
    loss_sum = 0

    while curr_iters < num_iters:
        batch_kwargs = {}
        if cond_dim:
            samples, cond = dataset_env.fetch_obs_demo(batch_size, return_condition=True)
            batch_kwargs["cond"] = cond.to(device)
        else:
            samples = dataset_env.fetch_obs_demo(batch_size)
        samples = samples.clone().detach()
        samples = samples.to(device)
        samples = model.normalize(samples.reshape(batch_size, -1, config["input_channel"])).reshape(batch_size, -1)

        loss = model(samples, **batch_kwargs)
        loss_sum += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        if model.model_ema is not False:
            model.ema_dmodel.update()

        if (curr_iters % output_iter == 0 and curr_iters != 0) or (curr_iters == num_iters - 1):
            model.eval()
            generate(model, dataset_env, obs_space, config, device, out_motion_dir=out_motion_dir,
                    enable_ema=model.model_ema, num_samples=16)
            torch.save(model.state_dict(), out_model_file)
            model.train()

            log.log("Iteration", curr_iters, collection="0_Main")
            log.log("Loss", loss_sum / output_iter, collection="0_Main")
            if eval_sets:
                model.eval()
                for key, value in eval_losses(model, eval_sets, config, device, use_ema=False).items():
                    log.log(f"Eval_{key}", value, collection="1_Eval")
                if model.model_ema is not False:
                    # The logger fixes its columns at the first row: log NaN until EMA updates begin.
                    if curr_iters >= config.get("model_ema_update_after", 0):
                        ema = eval_losses(model, eval_sets, config, device, use_ema=True)
                    else:
                        ema = dict.fromkeys(eval_sets, float("nan"))
                    for key, value in ema.items():
                        log.log(f"EvalEMA_{key}", value, collection="2_EvalEMA")
                model.train()
            log.print_log()
            log.write_log()

            loss_sum = 0

        if checkpoint_every and curr_iters and curr_iters % checkpoint_every == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, f"model_{curr_iters:07d}.pt"))

        curr_iters += 1

    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--cfg_path", type=str, default="tools/diffusion_model/config/tinymdm.yaml")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model_file", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.mode == "train":
        print("Training new model...")
        train(args.cfg_path, out_dir=args.out_dir, device=args.device)
    else:
        print("Testing model...")
        test(args.cfg_path, args.model_file, out_dir=args.out_dir, device=args.device)
