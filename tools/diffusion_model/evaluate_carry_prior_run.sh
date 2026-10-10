#!/usr/bin/env bash
# Benchmark, sample and check one finished carry-prior run on the original (unmirrored) test split.
#   tools/diffusion_model/evaluate_carry_prior_run.sh <run dir> <gpu> <eval cache index.json>
# Epsilon priors sample with 10-step DDIM (50-step DDPM diverges at t = 49); sample (x0)
# priors also with 50-step DDPM, which they handle.
set -euo pipefail
run=$1 gpu=$2 index=$3
cd "$(dirname "$0")/../.."
source scripts/source_smp_reward.sh >/dev/null
py=/home/sky/miniconda3/envs/ei_world_model/bin/python
mujoco_py=/data/sky/projects/FAR-Holosoma/venvs/hsmujoco/bin/python
mkdir -p "$run/eval"
mode=$(grep -E "^estimate_mode:" "$run/output/diffusion_config.yaml" | awk '{print $2}')
CUDA_VISIBLE_DEVICES=$gpu $py tools/diffusion_model/eval_carry_prior.py --model_dir "$run/output" \
    --index "$index" --split test --num 2048 --out "$run/eval/bench_test_final.json" > "$run/eval/bench_test_final.log" 2>&1
samplers=("ddim 10")
[ "$mode" = sample ] && samplers=("ddpm 50" "ddim 10")
for spec in "${samplers[@]}"; do
    set -- $spec
    out="$run/eval/samples_test_final_$1$2_x5.npz"
    CUDA_VISIBLE_DEVICES=$gpu $py tools/diffusion_model/sample_carry_prior.py --model_dir "$run/output" --index "$index" \
        --num 2048 --samples_per_size 5 --sampler "$1" --steps "$2" --out "$out" > "$run/eval/sample_$1$2.log" 2>&1
    PYTHONPATH=$PWD $mujoco_py tools/diffusion_model/check_carry_samples.py --samples "$out" \
        --out "$run/eval/sample_checks_final_$1$2_x5.json" > "$run/eval/sample_checks_final_$1$2_x5.log" 2>&1
done
echo "evaluated $run ($mode)"
