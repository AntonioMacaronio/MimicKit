#!/usr/bin/env bash
# Benchmark, sample and check one finished prior run on unmirrored held-out splits.
#   tools/diffusion_model/evaluate_carry_prior_run.sh <run dir> <gpu> <eval cache index.json> [splits...]
# Splits default to test. The skill prior also needs validation: most held-out push and
# all held-out slide windows are validation actors. EVAL_DIR (default <run dir>/eval)
# keeps a re-evaluation apart from earlier results.
# Epsilon priors sample with 10-step DDIM (50-step DDPM diverges at t = 49); sample (x0)
# priors also with 50-step DDPM, which they handle.
set -euo pipefail
run=$1 gpu=$2 index=$3
shift 3
splits=("${@:-test}")
cd "$(dirname "$0")/../.."
source scripts/source_smp_reward.sh >/dev/null
py=/home/sky/miniconda3/envs/ei_world_model/bin/python
mujoco_py=/data/sky/projects/FAR-Holosoma/venvs/hsmujoco/bin/python
eval_dir=${EVAL_DIR:-$run/eval}
mkdir -p "$eval_dir"
mode=$(grep -E "^estimate_mode:" "$run/output/diffusion_config.yaml" | awk '{print $2}')
samplers=("ddim 10")
[ "$mode" = sample ] && samplers=("ddpm 50" "ddim 10")
for split in "${splits[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu $py tools/diffusion_model/eval_carry_prior.py --model_dir "$run/output" \
        --index "$index" --split "$split" --num 2048 --out "$eval_dir/bench_${split}_final.json" \
        > "$eval_dir/bench_${split}_final.log" 2>&1
    for spec in "${samplers[@]}"; do
        set -- $spec
        out="$eval_dir/samples_${split}_final_$1$2_x5.npz"
        CUDA_VISIBLE_DEVICES=$gpu $py tools/diffusion_model/sample_carry_prior.py --model_dir "$run/output" \
            --index "$index" --split "$split" --num 2048 --samples_per_size 5 --sampler "$1" --steps "$2" \
            --out "$out" > "$eval_dir/sample_${split}_$1$2.log" 2>&1
        PYTHONPATH=$PWD $mujoco_py tools/diffusion_model/check_carry_samples.py --samples "$out" \
            --out "$eval_dir/sample_checks_${split}_final_$1$2_x5.json" \
            > "$eval_dir/sample_checks_${split}_final_$1$2_x5.log" 2>&1
    done
done
echo "evaluated $run ($mode; ${splits[*]})"
