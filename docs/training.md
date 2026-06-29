# Training

This repository exposes the Reflect-R1 cold-start SFT launcher and the SD-GRPO launchers used for the two reinforcement-learning stages described in the paper.

## Data Paths

Public training JSONs use relative video paths. Store `video_path` as one of:

```text
short/<relative path inside the short-video archive>
long/<relative path inside the long-video archive>
```

If your existing JSONs use another local layout, convert them with your current source roots:

```bash
export SHORT_SOURCE_PREFIX=/path/to/old/short-video/root
export LONG_SOURCE_PREFIX=/path/to/old/long-video/root

python scripts/open_source/prepare_reflect_r1_data.py strip \
  --input /path/to/source_sft.jsonl \
  --output /path/to/public_sft.jsonl \
  --short-prefix /path/to/old/short-video/root \
  --long-prefix /path/to/old/long-video/root
```

At training time, map the relative paths to the two local video folders:

```bash
export SHORT_VIDEO_DIR=/path/to/short
export LONG_VIDEO_DIR=/path/to/long
```

The launch scripts localize into `.cache/reflect-r1/` automatically. Localized JSONs, outputs, and caches are ignored by git.

## Environment

Create a Python 3.11 conda environment, install the verified PyTorch/CUDA stack, then install the Reflect-R1 dependencies and local CLIP client/server packages:

```bash
conda create -n reflect-r1 python=3.11 -y
conda activate reflect-r1

python -m pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e clip_as_service/client
pip install -e clip_as_service/server

export PYTHONPATH=$PWD:$PWD/clip_as_service/client:$PWD/clip_as_service/server:$PYTHONPATH
```

The launch scripts use `flash_attention_2` by default. `requirements.txt` pins a FlashAttention wheel verified with Python 3.11, PyTorch 2.6, and CUDA 12.4. If you use a different CUDA/PyTorch stack, install the matching FlashAttention wheel for that stack. For CPU-light local environment checks only, you can temporarily fall back to PyTorch SDPA:

```bash
export ATTN_IMPLEMENTATION=sdpa
```

Local validation for this release used PyTorch, TRL, DeepSpeed, vLLM, Transformers, and the local CLIP client/server packages.

## Reflect-R1 Cold-Start SFT

Entrypoint:

```text
reflect_r1/sft_video_r1.py
```

Launch:

```bash
export MODEL_NAME_OR_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export SFT_DATA_JSONL=/path/to/public_sft.jsonl
export SHORT_VIDEO_DIR=/path/to/short
export LONG_VIDEO_DIR=/path/to/long
export HOSTFILE=/path/to/mpi_hostfile
export MASTER_ADDR=10.0.0.1

bash scripts/open_source/train_reflect_r1_cold_start_sft.sh
```

SFT is intended to run with MPI across multiple nodes and GPUs. The full cold-start SFT run was designed for 2 H200 nodes; a single machine is only suitable for local debugging. The launcher follows the original training layout:

```text
mpirun -np $NP -> scripts/open_source/_sft_mpi_worker.sh -> reflect_r1/sft_video_r1.py
```

The worker maps `OMPI_COMM_WORLD_RANK`, `OMPI_COMM_WORLD_LOCAL_RANK`, and `OMPI_COMM_WORLD_SIZE` to the PyTorch distributed variables `RANK`, `LOCAL_RANK`, and `WORLD_SIZE`.

Useful overrides:

```bash
export MPIRUN=/path/to/mpirun
export NP=16
export OUTPUT_DIR=outputs/reflect-r1-sft
export LEARNING_RATE=1e-6
export REPORT_TO=none
```

`HOSTFILE` should be an OpenMPI hostfile with `slots=` entries. If `NP` is not set, the script sums those slots. `MASTER_ADDR` can also be provided through `MY_NODE_IP`, matching the original cluster launcher. The script intentionally requires a hostfile by default because the full SFT run is a multi-node multi-GPU job.

## SD-GRPO

Entrypoint:

```text
reflect_r1/train_VLLM_stage_1_split.py
```

The dataset loader uses the localized absolute `video_path` stored inside each JSON record. The `--video_folder` argument is retained for compatibility and should not be used as the source of truth for Panda paths.

Start the temporal evidence server first. It serves SigLIP features for the retrieval tool used by the intuition and arbitration stages:

```bash
export SIGLIP_MODEL_PATH=/path/to/siglip-so400m-patch14-384
export SIGLIP_DEVICE=cuda
export CLIP_PORT=52000
bash scripts/open_source/start_temporal_evidence_server.sh
```

Then launch SD-GRPO Stage I. This stage warms up the arbitration behavior with the S3 reward:

```bash
export MODEL_NAME_OR_PATH=/path/to/stage1-or-sft-checkpoint
export GRPO_SHORT_JSON=/path/to/reflect_r1_rl_30k_short.json
export GRPO_LONG_JSON=/path/to/reflect_r1_rl_30k_long.json
export SHORT_VIDEO_DIR=/path/to/short
export LONG_VIDEO_DIR=/path/to/long

bash scripts/open_source/train_sd_grpo_stage1_arbitration_warmup.sh
```

Launch SD-GRPO Stage II. This stage performs full-chain optimization over intuition, verification, and arbitration:

```bash
bash scripts/open_source/train_sd_grpo_stage2_full_chain.sh
```

Reward functions:

```text
Stage I arbitration warm-up: v11_valid_tool_split_S3_wandb_no_reasoning_fix
Stage II full-chain:         v11_valid_tool_split_S123_no_reasoning
```

Common SD-GRPO overrides:

```bash
export NPROC_PER_NODE=8
export OUTPUT_DIR=outputs/reflect-r1-grpo
export LEARNING_RATE=1e-6
export REPORT_TO=none
export SIGLIP_URL=grpc://127.0.0.1:52000
export DATALOADER_PIN_MEMORY=false  # optional for machines with restrictive memlock settings
```

Run a small GPU validation on your own machine before launching a full job.
