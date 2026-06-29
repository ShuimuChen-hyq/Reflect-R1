# Reflect-R1：面向长视频理解中自我纠错的证据驱动反思

*结合时间搜索和阶段解耦强化学习的长视频理解证据驱动自我纠错框架。*

[📄 [论文](https://arxiv.org/abs/2606.27922)] [🤗 [模型](https://huggingface.co/CSDDSFSFSAFSAF/Reflect-R1)] [📦 [数据集](https://huggingface.co/datasets/CSDDSFSFSAFSAF/Reflect-R1-data)] [💻 [代码](https://github.com/ShuimuChen-hyq/Reflect-R1)]

## 📰 最新消息

🔥 **[2026/06/26]** 我们的 [Reflect-R1](https://arxiv.org/abs/2606.27922) 论文已发布在 arXiv。

🔥 **[2026/06/17]** Reflect-R1 已被 ECCV 2026 接收。

## 👁️ 方法概览

Reflect-R1 是一个面向长视频理解的证据驱动反思框架。它将自我纠错分解为直觉、验证和仲裁三个阶段。不同于仅依赖模型内部闭环反思的方法，Reflect-R1 会检索时间维度上的视觉证据，用独立证据验证初始直觉，并在存在冲突时进行仲裁，最终生成答案。

![Reflect-R1 overview](assets/fig1.png)

我们进一步提出 Stage-Decoupled GRPO（SD-GRPO），在不同推理阶段分别计算优势函数。该机制可以缓解长链条多阶段推理中的策略耦合问题，使模型学习真实的纠错行为，而不是寻找优化捷径。

## 🚀 快速开始

### 🏝️ 环境配置

**步骤 1：** 准备运行环境。

安装 Python 依赖并暴露本地包：

```bash
pip install -r requirements.txt
pip install -e clip_as_service/client
pip install -e clip_as_service/server

export PYTHONPATH=$PWD:$PWD/clip_as_service/client:$PWD/clip_as_service/server:$PYTHONPATH
```

**步骤 2：** 启动时间证据服务。

下载预训练 SigLIP 模型：

```bash
hf download google/siglip-so400m-patch14-384 --local-dir /path/to/siglip-so400m-patch14-384
```

启动服务：

```bash
export SIGLIP_MODEL_PATH=/path/to/siglip-so400m-patch14-384
export SIGLIP_DEVICE=cuda
export CLIP_PORT=52000

bash scripts/open_source/start_temporal_evidence_server.sh
```

### 📦️ 数据集

Reflect-R1 的公开训练数据托管在 Hugging Face：

```bash
hf download CSDDSFSFSAFSAF/Reflect-R1-data \
  --repo-type dataset \
  --local-dir /path/to/Reflect-R1-data
```

数据集仓库包含：

```text
data/reflect_r1_cot_90k.jsonl        Reflect-R1-CoT-90k 冷启动 SFT 数据
data/reflect_r1_rl_30k_short.json    Reflect-R1-RL-30k 短视频划分
data/reflect_r1_rl_30k_long.json     Reflect-R1-RL-30k 长视频划分
archives/short.tar.zst               解压后位于 short/ 下的视频
archives/long.tar.zst                解压后位于 long/ 下的视频
```

解压视频文件：

```bash
cd /path/to/Reflect-R1-data
tar -I zstd -xf archives/short.tar.zst
tar -I zstd -xf archives/long.tar.zst

export SHORT_VIDEO_DIR=/path/to/Reflect-R1-data/short
export LONG_VIDEO_DIR=/path/to/Reflect-R1-data/long
```

JSON 中的 `video_path` 字段按 `short/` 和 `long/` 目录组织。本地路径准备细节见 [docs/training.md](docs/training.md)。

### 🏗️ 冷启动 SFT

冷启动 SFT 用于让模型学习结构化反思格式。完整训练是多节点多 GPU 任务，按 2 台 H200 节点设计。

```bash
export MODEL_NAME_OR_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export SFT_DATA_JSONL=/path/to/Reflect-R1-data/data/reflect_r1_cot_90k.jsonl
export SHORT_VIDEO_DIR=/path/to/Reflect-R1-data/short
export LONG_VIDEO_DIR=/path/to/Reflect-R1-data/long
export HOSTFILE=/path/to/mpi_hostfile
export MASTER_ADDR=10.0.0.1

bash scripts/open_source/train_reflect_r1_cold_start_sft.sh
```

### 📋️ SD-GRPO 训练

启动 SD-GRPO 前需要先运行时间证据服务：

```bash
export SIGLIP_URL=grpc://127.0.0.1:52000
```

**步骤 1：** 运行 SD-GRPO Stage I，用于仲裁阶段热身。

```bash
export MODEL_NAME_OR_PATH=/path/to/sft-checkpoint
export GRPO_SHORT_JSON=/path/to/Reflect-R1-data/data/reflect_r1_rl_30k_short.json
export GRPO_LONG_JSON=/path/to/Reflect-R1-data/data/reflect_r1_rl_30k_long.json
export SHORT_VIDEO_DIR=/path/to/Reflect-R1-data/short
export LONG_VIDEO_DIR=/path/to/Reflect-R1-data/long

bash scripts/open_source/train_sd_grpo_stage1_arbitration_warmup.sh
```

**步骤 2：** 运行 SD-GRPO Stage II，用于完整链路优化。

```bash
export MODEL_NAME_OR_PATH=/path/to/stage1-checkpoint

bash scripts/open_source/train_sd_grpo_stage2_full_chain.sh
```

开源启动脚本使用的奖励函数如下：

```text
Stage I:  v11_valid_tool_split_S3_wandb_no_reasoning_fix
Stage II: v11_valid_tool_split_S123_no_reasoning
```

更多训练细节和常用覆盖参数见 [docs/training.md](docs/training.md)。

## 🔖 引用

如果 Reflect-R1 对你的研究或应用有帮助，请使用以下 BibTeX 引用：

```bibtex
@article{chen2026reflectr1,
  title   = {Reflect-R1: Evidence-Driven Reflection for Self-Correction in Long Video Understanding},
  author  = {Shuimu Chen and Yuteng Chen and Yuanshen Guan and Zebang Cheng and Zeyu Zhang and Shengqian Qin and Bin Xia and Jiaran Li and Wenming Yang and Fei Ma},
  journal = {arXiv preprint arXiv:2606.27922},
  year    = {2026}
}
```

## 🎟️ 许可证

本项目基于 [Apache 2.0 license](LICENSE) 发布。

## 🏅 致谢

感谢以下项目作者的贡献：

* [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
* [trl](https://github.com/huggingface/trl)
* [vLLM](https://github.com/vllm-project/vllm)
* [DeepSpeed](https://github.com/microsoft/DeepSpeed)
* [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384)
