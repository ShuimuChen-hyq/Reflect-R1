# Copyright 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Example usage:
accelerate launch \
    --config_file=deepspeed_zero2.yaml \
    train_video_llm.py \
    --dataset_name mfarre/simplevideoshorts \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --output_dir video-llm-output \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing
"""

import os
import json
import random
import statistics
import requests
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from datasets import load_dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
)
from accelerate import Accelerator
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from reflect_r1.utils.qwen_vl_utils import process_vision_info

from datasets import Dataset, DatasetDict
from copy import deepcopy
try:
    import wandb
except ImportError:
    wandb = None

import base64
from io import BytesIO
from PIL import Image
import copy
from typing import List, Dict, Any

if os.environ.get("ENABLE_SWANLAB") == "1":
    try:
        import swanlab
    except ImportError as exc:
        raise ImportError("ENABLE_SWANLAB=1 requires swanlab to be installed.") from exc
    swanlab.sync_wandb()


def _reports_to_wandb(report_to):
    if isinstance(report_to, str):
        return report_to == "wandb"
    if report_to is None:
        return False
    return "wandb" in report_to

def sanitize_content_item(ele: dict) -> dict:
    """清理 PyArrow schema 对齐产生的 None 键，并按 type 保留合法字段。"""
    if not isinstance(ele, dict):
        return ele

    # 先删掉所有 None 值字段（PyArrow 补齐的 key 大多在这里被清掉）
    ele = {k: v for k, v in ele.items() if v is not None}

    t = ele.get("type")
    if t == "text":
        keep = {"type", "text"}
        return {k: v for k, v in ele.items() if k in keep}

    elif t == "image":
        keep = {"type", "image"}
        return {k: v for k, v in ele.items() if k in keep}

    elif t == "video":
        keep = {"type", "video", "fps", "total_pixels", "min_pixels", "max_pixels", "max_frames"}
        return {k: v for k, v in ele.items() if k in keep}

    # 其他类型先只做 None 清理
    return ele


def get_current_device():
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"

def download_video(url: str, folder: str = '/tmp/videos/') -> str:
    """Download video if not already present locally."""
    filename = url.split("/")[-1]
    local_path = os.path.join(folder, filename)

    if os.path.exists(local_path):
        return local_path

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return local_path
    except requests.RequestException as e:
        raise Exception(f"Failed to download video: {e}")

# =========================================================================
# 新增的图像处理辅助函数
# =========================================================================
def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")

def decode_base64_to_pil(base64_str: str) -> Image.Image:
    """将 Base64 字符串还原为 PIL Image"""
    if base64_str.startswith("data:image"):
        # 去掉前缀 data:image/jpeg;base64, 等
        base64_str = base64_str.split("base64,")[1]
    
    img_data = base64.b64decode(base64_str)
    image_obj = Image.open(BytesIO(img_data))
    image = to_rgb(image_obj)
    return image


def _is_removed_base64_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.strip() in {"<base64_removed>", "base64_removed", ""}



# =========================================================================
# 替换后的 prepare_dataset 函数
# =========================================================================
# def prepare_dataset(example: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     SFT 专用预处理：
#     只做“还原”，不做“转换”。把 Tensor 化留给 Collator 统一处理。
#     """
#     # 1. 深度拷贝 messages，防止干扰原始数据
#     messages = copy.deepcopy(example.get('messages', []))
#     split_source = example.get('split_source', 'unknown')
#     video_path = example.get('video_path', '')

#     # 你可以在这里统一定义视频的采样参数 kwargs（原代码中的 self.video_kwargs）
#     # 如果不需要覆盖默认的加载参数，可以直接留空字典
#     video_kwargs = {
#         "total_pixels": 24000 * 28 * 28,
#         "min_pixels": 4 * 28 * 28,
#         "max_pixels": 192 * 28 * 28,
#         "max_frames": 60,
#     }
#     for msg in messages:
#         new_content = []
#         if "content" in msg and isinstance(msg["content"], list):
#             for ele in msg["content"]:
#                 # # --- A. 还原图片：字符串 -> PIL 对象 ---
#                 # if ele.get("type") == "image" and isinstance(ele.get("image"), str):
#                 #     ele["image"] = decode_base64_to_pil(ele["image"])
                
#                 # --- B. 还原视频：占位符 -> 路径字符串 ---
#                 if ele.get("type") == "video" and ele.get("video") == "<video_placeholder>":
#                     # 确保这里赋值的是字符串路径！这样 Collator 里的 process_vision_info 才能工作
#                     ele["video"] = video_path 
#                     # 注入采样参数
#                     ele.update(video_kwargs)
#                     ele = sanitize_content_item(ele)
#                     new_content.append(ele)
#         msg["content"] = new_content

#     # ⚠️ 注意：这里不调用 process_vision_info，而是将干净的结构返回，交由 collate_fn 统一处理
#     return {
#         "messages": messages,
#         "split_source": split_source  # 保留此字段，供 collate_fn 中的 s3 mask 逻辑使用
#     }

def prepare_dataset(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    SFT 专用预处理：
    只做“还原”，不做“转换”。把 Tensor 化留给 Collator 统一处理。
    """
    # 1. 深度拷贝 messages，防止干扰原始数据
    messages = copy.deepcopy(example.get('messages', []))
    split_source = example.get('split_source', 'unknown')
    video_path = example.get('video_path', '')
    # print("***"*5)
    # print(video_path)

    # video_kwargs = {
    #     "total_pixels": 24000 * 28 * 28,
    #     "min_pixels": 4 * 28 * 28,
    #     "max_pixels": 192 * 28 * 28,
    #     "max_frames": 60,
    # }
    video_kwargs = {
        "total_pixels": int(os.environ.get("SFT_TOTAL_VIDEO_TOKENS", "24000")) * 28 * 28,
        "min_pixels": int(os.environ.get("SFT_MIN_PER_FRAME_TOKENS", "4")) * 28 * 28,
        "max_pixels": int(os.environ.get("SFT_MAX_PER_FRAME_TOKENS", "192")) * 28 * 28,
        "max_frames": int(os.environ.get("SFT_MAX_FRAMES", "60")),
    }

    for msg in messages:
        if "content" in msg and isinstance(msg["content"], list):
            new_content = []
            for ele in msg["content"]:
                # 1. 如果是视频占位符，先替换路径和参数
                if ele.get("type") == "video" and ele.get("video") == "<video_placeholder>":
                    ele["video"] = video_path 
                    ele.update(video_kwargs)
                
                # 2. 【核心修复】无论是 text、image 还是 video，全都要经过清洗！
                # 清洗掉 PyArrow 带来的 None 和不需要的杂质键
                clean_ele = sanitize_content_item(ele)
                
                # 3. 把清洗干净的元素放回新列表
                new_content.append(clean_ele)
                
            # 替换掉原本可能被污染的 content
            msg["content"] = new_content

    return {
        "messages": messages,
        "split_source": split_source 
    }


def extract_durations(hf_dataset) -> List[int]:
    """Extract video durations (seconds) from the explicit 'duration' field in each sample.

    The JSONL data file contains a top-level 'duration' key (float, in seconds)
    for every sample. We read it directly. Samples with missing or None duration
    fall back to the dataset median.
    """
    n = len(hf_dataset)
    raw_durations: List[float] = []
    fallback_indices: List[int] = []

    for idx in range(n):
        dur = hf_dataset[idx].get("duration")
        if dur is not None:
            raw_durations.append(float(dur))
        else:
            raw_durations.append(-1.0)  # placeholder, will be replaced
            fallback_indices.append(idx)

    # Compute median from valid durations for fallback
    valid = [d for d in raw_durations if d > 0]
    median_dur = statistics.median(valid) if valid else 60.0

    for idx in fallback_indices:
        raw_durations[idx] = median_dur

    durations = [int(round(d)) for d in raw_durations]

    # Log stats on rank 0
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"[DurationSampler] Total samples: {n}")
        print(f"[DurationSampler] Resolved from 'duration' field: {n - len(fallback_indices)}, "
              f"fallback to median: {len(fallback_indices)}")
        print(f"[DurationSampler] Duration stats — "
              f"min: {min(durations)}s, max: {max(durations)}s, "
              f"mean: {sum(durations)/len(durations):.1f}s, "
              f"median: {statistics.median(durations):.1f}s")

    return durations


def compute_tool_response_mask(seq, id_im_start=151644, id_tool=14172):
    """
    向量化计算掩码：从 seq (1, T) 形状的 LongTensor 中屏蔽所有 `<|im_start|>tool…<|im_end|>` 区域。
    返回长度 T 的 IntTensor，其中 1 表示保留，0 表示屏蔽。
    这里 id_im_start 和 id_tool 与模型相关，需要从 tokenizer 中获取。
    id_im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    id_tool     = tokenizer.convert_tokens_to_ids("tool")
    """
    # 1) 标记所有 "<|im_start|>"
    is_im_start = seq == id_im_start
    # 2) 计算 region_id
    region_id = is_im_start.int().cumsum(dim=1)
    # 3) 标记段开头是否为工具段
    next_is_tool = torch.zeros_like(seq, dtype=torch.bool)
    next_is_tool[:, :-1] = is_im_start[:, :-1] & (seq[:, 1:] == id_tool)
    # 4) 累加得到每段是否为工具段
    region_flag = torch.zeros_like(region_id)
    region_flag = region_flag.scatter_add(
        dim=1,
        index=region_id,
        src=next_is_tool.int().to(region_flag.dtype)
    )
    # 5) 映射回每个 token 是否在工具段
    tool_region_mask = region_flag.gather(dim=1, index=region_id)
    # 6) 最终掩码：非工具段为 1，工具段为 0
    completion_mask = (~tool_region_mask.bool()).int()
    return completion_mask # 最终工具段由于为0被屏蔽，非工具段由于为1被保留

def collate_fn(examples, processor=None):
    # 1. 常规处理：生成 input_ids 和 基础 mask
    # 安全检查
    import time as _time
    _t0 = _time.time()
    _rank = int(os.environ.get("LOCAL_RANK", 0))

    messages_list = [deepcopy(example["messages"]) for example in examples] ## 深度拷贝防止报错，超1个epoch报错
    if processor is None:
        raise ValueError("严重错误: collate_fn 里的 processor 是 None！")

    # 1. 动态解码当前 Batch 的 Base64 图片 (防止 CPU OOM)
    for messages in messages_list:
        for msg in messages:
            if "content" in msg and isinstance(msg["content"], list):
                cleaned_content = []
                for ele in msg["content"]:
                    # --- A. 还原图片：字符串 -> PIL 对象 ---
                    if ele.get("type") == "image" and isinstance(ele.get("image"), str):
                        if _is_removed_base64_placeholder(ele["image"]):
                            continue
                        ele["image"] = decode_base64_to_pil(ele["image"])
                    cleaned_content.append(ele)
                msg["content"] = cleaned_content

    # 🚀 核心修复：先提取视觉特征！此时 messages_list 里的 image 键还没被删掉
    images, videos, video_kwargs = process_vision_info(messages_list, return_video_kwargs=True)

    # === 诊断日志: 记录每个 rank 的 image/video 数量 ===
    _n_img = len(images) if images else 0
    _n_vid = len(videos) if videos else 0
    print(f"[Rank {_rank}] collate: images={_n_img}, videos={_n_vid}, decode_time={_time.time()-_t0:.1f}s", flush=True)

    # 🚀 核心修复：再渲染文本模板！并且传入 deepcopy，彻底隔绝污染
    prompts_text = [
        processor.apply_chat_template(deepcopy(messages), tokenize=False, add_generation_prompt=False)
        for messages in messages_list
    ]

    batch = processor(
        text=prompts_text,
        images=images,
        videos=videos,
        fps=video_kwargs["fps"],
        padding=True,
        return_tensors="pt",
    )

    # === 诊断日志: 验证 processor 输出中 pixel_values 的存在性 ===
    _has_pv = "pixel_values" in batch and batch["pixel_values"] is not None
    _has_pvv = "pixel_values_videos" in batch and batch["pixel_values_videos"] is not None
    _seq_len = batch["input_ids"].shape[-1]
    print(f"[Rank {_rank}] collate: pixel_values={_has_pv}, pixel_values_videos={_has_pvv}, "
          f"seq_len={_seq_len}, total_time={_time.time()-_t0:.1f}s", flush=True)

    labels = batch["input_ids"].clone()
    # ... 后面的 tokenizer 和 mask 逻辑保持不变 ...

    
    # 2. 获取特殊 token ID
    tokenizer = processor.tokenizer
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>") # 151644
    assistant_id = tokenizer.convert_tokens_to_ids("assistant")   # 77091
    
    # 3. 基础 Mask：屏蔽所有 System 和 User 部分 (保留所有 Assistant)
    non_assistant_response_mask = compute_tool_response_mask(labels, im_start_id, assistant_id)
    labels[non_assistant_response_mask == 1] = -100 
    labels[labels == tokenizer.pad_token_id] = -100
    
    # 屏蔽视觉占位符
    video_token_id = tokenizer.convert_tokens_to_ids(processor.video_token)
    image_token_id = tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == video_token_id] = -100
    labels[labels == image_token_id] = -100

    # =========================================================================
    # 【核心新增】针对 S3 (Arbitrator) 的历史屏蔽逻辑 (Updated Anchor)
    # =========================================================================
    # 目标：屏蔽掉 Phase 1 (S1) 和 Phase 2 (S2) 的 Assistant 回答。
    # 方法：找到 S3 User Prompt 的起始句，屏蔽其之前的所有内容。
    
    # 使用你指定的长句子作为锚点。
    # 为了防止 Tokenizer 对长句编码产生细微差异（如空格），我们取前这句独特的话即可。
    anchor_text = "interaction history above presents two distinct analysis"
    marker_ids = tokenizer.encode(anchor_text, add_special_tokens=False)
    marker_tensor = torch.tensor(marker_ids, dtype=torch.long)
    pat_len = len(marker_ids)
    for i, example in enumerate(examples):
        # 仅对 S3 样本执行此逻辑
        if example.get("split_source") == "s3":
            seq = batch["input_ids"][i]
            seq_len = seq.shape[0]
            
            # 1. 在 input_ids 中搜索 S3 User Prompt 的起始位置
            found_idx = -1
            
            # CPU 滑动窗口搜索
            # 优化：因为这句话出现在 prompt 的后半部分，其实可以从后往前搜，或者直接从头搜
            # 为了稳妥，我们还是从头搜，找到第一个匹配项（因为这句话是 S3 的开头，理论上只会出现一次）
            seq_cpu = seq.cpu() # 确保在 CPU 上操作
            for pos in range(seq_len - pat_len):
                if (seq_cpu[pos:pos+pat_len] == marker_tensor).all():
                    found_idx = pos
                    print(f"[S3] Found anchor at index {found_idx}")
                    break
            
            # 2. 如果找到了锚点，继续向后找 S3 Assistant 的开始
            if found_idx != -1:
                s3_assist_start = -1
                # 从锚点之后开始搜索 "<|im_start|>assistant"
                # 这必定是 S3 的 Assistant，因为 S1/S2 的 assistant 都在这个锚点之前
                for pos in range(found_idx, seq_len - 1):
                    if seq[pos] == im_start_id and seq[pos+1] == assistant_id:
                        s3_assist_start = pos
                        break
                
                # 3. 执行屏蔽：将 S3 Assistant 之前的所有 label 设为 -100
                if s3_assist_start != -1:
                    # 关键：这行代码会把 S1, S2 的 loss 全部抹去，只保留 S3
                    labels[i, :s3_assist_start] = -100
            else:
                # 如果没找到锚点，说明 Prompt 模板可能不对，或者被截断了
                # 这种情况下最好打印个 warning 或者跳过 masking (保留默认行为)
                # logger.warning(f"S3 Anchor not found in sample {i}")
                pass

    batch["labels"] = labels
    return batch


class DurationGroupedSampler(Sampler):
    """Sampler that groups consecutive indices by similar duration.

    Indices are sorted by duration and then grouped into chunks of
    ``world_size``.  When accelerate's ``BatchSamplerShard`` applies
    interleaved sharding (rank *i* receives every *world_size*-th sample),
    all ranks within the same micro-step end up processing samples of
    comparable duration.  This is critical for ZeRO-3 where every layer's
    forward pass requires an all-gather across all ranks — any cross-rank
    timing skew in data loading / collation directly blocks the entire group.

    Falls back to a plain sorted sampler when ``world_size=1``.
    """

    def __init__(self, lengths, world_size=None, shuffle=True, seed=42):
        super().__init__()
        self.lengths = lengths
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        if world_size is None:
            world_size = (
                dist.get_world_size()
                if dist.is_available() and dist.is_initialized()
                else 1
            )
        self.world_size = world_size

        # Sort by duration so consecutive indices are similar
        sorted_idx = sorted(range(len(lengths)), key=lambda i: lengths[i])

        # Pad to a multiple of world_size for even distribution
        rem = len(sorted_idx) % self.world_size
        if rem:
            pad_count = self.world_size - rem
            sorted_idx += sorted_idx[-pad_count:]

        self._sorted = sorted_idx
        self._total = len(sorted_idx)

    # ------------------------------------------------------------------
    def __iter__(self):
        ws = self.world_size
        chunks = [self._sorted[i : i + ws] for i in range(0, self._total, ws)]

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(chunks), generator=g).tolist()
            chunks = [chunks[o] for o in order]

        return iter(idx for chunk in chunks for idx in chunk)

    def __len__(self):
        return self._total

    def set_epoch(self, epoch):
        self.epoch = epoch


class DurationAwareSFTTrainer(SFTTrainer):
    """SFTTrainer subclass that groups samples by video duration across ranks."""

    def __init__(self, durations=None, **kwargs):
        super().__init__(**kwargs)
        self._durations = durations

    def _get_train_sampler(self, dataset=None):
        if self._durations is not None:
            return DurationGroupedSampler(
                lengths=self._durations,
                shuffle=True,
            )
        return super()._get_train_sampler(dataset)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to add dist.barrier() before forward — ensures all ranks
        have finished data loading before any rank triggers ZeRO-3 all-gathers."""
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return super().training_step(model, inputs, num_items_in_batch)


if __name__ == "__main__":
    # Extend NCCL timeout to 3 hours (default 10 min is too short for large video batches)
    os.environ.setdefault("NCCL_TIMEOUT", "10800")
    # Enable NCCL flight recorder for better stack traces on deadlock
    os.environ.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "1000")

    # Force spawn mode to avoid CUDA context corruption with forked DataLoader workers
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Parse arguments
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    
    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Load dataset
    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)



    # ===== DEBUG: 检查 HF/Arrow 读入后 content 是否被补齐 None 键 =====
    print("\n[DEBUG] Inspect raw dataset sample after Dataset.from_json/load_dataset")
    raw0 = dataset["train"][0]
    print("[DEBUG] raw keys:", list(raw0.keys()))
    msgs = raw0.get("messages", [])
    for mi, msg in enumerate(msgs[:5]):
        print(f"[DEBUG] msg[{mi}] role={msg.get('role')} content_type={type(msg.get('content'))}")
        content = msg.get("content")
        if isinstance(content, list):
            for ci, ele in enumerate(content[:10]):
                if isinstance(ele, dict):
                    print(f"  content[{ci}] =", ele)
                    print(f"    keys={list(ele.keys())}")
                else:
                    print(f"  content[{ci}] non-dict:", type(ele), ele)











    # Setup model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    # # Quantization configuration for 4-bit training
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    # Model initialization
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        # quantization_config=bnb_config,
    )
    # ZeRO-3 manages device placement itself; device_map is incompatible
    if not is_deepspeed_zero3_enabled():
        model_kwargs["device_map"] = get_kbit_device_map()
    
    
    if "Qwen2-VL" in model_config.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif "Qwen2.5-VL" in model_config.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_config.model_name_or_path, **model_kwargs)

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code
    )

    # =========================================================================
    # ZeRO-3 FIX: Monkey-patch the inner model forward to ALWAYS call
    # self.visual exactly twice (images + videos), even when pixel_values
    # or pixel_values_videos is None.
    #
    # In ZeRO-3, each module forward triggers parameter all-gather across all
    # ranks. If some ranks skip get_image_features (pixel_values=None) while
    # others don't, the all-gather sequence DIVERGES → NCCL deadlock.
    #
    # This patch ensures all ranks execute the same sequence of all-gathers
    # by running a dummy visual forward when the real one is skipped.
    # The dummy result is discarded.
    # =========================================================================
    if is_deepspeed_zero3_enabled():
        import types

        _inner_model = model.model  # Qwen2_5_VLModel
        _orig_forward = _inner_model.__class__.forward  # unbound method

        def _zero3_safe_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                                past_key_values=None, inputs_embeds=None, use_cache=None,
                                output_attentions=None, output_hidden_states=None,
                                pixel_values=None, pixel_values_videos=None,
                                image_grid_thw=None, video_grid_thw=None,
                                rope_deltas=None, cache_position=None,
                                second_per_grid_ts=None, return_dict=None,
                                **kwargs):
            """Wrapper that forces self.visual to be called even when pixel_values
            or pixel_values_videos is None, keeping ZeRO-3 all-gather sequences
            consistent across ranks."""

            # --- Phase 1: Build inputs_embeds (same as original) ---
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)

            # --- Phase 2: ALWAYS call visual encoder for images ---
            if pixel_values is not None:
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask, _ = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            else:
                # Dummy forward through visual encoder to match all-gather sequence
                # patch_embed expects (N, C*T*P*P) where C=3, T=2, P=14 → 1176 per patch
                # Need at least 4 patches for spatial_merge_unit=4, grid_thw=[[1,2,2]]
                _dummy_pv = torch.zeros(
                    4, 3 * 2 * 14 * 14,
                    device=inputs_embeds.device, dtype=self.visual.dtype
                )
                _dummy_thw = torch.tensor([[1, 2, 2]], device=inputs_embeds.device, dtype=torch.long)
                _dummy_out = self.visual(_dummy_pv, grid_thw=_dummy_thw)
                # Connect to computation graph so backward traverses visual encoder
                # (adds zero, doesn't change values, but ensures gradient hooks fire)
                inputs_embeds = inputs_embeds + _dummy_out.sum() * 0

            # --- Phase 3: ALWAYS call visual encoder for videos ---
            if pixel_values_videos is not None:
                video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
                video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                _, video_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            else:
                # Dummy forward through visual encoder to match all-gather sequence
                _dummy_pv = torch.zeros(
                    4, 3 * 2 * 14 * 14,
                    device=inputs_embeds.device, dtype=self.visual.dtype
                )
                _dummy_thw = torch.tensor([[1, 2, 2]], device=inputs_embeds.device, dtype=torch.long)
                _dummy_out = self.visual(_dummy_pv, grid_thw=_dummy_thw)
                # Connect to computation graph so backward traverses visual encoder
                inputs_embeds = inputs_embeds + _dummy_out.sum() * 0

            # --- Phase 4: Position encoding + language model (unchanged) ---
            from transformers.utils import is_torchdynamo_compiling
            if position_ids is None:
                prefill_compiled_stage = is_torchdynamo_compiling() and (
                    (input_ids is not None and input_ids.shape[1] != 1)
                    or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
                )
                prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                    (cache_position is not None and cache_position[0] == 0)
                    or (past_key_values is None or past_key_values.get_seq_length() == 0)
                )
                if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                    position_ids, rope_deltas = self.get_rope_index(
                        input_ids,
                        image_grid_thw,
                        video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                        attention_mask=attention_mask,
                    )
                    self.rope_deltas = rope_deltas
                else:
                    batch_size, seq_length, _ = inputs_embeds.shape
                    position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                    position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                    if cache_position is not None:
                        delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    else:
                        delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                    position_ids = position_ids + delta.to(position_ids.device)

            outputs = self.language_model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=cache_position,
                **kwargs,
            )

            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModelOutputWithPast
            output = Qwen2_5_VLModelOutputWithPast(
                last_hidden_state=outputs.last_hidden_state,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=self.rope_deltas,
            )
            return output if return_dict else output.to_tuple()

        _inner_model.forward = types.MethodType(_zero3_safe_forward, _inner_model)
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print("[ZeRO-3 FIX] Monkey-patched Qwen2_5_VLModel.forward to force "
                  "consistent visual encoder calls across all ranks")

    # Prepare dataset
    prepared_dataset = [prepare_dataset(example) for example in dataset['train']]

    # Extract video durations for length-grouped sampling
    durations = extract_durations(dataset['train'])


    # # ===== DEBUG: 检查 prepare_dataset 后是否仍然有补齐的 None 键 =====
    # print("\n[DEBUG] Inspect prepared_dataset sample")
    # ex0 = prepared_dataset[0]
    # msgs = ex0.get("messages", [])
    # for mi, msg in enumerate(msgs[:5]):
    #     print(f"[DEBUG] prepared msg[{mi}] role={msg.get('role')} content_type={type(msg.get('content'))}")
    #     content = msg.get("content")
    #     if isinstance(content, list):
    #         for ci, ele in enumerate(content[:10]):
    #             if isinstance(ele, dict):
    #                 print(f"  content[{ci}] =", ele)
    #                 print(f"    keys={list(ele.keys())}")



    # Initialize wandb if specified
    if _reports_to_wandb(training_args.report_to):
        if wandb is None:
            raise ImportError("report_to=wandb requires wandb to be installed.")
        wandb.init(project="video-sft")

    # Initialize trainer — bind processor to collate_fn so spawned workers can access it
    from functools import partial
    my_collator = partial(collate_fn, processor=processor)

    trainer = DurationAwareSFTTrainer(
        durations=durations,
        model=model,
        args=training_args,
        train_dataset=prepared_dataset,
        data_collator=my_collator,
        peft_config=get_peft_config(model_config),
        # tokenizer=processor.tokenizer
    )

    # =========================================================================
    # Resume-safe training: detect last checkpoint and resume if available
    # =========================================================================
    from transformers.trainer_utils import get_last_checkpoint
    import logging as _logging
    _logger = _logging.getLogger(__name__)

    checkpoint = None
    if training_args.resume_from_checkpoint is not None and \
       str(training_args.resume_from_checkpoint).lower() not in ("false", "no", "none", "0"):
        checkpoint = training_args.resume_from_checkpoint
    elif training_args.resume_from_checkpoint is None and os.path.isdir(training_args.output_dir):
        last_ckpt = get_last_checkpoint(training_args.output_dir)
        if last_ckpt is not None:
            _logger.info(f"[Resume] Checkpoint detected, resuming training from {last_ckpt}")
            checkpoint = last_ckpt

    # Train model (with resume support)
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # Save final model and trainer state
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    if wandb is not None:
        wandb.finish()
