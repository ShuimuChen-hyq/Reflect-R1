import json
import os
import re
import math
import random
import signal
import time
import copy
import torch
import yaml
from collections import defaultdict
from typing import Dict
from reflect_r1.utils.qwen_vl_utils import process_vision_info, fetch_video, smart_resize, replace_vision_info_with_placeholder, IMAGE_FACTOR, FRAME_FACTOR
from reflect_r1.utils import rank0_print, rank_print, parse_dataset_yaml, load_jsonl, load_json
from reflect_r1.prompt import get_prompt_fn
from reflect_r1.utils.visualize_frames import tensor_to_pil
from reflect_r1.prompt.tool_use import get_tool_use_prompt
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from reflect_r1.utils.video_tools import video_tool_call
import numpy as np
from reflect_r1.utils.clip_service import SiglipClient
from reflect_r1.utils.tafr import construct_temporal_augmented_frames

# --- Timeout protection for video loading and SigLIP encoding ---
VIDEO_LOAD_TIMEOUT = int(os.environ.get("VIDEO_LOAD_TIMEOUT", "120"))


class VideoLoadTimeoutError(TimeoutError):
    pass


def _timeout_handler(signum, frame):
    raise VideoLoadTimeoutError("Video loading timed out")


def fetch_video_with_timeout(video_ele, timeout_sec=VIDEO_LOAD_TIMEOUT, **kwargs):
    """Wrap fetch_video with a signal-based timeout (works in DataLoader worker processes)."""
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        result = fetch_video(video_ele, **kwargs)
        signal.alarm(0)
        return result
    except VideoLoadTimeoutError:
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def encode_images_with_timeout(clip_model_inst, frames, timeout_sec=VIDEO_LOAD_TIMEOUT):
    """Wrap clip_model.encode_images with a signal-based timeout."""
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        result = clip_model_inst.encode_images(frames)
        signal.alarm(0)
        return result
    except VideoLoadTimeoutError:
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

TOTAL_PIXELS = 256 * 60 * 28 * 28
MIN_PIXELS = 16 * 28 * 28
MAX_PIXELS = 192 * 28 * 28
MAX_CACHE_FRAMES = 2048

DEFAULT_VIDEO_KWARGS = {
    "total_pixels": TOTAL_PIXELS,
    "min_pixels": MIN_PIXELS,
    "max_pixels": MAX_PIXELS,
}

MIN_CACHE_PIXELS = 32 * 28 * 28
MAX_CACHE_PIXELS = 256 * 28 * 28
TOTAL_CACHE_PIXELS = MAX_CACHE_FRAMES * MAX_CACHE_PIXELS
DEFAULT_CACHE_VIDEO_KWARGS = {
    "max_frames": MAX_CACHE_FRAMES,
    "fps": 2,
    "min_pixels": MIN_CACHE_PIXELS,
    "max_pixels": MAX_CACHE_PIXELS,
    "total_pixels": TOTAL_CACHE_PIXELS,
}

clip_model = SiglipClient()
DEFAULT_SYSTEM_PROMPT = "You are a critical reasoning arbitrator capable of analyzing multiple reasoning paths and deriving the most accurate conclusion through tool-assisted verification."


def load_video_with_decord(video_path, max_frames_num=7200, fps=1, force_sample=False):
    import decord
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = decord.VideoReader(video_path)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    actual_fps = fps
    if len(frame_idx) > max_frames_num or force_sample:
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
        actual_fps = max_frames_num / video_time
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    spare_frames = torch.from_numpy(spare_frames).permute(0, 3, 1, 2)
    return spare_frames, frame_time, actual_fps


class LazyVLDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str,
                 data_root: str,
                 prompt_template: str, 
                 tool_name_list: list[str]=[],
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 video_kwargs: dict = {},
                 cache_video_kwargs: dict = {},
                 append_time_instruction: bool = False,
                ):
        super(LazyVLDataset, self).__init__()
        self.data_root = data_root
        self.prompt_template = get_prompt_fn(prompt_template)
        if tool_name_list:
            self.tool_use_prompt = get_tool_use_prompt(tool_name_list) # 内置工具提示
        else:
            self.tool_use_prompt = ""
        self.system_prompt = f"{system_prompt}\n{self.tool_use_prompt}"
        rank0_print(f"System prompt: {self.system_prompt}")
        self.video_kwargs = copy.deepcopy(DEFAULT_VIDEO_KWARGS)
        self.video_kwargs.update(video_kwargs)
        self.cache_video_kwargs = copy.deepcopy(DEFAULT_CACHE_VIDEO_KWARGS)
        self.cache_video_kwargs.update(cache_video_kwargs)
        self.append_time_instruction = append_time_instruction # Flase
        rank0_print(f"Append time instruction: {self.append_time_instruction}")
        rank0_print(f"Video kwargs: {self.video_kwargs}")
        rank0_print(f"Cache video kwargs: {self.cache_video_kwargs}")
        if data_path.endswith(".yaml"):
            dataset_info = parse_dataset_yaml(data_path) ## 执行这个数据集合并
            self.list_data_dict = dataset_info["data_dict_list"]
            json_file_list = dataset_info["json_file_list"]
            rank0_print(f"Loaded {len(self.list_data_dict)} samples from {json_file_list}")
        elif data_path.endswith(".jsonl"):
            rank0_print(f"Loading {data_path}")
            self.list_data_dict = load_jsonl(data_path)
        elif data_path.endswith(".json"):
            self.list_data_dict = load_json(data_path)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")
        
        # ### 筛选
        # initial_count = len(self.list_data_dict)
        # self.list_data_dict = [
        #     item for item in self.list_data_dict 
        #     if item.get('select', True)  # 如果没有 select 字段，默认保留
        # ]
        # new_count = len(self.list_data_dict)
        # rank0_print(f"SFT Dataset Filter: {initial_count} -> {new_count} (Dropped {initial_count - new_count})")    
        # ## 筛选结束   

        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")

        # --- Video blacklist filtering ---
        VIDEO_BLACKLIST_PATH = os.environ.get("VIDEO_BLACKLIST", "")
        if VIDEO_BLACKLIST_PATH and os.path.exists(VIDEO_BLACKLIST_PATH):
            blacklisted = set()
            with open(VIDEO_BLACKLIST_PATH) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        blacklisted.add(parts[1])
                    elif parts:
                        blacklisted.add(parts[0])
            initial_count = len(self.list_data_dict)
            self.list_data_dict = [
                item for item in self.list_data_dict
                if item.get("video_path", item.get("video", "")) not in blacklisted
            ]
            filtered_count = initial_count - len(self.list_data_dict)
            rank0_print(f"Video blacklist: filtered {filtered_count}/{initial_count} samples "
                        f"using {VIDEO_BLACKLIST_PATH}")

        rank0_print("Formatting inputs...Skip in lazy mode")
        status = defaultdict(int)
        for data in self.list_data_dict:
            if "image" in data:
                status["image"] += 1
            elif "multi_video" in data:
                status["multi_video"] += 1
            # elif "video" in data:
            #     status["video"] += 1
            elif 'video_path' in data:
                status["video"] += 1
            else:
                status["unknow"] += 1
        rank0_print(f"Dataset modalities status: {status}")
        rank0_print(f"Dataset template: {self.prompt_template}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        MAX_RETRIES = 10
        rank = os.environ.get('RANK', '0')

        for attempt in range(MAX_RETRIES):
            current_idx = (idx + attempt) % len(self.list_data_dict)
            item = copy.deepcopy(self.list_data_dict[current_idx])

            if 'video_path' in item:
                video_abs_path = item["video_path"]
                item["video"] = video_abs_path
            else:
                video_abs_path = item.get("video", "<unknown>")

            sample_id = item.get('id', current_idx)

            try:
                model_inputs = self.qwen_vl_preprocess_fn(item)
                print(f"[DEBUG][Rank {rank}] End loading: {video_abs_path}", flush=True)
                item.update(model_inputs)
                return item
            except Exception as e:
                print(f"[ERROR][Rank {rank}] Failed {video_abs_path} (id={sample_id}): {e}, attempt {attempt+1}/{MAX_RETRIES}", flush=True)
                continue

        raise RuntimeError(f"[Rank {rank}] Failed to load any sample after {MAX_RETRIES} attempts starting from idx {idx}")
    ## cache_video_frames用于帧检索特征提取，更多帧更大尺寸，preview_video_frames用于模型输入，帧数和尺寸相对较小
    def qwen_vl_preprocess_fn(self, item):
        """
        【任务无关】Qwen2.5-VL预处理, apply chat template, process_vision_info
        item: {
            "id": str,
            "video": str,
            "question": str,
            "target": list[float],
        }
        outputs:
            messages: list[dict],
            multimodal_cache: dict, for video interaction
        """
        cache_video_ele = {
            "type": "video", 
            "video": item["video"],
        }
        cache_video_ele.update(self.cache_video_kwargs)
        if os.path.exists(item["video"] + ".frame_cache"):
            # Limit max frames and per-frame tokens;
            # Maybe you should use fetch_video to do this
            try:
                cache_data = torch.load(item["video"] + ".frame_cache")
                cache_video_frames = cache_data["frame_tensor"]
                cache_video_sample_fps = cache_data["fps"]
                num_frames = len(cache_video_frames)
                if num_frames > MAX_CACHE_FRAMES:
                    sample_idx = torch.linspace(0, num_frames - 1, MAX_CACHE_FRAMES).round().long()
                    cache_video_frames = cache_video_frames[sample_idx]
                    cache_video_sample_fps = MAX_CACHE_FRAMES / num_frames * cache_video_sample_fps
                # smart resize
                nframes, _, height, width = cache_video_frames.shape
                min_pixels = cache_video_ele.get("min_pixels", MIN_CACHE_PIXELS)
                total_pixels = cache_video_ele.get("total_pixels", TOTAL_CACHE_PIXELS)
                max_pixels = max(min(MAX_CACHE_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
                max_pixels_supposed = cache_video_ele.get("max_pixels", max_pixels)
                if max_pixels_supposed > max_pixels:
                    print(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
                max_pixels = min(max_pixels_supposed, max_pixels)
                resized_height, resized_width = smart_resize(height,
                    width,
                    factor=IMAGE_FACTOR,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                cache_video_frames = transforms.functional.resize(
                    cache_video_frames,
                    [resized_height, resized_width],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ).float()
                
                # print(f"cache_video_frames: {cache_video_frames.shape}")
                # print(f"Loaded frame cache from {item['video'] + '.frame_cache'}")
            except Exception as e:
                print(f"Error loading frame cache: {e}")
                cache_video_frames, cache_video_sample_fps = fetch_video_with_timeout(cache_video_ele, return_video_sample_fps=True)
        else:
            cache_video_frames, cache_video_sample_fps = fetch_video_with_timeout(cache_video_ele, return_video_sample_fps=True)
        if os.path.exists(item["video"] + ".feature_cache"):
            try:
                cache_video_features = torch.load(item["video"] + ".feature_cache")
                num_frames = len(cache_video_features)
                if num_frames > MAX_CACHE_FRAMES:
                    sample_idx = torch.linspace(0, num_frames - 1, MAX_CACHE_FRAMES).round().long()
                    cache_video_features = cache_video_features[sample_idx]
                # print(f"cache_video_features: {cache_video_features.shape}")
                # print(f"Loaded feature cache from {item['video'] + '.feature_cache'}")
            except Exception as e:
                print(f"Error loading feature cache: {e}")
                cache_video_features = encode_images_with_timeout(clip_model, cache_video_frames)
        else:
            cache_video_features = encode_images_with_timeout(clip_model, cache_video_frames)
        multimodal_cache = {
            "video": cache_video_frames,
            "embedding": cache_video_features,
            "fps": cache_video_sample_fps,
        }
        preview_video_ele = {
            "type": "video", 
            "video": item["video"],
        }
        preview_video_ele.update(self.video_kwargs) # 用于模型实际输入的预览视频
        preview_video_frames, preview_video_sample_fps = fetch_video_with_timeout(preview_video_ele, return_video_sample_fps=True)
        # print(f"====>preview_video_frames: {preview_video_frames.shape}", f"cache_video_frames: {cache_video_frames.shape}")
        if "duration" not in item: ## 这是一个估算，还是建议用原视频时长
            item["duration"] = len(cache_video_frames) / cache_video_sample_fps
        multimodal_cache["duration"] = item["duration"]
        multimodal_cache["question"] = item["question"]

        if self.append_time_instruction:
            timestamps = [frame_id / preview_video_sample_fps for frame_id in range(len(preview_video_frames))]
            prompt_text = self.prompt_template({
                "question": item["question"],
                "timestamps": [frame_id / preview_video_sample_fps for frame_id in range(len(preview_video_frames))],
                "duration": item["duration"],
            })

            frames_and_prompt = construct_temporal_augmented_frames(timestamps, preview_video_frames)
            frames_and_prompt.append({"type": "text", "text": prompt_text})

            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.system_prompt}
                    ],
                },
                {
                    "role": "user", 
                    "content": frames_and_prompt,
                },
            ]
        else: ## 执行这个，不需要时间信息，后面调用工具才会用时间戳标识
            prompt_text = self.prompt_template(item)
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.system_prompt} ## 包含系统指令以及工具调用指令
                    ],
                },
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "video",   
                            "video": preview_video_frames,
                            "fps": preview_video_sample_fps,
                        },
                        {
                            "type": "text",
                            "text": prompt_text ## 这个是最终的用户问题+视频持续时间，见V4模板，但是没有选项hh
                        },
                    ]
                },
            ]
        return {
            "messages": messages,
            "multimodal_cache": multimodal_cache, # dict_keys(['video', 'embedding', 'fps', 'duration', 'question'])
        }

if __name__ == "__main__":
    data_path = "workdir/datasets/videomme/bes_videomme_withouttitle_test.json"
    data_root = "dataset/evaluation/llava_next/videomme/data/"
    from tqdm import tqdm
    from reflect_r1.prompt import get_prompt_fn
    dataset = LazyVLDataset(data_path, data_root, "v6", tool_name_list=["seek_video_frames"])
    item = dataset[0]
    print(item)
    print(replace_vision_info_with_placeholder(item["messages"]))
    # for k in item:
    #     if k == "messages":
    #         print(replace_vision_info_with_placeholder(item["messages"]))
    #     else:
    #         print(f"{k}: {item[k]}")
