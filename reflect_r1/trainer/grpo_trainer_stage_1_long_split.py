# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import os
import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Sized
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union, Tuple

import datasets
import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_peft_available, is_rich_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_liger_kernel_available, is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
from trl.models.utils import _ForwardRedirection
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import (
    disable_dropout_in_model,
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from copy import deepcopy
import re
import numpy as np
from reflect_r1.prompt.tool_use import get_tool_use_prompt
from reflect_r1.utils.qwen_vl_utils import replace_vision_info_with_placeholder


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


CSV_TEMPLATE_V1 = (
    "You must ALWAYS conduct thorough reasoning inside <think> and </think> tags BEFORE answering the question.\n"
    "Provide your answer within <answer> </answer> tags. Your answer should be supported by evidence from the given frame.\n"
    "When you don't have enough visual information, please say 'I don't know'.\n"
    "Your output must follow the format: <think>Your reasoning process</think><answer>Your answer</answer>"
    "Question: {question}\n"
    "Options:\n{options}\n"
    "Answer with the option's letter from the given choices within <answer> </answer> tags.\n"
)

REFLECT_PROMPT_WITH_TOOLS = (
    "The interaction history above presents two distinct analysis phases:\n"
    "- Phase 1 (S1): Initial Answer. (Derived from the full video and tool-retrieved frames.)\n"
    "- Phase 2 (S2): Blind Verification (based solely on retrieved frames from Initial Answer).\n"
    "Your task is to arbitrate and determine the correct answer. You must actively retrieve video frames using the tool to verify the evidence before making your final decision.\n"
    "Start your reasoning by explicitly quoting the conclusions of S1 and S2.\n"
    "You are PROHIBITED from answering directly in the first turn.\n"
    "Regardless of whether S1 and S2 agree or disagree, you MUST invoke the tool at least once to independently verify the key evidence or timestamp.\n"
    "Your output must follow the format: <think>Your reasoning process</think><tool_call>Parameters</tool_call> or <think>>Your reasoning process</think><answer>Your answer</answer>\n"
    "Question: {question}\n"
    "Options:\n{options}\n"
    "The video lasts for {duration} seconds.\n"
    "Please provide only the single option letter (e.g.,A, B, C, D, etc.) within the <answer></answer> tages.\n"
    "DO NOT output <answer></answer> tags in this turn. You MUST call a tool within <tool_call></tool_call> first.\n"
)

# REFLECT_PROMPT_WITH_TOOLS = (
#     "The interaction history above presents two distinct analysis phases:\n"
#     "Phase 1. **The Initial Answer**: Derived from the full video and tool-retrieved frames.\n"
#     "Phase 2. **The Blind Verification**: Based STRICTLY on the retrieved frames without access to the full video context.\n"
#     "\n"
#     "### YOUR MISSION:\n"
#     "You are the Lead Forensic Analyst. Your task is to resolve potential conflicts or verify the evidence with absolute certainty.\n"
#     "\n"
#     "### STRICT INSTRUCTIONS:\n"
#     "1. **EXTRACT & COMPARE**: Start your reasoning by explicitly quoting the conclusions of S1 and S2 (e.g., 'Extraction: S1=A, S2=B').\n"
#     "2. **MANDATORY VERIFICATION (Crucial)**: \n"
#     "   - **You are PROHIBITED from answering directly in the first turn.**\n"
#     "   - Regardless of whether S1 and S2 agree or disagree, you **MUST** invoke the tool at least once to independently verify the key evidence or timestamp.\n"
#     "   - Trust but verify. Even if both agree, check the frames yourself to ensure no shared hallucination.\n"
#     "3. **Reasoning & Action**: Explain what specific detail or timeframe you need to check, then generate the tool call.\n"
#     "\n"
#     "Your output must follow the format: <think>Extraction: S1=..., S2=...Your reasoning process</think><tool_call>Parameters</tool_call> or <think>...</think><answer>Your answer</answer>\n"
#     "\n"
#     "Question: {question}\n"
#     "Options:\n{options}\n"
#     "The video lasts for {duration} seconds.\n"
#     "**REMINDER: DO NOT output <answer></answer> tags in this turn. You MUST call a tool within <tool_call></tool_call> first.**\n"
# )



# REFLECT_PROMPT_WITH_TOOLS = (
#     "The interaction history above presents two distinct analysis phases:\n"
#     "Phase 1. **The Initial Answer**: Derived from the full video and tool-retrieved frames.\n"
#     "Phase 2. **The Blind Verification**: Based STRICTLY on the retrieved frames without access to the full video context.\n"
#     "\n"
#     "### YOUR MISSION:\n"
#     "You are the Lead Forensic Analyst. Your task is to resolve potential conflicts or verify the evidence with absolute certainty.\n"
#     "\n"
#     "### STRICT INSTRUCTIONS:\n"
#     "1. **EXTRACT & COMPARE**: Start your reasoning by explicitly quoting the conclusions of S1 and S2 (e.g., 'Extraction: S1=A, S2=B').\n"
#     "2. **MANDATORY VERIFICATION (Crucial)**: \n"
#     "   - **You are PROHIBITED from answering directly in the first turn.**\n"
#     "   - Regardless of whether S1 and S2 agree or disagree, you **MUST** invoke the tool at least once to independently verify the key evidence or timestamp.\n"
#     "   - Trust but verify. Even if both agree, check the frames yourself to ensure no shared hallucination.\n"
#     "3. **Reasoning & Action**: Explain what specific detail or timeframe you need to check, then generate the tool call.\n"
#     "\n"
#     "Your output must follow the format: <think>Your reasoning process including extraction of S1/S2</think><tool_call>Parameters</tool_call>\n" # <--- 这里改成了你建议的格式
#     "\n"
#     "Question: {question}\n"
#     "Options:\n{options}\n"
#     "The video lasts for {duration} seconds.\n"
#     "**REMINDER: DO NOT output <answer></answer> tags in this turn. You MUST call a tool within <tool_call></tool_call> first.**\n"
# )



class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)


def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.

    Example:
        >>> x = torch.arange(12).reshape(6, 2)
        >>> y = torch.arange(6).reshape(6, 1)
        >>> tensor_dict = {"x": x, "y": y}
        >>> split_tensor_dict(tensor_dict, 3)
        [
            {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]])},
            {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]])},
            {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]])}
        ]
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    return [
        {
            key: tensor[i * chunk_size : (i + 1) * chunk_size] if tensor is not None else None
            for key, tensor in tensor_dict.items()
        }
        for i in range(num_chunks)
    ]


def shuffle_tensor_dict(tensor_dict: dict[str, Optional[torch.Tensor]]) -> dict[str, Optional[torch.Tensor]]:
    """
    Shuffles a dictionary of tensors along the first dimension in unison.

    Example:
        >>> x = torch.arange(6).reshape(3, 2)
        >>> y = torch.arange(3).reshape(3, 1)
        >>> tensor_dict = {"x": x, "y": y}
        >>> shuffle_tensor_dict(tensor_dict)
        {'x': tensor([[2, 3],
                      [0, 1],
                      [4, 5]]),
         'y': tensor([[1],
                      [0],
                      [2]])}
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    batch_size = first_tensor.shape[0]
    permutation = torch.randperm(batch_size)
    return {key: tensor[permutation] if tensor is not None else None for key, tensor in tensor_dict.items()}


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


class GRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. Custom reward
                  functions can also return None when the reward is not applicable to those samples. This is useful for
                  multi-task training where different reward functions apply to different types of samples. When a
                  reward function returns None for a sample, that reward function is excluded from the reward
                  calculation for that sample. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`]. A
            padding token, `processing_class.pad_token`, must be set. If the processing class has not set a padding
            token, `processing_class.eos_token` will be used as the default.
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        environment = None,
        process_vision_fn = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2")
        model_init_kwargs["torch_dtype"] = torch.bfloat16
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`.")
            model = get_peft_model(model, peft_config)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.bos_token_id = processing_class.tokenizer.bos_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                processing_class.pad_token = processing_class.tokenizer.pad_token
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
        if processing_class.pad_token is None:
            raise NotImplementedError("I don't know how to handle Qwen2.5VL model with padding token")
            processing_class.pad_token = processing_class.eos_token
        
        # vision process function
        self.process_vision_fn = process_vision_fn

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(reward_funcs[i], nn.Module):  # Use Module over PretrainedModel for compat w/ compiled models
                self.reward_func_names.append(reward_funcs[i].config._name_or_path.split("/")[-1])
            else:
                self.reward_func_names.append(reward_funcs[i].__name__) ## 执行这个
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32) # 执行这个
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.use_liger_loss = args.use_liger_loss
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.mask_truncated_completions = args.mask_truncated_completions

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None
        self.replay_buffer = get_replay_buffer(args.replay_buffer_type, capacity=args.replay_buffer_capacity, alpha=args.replay_buffer_alpha)

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Reference model
        self.beta = args.beta
        # assert self.beta == 0.0, "Reference model is not supported for Qwen2-VL and Qwen2.5-VL"
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled() or self.is_fsdp_enabled:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Liger loss
        if self.use_liger_loss:
            if not is_liger_kernel_available():
                raise ImportError(
                    "Liger is required to use `liger_loss` as the GRPO loss. Run `pip install liger-kernel`."
                )
            # redirect the model.module forward to the model forward to ensure pre-forward hooks are called
            self._forward_redirection = _ForwardRedirection()

            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                use_ref_model=self.beta != 0.0,
                loss_type=self.loss_type,
                max_completion_length=self.max_completion_length,
            )

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.steps_per_generation
        self._textual_logs = {
            "prompt_1": deque(maxlen=maxlen),
            "completion_1": deque(maxlen=maxlen),

            
            "prompt_2": deque(maxlen=maxlen),
            "completion_2": deque(maxlen=maxlen),


            "prompt_3": deque(maxlen=maxlen),
            "completion_3": deque(maxlen=maxlen),

            "rewards_1": defaultdict(lambda: deque(maxlen=maxlen)),
            "rewards_2": defaultdict(lambda: deque(maxlen=maxlen)),
            "rewards_3": defaultdict(lambda: deque(maxlen=maxlen)),
        }

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)
        self._last_loaded_step = -1
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                self.vllm_client = VLLMClient(
                    args.vllm_server_host, args.vllm_server_port, connection_timeout=args.vllm_server_timeout
                )
                self.vllm_client.init_communicator()

            elif self.vllm_mode == "colocate":
                # Make sure vllm_tensor_parallel_size group size evenly divides the world size - each group should have
                # the same number of ranks
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                    # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                self.llm = LLM(
                    model=model.name_or_path,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                    * self.vllm_tensor_parallel_size
                    * self.args.gradient_accumulation_steps,
                    max_model_len=self.max_prompt_length + self.max_completion_length,
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    limit_mm_per_prompt={"image": 1024, "video": 10},
                )
            # vLLM specific sampling arguments
            self.guided_decoding_regex = args.vllm_guided_decoding_regex

            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                pad_token_id=processing_class.pad_token_id,
                bos_token_id=processing_class.bos_token_id,
                eos_token_id=processing_class.eos_token_id,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                repetition_penalty=self.repetition_penalty,
                cache_implementation=args.cache_implementation,
            )
        if environment is not None and not args.use_vllm and self.vllm_mode != "colocate":
            raise ValueError(
                f"You provided an environment, but `use_vllm` is set to {args.use_vllm} and `vllm_mode` is set to {self.vllm_mode}."
                f"Environments are only supported with colocate mode of vLLM."
            )
        # Initialize the environment
        self.environment = environment
        if self.environment is not None and self.use_vllm:
            self.environment.update_model_and_processor(
                model=self.llm,
                processor=self.processing_class
            )
        else:
            self.environment.update_model_and_processor(
                model=self.model,
                processor=processing_class
            )
        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    # set device placement to True to make `prepare_model` move `reward_func` to device when using fsdp
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True
                    )

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × steps_per_generation`). This allows us to generate completions
    # once every steps_per_generation step—rather than once per accumulation step—which is significantly more
    # efficient. The only change from the original implementation is multiplying the batch size by
    # `steps_per_generation`. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the
    # splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification. As a result, some parts of the method aren't relevant to GRPO, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.steps_per_generation,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            # dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

            from functools import partial 

            # 3. 改成下面这样：
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, 
                num_workers=self.args.dataloader_num_workers, 
                rank=self.accelerator.process_index)

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |    Accum step 0     |
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...

        return RepeatSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    @profiling_decorator
    def _get_last_hidden_state(self, unwrapped_model, input_ids, attention_mask, logits_to_keep=None):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model
        last_hidden_state = unwrapped_model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, multimodal_inputs, logits_to_keep, batch_size=None) -> torch.Tensor:
        assert batch_size is None, "I don't want to handle batch_size in _get_per_token_logps"
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]

            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids_batch, attention_mask=attention_mask_batch, **multimodal_inputs
            ).logits # (Batch_Size, Sequence_Length, Vocab_Size) 它代表模型在每一个位置预测词表中每一个词的分数（未归一化）。
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred 最后一个 logit 预测的是序列结束后的下一个词（我们没有这个标签），因此将它切除 (:-1)。自回归操作
            input_ids_batch = input_ids_batch[:, -logits_to_keep:] # 最后 logits_to_keep 个 token 的标签是计算损失的 torch.Size([2, 2298])
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -logits_to_keep:] # torch.Size([2, 2298, 152064])
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature # 1.0 temperature
            logps = selective_log_softmax(logits, input_ids_batch)  # 对 logits 做 log_softmax，得到所有词的概率分布。只取出 input_ids_batch 中实际出现的那个 Token 对应的 Log 概率。 返回一个形状为 (Batch, logits_to_keep) 的 Tensor，里面的每个值代表模型生成该 Token 的“信心”有多大（负数，越接近0信心越大）。
            all_logps.append(logps)
        return torch.cat(all_logps, dim=0)

    def _sync_fsdp_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
        if visited is None:
            visited = set()

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp_params_to_vllm(
                child_module, prefix=child_prefix, visited=visited
            )  # recurse into the child

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
                        full_name = full_name.replace(extra, "")

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    @profiling_decorator
    @profiling_decorator
    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if is_peft_model(self.model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    self._sync_fsdp_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = name.replace("modules_to_save.default.", "")
                        # Convert key format: model.visual.xxx -> visual.xxx
                        # model.language_model.xxx -> model.xxx
                        name = name.removeprefix("model.")
                        if name.startswith("language_model."):
                            name = name.replace("language_model.", "model.", 1)

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                self._sync_fsdp_params_to_vllm(self.model)  # use memory-efficient post-order traversal for FSDP
            else:
                for name, param in self.model.named_parameters():
                    with gather_if_zero3([param]):
                        # Convert key format: model.visual.xxx -> visual.xxx
                        # model.language_model.xxx -> model.xxx
                        converted_name = name.removeprefix("model.")
                        if converted_name.startswith("language_model."):
                            converted_name = converted_name.replace("language_model.", "model.", 1)
                        
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(converted_name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(converted_name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        # TODO: remove this
        assert self.args.steps_per_generation == 1 and self.num_iterations == 1, f"I don't want to generate more than 1 completion per step, but got steps_per_generation: {self.args.steps_per_generation} and num_iterations: {self.num_iterations}"
        if mode == "train":
            # generate_every = self.args.steps_per_generation * self.num_iterations
            # if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                # generation_batch = self._generate_and_score_completions(generation_batch)
                # generation_batch = shuffle_tensor_dict(generation_batch)
                # self._buffered_inputs = split_tensor_dict(generation_batch, self.args.steps_per_generation)

            # inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            # We use on-policy training, so we don't need to buffer the inputs
            self._step += 1
            inputs = self._generate_and_score_completions(generation_batch)
            if self.replay_buffer is not None:
                # single experience
                self.replay_buffer.add(inputs)
                adv1_is_zero = torch.all(inputs["advantages_1"] == 0)
                adv2_is_zero = torch.all(inputs["advantages_2"] == 0)
                adv3_is_zero = torch.all(inputs["advantages_3"] == 0)
                if adv1_is_zero and adv2_is_zero and adv3_is_zero:
                    inputs = self.replay_buffer.sample()
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # assert len(inputs) == 1, "I don't want to handle batch_size > 1 in VL-GRPO"
        options = [x["options"] for x in inputs]  
        questions = [x["question"] for x in inputs]
        prompts = [x["messages"] for x in inputs]
        durations = [x["duration"] for x in inputs]
        prompts_text = [self.processing_class.apply_chat_template(x, tokenize=False, add_generation_prompt=True) for x in prompts]
        images, videos, video_kwargs = self.process_vision_fn(prompts, return_video_kwargs=True)
        prompt_inputs = self.processing_class(
            text=deepcopy(prompts_text),    # bug fix: inplace operation will change the original prompts_text
            images=images,
            videos=videos,
            fps=video_kwargs["fps"],
            padding=True, 
            return_tensors="pt", 
            padding_side="left",  # # 结果: [PAD PAD PAD prompt_tokens] 生成时需要所有样本的prompt末尾对齐，方便模型从相同位置开始生成
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        assert prompt_ids.size(1) < self.max_prompt_length, f"Unexpected truncation: prompt_ids.size(1): {prompt_ids.size(1)}, self.max_prompt_length: {self.max_prompt_length}"
        # if self.max_prompt_length is not None:
        #     prompt_ids = prompt_ids[:, -self.max_prompt_length :]
        #     prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        # we need multimodal_inputs for rollout
        multimodal_inputs = {
            k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]
        }
        # Generate completions using either vLLM or regular generation
        if self.environment is not None:
            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step
            if self.guided_decoding_regex: # None
                guided_decoding = GuidedDecodingParams(backend="outlines", regex=self.guided_decoding_regex)
            else:
                guided_decoding = None
            sampling_params = SamplingParams(
                n=1,  # vLLM on each GPU generates only 1 in colocate mode # 每个GPU只生成1个补全
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=-1 if self.top_k is None else self.top_k,
                min_p=0.0 if self.min_p is None else self.min_p,
                max_tokens=self.max_completion_length,
                guided_decoding=guided_decoding,
                # TODO: temporary bad words for Qwen2.5VL
                bad_words=["matchCondition", "addCriterion", "_Parms", "actionDate", "fkk", "↤\n↤", " addCriterion"]
            )
            if self.vllm_tensor_parallel_size > 1:
                orig_size = len(prompts)
                # We prefer the messages for convinience and brevity for environment interface
                gathered_messages = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_messages, prompts, group=self.tp_group)
                gathered_messages = [p for sublist in gathered_messages for p in sublist]
                print(f"num of gathered_messages: {len(gathered_messages)}")
            else:
                # We prefer the messages for convinience and brevity for environment interface
                gathered_messages = prompts
            with profiling_context(self, "environment.generate"):
                multimodal_cache = [x['multimodal_cache'] for x in inputs]       # Cached vision tensors for interaction
                env_profiling_metrics = {}

                ### S1 ###
                generated_messages = self.environment.generate(gathered_messages, multimodal_cache, 
                                                            profiling_metrics=env_profiling_metrics, sampling_params=sampling_params,) ### S1
                # log to wandb
                if "wandb" in self.args.report_to and wandb.run is not None and self.accelerator.is_main_process:
                    for k, v in env_profiling_metrics.items():
                        wandb.log({f"{k}": v})
            if self.args.use_counterfactual_reasoning:
                # 反事实推理，需要额外的一次prediction
                ### s2 ###
                with profiling_context(self, "environment.counterfactual_reasoning"): ### S2
                    visual_trace_batch, has_visual_trace_batch = get_visual_trace(generated_messages, questions, options) # # 获取视觉轨迹（仅包含视觉信息的消息）
                    visual_trace_batch_filtered = [visual_trace_batch[i] for i in range(len(visual_trace_batch)) if has_visual_trace_batch[i]] #  # 过滤出有视觉轨迹的样本,这里是这么写，实际上环境已经强制设置至少调用一次工具了
                    # visual_trace_batch_prediction_texts = ["I don't know"] * len(visual_trace_batch) # # 初始化预测文本列表 对所有样本默认回答I don't know
                    if len(visual_trace_batch_filtered) > 0:
                        generated_messages_visual = self.environment._single_turn_generate_vllm_S2(visual_trace_batch_filtered, sampling_params=sampling_params) ### S2
                    prompts_text_visual = [self.processing_class.apply_chat_template(x, tokenize=False, add_generation_prompt=True) for x in visual_trace_batch_filtered]
                    images_visual, videos_visual, video_kwargs_visual = self.process_vision_fn(visual_trace_batch_filtered, return_video_kwargs=True)
                    prompt_inputs_visual = self.processing_class(
                        text=deepcopy(prompts_text_visual),    # bug fix: inplace operation will change the original prompts_text
                        images=images_visual,
                        videos=videos_visual,
                        fps=video_kwargs_visual["fps"],
                        padding=True, 
                        return_tensors="pt", 
                        padding_side="left",  # # 结果: [PAD PAD PAD prompt_tokens] 生成时需要所有样本的prompt末尾对齐，方便模型从相同位置开始生成
                        add_special_tokens=False,
                    )
                    prompt_inputs_visual = super()._prepare_inputs(prompt_inputs_visual)
                    prompt_ids_visual, prompt_mask_visual = prompt_inputs_visual["input_ids"], prompt_inputs_visual["attention_mask"]
            ### S3 ###
            reflection_messages_v2 = merge_and_reflect_v2(generated_messages, generated_messages_visual, questions, options, durations)  # 基于生成的响应进行反思，生成新的消息列表
            generated_messages_3 = self.environment.generate(reflection_messages_v2, multimodal_cache=multimodal_cache, sampling_params=sampling_params, profiling_metrics=env_profiling_metrics)
            prompts_text_3 = [self.processing_class.apply_chat_template(x, tokenize=False, add_generation_prompt=True) for x in reflection_messages_v2]
            images_visual_3, videos_visual_3, video_kwargs_visual_3 = self.process_vision_fn(reflection_messages_v2, return_video_kwargs=True)
            prompt_inputs_3 = self.processing_class(
                text=deepcopy(prompts_text_3),    # bug fix: inplace operation will change the original prompts_text
                images=images_visual_3,
                videos=videos_visual_3,
                fps=video_kwargs_visual_3["fps"],
                padding=True, 
                return_tensors="pt", 
                padding_side="left",  # # 结果: [PAD PAD PAD prompt_tokens] 生成时需要所有样本的prompt末尾对齐，方便模型从相同位置开始生成
                add_special_tokens=False,
            )
            prompt_inputs_3 = super()._prepare_inputs(prompt_inputs_3)
            prompt_ids_3, prompt_mask_3 = prompt_inputs_3["input_ids"], prompt_inputs_3["attention_mask"]

             # =================== 处理生成的响应 ===================

            ##### S1 ##### 
            generated_texts = self.processing_class.apply_chat_template(generated_messages, tokenize=False, add_generation_prompt=False)  # 将生成的多轮对话消息转换为纯文本格式,模版
            # extract multimodal inputs
            image_inputs, video_inputs, video_kwargs = self.process_vision_fn(generated_messages, return_video_kwargs=True)
            # Convert to token ids
            prompt_completion_inputs = self.processing_class( # 将生成的文本和多模态输入转换为token IDs
                text=generated_texts, # 包含 prompt + completion
                images=image_inputs,
                videos=video_inputs,
                fps=video_kwargs["fps"],
                padding=True,
                return_tensors="pt",
                padding_side="right",   # NOTE: padding_side is right, because we want to keep the model response tokens # 结果: [prompt_tokens completion_tokens PAD PAD]
                add_special_tokens=False,
            )   # shape: (1, N)
            # Move to device
            prompt_completion_inputs = super()._prepare_inputs(prompt_completion_inputs)
            # Don't forget to update multimodal_inputs # 更新多模态输入（包含生成响应中的多模态数据）
            multimodal_inputs = {
                k: v for k, v in prompt_completion_inputs.items() if k not in ["input_ids", "attention_mask"]
            }
            prompt_completion_ids = prompt_completion_inputs["input_ids"] # 可以用self.processing_class.batch_decode(completion_ids, skip_special_tokens=True) 验证
            prompt_completion_mask = prompt_completion_inputs["attention_mask"]

            ##### S2 ##### 
            generated_texts_visual = self.processing_class.apply_chat_template(generated_messages_visual, tokenize=False, add_generation_prompt=False)  # 将生成的多轮对话消息转换为纯文本格式,模版
            # extract multimodal inputs
            image_inputs_visual, video_inputs_visual, video_kwargs_visual = self.process_vision_fn(generated_messages_visual, return_video_kwargs=True)
            # Convert to token ids
            prompt_completion_inputs_visual = self.processing_class( # 将生成的文本和多模态输入转换为token IDs
                text=generated_texts_visual, # 包含 prompt + completion
                images=image_inputs_visual,
                videos=video_inputs_visual,
                fps=video_kwargs_visual["fps"],
                padding=True,
                return_tensors="pt",
                padding_side="right",   # NOTE: padding_side is right, because we want to keep the model response tokens # 结果: [prompt_tokens completion_tokens PAD PAD]
                add_special_tokens=False,
            )   # shape: (1, N)
            # Move to device
            prompt_completion_inputs_visual = super()._prepare_inputs(prompt_completion_inputs_visual)
            # Don't forget to update multimodal_inputs # 更新多模态输入（包含生成响应中的多模态数据）
            multimodal_inputs_visual = {
                k: v for k, v in prompt_completion_inputs_visual.items() if k not in ["input_ids", "attention_mask"]
            }
            prompt_completion_ids_visual = prompt_completion_inputs_visual["input_ids"] # 可以用self.processing_class.batch_decode(completion_ids, skip_special_tokens=True) 验证
            prompt_completion_mask_visual = prompt_completion_inputs_visual["attention_mask"]


            ##### S3 #####  
            generated_texts_3 = self.processing_class.apply_chat_template(generated_messages_3, tokenize=False, add_generation_prompt=False)  # 将生成的多轮对话消息转换为纯文本格式,模版
            # extract multimodal inputs
            image_inputs_3, video_inputs_3, video_kwargs_3 = self.process_vision_fn(generated_messages_3, return_video_kwargs=True)
            # Convert to token ids
            prompt_completion_inputs_3 = self.processing_class( # 将生成的文本和多模态输入转换为token IDs
                text=generated_texts_3, # 包含 prompt + completion
                images=image_inputs_3,
                videos=video_inputs_3,
                fps=video_kwargs_3["fps"],
                padding=True,
                return_tensors="pt",
                padding_side="right",   # NOTE: padding_side is right, because we want to keep the model response tokens # 结果: [prompt_tokens completion_tokens PAD PAD]
                add_special_tokens=False,
            )   # shape: (1, N)
            # Move to device
            prompt_completion_inputs_3 = super()._prepare_inputs(prompt_completion_inputs_3)
            # Don't forget to update multimodal_inputs # 更新多模态输入（包含生成响应中的多模态数据）
            multimodal_inputs_3 = {
                k: v for k, v in prompt_completion_inputs_3.items() if k not in ["input_ids", "attention_mask"]
            }
            prompt_completion_ids_3 = prompt_completion_inputs_3["input_ids"] # 可以用self.processing_class.batch_decode(completion_ids, skip_special_tokens=True) 验证
            prompt_completion_mask_3 = prompt_completion_inputs_3["attention_mask"]










             # =================== 提取补全部分 ===================

            ############## S1 ################
            prompt_length = prompt_inputs["input_ids"].size(1) # [2,3043]
            left_paddings = (prompt_inputs["attention_mask"] == 0).sum(dim=1)   # left padding size, before max_prompt_length truncation [0,0]
            prompt_actual_lengths = prompt_length - left_paddings  # 实际的prompt长度，注意prompt前面是左填充
            completion_ids, completion_mask = extract_completion_from_full_sequence( # # 从完整序列中提取补全部分（模型响应部分）包含id和mask
                prompt_completion_ids, prompt_completion_mask, prompt_actual_lengths, self.processing_class.pad_token_id
            )
            ### 下面这部分没用到 ####
            prompt_ids_no_padding = [prompt_completion_ids[i][:prompt_actual_lengths[i]] for i in range(len(prompt_completion_ids))]
            prompt_attention_mask_no_padding = [prompt_completion_mask[i][:prompt_actual_lengths[i]] for i in range(len(prompt_completion_mask))]
            prompt_ids_new = pad(prompt_ids_no_padding, padding_value=self.processing_class.pad_token_id, padding_side="left") # padding的
            prompt_attention_mask_new = pad(prompt_attention_mask_no_padding, padding_value=0, padding_side="left") # padding的
            ### 上面这部分没用到 ###
        
            # assert torch.all(prompt_ids_new == prompt_ids), f"prompt_ids_new: {prompt_ids_new.shape}, prompt_ids: {prompt_ids.sha pe}"
            # assert torch.all(prompt_attention_mask_new == prompt_mask), f"prompt_attention_mask_new: {prompt_attention_mask_new.shape}, prompt_mask: {prompt_mask.shape}"
            # ------------------------------
            # Mask tool responsed tokens, keep all model response tokens
            # =================== 处理工具响应掩码 ===================
            # 计算工具响应掩码：将工具调用部分的token掩码掉，只保留模型响应
            completion_mask = compute_tool_response_mask(completion_ids) * completion_mask ## 工具调用<|im_start|>tool ... <|im_end|> 是外部工具生成的不能计算损失
            # if self.accelerator.is_main_process:
            #     for i in range(len(completion_ids)):
            #         print(f"[DEBUG][RANK: {self.accelerator.process_index}], {clear_output(self.processing_class.decode(prompt_ids[i], clean_up_tokenization_spaces=False))=}")
            #         print(f"[DEBUG][RANK: {self.accelerator.process_index}], {clear_output(self.processing_class.decode(prompt_ids_new[i], clean_up_tokenization_spaces=False))=}")
            #         print(f"[DEBUG][RANK: {self.accelerator.process_index}], {clear_output(self.processing_class.decode(completion_ids[i], clean_up_tokenization_spaces=False))=}")
            prompt_completion_ids = torch.cat([prompt_inputs['input_ids'], completion_ids], dim=1) # 重新拼接下，包含padding部分，prompt_inputs是左pad，而completion_ids是右pad，并且有mask就足够了
            prompt_completion_mask = torch.cat([prompt_inputs['attention_mask'], completion_mask], dim=1)



            ############## S2 ################
            prompt_length = prompt_inputs_visual["input_ids"].size(1)
            left_paddings = (prompt_inputs_visual["attention_mask"] == 0).sum(dim=1)
            prompt_actual_lengths = prompt_length - left_paddings  # 实际的prompt长度，注意prompt前面是左填充
            completion_ids_visual, completion_mask_visual = extract_completion_from_full_sequence( # # 从完整序列中提取补全部分（模型响应部分）包含id和mask
                prompt_completion_ids_visual, prompt_completion_mask_visual, prompt_actual_lengths, self.processing_class.pad_token_id
            )
            completion_mask_visual = compute_tool_response_mask(completion_ids_visual) * completion_mask_visual
            prompt_completion_ids_visual = torch.cat([prompt_inputs_visual['input_ids'], completion_ids_visual], dim=1)
            prompt_completion_mask_visual = torch.cat([prompt_inputs_visual['attention_mask'], completion_mask_visual], dim=1)

            ############## S3 ################
            prompt_length = prompt_inputs_3["input_ids"].size(1)
            left_paddings = (prompt_inputs_3["attention_mask"] == 0).sum(dim=1)
            prompt_actual_lengths = prompt_length - left_paddings  # 实际的prompt长度，注意prompt前面是左填充
            completion_ids_3, completion_mask_3 = extract_completion_from_full_sequence( # # 从完整序列中提取补全部分（模型响应部分）包含id和mask
                prompt_completion_ids_3, prompt_completion_mask_3, prompt_actual_lengths, self.processing_class.pad_token_id
            )
            completion_mask_3 = compute_tool_response_mask(completion_ids_3) * completion_mask_3
            prompt_completion_ids_3 = torch.cat([prompt_inputs_3['input_ids'], completion_ids_3], dim=1)
            prompt_completion_mask_3 = torch.cat([prompt_inputs_3['attention_mask'], completion_mask_3], dim=1)


        ############## 从这里开始！！！！！！    
        # =================== 掩码处理（EOS和截断） ===================
        # Mask everything after the first EOS token
        # 找到第一个EOS（结束）token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)] # 对于有 EOS 的序列，找到第一个 EOS 的位置
        # 创建序列索引
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        # 创建补全掩码：将EOS之后的部分掩码掉
        if self.environment is None: # 多轮对话丢弃这个EOS掩码，不执行掩码
            # NOTE: Do NOT apply single-turn completion mask to multi-turn completions 注意：对于多轮对话，不应用单轮完成掩码
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        # 如果启用了掩码截断的补全，将未以EOS结束的补全掩码掉
        if self.mask_truncated_completions:# False
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation 将prompt掩码和补全掩码拼接起来
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C) # 和前面一模一样，不用担心换一个名字罢了

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens # 计算需要保留的logits数量（只计算补全token的logits）
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        # =================== 计算旧策略的对数概率 ===================
        with torch.no_grad():
            # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
            # old_per_token_logps == per_token_logps, so we can skip it's computation here, and use
            # per_token_logps.detach() instead.
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps: # off-policy
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, multimodal_inputs, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None # on-policy 执行这个 S1 S2 S3都是None

        # =================== 解码补全文本 ===================
        ### S1
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]): # 执行这个
            completions = []
            # 根据输入格式调整补全格式
            # 检查 prompt 最后一条消息是否是 assistant（部分回复的情况）
            # 如果是，取出其内容作为 "bootstrap"（已有的开头部分）
            for prompt, completion in zip(prompts, completions_text): ## batch 处理
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                assert bootstrap == "", "I don't want to handle bootstrap in VL-GRPO"
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else: 
            completions = completions_text
        
        ### S2
        completions_text_visual = self.processing_class.batch_decode(completion_ids_visual, skip_special_tokens=True)
        # if is_conversational(inputs[0]): # 执行这个
        #     completions = []
        #     # 根据输入格式调整补全格式
        #     # 检查 prompt 最后一条消息是否是 assistant（部分回复的情况）
        #     # 如果是，取出其内容作为 "bootstrap"（已有的开头部分）
        #     for prompt, completion in zip(prompts, completions_text_visual): ## batch 处理
        #         bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
        #         assert bootstrap == "", "I don't want to handle bootstrap in VL-GRPO"
        #         completions.append([{"role": "assistant", "content": bootstrap + completion}])
        # else: 
        #     completions = completions_text_visual

        ### S3
        completions_text_3 = self.processing_class.batch_decode(completion_ids_3, skip_special_tokens=True)
        # if is_conversational(inputs[0]): # 执行这个
        #     completions = []
        #     # 根据输入格式调整补全格式
        #     # 检查 prompt 最后一条消息是否是 assistant（部分回复的情况）
        #     # 如果是，取出其内容作为 "bootstrap"（已有的开头部分）
        #     for prompt, completion in zip(prompts, completions_text_visual): ## batch 处理
        #         bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
        #         assert bootstrap == "", "I don't want to handle bootstrap in VL-GRPO"
        #         completions.append([{"role": "assistant", "content": bootstrap + completion}])
        # else: 
        #     completions = completions_text_3
        # =================== 计算奖励 ===================
        # 初始化奖励张量
        rewards_per_func_S1 = torch.zeros(len(prompts), len(self.reward_funcs), device=device) # 【B,3】
        rewards_per_func_S2 = torch.zeros(len(prompts), len(self.reward_funcs), device=device) # 【B,3】
        rewards_per_func_S3 = torch.zeros(len(prompts), len(self.reward_funcs), device=device) # 【B,3】

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance( # 不执行这个，自定义奖励函数而不是模型
                    reward_func, nn.Module
                ):  # Module instead of PretrainedModel for compat with compiled models
                    if is_conversational(inputs[0]): # 是否是对话，是得，包含user和assistant角色的消息
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else: # 自定义奖励函数
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion", "messages", "multimodal_cache"]] # ['video_id', 'video_path', 'question', 'options', 'answer', 'duration_group', 'gt_frame_index', 'duration', 'video']
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys} # 上述键的字典
                    if self.environment is None:
                        output_reward_func = reward_func(prompts=prompts, completions=completions_text, **reward_kwargs)
                    elif self.args.use_counterfactual_reasoning: # 执行这个
                        # 反事实推理 得到每个奖励函数的奖励 prompts, completions, messages, **kwargs
                        # output_reward_func = reward_func(prompts=prompts, completions=generated_messages, completions_visual=generated_messages_visual, final_msg=generated_messages_3, **reward_kwargs) # # 原始 prompt（对话格式） # 解码后的纯文本补全  # 完整的多轮对话消息 # 反事实推理结果
                        output_reward_func_S1 = reward_func(prompts=prompts, completions=generated_messages, completions_visual=generated_messages_visual, final_msg=generated_messages_3, S1=True, **reward_kwargs)
                        output_reward_func_S2 = reward_func(prompts=prompts, completions=generated_messages, completions_visual=generated_messages_visual, final_msg=generated_messages_3, S2=True, **reward_kwargs)
                        output_reward_func_S3 = reward_func(prompts=prompts, completions=generated_messages, completions_visual=generated_messages_visual, final_msg=generated_messages_3, S3=True, **reward_kwargs)

                    else:
                        # pass generated_messages for environment
                        output_reward_func = reward_func(prompts=prompts, completions=completions_text, messages=generated_messages, **reward_kwargs)
                    # Convert None values to NaN output_reward_func
                    # output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func] # # 将None值转换为NaN


                    # # ==================== 【插入修复开始】 ====================
                    # expected_len = len(prompts)

                    # # 检查 S1
                    # if len(output_reward_func_S1) != expected_len:
                    #     print(f"\n[ERROR FIX] Reward Func '{self.reward_func_names[i]}' (S1) returned length {len(output_reward_func_S1)}, expected {expected_len}. Filling with 0.0.")
                    #     output_reward_func_S1 = [0.0] * expected_len

                    # # 检查 S2
                    # if len(output_reward_func_S2) != expected_len:
                    #     print(f"\n[ERROR FIX] Reward Func '{self.reward_func_names[i]}' (S2) returned length {len(output_reward_func_S2)}, expected {expected_len}. Filling with 0.0.")
                    #     output_reward_func_S2 = [0.0] * expected_len

                    # # 检查 S3
                    # if len(output_reward_func_S3) != expected_len:
                    #     print(f"\n[ERROR FIX] Reward Func '{self.reward_func_names[i]}' (S3) returned length {len(output_reward_func_S3)}, expected {expected_len}. Filling with 0.0.")
                    #     output_reward_func_S3 = [0.0] * expected_len
                    # # ==================== 【插入修复结束】 ====================
                    
                    # rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
                    output_reward_func_S1 = [reward if reward is not None else torch.nan for reward in output_reward_func_S1] # # 将None值转换为NaN
                    output_reward_func_S2 = [reward if reward is not None else torch.nan for reward in output_reward_func_S2]
                    output_reward_func_S3 = [reward if reward is not None else torch.nan for reward in output_reward_func_S3]

                    rewards_per_func_S1[:, i] = torch.tensor(output_reward_func_S1, dtype=torch.float32, device=device)
                    rewards_per_func_S2[:, i] = torch.tensor(output_reward_func_S2, dtype=torch.float32, device=device)
                    rewards_per_func_S3[:, i] = torch.tensor(output_reward_func_S3, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func_S1).all(dim=1).any(): # 某些行所有奖励函数都返回None，则发出详细警告
            nan_row_idx = torch.isnan(rewards_per_func_S1).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func_S1 = gather(rewards_per_func_S1) # 将每个GPU上的奖励张量收集到所有GPU上
        rewards_per_func_S2 = gather(rewards_per_func_S2)
        rewards_per_func_S3 = gather(rewards_per_func_S3)
        # Apply weights to each reward function's output and sum

        rewards_S1 = (rewards_per_func_S1 * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1) # 叠加 [global_batch_size]
        rewards_S2 = (rewards_per_func_S2 * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        rewards_S3 = (rewards_per_func_S3 * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards_S1 = rewards_S1.view(-1, self.num_generations).mean(dim=1) # 【1,2】，代表一个指令两个补全，只有一个组，所以这个组的均值为torch.Size([1])
        std_grouped_rewards_S1 = rewards_S1.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards_S2 = rewards_S2.view(-1, self.num_generations).mean(dim=1) # 【1,2】，代表一个指令两个补全，只有一个组，所以这个组的均值为torch.Size([1])
        std_grouped_rewards_S2 = rewards_S2.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards_S3 = rewards_S3.view(-1, self.num_generations).mean(dim=1) # 【1,2】，代表一个指令两个补全，只有一个组，所以这个组的均值为torch.Size([1])
        std_grouped_rewards_S3 = rewards_S3.view(-1, self.num_generations).std(dim=1)        


        # Normalize the rewards to compute the advantages
        mean_grouped_rewards_S1 = mean_grouped_rewards_S1.repeat_interleave(self.num_generations, dim=0) # 重新恢复到[global_batch_size]维度，重复一下就行
        std_grouped_rewards_S1 = std_grouped_rewards_S1.repeat_interleave(self.num_generations, dim=0)
        mean_grouped_rewards_S2 = mean_grouped_rewards_S2.repeat_interleave(self.num_generations, dim=0) # 重新恢复到[global_batch_size]维度，重复一下就行
        std_grouped_rewards_S2 = std_grouped_rewards_S2.repeat_interleave(self.num_generations, dim=0)
        mean_grouped_rewards_S3 = mean_grouped_rewards_S3.repeat_interleave(self.num_generations, dim=0) # 重新恢复到[global_batch_size]维度，重复一下就行
        std_grouped_rewards_S3 = std_grouped_rewards_S3.repeat_interleave(self.num_generations, dim=0)

        advantages_1 = rewards_S1 - mean_grouped_rewards_S1
        advantages_2 = rewards_S2 - mean_grouped_rewards_S2
        advantages_3 = rewards_S3 - mean_grouped_rewards_S3


        if self.scale_rewards: # 不执行
            advantages_1 = advantages_1 / (std_grouped_rewards_S1 + 1e-4)
            advantages_2 = advantages_2 / (std_grouped_rewards_S2 + 1e-4)
            advantages_3 = advantages_3 / (std_grouped_rewards_S3 + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages_1 = advantages_1[process_slice] # 每个GPU上的优势函数
        advantages_2 = advantages_2[process_slice] # 每个GPU上的优势函数
        advantages_3 = advantages_3[process_slice] # 每个GPU上的优势函数

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item() # 统计训练看到的token总数
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # log completion lengths, mean, min, max
        agg_completion_mask = self.accelerator.gather_for_metrics(completion_mask.sum(1)) # 每个样本的completion长度
        self._metrics[mode]["completions/mean_length"].append(agg_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_mask.float().max().item())

        # identify sequences that terminated with EOS and log their lengths 这个不用管
        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(agg_completion_mask)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio) # 0
        if len(term_completion_mask) == 0:
            # edge case where no completed sequences are found
            term_completion_mask = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_mask.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards_1 = torch.nanmean(rewards_per_func_S1[:, i]).item()
            self._metrics[mode][f"rewards_1/{reward_func_name}/mean"].append(mean_rewards_1)
            std_rewards_1 = nanstd(rewards_per_func_S1[:, i]).item()
            self._metrics[mode][f"rewards_1/{reward_func_name}/std"].append(std_rewards_1)
        self._metrics[mode]["reward_1"].append(mean_grouped_rewards_S1.mean().item())
        self._metrics[mode]["reward_std_1"].append(std_grouped_rewards_S1.mean().item())

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards_2 = torch.nanmean(rewards_per_func_S2[:, i]).item()
            self._metrics[mode][f"rewards_2/{reward_func_name}/mean"].append(mean_rewards_2)
            std_rewards_2 = nanstd(rewards_per_func_S2[:, i]).item()
            self._metrics[mode][f"rewards_2/{reward_func_name}/std"].append(std_rewards_2)
        self._metrics[mode]["reward_2"].append(mean_grouped_rewards_S2.mean().item())
        self._metrics[mode]["reward_std_2"].append(std_grouped_rewards_S2.mean().item())

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards_3 = torch.nanmean(rewards_per_func_S3[:, i]).item()
            self._metrics[mode][f"rewards_3/{reward_func_name}/mean"].append(mean_rewards_3)
            std_rewards_3 = nanstd(rewards_per_func_S3[:, i]).item()
            self._metrics[mode][f"rewards_3/{reward_func_name}/std"].append(std_rewards_3)
        self._metrics[mode]["reward_3"].append(mean_grouped_rewards_S3.mean().item())
        self._metrics[mode]["reward_std_3"].append(std_grouped_rewards_S3.mean().item())

        # Log prompt and completion texts
        self._textual_logs["prompt_1"].extend(gather_object(prompts_text))
        self._textual_logs["completion_1"].extend(gather_object(completions_text))
        self._textual_logs["prompt_2"].extend(gather_object(prompts_text_visual))
        self._textual_logs["completion_2"].extend(gather_object(completions_text_visual))
        self._textual_logs["prompt_3"].extend(gather_object(prompts_text_3))
        self._textual_logs["completion_3"].extend(gather_object(completions_text_3))



        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards_1"][name].extend(rewards_per_func_S1[:, i].tolist())
            self._textual_logs["rewards_2"][name].extend(rewards_per_func_S2[:, i].tolist())
            self._textual_logs["rewards_3"][name].extend(rewards_per_func_S3[:, i].tolist())

        return {
            "prompt_ids": prompt_ids, # S1
            "prompt_mask": prompt_mask,# S1
            "prompt_ids_visual": prompt_ids_visual,   # S2
            "prompt_mask_visual": prompt_mask_visual, # S2
            "prompt_ids_3": prompt_ids_3, # S3
            "prompt_mask_3": prompt_mask_3, # S3
            "completion_ids": completion_ids, # S1
            "completion_mask": completion_mask, # S1
            "completion_ids_visual": completion_ids_visual,   # S2
            "completion_mask_visual": completion_mask_visual, # S2
            "completion_ids_3": completion_ids_3, # S3
            "completion_mask_3": completion_mask_3, # S3
            "advantages_1": advantages_1, # S1
            "advantages_2": advantages_2, # S2
            "advantages_3": advantages_3, # S3
            "old_per_token_logps": old_per_token_logps, # 这个为None不用管
            "multimodal_inputs": multimodal_inputs, # S1
            "multimodal_inputs_visual": multimodal_inputs_visual, # S2
            "multimodal_inputs_3": multimodal_inputs_3, # S3
            "final_output_msgs": generated_messages_3, # 最终生成的多轮对话消息
        }

    def compute_liger_loss(self, unwrapped_model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = None
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, input_ids, attention_mask, multimodal_inputs=inputs["multimodal_inputs"], 
                        logits_to_keep=logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, input_ids, attention_mask, multimodal_inputs=inputs["multimodal_inputs"], 
                            logits_to_keep=logits_to_keep
                        )

        # get the last hidden state of the model
        last_hidden_state = self._get_last_hidden_state(unwrapped_model, input_ids, attention_mask, logits_to_keep)

        # compute loss and metrics using liger grpo loss
        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            attention_mask=completion_mask,
            advantages=inputs["advantages"],
            bias=unwrapped_model.lm_head.bias,
            old_per_token_logps=inputs["old_per_token_logps"],
            ref_per_token_logps=ref_per_token_logps,
        )
        # Extract metrics from the liger_grpo_loss output
        # KL divergence is the first metric when beta is non-zero
        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]

        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        if self.use_liger_loss:
            # Compute the loss using the liger grpo loss
            unwrapped_model = self.accelerator.unwrap_model(model)
            return self._forward_redirection(model, unwrapped_model, self.compute_liger_loss, unwrapped_model, inputs)
        else:
            return self._compute_loss(model, inputs) # 直接执行这一步

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        prompt_ids_visual, prompt_mask_visual = inputs["prompt_ids_visual"], inputs["prompt_mask_visual"]
        completion_ids_visual, completion_mask_visual = inputs["completion_ids_visual"], inputs["completion_mask_visual"]
        prompt_ids_3, prompt_mask_3 = inputs["prompt_ids_3"], inputs["prompt_mask_3"]
        completion_ids_3, completion_mask_3 = inputs["completion_ids_3"], inputs["completion_mask_3"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        input_ids_visual = torch.cat([prompt_ids_visual, completion_ids_visual], dim=1)
        attention_mask_visual = torch.cat([prompt_mask_visual, completion_mask_visual], dim=1)
        input_ids_3 = torch.cat([prompt_ids_3, completion_ids_3], dim=1)
        attention_mask_3 = torch.cat([prompt_mask_3, completion_mask_3], dim=1)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        logits_to_keep_visual = completion_ids_visual.size(1)
        logits_to_keep_s3 = completion_ids_3.size(1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, 
                                                    multimodal_inputs=inputs["multimodal_inputs"], 
                                                    logits_to_keep=logits_to_keep) # torch.Size([2, 2298]) [batch_size, completion_length]
        per_token_logps_visual = self._get_per_token_logps(model, input_ids_visual, attention_mask_visual, 
                                                            multimodal_inputs=inputs["multimodal_inputs_visual"], 
                                                            logits_to_keep=logits_to_keep_visual)
        per_token_logps_3 = self._get_per_token_logps(model, input_ids_3, attention_mask_3, 
                                                            multimodal_inputs=inputs["multimodal_inputs_3"], 
                                                            logits_to_keep=logits_to_keep_s3)
 
        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, input_ids, attention_mask, multimodal_inputs=inputs["multimodal_inputs"], 
                        logits_to_keep=logits_to_keep
                    )
                    ref_per_token_logps_visual = self._get_per_token_logps(
                        self.ref_model, input_ids_visual, attention_mask_visual, multimodal_inputs=inputs["multimodal_inputs_visual"], 
                        logits_to_keep=logits_to_keep_visual
                    )
                    ref_per_token_logps_3 = self._get_per_token_logps(
                        self.ref_model, input_ids_3, attention_mask_3, multimodal_inputs=inputs["multimodal_inputs_3"], 
                        logits_to_keep=logits_to_keep_s3
                    )
                    
                else: ## 这部分不执行不用管
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, input_ids, attention_mask, multimodal_inputs=inputs["multimodal_inputs"], 
                            logits_to_keep=logits_to_keep
                        )
            # per_token_kl = ( #  # 计算KL散度：KL(q||p) = exp(log_q - log_p) - (log_q - log_p) - 1
            #     torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            # )
            # per_token_kl_visual = ( #  # 计算KL散度：KL(q||p) = exp(log_q - log_p) - (log_q - log_p) - 1
            #     torch.exp(ref_per_token_logps_visual - per_token_logps_visual) - (ref_per_token_logps_visual - per_token_logps_visual) - 1
            # )


            # =================== [S1] KL 计算 (带数值保护) ===================
            # 先计算 log_q - log_p
            diff_1 = ref_per_token_logps - per_token_logps
            # 关键：截断差异范围到 [-10, 10]，防止 exp(diff) 溢出
            # exp(10) ≈ 22026，是一个很大但安全的数字；如果不截断，exp(20) ≈ 4.8亿 会导致 NaN
            x_clamped_1 = torch.clamp(diff_1, min=-10, max=10)
            per_token_kl = torch.exp(x_clamped_1) - x_clamped_1 - 1

            # =================== [S2] Visual KL 计算 (带数值保护) ===================
            # 先计算 log_q - log_p
            diff_2 = ref_per_token_logps_visual - per_token_logps_visual
            # 同样截断
            x_clamped_2 = torch.clamp(diff_2, min=-10, max=10)
            per_token_kl_visual = torch.exp(x_clamped_2) - x_clamped_2 - 1  

            # =================== [S3] Visual KL 计算 (带数值保护) ===================
            # 先计算 log_q - log_p
            diff_3= ref_per_token_logps_3 - per_token_logps_3
            # 同样截断
            x_clamped_3 = torch.clamp(diff_3, min=-10, max=10)
            per_token_kl_3 = torch.exp(x_clamped_3) - x_clamped_3 - 1  
        # Compute the loss
        advantages_1 = inputs["advantages_1"]
        advantages_2 = inputs["advantages_2"]
        advantages_3 = inputs["advantages_3"]
        # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        )

        # old_per_token_logps_visual = (
        #     per_token_logps_visual.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        # )

        # old_per_token_logps_3 = (
        #     per_token_logps_3.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        # )
        # 计算 Ratio S1
        coef_1 = torch.exp(per_token_logps - old_per_token_logps) # 新策略概率 / 旧策略概率 on-policy是1.0
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        loss1_unclipped = coef_1 * advantages_1.unsqueeze(1)
        loss1_clipped = coef_2 * advantages_1.unsqueeze(1)
        per_token_loss_S1 = -torch.min(loss1_unclipped, loss1_clipped)
        if self.beta != 0.0:
            per_token_loss_S1 = per_token_loss_S1 + 0.05 * per_token_kl


        # 计算 Ratio S2
        old_per_token_logps_visual = per_token_logps_visual.detach()
        coef_1_v = torch.exp(per_token_logps_visual - old_per_token_logps_visual)
        coef_2_v = torch.clamp(coef_1_v, 1 - self.epsilon_low, 1 + self.epsilon_high)
        loss2_unclipped = coef_1_v * advantages_2.unsqueeze(1)
        loss2_clipped = coef_2_v * advantages_2.unsqueeze(1)
        per_token_loss_S2 = -torch.min(loss2_unclipped, loss2_clipped)
        if self.beta != 0.0:
            per_token_loss_S2 = per_token_loss_S2 + 0.05 * per_token_kl_visual # 负数+正数


        # 计算 Ratio S3, 但是不加入KL暂时
        old_per_token_logps_3 = per_token_logps_3.detach()
        coef_1_3 = torch.exp(per_token_logps_3 - old_per_token_logps_3)
        coef_2_3 = torch.clamp(coef_1_3, 1 - self.epsilon_low, 1 + self.epsilon_high)
        loss3_unclipped = coef_1_3 * advantages_3.unsqueeze(1)
        loss3_clipped = coef_2_3 * advantages_3.unsqueeze(1)
        per_token_loss_S3 = -torch.min(loss3_unclipped, loss3_clipped)+ 0.005 * per_token_kl_3



        # 聚合总 Loss (Mean reduction)
        if self.loss_type == "grpo":
            # S1 Loss Mean # .sum(-1) 从[b,s]变成[b],故而是除以batch_size  对所有样本求平均
            loss_s1 = ((per_token_loss_S1 * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            # S2 Loss Mean
            loss_s2 = ((per_token_loss_S2 * completion_mask_visual).sum(-1) / completion_mask_visual.sum(-1).clamp(min=1.0)).mean()
            # S3 Loss Mean
            loss_s3 = ((per_token_loss_S3 * completion_mask_3).sum(-1) / completion_mask_3.sum(-1).clamp(min=1.0)).mean()
            
            # Total Loss
            loss = loss_s1 + loss_s2 + loss_s3
            
        elif self.loss_type == "bnpo": # 整个batch的总损失 ÷ 整个batch的总token数
            # 分母直接是所有 mask 的总和，不再区分 S1/S2/S3 的样本长度差异
            loss_s1 = ((per_token_loss_S1 * completion_mask).sum() / completion_mask.sum().clamp(min=1.0))
            # S2 Loss Mean
            loss_s2 = ((per_token_loss_S2 * completion_mask_visual).sum() / completion_mask_visual.sum().clamp(min=1.0))
            # S3 Loss Mean
            loss_s3 = ((per_token_loss_S3 * completion_mask_3).sum() / completion_mask_3.sum().clamp(min=1.0))
            loss = loss_s1 + loss_s2 + loss_s3
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


        # Log the metrics
        mode = "train" if self.model.training else "eval"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())
            mean_kl_visual = (per_token_kl_visual * completion_mask_visual).sum() / completion_mask_visual.sum()
            self._metrics[mode]["kl_visual"].append(self.accelerator.gather_for_metrics(mean_kl_visual).nanmean().item())
            mean_kl_3 = (per_token_kl_3 * completion_mask_3).sum() / completion_mask_3.sum()
            self._metrics[mode]["kl_3"].append(self.accelerator.gather_for_metrics(mean_kl_3).nanmean().item())

        # Compute the clipped probability ratios 这个好像没啥意义，因为是on-policy
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages_1.unsqueeze(1) < 0) # 概率比低于下限且优势为负（应该减少概率）
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages_1.unsqueeze(1) > 0)#  概率比高于上限且优势为正（应该增加概率）
        is_region_clipped = is_low_clipped | is_high_clipped # 两种裁剪情况的并集

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()


        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd
                print(f"DEBUG: Buffer S1 len: {len(self._textual_logs['prompt_1'])}")
                print(f"DEBUG: Buffer S2 len: {len(self._textual_logs['prompt_2'])}")
                print(f"DEBUG: Buffer S2 len: {len(self._textual_logs['prompt_3'])}")
                # 1. 记录 S1 阶段 (原始对话)
                # if len(self._textual_logs["prompt_1"]) > 0:
                #     table_s1 = {
                #         "step": [str(self.state.global_step)] * len(self._textual_logs["prompt_1"]),
                #         "prompt": self._textual_logs["prompt_1"],
                #         "completion": self._textual_logs["completion_1"],
                #         **self._textual_logs["rewards"],
                #     }
                #     df_s1 = pd.DataFrame(table_s1)
                #     if self.wandb_log_unique_prompts:
                #         df_s1 = df_s1.drop_duplicates(subset=["prompt"])
                #     wandb.log({"completions_S1": wandb.Table(dataframe=df_s1)})

                # # 2. 记录 S2 阶段 (Visual Trace)
                # if len(self._textual_logs["prompt_2"]) > 0:
                #     table_s2 = {
                #         "step": [str(self.state.global_step)] * len(self._textual_logs["prompt_2"]),
                #         "prompt": self._textual_logs["prompt_2"],
                #         "completion": self._textual_logs["completion_2"],
                #         **self._textual_logs["rewards"],
                #     }
                #     df_s2 = pd.DataFrame(table_s2)
                #     if self.wandb_log_unique_prompts:
                #         df_s2 = df_s2.drop_duplicates(subset=["prompt"])
                #     wandb.log({"completions_S2": wandb.Table(dataframe=df_s2)})
                # # 3. 记录 S3 阶段 (Counterfactual Reasoning)
                # if len(self._textual_logs["prompt_3"]) > 0:
                #     table_s3 = {
                #         "step": [str(self.state.global_step)] * len(self._textual_logs["prompt_3"]),
                #         "prompt": self._textual_logs["prompt_3"],
                #         "completion": self._textual_logs["completion_3"],
                #         **self._textual_logs["rewards"],
                #     }
                #     df_s3 = pd.DataFrame(table_s3)
                #     if self.wandb_log_unique_prompts:
                #         df_s3 = df_s3.drop_duplicates(subset=["prompt"])
                #     wandb.log({"completions_S3": wandb.Table(dataframe=df_s3)})
                ## 合并记录
                if len(self._textual_logs["prompt_1"]) > 0 and len(self._textual_logs["prompt_2"]) > 0 and len(self._textual_logs["prompt_3"]) > 0:
                    assert len(self._textual_logs["prompt_1"]) == len(self._textual_logs["prompt_2"]), "S1 and S2 textual logs must have the same length for combined logging."
                    table_combined = {
                        "step": [str(self.state.global_step)] * len(self._textual_logs["prompt_1"]),
                        "prompt_1": self._textual_logs["prompt_1"],
                        "prompt_2": self._textual_logs["prompt_2"],
                        "prompt_3": self._textual_logs["prompt_3"],
                        "completion_1": self._textual_logs["completion_1"],
                        "completion_2": self._textual_logs["completion_2"],
                        "completion_3": self._textual_logs["completion_3"],
                    }

                    # 2. 显式添加 S1 奖励 (加后缀)
                    for key, val in self._textual_logs["rewards_1"].items():
                        table_combined[f"{key}_S1"] = val
                    
                    # 3. 显式添加 S2 奖励 (加后缀)
                    for key, val in self._textual_logs["rewards_2"].items():
                        table_combined[f"{key}_S2"] = val

                    # 4. 显式添加 S3 奖励 (加后缀)
                    for key, val in self._textual_logs["rewards_3"].items():
                        table_combined[f"{key}_S3"] = val

                    df = pd.DataFrame(table_combined)
                    # if self.wandb_log_unique_prompts:
                    wandb.log({"completions_S1S2": wandb.Table(dataframe=df)})

                    # Save completions to output_dir as JSON
                    json_path = os.path.join(self.args.output_dir, f"completions_S1S2_step{self.state.global_step}.json")
                    df.to_json(json_path, orient="records", force_ascii=False)

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))


def compute_tool_response_mask(seq, id_im_start=151644, id_tool=14172):
    """
    向量化计算掩码：从 seq (1, T) 形状的 LongTensor 中屏蔽所有 `<|im_start|>tool…<|im_end|>` 区域。
    返回长度 T 的 IntTensor，其中 1 表示保留，0 表示屏蔽。
    这里 id_im_start 和 id_tool 与模型相关，需要从 tokenizer 中获取。
    id_im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    id_tool     = tokenizer.convert_tokens_to_ids("tool") tool 还有 <tool_call> 都有专门的id分配
    """
    # 1) 标记所有 "<|im_start|>"
    is_im_start = seq == id_im_start # 151644 torch.Size([2, 2393])
    # 2) 计算 region_id
    region_id = is_im_start.int().cumsum(dim=1)
    # 3) 标记段开头是否为工具段
    next_is_tool = torch.zeros_like(seq, dtype=torch.bool) # 全零张量，False
    next_is_tool[:, :-1] = is_im_start[:, :-1] & (seq[:, 1:] == id_tool) # 当前序列前一个为"<|im_start|>"后一个为tool，则标记为True
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
    return completion_mask


def clear_output(text, reserve_pad=True):
    if reserve_pad:
        text = replace_n_pattern(text, "<|endoftext|>", "{N} * <|endoftext|>")
    else:
        text = replace_n_pattern(text, "<|endoftext|>", "")
    text = replace_n_pattern(text, "<|video_pad|>", "{N} * <|video_pad|>")
    text = replace_n_pattern(text, "<|image_pad|>", "{N} * <|image_pad|>")
    return text.strip("\n")

def replace_n_pattern(text, pattern, replace_pattern):
    """
    替换字符串中连续出现的指定模式。

    参数:
        text (str): 原始字符串。
        pattern (str): 需要匹配的模式。
        replace_pattern (str): 替换模式，其中 `{N}` 会被替换为连续匹配的数量。

    返回:
        str: 替换后的字符串。
    """
    regex = re.compile(f'({re.escape(pattern)})+')
    
    def replace_match(match):
        matched_text = match.group()
        count = matched_text.count(pattern)
        return replace_pattern.replace('{N}', str(count))
    
    result = regex.sub(replace_match, text)
    return result


def extract_completion_from_full_sequence(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_actual_lengths: torch.Tensor,
    pad_token_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从完整序列中提取completion部分，
    注意：这里假设input_ids和attention_mask是已经填充过的，并且padding_side是right
    
    参数:
    - input_ids: 完整序列的input_ids，形状为[batch_size, seq_len]
    - attention_mask: 完整序列的attention_mask，形状为[batch_size, seq_len]
    - prompt_actual_lengths: 每个样本中prompt部分的实际长度，形状为[batch_size]
    - pad_token_id: 用于填充的token ID
    
    返回:
    - completion_input_ids: completion部分的input_ids
    - completion_attention_mask: completion部分的attention_mask
    """
    batch_size = input_ids.size(0)
    completion_input_ids_list = []
    completion_attention_mask_list = []
    
    for i in range(batch_size):
        # 提取当前样本的completion部分
        split_point = prompt_actual_lengths[i]
        curr_completion_ids = input_ids[i, split_point:]
        curr_completion_mask = attention_mask[i, split_point:]
        # print("curr shape", curr_completion_ids.shape, curr_completion_mask.shape)
        # 过滤掉尾部的填充token (只保留有效内容)
        valid_indices = curr_completion_mask.nonzero(as_tuple=True)[0] #  会返回所有非零元素的下标（即有效token的位置）。加上 as_tuple=True，返回的是一个元组，第一个元素是下标tensor。
        if len(valid_indices) > 0:
            last_valid_idx = valid_indices[-1].item() + 1
            curr_completion_ids = curr_completion_ids[:last_valid_idx]
            curr_completion_mask = curr_completion_mask[:last_valid_idx]
        # print("valid shape", curr_completion_ids.shape, curr_completion_mask.shape)
        # 添加到列表
        completion_input_ids_list.append(curr_completion_ids)
        completion_attention_mask_list.append(curr_completion_mask)
    
    # 对completion部分进行右侧填充
    completion_input_ids = pad(completion_input_ids_list, padding_value=pad_token_id)
    completion_attention_mask = pad(completion_attention_mask_list, padding_value=0)
    return completion_input_ids, completion_attention_mask


class ReplayBuffer:
    """
    assume the buffer is a list of dicts, each dict is a single experience
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, experience: dict):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer.pop(0)
            self.buffer.append(experience)

    def sample(self):
        p = np.ones(len(self.buffer)) / len(self.buffer)
        selection = np.random.choice(np.arange(len(self.buffer)), size=1, p=p)
        selection = selection[0]
        return self.buffer[selection]

    def __len__(self):
        return len(self.buffer)


class SSRReplayBuffer(ReplayBuffer):
    # implementation of the SSR replay buffer from https://arxiv.org/pdf/2504.08837
    def __init__(self, capacity, alpha=1.0):
        super().__init__(capacity)
        self.alpha = alpha
        self.advantages = []

    def add(self, experience):
        EPS = 0.0001  # ensures we get non-zero advs when the buffer contains all 0 advantages
        advantage = experience["advantages"]
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.advantages.append(abs(advantage) + EPS)  # Store absolute advantage
        elif torch.any(advantage.abs() > EPS):
            # Replace the oldest entry if the buffer is full and adv is non zero
            self.buffer.pop(0)
            self.advantages.pop(0)
            self.buffer.append(experience)
            self.advantages.append(advantage.abs().mean())

    def sample(self):
        if not self.buffer:
            raise ValueError("Buffer is empty. Cannot sample from an empty buffer.")

        # Convert advantages to priorities
        scaled_priorities = np.power(self.advantages, self.alpha)
        total_priority = np.sum(scaled_priorities)
        probabilities = scaled_priorities / total_priority

        selection = np.random.choice(np.arange(len(self.buffer)), size=1, p=probabilities)
        selection = selection[0]
        return self.buffer[selection]


class DapoReplayBuffer(ReplayBuffer):
    # implementation of the SSR replay buffer from https://arxiv.org/pdf/2504.08837
    def __init__(self, capacity, alpha=1.0):
        super().__init__(capacity)
        self.alpha = alpha
        self.weights = []

    def add(self, experience):
        EPS = 0.0001  # ensures we get non-zero advs when the buffer contains all 0 advantages
        adv1 = experience["advantages_1"]
        adv2 = experience["advantages_2"]   
        adv3 = experience["advantages_3"]
        # 先比前两个，结果再和第三个比
        combined_abs_advantage = torch.maximum(adv1.abs(), torch.maximum(adv2.abs(), adv3.abs()))
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.weights.append(1.0)  # Store absolute advantage
        elif torch.any(combined_abs_advantage.abs() > EPS):
            # Replace the oldest entry if the buffer is full and adv is positive
            self.buffer.pop(0)
            self.weights.pop(0)
            self.buffer.append(experience)
            self.weights.append(1.0)

    def sample(self):
        if not self.buffer:
            raise ValueError("Buffer is empty. Cannot sample from an empty buffer.")

        # Convert advantages to priorities
        scaled_priorities = np.power(self.weights, self.alpha)
        total_priority = np.sum(scaled_priorities)
        probabilities = scaled_priorities / total_priority

        selection = np.random.choice(np.arange(len(self.buffer)), size=1, p=probabilities)
        selection = selection[0]
        return self.buffer[selection]


def get_replay_buffer(buffer_type: str, capacity: int, alpha: float = 1.0):
    if buffer_type == "ssr":
        return SSRReplayBuffer(capacity, alpha=alpha)
    elif buffer_type == "dapo":
        return DapoReplayBuffer(capacity, alpha=alpha)
    elif buffer_type == "none":
        return None
    else:
        raise ValueError(f"Invalid buffer type: {buffer_type}")


def get_visual_trace(batch_messages, batch_questions, batch_options):
    """
    get the visual only reasoning messages from the messages
    messages: batch
    NOTE: need deepcopy
    """
    filtered_batch_messages = []
    has_visual_trace_batch = []
    for messages, question ,option in zip(batch_messages, batch_questions, batch_options):
        filtered_messages = []
        filtered_messages.append({
            "role": "system", 
            "content": f"You are a strict Visual Evidence Analyst. Your task is to answer a multiple-choice question based STRICTLY on the provided video frames. When you don't have enough visual information, please say 'I don't know'."
        })
        content = []
        has_visual_trace = False
        for message in messages:
            if message['role'] == 'tool':
                for c in message['content']:
                    if isinstance(c, dict) and c.get("type") in ["text"] and c.get("text").startswith("Here are selected frames."):
                        # b跳过tool response 中的text prompt，也就是"Here are selected frames."开头的prompt不要，只保留时间戳text以及图片，一般没有视频
                        continue
                    content.append(c)
                    if isinstance(c, dict) and c.get("type") in ["video", "image"]:
                        # 检查是否有visual trace，不要跳出循环
                        has_visual_trace = True
        content.append(
            {
                "type": "text",
                "text": CSV_TEMPLATE_V1.format(
                            question=question, 
                            options=option,
                        )
            }
        )
        filtered_messages.append({
            "role": "user", "content": content
        })
        has_visual_trace_batch.append(has_visual_trace)
        filtered_batch_messages.append(filtered_messages)
    return filtered_batch_messages, has_visual_trace_batch

## [{'role': 'system', 'content': 'You are a helpful assistant. Please answer visual questions as briefly as possible. W...I don't know'."}, {'role': 'user', 'content': [...]}]
# 前面是系统指令，后面是对应的时间戳和选择的图像帧


def merge_and_reflect_v2(output_msgs, visual_trace_msgs, questions_batch, options_batch, durations_batch):
    reflection_messages = []
    
    for output_msg, visual_msg, question, option, duration in zip(output_msgs, visual_trace_msgs, questions_batch, options_batch, durations_batch):
        full_history = [msg for msg in output_msg if msg['role'] != 'system']
        visual_history = [msg for msg in visual_msg if msg['role'] != 'system']
        
        reflection_msg = []
        tool_use_prompt = get_tool_use_prompt(['seek_video_frames'])

        reflection_msg.append({
            "role": "system",
            "content": "You are a critical reasoning arbitrator capable of analyzing multiple reasoning paths and deriving the most accurate conclusion through tool-assisted verification.\n"+tool_use_prompt
        })
        # 添加标记消息
        reflection_msg.append({
            "role": "user",
            "content": [{"type": "text", "text": "Phase 1: Initial Answer"}]
        })
        
        # 直接添加原始消息（保留role）
        reflection_msg.extend(full_history)
        
        # 添加第二阶段标记
        reflection_msg.append({
            "role": "user",
            "content": [{"type": "text", "text": "Phase 2: Blind Verification"}]
        })
        
        reflection_msg.extend(visual_history)
        
        # 添加反思prompt
        reflection_msg.append({
            "role": "user",
            "content": [{
                "type": "text",
                "text": REFLECT_PROMPT_WITH_TOOLS.format(question=question, options=option, duration=duration)
            }]
        })
        
        reflection_messages.append(reflection_msg)
    
    return reflection_messages
