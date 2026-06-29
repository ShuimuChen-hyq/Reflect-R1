from .v11_valid_tool_split_S3_wandb_no_reasoning_fix import (
    reward_functions as v11_valid_tool_split_S3_wandb_no_reasoning_fix,
    reward_weights as v11_valid_tool_split_S3_wandb_no_reasoning_fix_weights,
)
from .v11_valid_tool_split_S123_no_reasoning import (
    reward_functions as v11_valid_tool_split_S123_no_reasoning,
    reward_weights as v11_valid_tool_split_S123_no_reasoning_weights,
)


REWARD_FUNCTIONS_REGISTRY = {
    "v11_valid_tool_split_S3_wandb_no_reasoning_fix": (
        v11_valid_tool_split_S3_wandb_no_reasoning_fix,
        v11_valid_tool_split_S3_wandb_no_reasoning_fix_weights,
    ),
    "v11_valid_tool_split_S123_no_reasoning": (
        v11_valid_tool_split_S123_no_reasoning,
        v11_valid_tool_split_S123_no_reasoning_weights,
    ),
}


def get_reward_functions(version: str):
    if version not in REWARD_FUNCTIONS_REGISTRY:
        raise ValueError(f"Invalid reward version: {version}")
    return REWARD_FUNCTIONS_REGISTRY[version]
