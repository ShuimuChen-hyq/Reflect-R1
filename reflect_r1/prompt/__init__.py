from .v3 import make_prompt as v3_prompt
from .tool_response import get_tool_response_prompt

PROMPT_TEMPLATES = {
    "v3": v3_prompt,
    "tool_response": get_tool_response_prompt,
}

def get_prompt_fn(prompt_name):
    return PROMPT_TEMPLATES[prompt_name]
