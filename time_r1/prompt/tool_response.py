# TOOL_RESPONSE_PROMPT = "Here are selected frames. They are located at {timestamps}.\n" \
# "If the frames provided above are sufficient to answer the user's question, please put your final answer within <answer></answer>. Otherwise invoke the tool again with different parameters in JSON format.\n"

TOOL_RESPONSE_PROMPT = (
    "Here are selected frames. They are located at {timestamps}.\n"
    "You must conduct reasoning inside <think> and </think> tags first, "
    "If the frames provided above are sufficient to answer the user's question, please put your final answer within <answer></answer>. Otherwise invoke the tool again by wrapping the JSON object within <tool_call> and </tool_call> tags.\n"
)

def get_tool_response_prompt(item: dict):
    return TOOL_RESPONSE_PROMPT.format(timestamps=item["timestamps"])
