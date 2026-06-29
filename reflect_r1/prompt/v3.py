# TEMPLATE_V3 = (
#     "You must conduct reasoning inside <think> and </think> first every time you call the tool or answer the question. "
#     "You must invoke tools to explore any video content you are interested in within <tool_call> </tool_call> tags.\n"
#     "You will then receive the tool response along with the corresponding video frames. You must wait for the response from the tool before answering or invoking the tool again.\n"
#     "If you are not sure, invoke again before answering."
#     "When you have enough information to answer the question, provide your answer within <answer> </answer> tags. Your answer should be supported by evidence from the video.\n"
#     "Question: {question}\n"
#     "Options:\n{options}\n"
#     "The video lasts for {duration} seconds.\n"
# )

TEMPLATE_V3 = (
    "You must conduct reasoning inside <think> and </think> first every time you call the tool or answer the question. "
    "You must invoke tools to explore any video content you are interested in within <tool_call> </tool_call> tags.\n"
    # --- 新增的强约束指令 ---
    "You cannot answer the question directly. You are required to invoke the tool at least once to explore the video content before providing your final answer.\n"
    # -----------------------
    "You will then receive the tool response along with the corresponding video frames. You must wait for the response from the tool before answering or invoking the tool again.\n"
    "If you are not sure, invoke again before answering."
    "You are allowed to use <tool_call></tool_call> tags for a maximum of 4 rounds.\n"
    "When you have enough information to answer the question, provide your answer within <answer> </answer> tags. Your answer should be supported by evidence from the video.\n"
    "Your output must follow the format: <think>Your reasoning process</think><tool_call>Parameters</tool_call> or <think>Your reasoning process</think><answer>Your answer</answer></answer>"
    "Question: {question}\n"
    "Options:\n{options}\n"
    "Please provide only the single option letter (e.g.,A, B, C, D, etc.) within the <answer></answer> tages.\n"
    "The video lasts for {duration} seconds.\n"
)
# Reflect-R1/reflect_r1/eval/select_showcase.py

def make_prompt(example):
    options_text = example.get('options', '')
    return TEMPLATE_V3.format(question=example['question'], options=options_text ,duration=example['duration'])