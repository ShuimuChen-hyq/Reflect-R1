import json
import re
from datetime import datetime
import os
import math
# from time_r1.reward.llm_judge import llm_judge_score
import ast
import numpy as np
MAX_TOOL_USE_NUM=10


# def general_answer_score(answer, ground_truth, question, type):
#     """
#     General answer score function.
#     """
#     if type == 'multiple_choice':
#         s = 1.0 if extract_characters_regex(answer) == extract_characters_regex(ground_truth) else 0.0

#     return s

def slice_s3_phase(messages):
    """
    专门用于 S3 阶段：逆向寻找最后一个 'user' 角色，
    提取它之后的所有消息（即 S3 阶段模型生成的 Assistant 和 Tool 交互），
    彻底阻断去 S1/S2 历史记录里“偷答案”的可能。
    """
    if not messages:
        return []

    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx != -1:
        # 只返回最后一个 user 之后产生的消息！
        return messages[last_user_idx + 1:]

    return messages

def accuracy_reward_S2(S1, S2, tool_use, **kwargs):
    # completions: 模型生成的完整响应列表

    # messages: 对话消息列表

    # target: 目标/参考答案列表（每个元素是包含"answer"键的字典）

    # question: 问题列表

    # type: 问题类型列表

    # **kwargs: 额外参数
    """
    Calculate the llm judge reward.
    """
    if tool_use:
        return [0.0] * len(S1)
    ## IDK模式
    idk_pattern = r"i don'?t know|unsure|not sure|insufficient|no information|cannot determine|can'?t determine|unclear"
    messages = S2
    reward_list = []
    for msg, sol, q, options in zip(messages, kwargs["answer"], kwargs["question"], kwargs["options"]):
        question = q
        # 解析选项字符串
        options_str = options
        candidates = parse_options_string(options_str)
        # 获取正确答案（字母形式，如 "A", "B", "C"）
        correct_answer = sol
        # 提取模型预测的文本
        pred_text = extract_prediction_from_message(msg)
        if pred_text is None or pred_text == "":
            reward_list.append(0.0) ## 不能同时出现答案和工具调用，否则直接判0分
            continue # 跳过后续处理

        ### IDK 检查
        text_clean = re.sub(r'[.!?]+$', '', pred_text.strip().lower())
        if re.search(idk_pattern, text_clean):
            reward_list.append(0.0)
            continue
        # 解析预测结果
        all_choices = []
        index2ans = {}
        for i, option in enumerate(candidates):
            index2ans[chr(ord("A") + i)] = option
            all_choices.append(chr(ord("A") + i))
        parsed_pred = parse_multi_choice_response(pred_text, all_choices, index2ans) ## 从文本中提取出 "A", "B" 等字母
        if parsed_pred is None:
            # 解析失败，给 0 分
            reward_list.append(-1.0)
        else:
            # ---------------------------------------------------------
            # 2. 准确率检查 (Accuracy)
            # ---------------------------------------------------------
            is_correct = (parsed_pred.strip().upper() == sol.strip().upper())
            acc_reward = 1.0 if is_correct else -1.0

            reward_list.append(acc_reward)
    return reward_list

def parse_options_string(options_str):
    """
    将选项字符串解析为candidates列表
    例如: "A) First, a cartoon...\nB) First, an image..." -> ["First, a cartoon...", "First, an image..."]
    """
    candidates = []
    lines = options_str.strip().split('\n')

    current_option = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 检查是否是新的选项开始（以A)、B)等开头）
        if len(line) >= 2 and line[0].isalpha() and line[1] == ')':
            # 如果已经有收集的选项内容，添加到candidates
            if current_option:
                candidates.append(current_option.strip())
            # 开始新的选项，去掉"A) "前缀
            current_option = line[3:] if len(line) > 3 else ""
        else:
            # 继续当前选项的内容
            if current_option:
                current_option += " " + line

    # 添加最后一个选项
    if current_option:
        candidates.append(current_option.strip())

    return candidates


# def general_answer_score(answer, ground_truth, question, type):
#     """
#     General answer score function.
#     """
#     if type == 'open_ended':
#         s = llm_judge_score(answer, ground_truth, question)
#     elif type == 'grounding':
#         if answer.startswith("```json"):
#             answer = answer.replace("```json", "").replace("```", "").strip()
#         if is_valid_json_time_format(answer):
#             pred_time = json.loads(answer)
#             gt_time = json.loads(ground_truth)
#             pred_start_time = pred_time.get("start_time")
#             pred_end_time = pred_time.get("end_time")
#             gt_start_time = gt_time.get("start_time")
#             gt_end_time = gt_time.get("end_time")
#             s = compute_iou([[pred_start_time, pred_end_time]], [[gt_start_time, gt_end_time]])
#         else:
#             s = 0.0
#     elif type == 'sequence':
#         s = 1.0 if extract_sequence_index(answer) == extract_sequence_index(ground_truth) else 0.0
#     elif type == 'multiple_choice':
#         s = 1.0 if extract_characters_regex(answer) == extract_characters_regex(ground_truth) else 0.0
#     else:
#         print(f"Error type in general_answer_score: {type}, question: {question}, answer: {answer}, ground_truth: {ground_truth}")
#         s = llm_judge_score(answer, ground_truth, question)
#     return s



def extract_prediction_from_message(messages):
    """
    最高优先级（格式错误判定）：如果某条消息同时包含答案和工具调用，视为无效，返回空。

    次优先级（标准格式）：优先寻找被 <answer> 标签包裹的内容。如果有多个，取对话中最后一条消息里的标签内容。

    低优先级（非标准/纯文本）：如果全程没有标签，则直接把模型说的**最后一句话（整段文本）**当作答案返回。
    """
    answers = []
    patterns = [
        r'<answer>(.*?)</answer>',  # 原始模式
    ]

    for message in messages:
        if message['role'] == 'assistant':
            for content in message['content']:
                if content['type'] == 'text':
                    text = content['text']
                    if has_tag(text, "answer") and has_tag(text, "tool_call"):
                        return ""
                    # 尝试所有正则表达式模式
                    for pattern in patterns:
                        all_answers = re.findall(pattern, text, re.DOTALL)
                        if all_answers:
                            answer = all_answers[-1].strip()
                            answers.append(answer)
                            break

    if len(answers) > 0:
        result = answers[-1]
    else:
        result = ""
        # for message in messages:
        #     if message['role'] == 'assistant':
        #         for content in message['content']:
        #             if content['type'] == 'text':
        #                 text = content['text']
        #                 for pattern in patterns:
        #                     match = re.search(pattern, text)
        #                     if match:
        #                         result = match.group(1).strip()
        #                         break
        #                 if not result:
        #                     result = text.strip()

    return result




def is_valid_json_time_format(s):
    """检查JSON格式是否正确"""
    try:
        item = json.loads(s)
        start_time = item.get("start_time")
        end_time = item.get("end_time")
        if start_time is None or end_time is None:
            return False
        if start_time < 0 or end_time < 0:
            return False
        if start_time > end_time:
            return False
        if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
            return False
        return True
    except Exception as e:
        print(f"Error in is_valid_json_time_format: {e}, s: {s}")
        return False


def merge_intervals(intervals):
    """合并重叠或相邻的时间区间"""
    if not intervals:
        return []
    intervals = [list(i) for i in intervals] # tuple to list
    # 按起始时间排序
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0][:]]  # 复制第一个区间
    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            # 合并区间
            merged[-1][1] = max(last[1], current[1])
        else:
            merged.append(current[:])
    return merged


def compute_iou(list_a, list_b):
    # # 示例用法
    # list_a = [[0, 3], [2, 4], [22, 25]]
    # list_b = [[1, 5], [2, 2], [2, 4]]
    # iou = compute_iou(list_a, list_b)
    # 合并两个列表的区间
    merged_a = merge_intervals(list_a)
    merged_b = merge_intervals(list_b)

    # 计算各自的总长度
    len_a = sum(end - start for start, end in merged_a)
    len_b = sum(end - start for start, end in merged_b)

    # 计算交集的总长度
    intersection = 0
    i = j = 0
    while i < len(merged_a) and j < len(merged_b):
        a_start, a_end = merged_a[i]
        b_start, b_end = merged_b[j]

        # 计算当前两个区间的重叠部分
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        if start < end:
            intersection += end - start

        # 移动指针
        if a_end < b_end:
            i += 1
        else:
            j += 1
    # 计算并集总长度
    union = len_a + len_b - intersection
    if union == 0:
        return 1.0
    return intersection / union


def extract_sequence_index(answer):
    # input: The sequence of the topics introduced in this video is (a) Men are setting up a tent in the dark, (c) Women do their beauty routine in the bathroom, (b) A baby is eating from a large platter of french fries on a black tray.
    # 输出: (a)(c)(b)
    pattern = r'(\([a-g,1-6]\))'
    matches = re.findall(pattern, answer)
    return ''.join(matches)


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice})" in response:  # 匹配 "C)" 格式
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:
            if f"({choice}" in response:  # 匹配 "(C" 格式
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    # if len(candidates) == 0 and len(response.split()) > 5:
    #     for index, ans in index2ans.items():
    #         if ans.lower() in response.lower():
    #             candidates.append(index)
    #             index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = None
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def has_tag(text: str, tag: str) -> bool:
    return re.search(fr"<{tag}>", text)


def answer_format_check(text):
    pattern = re.compile(r'<think>.*?</think>\s*<answer>.*?</answer>', re.DOTALL)
    match = re.fullmatch(pattern, text.strip())
    return 1.0 if match else 0.0


def tool_call_format_check(text):
    pattern = re.compile(r'<think>.*?</think>\s*<tool_call>.*?</tool_call>', re.DOTALL)
    match = re.fullmatch(pattern, text.strip())
    return 1.0 if match else 0.0


def multiturn_format_check(messages, **kwargs):
    """
    检查多轮对话中每条 assistant 消息的格式是否严格符合要求：
    0. 必须有answer，且answer/tool_call都符合格式
    1. 如果包含 answer，必须符合 answer_format_check
    2. 如果包含 tool_call，必须符合 tool_call_format_check
    3. answer 和 tool_call 不能同时出现在同一条消息中
    """
    answer_format_stats = []
    tool_call_format_stats = []

    for message in messages:
        if message["role"] == "assistant":
            for content in message["content"]:
                if isinstance(content, dict) and content["type"] == "text":
                    text = content["text"]
                    # 检查 answer 和 tool_call 不能同时出现
                    if has_tag(text, "answer") and has_tag(text, "tool_call"):
                        return 0.0
                    if has_tag(text, "answer"):
                        answer_format_stats.append(answer_format_check(text))
                    elif has_tag(text, "tool_call"):
                        tool_call_format_stats.append(tool_call_format_check(text))
    if len(answer_format_stats) > 0 and all(answer_format_stats) and all(tool_call_format_stats):
        return 1.0
    else:
        return 0.0


def multiturn_format_reward(prompts, completions, completions_visual, final_msg, S1=False, S2=False, S3=False, **kwargs):
    """
    计算三阶段联合格式奖励 (Hard Constraint)。
    逻辑：S1 AND S2 AND S3 必须全部格式正确，否则为 0 分。
    """
    rewards = []

    # 使用 zip 将三个阶段的数据对齐，按样本(sample)进行遍历
    # s1, s2, s3 分别对应同一个样本的三个阶段输出
    for s1, s2, s3 in zip(completions, completions_visual, final_msg):

        # 分别检查三个阶段的格式
        check_s1 = multiturn_format_check(s1)
        check_s2 = multiturn_format_check(s2)
        s3_slice = slice_s3_phase(s3)
        check_s3 = multiturn_format_check(s3_slice)

        if S1 and check_s1 == 1.0:
            rewards.append(1.0)
        elif S2 and check_s2 == 1.0:
            rewards.append(1.0)
        elif S3 and check_s3 == 1.0:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards




# def advanced_tool_success_check(messages):
#     """
#     综合评估工具调用成功情况，包括：
#     1. 基础工具调用成功检查
#     2. 工具多样性和数量评估
#     3. 重复调用惩罚
#     4. 调用失败惩罚
#     NOTE:  VideoInteraction.avoid_mm_missing=True时，这项永远为1；当使用counterfactual reasoning时，这项不再重要

#     """
#     if not messages:
#         return 0.0

#     # 基础工具调用成功检查
#     successful_tools = 0
#     total_tool_calls = 0
#     response_signitures_count = dict()
#     tool_failure_count = 0

#     for message in messages:
#         if message.get("role") == "tool" and message.get("name") == "parse_error":
#             tool_failure_count += 1
#         if message.get("role") == "tool" and message.get("name") != "parse_error":
#             total_tool_calls += 1
#             content = message.get("content", [])
#             if not isinstance(content, list):
#                 continue
#             for item in content:
#                 if isinstance(item, dict) and item.get("type") in ["video", "image"]:
#                     successful_tools += 1
#                     break
#                 elif not isinstance(item, dict):
#                     print(f"Error in tool_success_check: {item}, content: {content}")
#     tool_score = 1.0 / (1.0 + math.exp(-(successful_tools - 2)))
#     if successful_tools == 0:
#         tool_score = 0.0
#     return tool_score

def advanced_tool_success_check(messages):
    """
    综合评估工具调用成功情况。

    【核心修复】：必须剔除 query="any" 的系统兜底调用，防止模型“偷懒”骗分。
    """
    if not messages:
        return 0.0

    successful_tools = 0
    # tool_failure_count = 0 # 暂时没用到，可以根据需要决定是否启用

    for message in messages:
        # 1. 检查角色是否为 tool
        if message.get("role") != "tool":
            continue

        # 2. 检查是否为解析错误 (parse_error)
        if message.get("name") == "parse_error":
            # tool_failure_count += 1
            continue

        # 3. 【关键修改】检查是否为系统自动填充的“兜底”调用
        # 系统兜底调用的特征：name="seek_video_frames" 且 query="any" (且 start_time=0)
        args = message.get("arguments", {})
        # 注意：args 可能是字典，也可能是字符串（取决于之前的处理），这里假设是字典
        if isinstance(args, dict):
            query = args.get("query", "")
            # 如果是系统强制填充的 'any'，直接跳过，不算作成功调用
            if query == "any":
                print("DEBUG: Detected placeholder tool call, ignoring reward.")
                continue

        # 4. 检查内容是否包含有效的视频/图片
        content = message.get("content", [])
        if not isinstance(content, list):
            continue

        has_valid_media = False
        for item in content:
            # 只有当返回了真正的 image 或 video 数据时才算成功
            if isinstance(item, dict) and item.get("type") in ["video", "image"]:
                has_valid_media = True
                break
            # 这里的 else print error 可以保留用于调试，但在训练中最好注释掉以免刷屏
            # elif not isinstance(item, dict):
            #     print(f"Error in tool_success_check: {item}, content: {content}")

        if has_valid_media:
            successful_tools += 1

    # 计算分数
    # 逻辑：只要有至少 1 次“非兜底”的有效调用，得分就 > 0
    tool_score = 1.0 / (1.0 + math.exp(-(successful_tools - 2)))

    if successful_tools == 0:
        tool_score = 0.0

    return tool_score



def reflect_reward(prompts, completions, completions_visual, final_msg, S1=False, S2= False, S3 = False, **kwargs):
    rewards = []

    iterator = zip(
        completions, completions_visual, final_msg,
        kwargs["answer"], kwargs["question"], kwargs["options"]
    )
    if S2:
        rewards = accuracy_reward_S2(S1=completions, S2=completions_visual, tool_use= False, **kwargs)
        return rewards
    for s1, s2, s3, sol, q, options in iterator:
        # ... (解析逻辑保持不变) ...
        # 解析选项、提取文本、转换为 A/B/C/D ...

        candidates = parse_options_string(options)
        all_choices = [chr(ord("A") + i) for i in range(len(candidates))]
        index2ans = {idx: opt for idx, opt in zip(all_choices, candidates)}

        s1_text = extract_prediction_from_message(s1)
        s2_text = extract_prediction_from_message(s2)
        s3_slice = slice_s3_phase(s3)
        s3_text = extract_prediction_from_message(s3_slice)
        # # 【严防死守】任何阶段格式崩坏(None) 或 没给答案("")，直接 0 分
        # # 这倒逼模型在 S1/S2 必须输出有效答案，否则 S3 再努力也没用
        # if not s1_text or not s2_text or not s3_text:
        #     rewards.append(0.0)
        #     continue

        pred_s1 = parse_multi_choice_response(s1_text, all_choices, index2ans)
        pred_s2 = parse_multi_choice_response(s2_text, all_choices, index2ans)
        pred_s3 = parse_multi_choice_response(s3_text, all_choices, index2ans)
        correct_answer = sol

        # # 检查 S3 内部是否包含 Extraction 格式 (作为保底分)
        # s3_raw_think = ""
        # for msg in s3:
        #     if msg['role'] == 'assistant':
        #         match = re.search(r'<think>(.*?)</think>', msg['content'][0]['text'], re.DOTALL)
        #         if match: s3_raw_think = match.group(1)
        #         break

        # ==================== 核心修正逻辑 ====================
        current_reward = 0.0

        # # 1. 格式分 (小额激励，防止格式崩塌)  可以之后试试看
        # has_s1 = re.search(r"s1\s*=", s3_raw_think, re.IGNORECASE)
        # has_s2 = re.search(r"s2\s*=", s3_raw_think, re.IGNORECASE)
        # if has_s1 and has_s2:
        #     current_reward += 0.1

        # 2. 正误逻辑
        # s1_right = (pred_s1.strip().upper() == correct_answer.strip().upper())
        # s2_right = (pred_s2.strip().upper() == correct_answer.strip().upper())
        # s3_right = (pred_s3.strip().upper() == correct_answer.strip().upper())
        s1_right = False
        if pred_s1 is not None and correct_answer is not None:
            s1_right = (pred_s1.strip().upper() == correct_answer.strip().upper())

        s2_right = False
        if pred_s2 is not None and correct_answer is not None:
            s2_right = (pred_s2.strip().upper() == correct_answer.strip().upper())

        s3_right = False
        if pred_s3 is not None and correct_answer is not None:
            s3_right = (pred_s3.strip().upper() == correct_answer.strip().upper())

        if S3:
            if s3_right:
                # 只要 S3 对了，给统一的高分。
                # 不区分 S1/S2 是否对，防止模型学会“故意献祭 S1 来骗取高分”。
                current_reward += 1.0

                # (可选) 如果你实在想鼓励“独立纠错”，可以给一个极小的 Bonus，
                # 但绝对不能超过 S1 本身做对带来的潜在收益。
                # if not s1_right and not s2_right:
                #     current_reward += 0.1
                # 但为了安全，建议不加。
            else:
                # S3 错了
                if s1_right or s2_right:
                    # 败家惩罚：明明有人做对了，你却改错了。
                    # 这是一个强烈的负信号。
                    current_reward -= 1.0
                else:
                    # 大家都错，S3 也没救回来。
                    current_reward += 0.0

            rewards.append(current_reward)
        elif S1:
            if s1_right:
                current_reward += 1.0
            else:
                current_reward += 0.0 ## 冻结S1的准确性
            rewards.append(current_reward)
    return rewards


import re

def reflect_format_reward(prompts, completions, completions_visual, final_msg, S1=False, S2=False, S3=False, **kwargs):
    if S2:
        return [0.0] * len(completions_visual)
    elif S1:
        return [0.0] * len(completions)
    elif S3:
        rewards = []
        # final_msg 是一个 batch，msg 是其中一个样本的对话历史 (List[Dict])
        for conversation in final_msg:
            found_format = False

            # 遍历该对话中的每一句话
            for turn in conversation:
                if turn['role'] == 'assistant':
                    # ==================== 1. 健壮的内容提取 ====================
                    content_str = ""
                    raw_content = turn.get('content', "")

                    if isinstance(raw_content, str):
                        content_str = raw_content
                    elif isinstance(raw_content, list):
                        # 处理 VLM 格式：提取 list 中 type='text' 的部分
                        for item in raw_content:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                content_str += item.get('text', "")

                    # ==================== 2. 提取 <think> ====================
                    match = re.search(r"<think>(.*?)</think>", content_str, re.DOTALL)
                    if not match:
                        continue # 这一轮没找到，继续找下一轮（比如 Tool 返回后的那一轮）

                    think_content = match.group(1)

                    # ==================== 3. 核心正则检查 ====================
                    has_s1 = re.search(r"s1\s*=", think_content, flags=re.IGNORECASE)
                    has_s2 = re.search(r"s2\s*=", think_content, flags=re.IGNORECASE)

                    if has_s1 and has_s2:
                        found_format = True
                        break # 找到了！不需要再看后面的轮次了

            # ==================== 4. 打分 ====================
            if found_format:
                rewards.append(1.0) # 建议格式分不要给 1.0 这么高，0.1~0.2 即可，否则模型会刷分
            else:
                rewards.append(0.0)

        return rewards





def accuracy_reward(S1, S2, tool_use, **kwargs):
    # completions: 模型生成的完整响应列表

    # messages: 对话消息列表

    # target: 目标/参考答案列表（每个元素是包含"answer"键的字典）

    # question: 问题列表

    # type: 问题类型列表

    # **kwargs: 额外参数
    """
    Calculate the llm judge reward.
    """
    if tool_use:
        messages = S1
    else:
        messages = S2
    reward_list = []
    for msg, sol, q, options in zip(messages, kwargs["answer"], kwargs["question"], kwargs["options"]):
        question = q
        # 解析选项字符串
        options_str = options
        candidates = parse_options_string(options_str)
        # 获取正确答案（字母形式，如 "A", "B", "C"）
        correct_answer = sol
        # 提取模型预测的文本
        pred_text = extract_prediction_from_message(msg)
        if pred_text is None:
            reward_list.append(0.0) ## 不能同时出现答案和工具调用，否则直接判0分
            continue # 跳过后续处理
        # 解析预测结果
        all_choices = []
        index2ans = {}
        for i, option in enumerate(candidates):
            index2ans[chr(ord("A") + i)] = option
            all_choices.append(chr(ord("A") + i))
        parsed_pred = parse_multi_choice_response(pred_text, all_choices, index2ans) ## 从文本中提取出 "A", "B" 等字母
        if parsed_pred is None:
            # 解析失败，给 0 分
            reward_list.append(0.0)
        else:
            if tool_use:
                # S1 模式：必须经过 advanced_tool_success_check 检查
                # 注意：假设 advanced_tool_success_check 返回值 > 0 表示成功调用
                tool_behavior_score = advanced_tool_success_check(msg)

                # 定义什么是“合规”：比如分数大于0，或者分数等于1.0
                # 如果你的 check 函数返回的是 sigmoid 分数，这里建议设个阈值 (例如 0.5)
                # 如果你的 check 函数返回的是 binary (0/1)，直接判定
                is_behavior_valid = tool_behavior_score > 0
            else:
                # S2 模式：Blind Mode，不强制工具调用，默认合规
                is_behavior_valid = True

            # ---------------------------------------------------------
            # 2. 准确率检查 (Accuracy)
            # ---------------------------------------------------------
            is_correct = (parsed_pred.strip().upper() == sol.strip().upper())
            acc_reward = 1.0 if is_correct else 0.0

            # ---------------------------------------------------------
            # 3. 最终奖励计算 (核心修改逻辑)
            # ---------------------------------------------------------
            if not is_behavior_valid:
                # 【核心逻辑】
                # 如果是 S1 且没调工具 (is_behavior_valid 为 False)
                # 无论答案对错，直接给 0 分 (或者建议给 -1.0 进行惩罚)
                # final_reward = 0.0
                final_reward = -1.0
            else:
                # 只有行为合规了，才有资格拿准确率的奖
                final_reward = acc_reward

            reward_list.append(final_reward)
    return reward_list

def tool_call_reward(prompts, completions, completions_visual, final_msg, **kwargs):
    """
    [Batch Reward] S1 gets rewarded if S2 (Blind) is correct.
    This creates the causal link: Good Search (S1) -> Good Blind Answer (S2).
    """

    return accuracy_reward(S1=completions, S2=completions_visual, tool_use=False, **kwargs)

def tool_call_num_reward(prompts, completions, completions_visual, final_msg, S1=False, S2=False, S3=False, **kwargs):  ## 鼓励S1多调用工具
    """
    [Batch Reward] S1 gets rewarded based on the number of valid tool calls.
    """
    if S2:
        return [0.0] * len(completions_visual)
    elif S3:
        return [0.0] * len(final_msg)
    reward_list = []
    S1 = completions
    for msg in S1:
        tool_score = advanced_tool_success_check(msg)
        score = tool_score > 0
        reward_list.append(score)
    return reward_list


def tool_call_num_reward_advance(prompts, completions, completions_visual, final_msg, S1=False, S2=False, S3=False, **kwargs):
    """
    工具调用奖励函数。
    支持 S1 和 S3。强制要求 S3 在最后阶段必须进行有效的工具验证。
    """
    reward_list = []

    # Case 1: S2 阶段（通常不涉及工具，或者是纯文本批判）
    if S2:
        return [0.0] * len(completions_visual)

    # Case 2: S3 阶段（最终反思与验证）
    elif S3:
        # 遍历每一个样本的历史记录
        for history in final_msg:
            recent_history = slice_s3_phase(history)

            tool_score = advanced_tool_success_check(recent_history)
            score = 1.0 if tool_score > 0 else 0.0
            # 如果 tool_score > 0，说明最后这几步里真的发生了有效调用
            reward_list.append(score)

        return reward_list

    # Case 3: S1 阶段（初始探索）
    # 注意：TRL 框架在 S1 阶段传进来的 completions 通常是 list of lists (messages)
    else:
        # S1 只需要检查它自己生成的那部分 completions 即可
        for msg in completions:
            tool_score = advanced_tool_success_check(msg)
            score = 1.0 if tool_score > 0 else 0.0
            reward_list.append(score)

        return reward_list


import re

def reasoning_length_score(length, min_len=120, max_len=700):
    """
    软边界长度打分 + 非对称惩罚。

    奖励曲线:
      太短 (0~min_len):           线性上升 0 → 1.0
      合格 (min_len~max_len):     满分 1.0
      偏长 (max_len~max_len*2):   线性下降 1.0 → 0
      过长 (>max_len*2):          固定惩罚 -1
    """
    if length <= 0:
        return 0.0
    elif length < min_len:
        return length / min_len
    elif length <= max_len:
        return 1.0
    elif length <= max_len * 2:
        return 1.0 - (length - max_len) / max_len
    else:
        return -1


def reasoning_length_reward_single(batch_messages):
    """
    长度奖励：支持 Batch 处理，且会检查对话中【每一条】Assistant 回复的思考长度。
    计算方式：该样本中所有 Assistant 回复的平均得分。
    使用软边界打分，区间内满分，超出边界平滑衰减，过长额外惩罚。
    """

    min_len = 120
    max_len = 700

    rewards = []

    # pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    # [FIX] 新增：匹配被 token 截断的未闭合 <think>，原正则要求闭合标签，
    # 截断时 </think> 丢失导致匹配失败，模型收不到过长惩罚信号
    pattern_closed = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    pattern_unclosed = re.compile(r'<think>(.*)', re.DOTALL)

    for conversation in batch_messages:
        turn_scores = []

        for msg in conversation:
            if msg.get("role") == "assistant":
                text = ""
                content = msg.get("content", "")

                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text += item.get("text", "")

                match = pattern_closed.search(text)
                if match:
                    think_content = match.group(1).strip()
                    length = len(think_content)
                    turn_scores.append(reasoning_length_score(length, min_len, max_len))
                else:
                    # [FIX] 检查未闭合的 <think>（被截断的情况）
                    # 截断意味着 think 过长导致 token 耗尽，直接给 -1 惩罚
                    unclosed = pattern_unclosed.search(text)
                    if unclosed:
                        print("[LENGTH_REWARD] truncated <think> detected, applying -1 penalty")
                        turn_scores.append(-1.0)
                    else:
                        turn_scores.append(0.0)

        if len(turn_scores) > 0:
            final_score = sum(turn_scores) / len(turn_scores)
        else:
            final_score = 0.0

        rewards.append(final_score)

    return rewards


# ==================== 旧版二值长度奖励（已废弃） ====================
# def reasoning_length_reward_single(batch_messages):
#     """
#     长度奖励：支持 Batch 处理，且会检查对话中【每一条】Assistant 回复的思考长度。
#     计算方式：该样本中所有 Assistant 回复的平均得分。
#     """
#
#     min_len = 120
#     max_len = 700
#
#     rewards = []
#
#     pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
#
#     for conversation in batch_messages:
#         turn_scores = []
#
#         for msg in conversation:
#             if msg.get("role") == "assistant":
#                 text = ""
#                 content = msg.get("content", "")
#
#                 if isinstance(content, str):
#                     text = content
#                 elif isinstance(content, list):
#                     for item in content:
#                         if isinstance(item, dict) and item.get("type") == "text":
#                             text += item.get("text", "")
#
#                 match = pattern.search(text)
#                 if match:
#                     think_content = match.group(1).strip()
#                     length = len(think_content)
#
#                     if min_len <= length <= max_len:
#                         turn_scores.append(1.0)
#                     else:
#                         turn_scores.append(0.0)
#                 else:
#                     turn_scores.append(0.0)
#
#         if len(turn_scores) > 0:
#             final_score = sum(turn_scores) / len(turn_scores)
#         else:
#             final_score = 0.0
#
#         rewards.append(final_score)
#
#     return rewards

# def reasoning_length_reward(prompts, completions, completions_visual, final_msg, S1=False, S2=False, S3=False, **kwargs):
#     """
#     主函数：分别计算三个阶段的得分，然后取平均值。
#     """

#     # 1. 分别计算三个阶段的得分 (得到的是三个分数列表)
#     # 注意：这里要把 S1, S2, S3 对应的变量传进去，不要传 prompts
#     score_s1_list = reasoning_length_reward_single(completions)        # S1
#     score_s2_list = reasoning_length_reward_single(completions_visual) # S2
#     score_s3_list = reasoning_length_reward_single(final_msg)          # S3

#     if S1:
#         rewards = score_s1_list
#     elif S2:
#         rewards = score_s2_list
#     elif S3:
#         rewards = score_s3_list
#     return rewards

def reasoning_length_reward(prompts, completions, completions_visual, final_msg, S1=False, S2=False, S3=False, **kwargs):
    """
    主函数：通过对 final_msg 进行切片预处理，确保 S3 阶段只评价增量部分的长度。
    """

    if S1:
        # S1 阶段 completions 通常只包含当前生成的回答，直接计算
        return reasoning_length_reward_single(completions)

    elif S2:
        # S2 阶段同理
        return reasoning_length_reward_single(completions_visual)

    elif S3:
        # === S3 阶段增量切片逻辑 ===
        s3_incremental_batch = []

        for conversation in final_msg:
            # 1. 逆向寻找最后一个 user 的索引位置
            last_user_idx = -1
            for i in range(len(conversation) - 1, -1, -1):
                if conversation[i].get("role") == "user":
                    last_user_idx = i
                    print("DEBUG:last_user_idx",last_user_idx)
                    break

            # 2. 截取从该 User 开始到对话结束的所有消息 (包含 S3 的 Assistant 和 Tool)
            # 如果没找到 User (理论上不会)，则回退到全量
            s3_increment = conversation[last_user_idx:] if last_user_idx != -1 else conversation
            s3_incremental_batch.append(s3_increment)

        # 将切片后的增量数据传给 single 函数进行评分
        # 这样 single 函数在遍历时，msg['role'] == 'assistant' 匹配到的就全是 S3 的回复了
        return reasoning_length_reward_single(s3_incremental_batch)

    return [0.0] * len(completions)



def ACC_FOR_WANDB(prompts, completions, completions_visual, final_msg, S1=False, S2= False, S3 = False, **kwargs):
    rewards = []

    iterator = zip(
        completions, completions_visual, final_msg,
        kwargs["answer"], kwargs["question"], kwargs["options"]
    )

    for s1, s2, s3, sol, q, options in iterator:
        # ... (解析逻辑保持不变) ...
        # 解析选项、提取文本、转换为 A/B/C/D ...

        candidates = parse_options_string(options)
        all_choices = [chr(ord("A") + i) for i in range(len(candidates))]
        index2ans = {idx: opt for idx, opt in zip(all_choices, candidates)}

        s1_text = extract_prediction_from_message(s1)
        s2_text = extract_prediction_from_message(s2)
        s3_slice = slice_s3_phase(s3)
        s3_text = extract_prediction_from_message(s3_slice)
        # # 【严防死守】任何阶段格式崩坏(None) 或 没给答案("")，直接 0 分
        # # 这倒逼模型在 S1/S2 必须输出有效答案，否则 S3 再努力也没用
        # if not s1_text or not s2_text or not s3_text:
        #     rewards.append(0.0)
        #     continue

        pred_s1 = parse_multi_choice_response(s1_text, all_choices, index2ans)
        pred_s2 = parse_multi_choice_response(s2_text, all_choices, index2ans)
        pred_s3 = parse_multi_choice_response(s3_text, all_choices, index2ans)
        correct_answer = sol

        # # 检查 S3 内部是否包含 Extraction 格式 (作为保底分)
        # s3_raw_think = ""
        # for msg in s3:
        #     if msg['role'] == 'assistant':
        #         match = re.search(r'<think>(.*?)</think>', msg['content'][0]['text'], re.DOTALL)
        #         if match: s3_raw_think = match.group(1)
        #         break

        # ==================== 核心修正逻辑 ====================
        current_reward = 0.0

        # # 1. 格式分 (小额激励，防止格式崩塌)  可以之后试试看
        # has_s1 = re.search(r"s1\s*=", s3_raw_think, re.IGNORECASE)
        # has_s2 = re.search(r"s2\s*=", s3_raw_think, re.IGNORECASE)
        # if has_s1 and has_s2:
        #     current_reward += 0.1

        # 2. 正误逻辑
        # s1_right = (pred_s1.strip().upper() == correct_answer.strip().upper())
        # s2_right = (pred_s2.strip().upper() == correct_answer.strip().upper())
        # s3_right = (pred_s3.strip().upper() == correct_answer.strip().upper())
        s1_right = False
        if pred_s1 is not None and correct_answer is not None:
            s1_right = (pred_s1.strip().upper() == correct_answer.strip().upper())

        s2_right = False
        if pred_s2 is not None and correct_answer is not None:
            s2_right = (pred_s2.strip().upper() == correct_answer.strip().upper())

        s3_right = False
        if pred_s3 is not None and correct_answer is not None:
            s3_right = (pred_s3.strip().upper() == correct_answer.strip().upper())

        if S3:  ## 纯记录
            if s3_right:
                current_reward += 1.0
            else:
                # 大家都错，S3 也没救回来。
                current_reward += 0.0

            rewards.append(current_reward)
        elif S2:
            if s2_right:
                current_reward += 1.0
            else:
                current_reward += 0.0
            rewards.append(current_reward)
        elif S1:
            if s1_right:
                current_reward += 1.0
            else:
                current_reward += 0.0 ## 冻结S1的准确性
            rewards.append(current_reward)
    return rewards



###
reward_functions = [ ## 对于S1和S2可能还需要加入长度奖励，参考video-r1
    multiturn_format_reward,
    reflect_reward,
    tool_call_num_reward_advance,
    reasoning_length_reward,
    reflect_format_reward, ## 仅仅监控一下而已
    ACC_FOR_WANDB,
]
## advanced_tool_success_check
##
## 奖励函数
reward_weights = [
    1.0,
    0.5,
    0.5,
    0.2,
    0.0,
    0.0,
]
