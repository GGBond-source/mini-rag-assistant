def build_prompt(question, contexts):
    """
    Day2 版 Prompt：
    更强调“只根据资料回答”，并约束输出格式。
    """
    if not contexts:
        context_text = "无可用资料。"
    else:
        context_text = "\n".join(
            [f"[{i + 1}] {text}" for i, text in enumerate(contexts)]
        )

    prompt = f"""你是一个严格基于资料回答问题的问答助手。

请遵守以下规则：
1. 只能根据“已知资料”回答，不允许使用资料外的常识补充。
2. 如果资料不足以回答问题，请输出：根据给定资料无法确定。
3. 回答尽量简洁、准确。
4. 优先整合多条资料，避免重复原句。
5. 给出回答时，尽量保留关键信息词。

已知资料：
{context_text}

问题：
{question}

请按以下格式输出：
answer: <你的回答>
evidence:
- <支撑回答的资料1>
- <支撑回答的资料2>
"""
    return prompt