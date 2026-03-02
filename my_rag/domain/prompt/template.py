"""
Prompt 模板管理

面试考点：
- Prompt 工程的核心原则：角色设定、规则约束、上下文注入、输出格式控制
- RAG Prompt 的关键设计：
  1. 明确指示只基于提供的文档回答，不得编造
  2. 无法回答时要明确告知用户（减少幻觉）
  3. 要求引用来源（可追溯性）
- 对话历史的注入方式：放在 context 之前还是之后、长度控制
"""

RAG_QA_PROMPT = """你是一个专业的文档问答助手。请严格根据以下【参考文档】回答用户问题。

## 规则
1. 只根据参考文档中的内容回答，不要编造或推测文档中没有的信息
2. 如果参考文档中没有相关信息，请明确回答"根据已有文档，暂时无法回答该问题"
3. 回答时请尽量引用文档内容，保持准确性
4. 使用清晰的结构化格式回答（列表、标题等）

## 参考文档
{context}

## 用户问题
{question}

## 回答"""

RAG_QA_WITH_HISTORY_PROMPT = """你是一个专业的文档问答助手。请严格根据以下【参考文档】回答用户问题。

## 规则
1. 只根据参考文档中的内容回答，不要编造或推测文档中没有的信息
2. 如果参考文档中没有相关信息，请明确回答"根据已有文档，暂时无法回答该问题"
3. 回答时请尽量引用文档内容，保持准确性
4. 结合对话历史理解用户的指代和上下文

## 参考文档
{context}

## 对话历史
{chat_history}

## 用户问题
{question}

## 回答"""


def build_context(chunks: list[dict]) -> str:
    """将检索到的文档块组装为 Prompt 中的 context 部分"""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "未知来源")
        content = chunk.get("content", "")
        parts.append(f"[文档{i}] (来源: {source})\n{content}")
    return "\n\n".join(parts)


def build_prompt(
    question: str,
    context: str,
    chat_history: str = "",
) -> str:
    """构建最终的 RAG Prompt"""
    if chat_history:
        return RAG_QA_WITH_HISTORY_PROMPT.format(
            context=context, chat_history=chat_history, question=question
        )
    return RAG_QA_PROMPT.format(context=context, question=question)
