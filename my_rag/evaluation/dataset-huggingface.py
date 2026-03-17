"""
HotpotQA 数据集加载 & 评估数据准备 - 独立运行脚本

用途：
    本脚本可独立运行（无需启动 FastAPI 服务），完成以下工作：
    1. 从 HuggingFace 拉取 hotpot_qa 数据集（使用镜像加速）
    2. 解析 context/question/answer 字段
    3. 将 context 文章导出为 docx 文件（每篇文章一个 docx，方便调试 chunk）
    4. 生成标准 EvalDataset JSON（可直接用于 /evaluations/batch 接口）
    5. 打印数据集统计信息

运行方式：
    cd d:/work-tianrun/project/my/my_rag
    python -m my_rag.evaluation.dataset-huggingface

HotpotQA 数据集字段说明：
    - id: 唯一标识
    - question: 需要多跳推理的问题
    - answer: 标准答案（通常是短语或实体）
    - type: "bridge"（桥接推理）| "comparison"（比较推理）
    - level: "easy" | "medium" | "hard"
    - context: [{title, sentences}] - 10 篇文章（2 篇支撑 + 8 篇干扰）
    - supporting_facts: [{title, sent_id}] - 支撑句子的位置标注

为什么用 HotpotQA 做 RAG 评估？
    - 真实数据：来自维基百科，无 LLM 造数据偏差
    - 多跳推理：测试 RAG 跨文档检索和综合推理能力
    - 干扰项：10 篇文章中有 8 篇干扰，测试检索精度
    - 标准答案：有 ground_truth，可计算 context_recall / answer_correctness
"""

import logging
import os
import json
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # ── 配置 ──────────────────────────────────────────────────────────
    NUM_SAMPLES = 20          # 取多少条样本（建议先用 20 条调试）
    SPLIT = "train"
    CONFIG_NAME = "distractor"  # 含干扰项，更贴近真实 RAG 场景
    ONLY_SUPPORTING = False    # False：保留全部 10 篇（含干扰）；True：只保留支撑文章

    # 输出目录（相对于项目根目录）
    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = project_root / "data" / "hotpotqa_docs" / f"hotpotqa_{CONFIG_NAME}_{NUM_SAMPLES}"
    eval_json_dir = project_root / "data" / "eval_datasets"

    # ── 加载数据集 ─────────────────────────────────────────────────────
    logger.info(f"从 HuggingFace 拉取 hotpot_qa ({CONFIG_NAME}) {SPLIT}[:{NUM_SAMPLES}] ...")

    from datasets import load_dataset
    ds = load_dataset(
        "hotpot_qa",
        CONFIG_NAME,
        split=f"{SPLIT}[:{NUM_SAMPLES}]",
        trust_remote_code=True,
    )

    logger.info(f"成功加载 {len(ds)} 条样本")

    # ── 打印第一条样本结构 ─────────────────────────────────────────────
    first = ds[0]
    print("\n" + "=" * 60)
    print("【第一条样本结构预览】")
    print(f"  id:       {first['id']}")
    print(f"  question: {first['question']}")
    print(f"  answer:   {first['answer']}")
    print(f"  type:     {first['type']}")
    print(f"  level:    {first['level']}")
    print(f"  context 文章数: {len(first['context']['title'])}")
    print(f"  context 标题: {first['context']['title']}")
    print(f"  supporting_facts 标题: {list(dict.fromkeys(first['supporting_facts']['title']))}")
    print("=" * 60 + "\n")

    # ── 解析样本 ───────────────────────────────────────────────────────
    from my_rag.evaluation.hotpotqa_loader import HotpotQALoader, HotpotQALoaderConfig

    cfg = HotpotQALoaderConfig(
        split=SPLIT,
        num_samples=NUM_SAMPLES,
        config_name=CONFIG_NAME,
        only_supporting=ONLY_SUPPORTING,
        output_dir=str(output_dir),
    )
    loader = HotpotQALoader(cfg)
    samples, eval_dataset = loader.load()

    # ── 统计信息 ───────────────────────────────────────────────────────
    type_counts: dict[str, int] = {}
    level_counts: dict[str, int] = {}
    for s in samples:
        type_counts[s.type] = type_counts.get(s.type, 0) + 1
        level_counts[s.level] = level_counts.get(s.level, 0) + 1

    # 统计去重后的文章数
    all_titles: set[str] = set()
    for s in samples:
        all_titles.update(s.context_titles)

    print("【数据集统计】")
    print(f"  样本总数:       {len(samples)}")
    print(f"  唯一文章数:     {len(all_titles)}")
    print(f"  问题类型分布:   {type_counts}")
    print(f"  难度分布:       {level_counts}")
    print(f"  docx 输出目录:  {output_dir}")

    # ── 保存评估数据集 JSON ────────────────────────────────────────────
    eval_json_dir.mkdir(parents=True, exist_ok=True)
    eval_json_path = eval_json_dir / f"{eval_dataset.name}.json"
    eval_dataset.save(str(eval_json_path))
    print(f"  评估数据集 JSON: {eval_json_path}")

    # ── 打印 docx 文件列表 ─────────────────────────────────────────────
    if output_dir.exists():
        docx_files = sorted(output_dir.glob("*.docx"))
        print(f"\n【生成的 docx 文件（共 {len(docx_files)} 个）】")
        for f in docx_files[:10]:
            print(f"  {f.name}")
        if len(docx_files) > 10:
            print(f"  ... 共 {len(docx_files)} 个文件")

    # ── 打印前 5 条评估样本 ────────────────────────────────────────────
    print("\n【前 5 条评估样本预览】")
    for i, s in enumerate(eval_dataset.samples[:5]):
        print(f"\n  [{i+1}] Q: {s.question}")
        print(f"       A: {s.ground_truth}")
        meta = s.metadata
        print(f"       type={meta.get('type')}, level={meta.get('level')}")
        print(f"       supporting: {meta.get('supporting_titles')}")

    print("\n" + "=" * 60)
    print("下一步操作：")
    print("  1. 创建知识库：POST /knowledge-bases")
    print("  2. 批量上传 docx：POST /knowledge-bases/{kb_id}/documents/batch")
    print(f"     文件目录：{output_dir}")
    print("  3. 更新评估数据集的 knowledge_base_id")
    print("  4. 运行批量评估：POST /evaluations/batch")
    print("     或直接调用：POST /evaluations/import-hotpotqa（一键完成 1-3 步）")
    print("=" * 60)


if __name__ == "__main__":
    main()
