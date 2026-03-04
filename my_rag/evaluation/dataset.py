"""
评估数据集管理

面试考点：
- 评估数据集的格式：Question + Ground Truth + 可选 Context
- 为什么需要 Ground Truth？量化评估需要有参考标准
- 数据集来源：人工标注、从文档中半自动生成、LLM 辅助生成
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from my_rag.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvalSample:
    """单条评估样本"""
    question: str
    ground_truth: str = ""
    knowledge_base_id: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalDataset:
    """评估数据集"""
    name: str
    samples: list[EvalSample] = field(default_factory=list)

    def add(self, question: str, ground_truth: str = "", knowledge_base_id: str = ""):
        self.samples.append(EvalSample(
            question=question,
            ground_truth=ground_truth,
            knowledge_base_id=knowledge_base_id,
        ))

    def save(self, path: str | Path) -> None:
        """保存为 JSON 文件"""
        data = {
            "name": self.name,
            "samples": [
                {
                    "question": s.question,
                    "ground_truth": s.ground_truth,
                    "knowledge_base_id": s.knowledge_base_id,
                    "metadata": s.metadata,
                }
                for s in self.samples
            ],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("dataset_saved", path=str(path), samples=len(self.samples))

    @classmethod
    def load(cls, path: str | Path) -> "EvalDataset":
        """从 JSON 文件加载"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        dataset = cls(name=data.get("name", "unnamed"))
        for item in data.get("samples", []):
            dataset.samples.append(EvalSample(
                question=item["question"],
                ground_truth=item.get("ground_truth", ""),
                knowledge_base_id=item.get("knowledge_base_id", ""),
                metadata=item.get("metadata", {}),
            ))
        logger.info("dataset_loaded", path=str(path), samples=len(dataset.samples))
        return dataset
