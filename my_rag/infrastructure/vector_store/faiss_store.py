"""
FAISS 向量存储实现

面试考点：
- FAISS 是 Meta 开源的向量相似度搜索库，纯 CPU 即可运行
- IndexFlatIP：暴力搜索 + 内积（归一化向量下等价于 cosine similarity）
- IndexIDMap：为每个向量绑定自定义 ID（默认 FAISS 用数组索引）
- 持久化：FAISS 原生支持 write_index / read_index 做磁盘持久化
- 局限：不支持原生元数据过滤，需自行维护 id→metadata 的映射表
"""

import asyncio
import pickle
from pathlib import Path

import numpy as np

from my_rag.infrastructure.vector_store.base import BaseVectorStore, VectorSearchResult
from my_rag.utils.logger import get_logger

logger = get_logger(__name__)


class FAISSVectorStore(BaseVectorStore):

    def __init__(self, dimension: int, persist_dir: str | None = None):
        import faiss

        self._dimension = dimension
        self._persist_dir = Path(persist_dir) if persist_dir else None

        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
        self._id_to_text: dict[str, str] = {}
        self._id_to_metadata: dict[str, dict] = {}
        self._str_to_int: dict[str, int] = {}
        self._int_to_str: dict[int, str] = {}
        self._next_int_id = 0

        if self._persist_dir:
            self._try_load()

    def _str_id_to_int(self, str_id: str) -> int:
        if str_id not in self._str_to_int:
            self._str_to_int[str_id] = self._next_int_id
            self._int_to_str[self._next_int_id] = str_id
            self._next_int_id += 1
        return self._str_to_int[str_id]

    async def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        vectors = np.array(embeddings, dtype=np.float32)
        int_ids = np.array([self._str_id_to_int(sid) for sid in ids], dtype=np.int64)

        await asyncio.to_thread(self._index.add_with_ids, vectors, int_ids)

        for i, sid in enumerate(ids):
            self._id_to_text[sid] = texts[i]
            self._id_to_metadata[sid] = metadatas[i] if metadatas else {}

        if self._persist_dir:
            await asyncio.to_thread(self._save)

        logger.info("faiss_vectors_added", count=len(ids), total=self._index.ntotal)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[VectorSearchResult]:
        if self._index.ntotal == 0:
            return []

        query = np.array([query_embedding], dtype=np.float32)
        fetch_k = min(top_k * 3, self._index.ntotal) if filter_metadata else min(top_k, self._index.ntotal)

        scores, int_ids = await asyncio.to_thread(self._index.search, query, fetch_k)

        results: list[VectorSearchResult] = []
        for score, int_id in zip(scores[0], int_ids[0]):
            if int_id == -1:
                continue
            str_id = self._int_to_str.get(int(int_id), "")
            if not str_id:
                continue

            meta = self._id_to_metadata.get(str_id, {})

            if filter_metadata:
                if not all(meta.get(k) == v for k, v in filter_metadata.items()):
                    continue

            results.append(VectorSearchResult(
                chunk_id=str_id,
                score=float(score),
                content=self._id_to_text.get(str_id, ""),
                metadata=meta,
            ))

            if len(results) >= top_k:
                break

        return results

    async def delete(self, ids: list[str]) -> None:
        int_ids = np.array(
            [self._str_to_int[sid] for sid in ids if sid in self._str_to_int],
            dtype=np.int64,
        )
        if len(int_ids) > 0:
            self._index.remove_ids(int_ids)
        for sid in ids:
            self._id_to_text.pop(sid, None)
            self._id_to_metadata.pop(sid, None)
            int_id = self._str_to_int.pop(sid, None)
            if int_id is not None:
                self._int_to_str.pop(int_id, None)

        if self._persist_dir:
            await asyncio.to_thread(self._save)

    async def delete_by_metadata(self, key: str, value: str) -> None:
        to_delete = [sid for sid, m in self._id_to_metadata.items() if m.get(key) == value]
        if to_delete:
            await self.delete(to_delete)

    def count(self) -> int:
        return self._index.ntotal

    def _save(self) -> None:
        import faiss
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._persist_dir / "index.faiss"))
        with open(self._persist_dir / "metadata.pkl", "wb") as f:
            pickle.dump({
                "id_to_text": self._id_to_text,
                "id_to_metadata": self._id_to_metadata,
                "str_to_int": self._str_to_int,
                "int_to_str": self._int_to_str,
                "next_int_id": self._next_int_id,
            }, f)

    def _try_load(self) -> None:
        import faiss
        index_path = self._persist_dir / "index.faiss"
        meta_path = self._persist_dir / "metadata.pkl"
        if index_path.exists() and meta_path.exists():
            self._index = faiss.read_index(str(index_path))
            with open(meta_path, "rb") as f:
                data = pickle.load(f)
            self._id_to_text = data["id_to_text"]
            self._id_to_metadata = data["id_to_metadata"]
            self._str_to_int = data["str_to_int"]
            self._int_to_str = data["int_to_str"]
            self._next_int_id = data["next_int_id"]
            logger.info("faiss_index_loaded", vectors=self._index.ntotal)
