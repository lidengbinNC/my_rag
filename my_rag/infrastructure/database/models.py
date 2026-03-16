"""
SQLAlchemy ORM 模型定义（MySQL 版）

面试考点：
- SQLAlchemy 2.0 Mapped 声明式模型
- 关系映射（一对多）
- UUID 主键 vs 自增主键的取舍
- 索引设计

MySQL 适配要点：
- VARCHAR 主键需显式指定 length（MySQL 索引对 VARCHAR 有长度限制）
- server_default=func.now() 在 MySQL 中映射为 DEFAULT CURRENT_TIMESTAMP
- onupdate=func.now() 映射为 ON UPDATE CURRENT_TIMESTAMP
- Text 类型在 MySQL 中为 MEDIUMTEXT（最大 16MB），适合存储 chunk 内容
- __table_args__ 中可通过 mysql_engine/mysql_charset 指定存储引擎和字符集
- 字符集统一使用 utf8mb4（支持 emoji 和完整 Unicode），排序规则 utf8mb4_unicode_ci
"""

import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, Index, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# MySQL 表级公共参数：InnoDB 引擎 + utf8mb4 字符集
_MYSQL_TABLE_KWARGS = {
    "mysql_engine": "InnoDB",
    "mysql_charset": "utf8mb4",
    "mysql_collate": "utf8mb4_unicode_ci",
}

# UUID 主键长度（36 字符，如 "550e8400-e29b-41d4-a716-446655440000"）
_UUID_LEN = 36


class Base(DeclarativeBase):
    pass


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"
    __table_args__ = (
        _MYSQL_TABLE_KWARGS,
    )

    id: Mapped[str] = mapped_column(
        String(_UUID_LEN), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(String(1000), default="")
    document_count: Mapped[int] = mapped_column(default=0)
    chunk_count: Mapped[int] = mapped_column(default=0)
    # server_default + onupdate 让 MySQL 在 DB 层面维护时间戳，无需应用层手动赋值
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )

    documents: Mapped[list["Document"]] = relationship(
        back_populates="knowledge_base", cascade="all, delete-orphan"
    )
    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="knowledge_base", cascade="all, delete-orphan"
    )


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        Index("idx_doc_kb_id", "knowledge_base_id"),
        _MYSQL_TABLE_KWARGS,
    )

    id: Mapped[str] = mapped_column(
        String(_UUID_LEN), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size: Mapped[int] = mapped_column(default=0)
    content_type: Mapped[str | None] = mapped_column(String(128), default=None)
    # status: pending → processing → completed / failed
    status: Mapped[str] = mapped_column(String(32), default="pending")
    chunk_count: Mapped[int] = mapped_column(default=0)
    knowledge_base_id: Mapped[str] = mapped_column(
        String(_UUID_LEN), ForeignKey("knowledge_bases.id")
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    knowledge_base: Mapped["KnowledgeBase"] = relationship(back_populates="documents")
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="Chunk.chunk_index"
    )


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (
        Index("idx_conv_kb_id", "knowledge_base_id"),
        _MYSQL_TABLE_KWARGS,
    )

    id: Mapped[str] = mapped_column(
        String(_UUID_LEN), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    title: Mapped[str | None] = mapped_column(String(512), default=None)
    knowledge_base_id: Mapped[str] = mapped_column(
        String(_UUID_LEN), ForeignKey("knowledge_bases.id")
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )

    knowledge_base: Mapped["KnowledgeBase"] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at"
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index("idx_chunk_doc_id", "document_id"),
        Index("idx_chunk_kb_id", "knowledge_base_id"),
        _MYSQL_TABLE_KWARGS,
    )

    id: Mapped[str] = mapped_column(
        String(_UUID_LEN), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    # MySQL MEDIUMTEXT 最大 16MB，足够存储任意 chunk 内容
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(default=0)
    token_count: Mapped[int] = mapped_column(default=0)
    metadata_json: Mapped[str | None] = mapped_column(Text, default=None)
    document_id: Mapped[str] = mapped_column(
        String(_UUID_LEN), ForeignKey("documents.id")
    )
    knowledge_base_id: Mapped[str] = mapped_column(
        String(_UUID_LEN), ForeignKey("knowledge_bases.id")
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    document: Mapped["Document"] = relationship(back_populates="chunks")


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        Index("idx_msg_conv_id", "conversation_id"),
        _MYSQL_TABLE_KWARGS,
    )

    id: Mapped[str] = mapped_column(
        String(_UUID_LEN), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources_json: Mapped[str | None] = mapped_column(Text, default=None)
    conversation_id: Mapped[str] = mapped_column(
        String(_UUID_LEN), ForeignKey("conversations.id")
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
