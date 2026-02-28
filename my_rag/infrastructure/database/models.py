"""
SQLAlchemy ORM 模型定义

面试考点：
- SQLAlchemy 2.0 Mapped 声明式模型
- 关系映射（一对多）
- UUID 主键 vs 自增主键的取舍
- 索引设计
"""

import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, Index, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(default="")
    document_count: Mapped[int] = mapped_column(default=0)
    chunk_count: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime] = mapped_column(default=func.now(), onupdate=func.now())

    documents: Mapped[list["Document"]] = relationship(
        back_populates="knowledge_base", cascade="all, delete-orphan"
    )
    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="knowledge_base", cascade="all, delete-orphan"
    )


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (Index("idx_doc_kb_id", "knowledge_base_id"),)

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    filename: Mapped[str] = mapped_column(nullable=False)
    file_path: Mapped[str] = mapped_column(nullable=False)
    file_size: Mapped[int] = mapped_column(default=0)
    content_type: Mapped[str | None] = mapped_column(default=None)
    status: Mapped[str] = mapped_column(default="pending")
    chunk_count: Mapped[int] = mapped_column(default=0)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id"))
    created_at: Mapped[datetime] = mapped_column(default=func.now())

    knowledge_base: Mapped["KnowledgeBase"] = relationship(back_populates="documents")
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="Chunk.chunk_index"
    )


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (Index("idx_conv_kb_id", "knowledge_base_id"),)

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    title: Mapped[str | None] = mapped_column(default=None)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id"))
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime] = mapped_column(default=func.now(), onupdate=func.now())

    knowledge_base: Mapped["KnowledgeBase"] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at"
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index("idx_chunk_doc_id", "document_id"),
        Index("idx_chunk_kb_id", "knowledge_base_id"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(default=0)
    token_count: Mapped[int] = mapped_column(default=0)
    metadata_json: Mapped[str | None] = mapped_column(Text, default=None)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id"))
    created_at: Mapped[datetime] = mapped_column(default=func.now())

    document: Mapped["Document"] = relationship(back_populates="chunks")


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (Index("idx_msg_conv_id", "conversation_id"),)

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    role: Mapped[str] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources_json: Mapped[str | None] = mapped_column(Text, default=None)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"))
    created_at: Mapped[datetime] = mapped_column(default=func.now())

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
