#!/usr/bin/env python3
"""
Script để index chunks vào Chroma vector store
Chạy 1 lần để tạo database
"""
import argparse
import json
import shutil
from pathlib import Path

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb

import config


def clean_text(text: str) -> str:
    """Remove surrogate characters that cause UTF-8 encoding errors"""
    return text.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='replace')


def load_chunks() -> list[Document]:
    """Load chunks từ JSON và chuyển thành LlamaIndex Documents"""
    print(f"📂 Loading chunks from: {config.CHUNKS_FILE}")

    with open(config.CHUNKS_FILE, 'r', encoding='utf-8', errors='surrogatepass') as f:
        chunks = json.load(f)

    documents = []
    for chunk in chunks:
        # Sử dụng content_with_context (có tóm tắt chương/mục)
        content = chunk.get('content_with_context', chunk.get('content', ''))
        content = clean_text(content)  # Clean surrogate characters

        # Metadata cho filtering và display
        metadata = {
            'id': chunk.get('id', ''),
            'path': clean_text(chunk.get('path', '')),
            'level': chunk.get('level', 0),
            'title': clean_text(chunk.get('title', '')),
        }
        # Clean metadata from chunk
        for k, v in chunk.get('metadata', {}).items():
            if isinstance(v, str):
                metadata[k] = clean_text(v)
            else:
                metadata[k] = v

        doc = Document(
            text=content,
            metadata=metadata,
            doc_id=chunk.get('id', '')
        )
        documents.append(doc)

    print(f"✅ Loaded {len(documents)} documents")
    return documents


def create_index(documents: list[Document]) -> VectorStoreIndex:
    """Tạo Chroma index từ documents"""

    # Xóa database cũ nếu có
    if config.CHROMA_DIR.exists():
        print(f"🗑️  Removing old database: {config.CHROMA_DIR}")
        shutil.rmtree(config.CHROMA_DIR)

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # Setup Chroma
    print(f"📦 Creating Chroma database at: {config.CHROMA_DIR}")
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    chroma_collection = chroma_client.get_or_create_collection("triet_hoc_mln")

    # Vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Embedding model
    print(f"🔤 Using embedding model: {config.EMBEDDING_MODEL}")
    embed_model = OpenAIEmbedding(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY
    )

    # Node parser with larger chunk size for metadata
    node_parser = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    # Create index
    print("🔄 Creating vector index (this may take a moment)...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[node_parser],
        show_progress=True
    )

    print("✅ Index created successfully!")
    return index


def test_query(index: VectorStoreIndex):
    """Test query để verify index hoạt động"""
    from llama_index.llms.openai import OpenAI

    print("\n" + "="*60)
    print("🧪 Testing query...")
    print("="*60)

    llm = OpenAI(
        model=config.LLM_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=0.1
    )

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=config.SIMILARITY_TOP_K
    )

    test_question = "Vật chất là gì theo định nghĩa của Lênin?"
    print(f"\n❓ Câu hỏi test: {test_question}\n")

    response = query_engine.query(test_question)
    print(f"💬 Trả lời:\n{response.response}")

    print("\n📚 Sources:")
    seen = set()
    for node in response.source_nodes:
        path = node.metadata.get('path', 'Unknown')
        if path not in seen:
            print(f"  - {path}")
            seen.add(path)


def main():
    parser = argparse.ArgumentParser(description="Index chunks vào Chroma vector store")
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Bỏ qua test query sau khi index (tránh gọi LLM)",
    )
    args = parser.parse_args()

    print("="*60)
    print("🚀 INDEXING CHUNKS VÀO CHROMA")
    print("="*60)

    # Check API key
    if not config.OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not set!")
        print("   Copy .env.example to .env and add your API key")
        return

    # Load and index
    documents = load_chunks()
    index = create_index(documents)

    # Test
    if not args.skip_test:
        test_query(index)

    print("\n" + "="*60)
    print("✅ DONE! Database ready at:", config.CHROMA_DIR)
    print("   Run 'uv run chat.py' to start chatting")
    print("="*60)


if __name__ == "__main__":
    main()
