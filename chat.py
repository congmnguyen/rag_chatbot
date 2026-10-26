#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat interface cho RAG Chatbot Triết học Mác-Lênin
"""
import os
import sys
import io
import chromadb

# Force UTF-8 encoding for stdin/stdout/stderr (WSL fix)
if sys.platform.startswith('linux'):
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

import config

# ANSI color codes
GREEN = "\033[92m"
WHITE = "\033[97m"
RESET = "\033[0m"


def load_index() -> VectorStoreIndex:
    """Load existing Chroma index"""
    if not config.CHROMA_DIR.exists():
        print("❌ Database not found! Run 'uv run index_chunks.py' first.")
        sys.exit(1)

    print(f"📂 Loading database from: {config.CHROMA_DIR}")

    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        chroma_collection = chroma_client.get_collection("triet_hoc_mln")
    except Exception:
        print("❌ Collection 'triet_hoc_mln' not found! Run 'uv run index_chunks.py' first.")
        sys.exit(1)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    embed_model = OpenAIEmbedding(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )

    return index


def create_query_engine(index: VectorStoreIndex, use_advanced_model: bool = False):
    """Tạo query engine với LLM"""
    model = config.LLM_MODEL_ADVANCED if use_advanced_model else config.LLM_MODEL

    llm = OpenAI(
        model=model,
        api_key=config.OPENAI_API_KEY,
        temperature=0.2,
        system_prompt=config.SYSTEM_PROMPT
    )

    # Custom prompt
    qa_prompt = PromptTemplate(config.QUERY_PROMPT_TEMPLATE)

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=config.SIMILARITY_TOP_K,
        text_qa_template=qa_prompt
    )

    return query_engine, model


def print_sources(response):
    """In nguồn tham khảo"""
    print("\n📚 Nguồn tham khảo:")
    seen = set()
    for node in response.source_nodes:
        path = node.metadata.get('path', 'Unknown')
        if path not in seen:
            print(f"  • {path}")
            seen.add(path)


COMMANDS = [
    ("/pro", "Chuyển sang model nâng cao cho câu hỏi phức tạp"),
    ("/normal", "Chuyển về model thường"),
    ("/source", "Bật/tắt hiển thị nguồn"),
    ("/clear", "Xóa màn hình"),
    ("/help", "Hiện trợ giúp"),
    ("/quit", "Thoát"),
]


def print_commands():
    """In danh sách lệnh (dùng chung cho banner và /help)"""
    print("Lệnh:")
    for name, desc in COMMANDS:
        print(f"  {name:<8} - {desc}")


def chat_loop(index: VectorStoreIndex):
    """Main chat loop"""
    print("\n" + "="*60)
    print("🎓 CHATBOT TRIẾT HỌC MÁC-LÊNIN")
    print("="*60)
    print(f"Model mặc định: {config.LLM_MODEL}")
    print(f"Model nâng cao: {config.LLM_MODEL_ADVANCED}")
    print("-"*60)
    print_commands()
    print("="*60)

    use_advanced = False
    show_sources = True
    query_engine = None
    engine_is_advanced = None

    while True:
        try:
            user_input = input("\n❯ ").strip()

            if not user_input:
                continue

            cmd = user_input.lower()

            # Commands
            if cmd == '/quit':
                print("👋 Tạm biệt!")
                break

            if cmd == '/pro':
                use_advanced = True
                print(f"✅ Đã chuyển sang model: {config.LLM_MODEL_ADVANCED}")
                continue

            if cmd == '/normal':
                use_advanced = False
                print(f"✅ Đã chuyển về model: {config.LLM_MODEL}")
                continue

            if cmd == '/source':
                show_sources = not show_sources
                status = "BẬT" if show_sources else "TẮT"
                print(f"✅ Hiển thị nguồn: {status}")
                continue

            if cmd == '/clear':
                os.system('clear' if os.name != 'nt' else 'cls')
                continue

            if cmd == '/help':
                print("""
📖 Hướng dẫn sử dụng:
- Đặt câu hỏi về Triết học Mác-Lênin
- Chatbot sẽ trả lời dựa trên giáo trình chính thống

Ví dụ câu hỏi:
- "Vật chất là gì?"
- "Giải thích quy luật lượng chất"
- "Mối quan hệ giữa tồn tại xã hội và ý thức xã hội"
- "So sánh chủ nghĩa duy vật và duy tâm"
                """)
                print_commands()
                continue

            # Query
            # Chỉ tạo lại query engine khi đổi model
            if query_engine is None or engine_is_advanced != use_advanced:
                query_engine, model = create_query_engine(index, use_advanced)
                engine_is_advanced = use_advanced
            print(f"\n{GREEN}● Đang suy nghĩ...{RESET}")

            response = query_engine.query(user_input)

            print(f"\n{WHITE}●{RESET} {response.response}")

            if show_sources:
                print_sources(response)

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"❌ Lỗi: {e}")


def main():
    # Check API key
    if not config.OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not set!")
        print("   Copy .env.example to .env and add your API key")
        return

    index = load_index()
    chat_loop(index)


if __name__ == "__main__":
    main()
