"""
Configuration cho RAG Chatbot Triết học
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR
CHROMA_DIR = Path(__file__).parent / "chroma_db"

# Data files
CHUNKS_FILE = DATA_DIR / "chunks_with_context.json"
SYLLABUS_FILE = DATA_DIR / "syllabus.json"

# OpenAI Models
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM Models - Adjust based on your quota
# Mini models (2.5M tokens/day): gpt-4o-mini, gpt-4.1-mini, o1-mini, o3-mini
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Advanced models (250k tokens/day): gpt-4o, gpt-4.1, gpt-5, o1, o3
LLM_MODEL_ADVANCED = os.getenv("LLM_MODEL_ADVANCED", "gpt-4o")

# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# RAG Settings
SIMILARITY_TOP_K = 5  # Số chunks retrieve
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200

# System prompts
SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên về môn Triết học Mác-Lênin, được thiết kế để hỗ trợ sinh viên học tập.

Nguyên tắc trả lời:
1. Trả lời dựa trên nội dung giáo trình được cung cấp trong context
2. Sử dụng ngôn ngữ dễ hiểu, phù hợp với sinh viên
3. Khi trích dẫn, ghi rõ nguồn từ Chương/Mục nào
4. Nếu câu hỏi nằm ngoài phạm vi giáo trình, hãy nói rõ
5. Có thể đưa ví dụ thực tế để minh họa khái niệm triết học
6. Khuyến khích tư duy phản biện

Lưu ý: Đây là giáo trình chính thống của Bộ Giáo dục Việt Nam."""

QUERY_PROMPT_TEMPLATE = """Dựa trên các đoạn trích từ giáo trình Triết học Mác-Lênin dưới đây, hãy trả lời câu hỏi của sinh viên.

Context từ giáo trình:
{context_str}

Câu hỏi: {query_str}

Hướng dẫn trả lời:
- Trả lời chính xác dựa trên context
- Ghi rõ thông tin từ Chương/Mục nào nếu có thể
- Nếu context không đủ thông tin, hãy nói rõ
"""
