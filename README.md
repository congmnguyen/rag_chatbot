# RAG Chatbot Triết học Mác-Lênin

Chatbot RAG (Retrieval-Augmented Generation) tiếng Việt cho môn Triết học Mác-Lênin - một môn học bắt buộc trong giáo dục đại học Việt Nam. Chatbot trả lời câu hỏi dựa trên nội dung giáo trình chính thống.

## Tính năng

- 🤖 Trả lời câu hỏi về Triết học Mác-Lênin bằng tiếng Việt
- 📚 Sử dụng RAG để truy xuất thông tin từ giáo trình
- 🔍 Hiển thị nguồn trích dẫn cho mỗi câu trả lời
- ⚡ Hai chế độ LLM: tiêu chuẩn (`gpt-4o-mini`) và nâng cao (`gpt-4o`)
- 💾 Vector database bền vững với ChromaDB

## Công nghệ sử dụng

- **Ngôn ngữ**: Python 3.10+
- **Quản lý package**: uv
- **LLM Framework**: LlamaIndex
- **Vector Store**: ChromaDB (persistent)
- **Embeddings**: OpenAI `text-embedding-3-small`
- **LLM Models**:
  - Mặc định: `gpt-4o-mini`
  - Nâng cao: `gpt-4o` 

## Cài đặt

### Yêu cầu

- Python 3.10 trở lên
- [uv](https://github.com/astral-sh/uv) package manager
- OpenAI API key

### Các bước cài đặt

1. Clone repository:
```bash
git clone https://github.com/congmnguyen/rag_chatbot.git
cd rag_chatbot
```

2. Cài đặt dependencies:
```bash
uv sync
```

3. Tạo file `.env` với nội dung:
```env
OPENAI_API_KEY=your_key_here
LLM_MODEL=gpt-4o-mini          # Tùy chọn, mặc định như trên
LLM_MODEL_ADVANCED=gpt-4o      # Tùy chọn, mặc định như trên
EMBEDDING_MODEL=text-embedding-3-small  # Tùy chọn
```

4. Chuẩn bị dữ liệu (đặt ở thư mục cha):
   - `chunks_with_context.json` - Các đoạn văn bản đã được xử lý trước với ngữ cảnh
   - `syllabus.json` - Cấu trúc đề cương môn học

5. Khởi tạo vector database (chỉ chạy một lần hoặc khi dữ liệu thay đổi):
```bash
uv run index_chunks.py
```

## Sử dụng

### Chạy chatbot

```bash
uv run chat.py
```

### Lệnh trong chat

- `/pro` - Chuyển sang model nâng cao (gpt-4o)
- `/normal` - Chuyển về model mặc định (gpt-4o-mini)
- `/source` - Bật/tắt hiển thị nguồn trích dẫn
- `/help` - Hiển thị trợ giúp
- `/quit` - Thoát chương trình

### Ví dụ sử dụng

```
Bạn: Chủ nghĩa duy vật biện chứng là gì?
Bot: [Trả lời dựa trên nội dung giáo trình với nguồn trích dẫn]

Bạn: /pro
Bot: Đã chuyển sang model gpt-4o

Bạn: Giải thích nguyên lý về mối liên hệ phổ biến
Bot: [Câu trả lời chi tiết hơn từ model nâng cao]
```

## Cấu trúc dự án

```
rag_chatbot/
├── chat.py           # Giao diện chat chính với các lệnh CLI
├── config.py         # Cấu hình, prompts và settings
├── index_chunks.py   # Script để index các đoạn văn bản vào ChromaDB
├── pyproject.toml    # Dependencies của dự án
├── chroma_db/        # Vector database được lưu trữ
├── .env              # API keys (không commit)
└── .env.example      # Mẫu file environment
```

## Cấu hình

Các cấu hình chính trong `config.py`:

- `SIMILARITY_TOP_K = 5` - Số lượng chunks được truy xuất
- `CHUNK_SIZE = 1024` - Kích thước mỗi chunk
- `CHUNK_OVERLAP = 200` - Độ chồng lấp giữa các chunks
- System prompt được tối ưu hóa cho việc học tập của sinh viên

## Kiến trúc

1. **Indexing** (`index_chunks.py`):
   - Tải chunks từ JSON → Tạo LlamaIndex Documents → Embedding với OpenAI → Lưu vào ChromaDB

2. **Querying** (`chat.py`):
   - Tải index từ ChromaDB → Tạo query engine với LLM → Truy xuất chunks liên quan → Tạo câu trả lời với nguồn trích dẫn

## Giấy phép

[MIT]

## Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.
