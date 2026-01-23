#!/usr/bin/env python3
"""
Improved chunking strategy for RAG Chatbot Triết học Mác-Lênin.

Features:
1. Hierarchical parsing: Chương > Phần (I/II/III) > Mục (1/2/3) > Tiểu mục (a/b/c)
2. Rich metadata with section hierarchy information
3. LLM-generated summaries for each section describing what it covers
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from openai import OpenAI

import config


# === REGEX PATTERNS ===
CHAPTER_RE = re.compile(r"^(Chương|CHƯƠNG)\s+(\d+)\b")
# Roman numerals like "I- TRIẾT HỌC..." or "II- VẬT CHẤT..."
ROMAN_SECTION_RE = re.compile(r"^([IVXLC]+)\s*[-–]\s*(.+)$")
# Numbered sections like "1. Khái lược về triết học" or "2. Vấn đề cơ bản"
NUMBERED_SECTION_RE = re.compile(r"^(\d+)\.\s+(.+)$")
# Letter sections like "a) Nguồn gốc của triết học" or "b) Khái niệm triết học"
LETTER_SECTION_RE = re.compile(r"^([a-z])\)\s+(.+)$")
# Uppercase letter headings like "A. MỤC TIÊU" or "B. NỘI DUNG"
# Must be followed by all uppercase content (to avoid matching "C. Mác...")
UPPER_LETTER_SECTION_RE = re.compile(r"^([A-Z])\.\s+([A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ\s]+)$")
# Star/bullet points like "* Nguồn gốc nhận thức"
STAR_SECTION_RE = re.compile(r"^\*\s+(.+)$")


@dataclass
class Section:
    """Represents a section in the document hierarchy."""
    level: int  # 0=chapter, 1=roman, 2=numbered, 3=letter, 4=star
    heading: str
    raw_heading: str
    content_lines: list[str] = field(default_factory=list)
    children: list["Section"] = field(default_factory=list)
    parent: Optional["Section"] = None

    # Hierarchy path
    chapter_num: int = 0
    chapter_title: str = ""
    roman_section: str = ""
    roman_title: str = ""
    numbered_section: str = ""
    numbered_title: str = ""
    letter_section: str = ""
    letter_title: str = ""
    star_title: str = ""

    def get_full_path(self) -> str:
        """Get the full hierarchical path."""
        parts: list[str] = []
        if self.chapter_num:
            parts.append(f"Chương {self.chapter_num}")
            if self.chapter_title:
                parts.append(self.chapter_title)
        if self.roman_section and self.roman_title:
            parts.append(f"{self.roman_section}- {self.roman_title}")
        if self.numbered_section and self.numbered_title:
            parts.append(f"{self.numbered_section}. {self.numbered_title}")
        if self.letter_section and self.letter_title:
            parts.append(f"{self.letter_section}) {self.letter_title}")
        if self.star_title:
            parts.append(f"* {self.star_title}")
        return " > ".join(parts)

    def get_content_text(self) -> str:
        """Get all content including heading."""
        lines = [self.raw_heading] if self.raw_heading else []
        lines.extend(self.content_lines)
        return "\n".join(lines).strip()


def is_all_caps_heading(line: str) -> bool:
    """Check if a line is an all-caps heading."""
    if len(line) < 6:
        return False
    # Remove punctuation and check if all letters are uppercase
    letters = [ch for ch in line if ch.isalpha()]
    if not letters:
        return False
    return all(ch.isupper() for ch in letters)


def parse_document(text: str) -> list[Section]:
    """Parse the document into a hierarchical structure."""
    lines = text.splitlines()

    chapters: list[Section] = []
    current_chapter: Optional[Section] = None
    current_roman: Optional[Section] = None
    current_numbered: Optional[Section] = None
    current_letter: Optional[Section] = None
    current_star: Optional[Section] = None

    chapter_title_pending = False

    for line in lines:
        line = line.strip()

        # Skip empty lines but preserve them in content
        if not line:
            if current_star:
                current_star.content_lines.append("")
            elif current_letter:
                current_letter.content_lines.append("")
            elif current_numbered:
                current_numbered.content_lines.append("")
            elif current_roman:
                current_roman.content_lines.append("")
            elif current_chapter:
                current_chapter.content_lines.append("")
            continue

        # Check for chapter heading
        chapter_match = CHAPTER_RE.match(line)
        if chapter_match:
            chapter_num = int(chapter_match.group(2))
            current_chapter = Section(
                level=0,
                heading=f"Chương {chapter_num}",
                raw_heading=line,
                chapter_num=chapter_num,
            )
            chapters.append(current_chapter)
            current_roman = None
            current_numbered = None
            current_letter = None
            current_star = None
            chapter_title_pending = True
            continue

        # Capture chapter title (usually the line after "Chương X")
        if chapter_title_pending and current_chapter:
            if is_all_caps_heading(line) and not UPPER_LETTER_SECTION_RE.match(line):
                current_chapter.chapter_title = line
                current_chapter.heading = f"Chương {current_chapter.chapter_num}: {line}"
            else:
                current_chapter.content_lines.append(line)
            chapter_title_pending = False
            continue

        # Check for Roman numeral section (I-, II-, III-)
        roman_match = ROMAN_SECTION_RE.match(line)
        if roman_match and current_chapter:
            roman_num = roman_match.group(1)
            roman_title = roman_match.group(2).strip()
            current_roman = Section(
                level=1,
                heading=f"{roman_num}- {roman_title}",
                raw_heading=line,
                parent=current_chapter,
                chapter_num=current_chapter.chapter_num,
                chapter_title=current_chapter.chapter_title,
                roman_section=roman_num,
                roman_title=roman_title,
            )
            current_chapter.children.append(current_roman)
            current_numbered = None
            current_letter = None
            current_star = None
            continue

        # Check for uppercase letter section (A., B., C.) - typically MỤC TIÊU, NỘI DUNG
        upper_letter_match = UPPER_LETTER_SECTION_RE.match(line)
        if upper_letter_match and current_chapter:
            # These are top-level sections within a chapter (like A. MỤC TIÊU, B. NỘI DUNG)
            letter = upper_letter_match.group(1)
            title = upper_letter_match.group(2).strip()
            # Treat as a special roman-level section
            current_roman = Section(
                level=1,
                heading=f"{letter}. {title}",
                raw_heading=line,
                parent=current_chapter,
                chapter_num=current_chapter.chapter_num,
                chapter_title=current_chapter.chapter_title,
                roman_section=letter,
                roman_title=title,
            )
            current_chapter.children.append(current_roman)
            current_numbered = None
            current_letter = None
            current_star = None
            continue

        # Check for numbered section (1., 2., 3.)
        numbered_match = NUMBERED_SECTION_RE.match(line)
        if numbered_match:
            num = numbered_match.group(1)
            title = numbered_match.group(2).strip()

            parent = current_roman or current_chapter
            if parent:
                current_numbered = Section(
                    level=2,
                    heading=f"{num}. {title}",
                    raw_heading=line,
                    parent=parent,
                    chapter_num=parent.chapter_num if hasattr(parent, 'chapter_num') else 0,
                    chapter_title=parent.chapter_title if hasattr(parent, 'chapter_title') else "",
                    roman_section=parent.roman_section if hasattr(parent, 'roman_section') else "",
                    roman_title=parent.roman_title if hasattr(parent, 'roman_title') else "",
                    numbered_section=num,
                    numbered_title=title,
                )
                parent.children.append(current_numbered)
                current_letter = None
                current_star = None
            continue

        # Check for letter section (a), b), c))
        letter_match = LETTER_SECTION_RE.match(line)
        if letter_match:
            letter = letter_match.group(1)
            title = letter_match.group(2).strip()

            parent = current_numbered or current_roman or current_chapter
            if parent:
                current_letter = Section(
                    level=3,
                    heading=f"{letter}) {title}",
                    raw_heading=line,
                    parent=parent,
                    chapter_num=getattr(parent, 'chapter_num', 0),
                    chapter_title=getattr(parent, 'chapter_title', ""),
                    roman_section=getattr(parent, 'roman_section', ""),
                    roman_title=getattr(parent, 'roman_title', ""),
                    numbered_section=getattr(parent, 'numbered_section', ""),
                    numbered_title=getattr(parent, 'numbered_title', ""),
                    letter_section=letter,
                    letter_title=title,
                )
                parent.children.append(current_letter)
                current_star = None
            continue

        # Check for star/bullet section (* Topic)
        star_match = STAR_SECTION_RE.match(line)
        if star_match:
            title = star_match.group(1).strip()

            parent = current_letter or current_numbered or current_roman or current_chapter
            if parent:
                current_star = Section(
                    level=4,
                    heading=f"* {title}",
                    raw_heading=line,
                    parent=parent,
                    chapter_num=getattr(parent, 'chapter_num', 0),
                    chapter_title=getattr(parent, 'chapter_title', ""),
                    roman_section=getattr(parent, 'roman_section', ""),
                    roman_title=getattr(parent, 'roman_title', ""),
                    numbered_section=getattr(parent, 'numbered_section', ""),
                    numbered_title=getattr(parent, 'numbered_title', ""),
                    letter_section=getattr(parent, 'letter_section', ""),
                    letter_title=getattr(parent, 'letter_title', ""),
                    star_title=title,
                )
                parent.children.append(current_star)
            continue

        # Add content to the deepest current section
        if current_star:
            current_star.content_lines.append(line)
        elif current_letter:
            current_letter.content_lines.append(line)
        elif current_numbered:
            current_numbered.content_lines.append(line)
        elif current_roman:
            current_roman.content_lines.append(line)
        elif current_chapter:
            current_chapter.content_lines.append(line)

    return chapters


def flatten_sections(chapters: list[Section]) -> list[Section]:
    """Flatten the hierarchical structure into a list of all sections."""
    result: list[Section] = []

    def traverse(section: Section):
        result.append(section)
        for child in section.children:
            traverse(child)

    for chapter in chapters:
        traverse(chapter)

    return result


def summarize_section_llm(
    section: Section,
    client: OpenAI,
    model: str,
    max_chars: int = 300,
    max_input_chars: int = 3000,
) -> str:
    """Generate a summary of what topics/concepts the section covers."""
    content = section.get_content_text()
    if not content.strip():
        return ""

    # Truncate input if too long
    content_truncated = content[:max_input_chars]

    path = section.get_full_path()

    system = """Bạn là trợ lý tóm tắt giáo trình triết học Mác-Lênin.
Nhiệm vụ: Tóm tắt ngắn gọn NỘI DUNG CHÍNH của đoạn văn, nêu các khái niệm/chủ đề quan trọng được đề cập.
Yêu cầu:
- Viết 1-2 câu ngắn gọn
- Liệt kê các khái niệm/thuật ngữ quan trọng
- Không thêm thông tin ngoài đoạn văn"""

    user = f"""Đường dẫn: {path}

Nội dung:
{content_truncated}

Hãy tóm tắt nội dung chính và các khái niệm quan trọng trong đoạn này (tối đa {max_chars} ký tự):"""

    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        summary = response.output_text.strip()
        if len(summary) > max_chars:
            summary = summary[:max_chars].rstrip()
        return summary
    except Exception as e:
        print(f"Warning: LLM summary failed for {path}: {e}")
        return ""


def simple_summarize(text: str, max_sentences: int = 2, max_chars: int = 280) -> str:
    """Simple fallback summary - extract first few sentences."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    summary_parts = []
    total = 0
    for sentence in sentences[:max_sentences]:
        if not sentence:
            continue
        summary_parts.append(sentence)
        total += len(sentence) + 1
        if total >= max_chars:
            break

    summary = " ".join(summary_parts).strip()
    if len(summary) > max_chars:
        summary = summary[:max_chars].rstrip()
    return summary


def build_chunks_v2(
    text: str,
    use_llm_summary: bool = False,
    summary_model: str = "gpt-4o-mini",
    summary_max_chars: int = 300,
    summary_max_input_chars: int = 3000,
) -> list[dict]:
    """Build chunks with hierarchical metadata and summaries."""

    # Parse document structure
    chapters = parse_document(text)
    all_sections = flatten_sections(chapters)

    # Initialize OpenAI client if using LLM
    client = None
    if use_llm_summary:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set for LLM summaries.")
        client = OpenAI(api_key=api_key)

    # Generate summaries for each section
    section_summaries: dict[int, str] = {}
    print("Generating section summaries...")
    for i, section in enumerate(all_sections):
        content = section.get_content_text()
        if not content.strip():
            section_summaries[i] = ""
            continue

        if use_llm_summary and client:
            summary = summarize_section_llm(
                section,
                client,
                summary_model,
                summary_max_chars,
                summary_max_input_chars,
            )
        else:
            summary = simple_summarize(content, max_chars=summary_max_chars)

        section_summaries[i] = summary
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(all_sections)} sections...")

    # Create chunks using sentence splitter
    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    chunks = []
    chunk_counter = 1

    for i, section in enumerate(all_sections):
        content = section.get_content_text()
        if not content.strip():
            continue

        # Get hierarchy info
        path = section.get_full_path()
        summary = section_summaries.get(i, "")

        # Split into chunks
        nodes = splitter.get_nodes_from_documents([Document(text=content)])

        for node in nodes:
            chunk_text = node.text.strip()
            if not chunk_text:
                continue

            chunk_id = f"chunk_{chunk_counter:05d}"
            chunk_counter += 1

            # Build content with context
            context_parts = [f"📍 {path}"]
            if summary:
                context_parts.append(f"📝 Tóm tắt: {summary}")
            context_parts.append("")
            context_parts.append(chunk_text)
            content_with_context = "\n".join(context_parts)

            # Build metadata
            metadata = {
                "chapter_num": section.chapter_num,
                "chapter_title": section.chapter_title,
                "roman_section": section.roman_section,
                "roman_title": section.roman_title,
                "numbered_section": section.numbered_section,
                "numbered_title": section.numbered_title,
                "letter_section": section.letter_section,
                "letter_title": section.letter_title,
                "star_title": section.star_title,
                "section_summary": summary,
                "hierarchy_path": path,
                "level": section.level,
            }

            chunks.append({
                "id": chunk_id,
                "content": chunk_text,
                "content_with_context": content_with_context,
                "path": path,
                "level": section.level,
                "title": section.heading,
                "metadata": metadata,
            })

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Improved chunking with hierarchical structure and LLM summaries."
    )
    parser.add_argument(
        "--input",
        default=Path(__file__).parent / "giao-trinh-triet-hoc-mac-lenin.clean.txt",
        type=Path,
        help="Path to cleaned textbook text file.",
    )
    parser.add_argument(
        "--output",
        default=Path(__file__).parent / "chunks_with_context.json",
        type=Path,
        help="Path to write chunks JSON.",
    )
    parser.add_argument(
        "--summarize-llm",
        action="store_true",
        help="Use LLM to summarize each section.",
    )
    parser.add_argument(
        "--summary-model",
        default=os.getenv("SUMMARY_MODEL", "gpt-4o-mini"),
        help="LLM model for section summaries.",
    )
    parser.add_argument(
        "--summary-max-chars",
        type=int,
        default=300,
        help="Max chars for section summary.",
    )
    parser.add_argument(
        "--summary-max-input-chars",
        type=int,
        default=3000,
        help="Max chars of section text fed to LLM.",
    )
    parser.add_argument(
        "--debug-structure",
        action="store_true",
        help="Print parsed document structure for debugging.",
    )
    args = parser.parse_args()

    # Read input
    text = args.input.read_text(encoding="utf-8", errors="ignore")

    # Debug: print structure
    if args.debug_structure:
        chapters = parse_document(text)
        print("\n=== DOCUMENT STRUCTURE ===\n")
        for chapter in chapters:
            print_section_tree(chapter, indent=0)
        return

    # Build chunks
    print(f"Building chunks from: {args.input}")
    chunks = build_chunks_v2(
        text,
        use_llm_summary=args.summarize_llm,
        summary_model=args.summary_model,
        summary_max_chars=args.summary_max_chars,
        summary_max_input_chars=args.summary_max_input_chars,
    )

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n✅ Chunks written to: {args.output}")
    print(f"✅ Total chunks: {len(chunks)}")

    # Print some statistics
    chapters_found = set()
    for chunk in chunks:
        if chunk["metadata"]["chapter_num"]:
            chapters_found.add(chunk["metadata"]["chapter_num"])
    print(f"✅ Chapters covered: {sorted(chapters_found)}")


def print_section_tree(section: Section, indent: int = 0):
    """Print section tree for debugging."""
    prefix = "  " * indent
    content_preview = section.get_content_text()[:50].replace("\n", " ")
    print(f"{prefix}[L{section.level}] {section.heading}")
    print(f"{prefix}     Path: {section.get_full_path()}")
    print(f"{prefix}     Content: {content_preview}...")
    for child in section.children:
        print_section_tree(child, indent + 1)


if __name__ == "__main__":
    main()
