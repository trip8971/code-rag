"""Markdown document chunking module."""

import hashlib
import re
from dataclasses import dataclass
from enum import Enum


class ChunkType(Enum):
    """文档片段类型枚举"""

    TEXT = "text"
    CODE = "code"


def generate_chunk_id(content: str, source_file: str, start_line: int = 0) -> str:
    """使用 content、source_file 和 start_line 的 SHA256 哈希生成唯一 chunk_id。

    Args:
        content: 片段文本内容
        source_file: 源文件路径
        start_line: 在原文档中的起始行号（避免 overlap 导致的内容重复冲突）

    Returns:
        SHA256 哈希字符串作为 chunk_id
    """
    hash_input = f"{content}{source_file}{start_line}"
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


@dataclass
class Chunk:
    """文档片段数据模型"""

    chunk_id: str  # 唯一标识符 (SHA256 of content + source_file)
    content: str  # 片段文本内容
    source_file: str  # 源文件路径
    heading_level: int  # 标题层级 0-6
    chunk_type: ChunkType  # 片段类型
    start_line: int  # 在原文档中的起始行号
    heading_text: str  # 所属标题文本
    context: str = ""  # 上下文文本（代码块前的描述，用于增强 embedding）

    @property
    def embedding_text(self) -> str:
        """用于生成 embedding / BM25 索引的文本。

        将 heading_text + context + content 拼接，确保即使 chunk 内容
        没有出现 heading 或上下文中的关键词，也能通过结构化信号被召回。
        若 content 已经以 heading_text 开头（代码 chunk 在索引阶段已把
        context 合并进 content），则跳过 heading 前缀以避免重复。
        """
        parts = []
        if self.heading_text and not self.content.lstrip().startswith(self.heading_text):
            parts.append(self.heading_text)
        if self.context:
            parts.append(self.context)
        parts.append(self.content)
        return "\n\n".join(parts)


class Chunker:
    """Markdown 文档分块器"""

    def __init__(self, max_chunk_size: int = 1500, overlap: int = 200):
        """初始化分块器。

        Args:
            max_chunk_size: 最大 Chunk 长度（字符）
            overlap: 相邻文本 Chunk 的重叠字符数
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        # 短于此长度的代码块不单独成 chunk，而是与前后文本合并
        self.min_code_chunk_size = 300
        # section 最小长度，短于此的 section 与相邻 section 合并
        self.min_section_size = 500
        # 短于此长度的「文本-代码」对中，文本段不单独成 chunk，
        # 而是作为代码块的 context（避免一行标题/解释成为孤立 chunk）
        self.min_text_anchor_size = 200

    def chunk_document(self, content: str, source_file: str) -> list[Chunk]:
        """将 Markdown 文档分块为 Chunk 列表。

        按标题层级和代码块边界拆分文档。长代码块（>= min_code_chunk_size）作为
        独立 Chunk 保留并附带上下文；短代码块与前后文本合并为混合 Chunk。
        超长文本在句子边界处拆分，相邻文本 Chunk 添加重叠内容。

        Args:
            content: Markdown 文档内容
            source_file: 源文件路径

        Returns:
            Chunk 列表，空文档返回空列表
        """
        if not content or not content.strip():
            return []

        sections = self._split_by_headings(content)
        # 合并过短的 section，避免产生碎片化 chunk
        sections = self._merge_short_sections(sections)
        chunks: list[Chunk] = []

        for section in sections:
            section_content = section["content"]
            heading_level = section["heading_level"]
            heading_text = section["heading_text"]
            start_line = section["start_line"]

            # Extract code blocks from the section content
            code_block_pattern = re.compile(r"^(```[^\n]*\n.*?^```)", re.MULTILINE | re.DOTALL)
            parts = code_block_pattern.split(section_content)

            # 第一遍：将短代码块与相邻文本合并
            merged_parts = self._merge_short_code_blocks(parts)

            # Track line offset within the section
            line_offset = 0
            # 记录最近的文本片段，作为代码块的上下文
            last_text_piece = ""

            for idx, (part_content, part_type) in enumerate(merged_parts):
                if not part_content.strip():
                    line_offset += part_content.count("\n")
                    continue

                if part_type == "code":
                    # 收尾碎片（CSS/Sandpack 闭合等）：内容太少，不单独成 chunk
                    if not self._is_substantial_code(part_content):
                        # 仍可作为后续上下文的延伸，但通常不希望污染 last_text_piece
                        line_offset += part_content.count("\n")
                        continue

                    # 长代码块：独立 chunk，把上下文直接合并进 content
                    # 这样 ChromaDB 嵌入、BM25、reranker 看到的都是带上下文的代码，
                    # 便于通过描述词召回；context 字段保持为空，避免重复拼接
                    context = self._build_code_context(heading_text, last_text_piece)
                    code_content = (
                        f"{context}\n\n{part_content}" if context else part_content
                    )

                    code_start_line = start_line + line_offset
                    chunk = Chunk(
                        chunk_id=generate_chunk_id(code_content, source_file, code_start_line),
                        content=code_content,
                        source_file=source_file,
                        heading_level=heading_level,
                        chunk_type=ChunkType.CODE,
                        start_line=code_start_line,
                        heading_text=heading_text,
                        context="",
                    )
                    chunks.append(chunk)
                    line_offset += part_content.count("\n")
                else:
                    # 文本（可能包含内联的短代码块）
                    stripped_text = part_content.strip()
                    last_text_piece = stripped_text

                    # 改动 2：短文本段紧邻后续代码块时，作为 context 而不是独立 chunk
                    next_is_code = (
                        idx + 1 < len(merged_parts)
                        and merged_parts[idx + 1][1] == "code"
                        and self._is_substantial_code(merged_parts[idx + 1][0])
                    )
                    if (
                        next_is_code
                        and len(stripped_text) < self.min_text_anchor_size
                        and "```" not in stripped_text
                    ):
                        line_offset += part_content.count("\n")
                        continue

                    # Text content: split if too long, apply overlap
                    text_pieces = self._split_long_text(part_content, start_line + line_offset)
                    text_pieces_with_overlap = self._apply_overlap(text_pieces)

                    piece_line_offset = line_offset
                    for piece in text_pieces_with_overlap:
                        stripped_piece = piece.strip()
                        if not stripped_piece:
                            piece_line_offset += piece.count("\n")
                            continue
                        # 过滤无意义碎片（纯标记标签、分隔符等）
                        has_code = "```" in stripped_piece
                        if has_code:
                            # 含代码 fence 的 piece：剥离 fence/JSX 后看实际有效内容长度，
                            # 过滤纯 CSS 收尾 + Sandpack/Solution 闭合标签这种碎片
                            if not self._is_substantial_code(stripped_piece):
                                piece_line_offset += piece.count("\n")
                                continue
                        elif not self._is_meaningful_content(stripped_piece):
                            piece_line_offset += piece.count("\n")
                            continue
                        piece_start_line = start_line + piece_line_offset
                        # 判断是否包含代码块来决定 chunk_type
                        chunk_type = ChunkType.CODE if has_code else ChunkType.TEXT
                        chunk = Chunk(
                            chunk_id=generate_chunk_id(stripped_piece, source_file, piece_start_line),
                            content=stripped_piece,
                            source_file=source_file,
                            heading_level=heading_level,
                            chunk_type=chunk_type,
                            start_line=piece_start_line,
                            heading_text=heading_text,
                        )
                        chunks.append(chunk)
                        piece_line_offset += piece.count("\n")

                    line_offset += part_content.count("\n")

        # 兜底：如果全部内容都被碎片过滤掉了（极短文档场景），
        # 回退为不过滤地产出至少一个 chunk
        if not chunks and content.strip():
            stripped = content.strip()
            fallback_type = ChunkType.CODE if "```" in stripped else ChunkType.TEXT
            chunks.append(
                Chunk(
                    chunk_id=generate_chunk_id(stripped, source_file, 1),
                    content=stripped,
                    source_file=source_file,
                    heading_level=sections[0]["heading_level"] if sections else 0,
                    chunk_type=fallback_type,
                    start_line=1,
                    heading_text=sections[0]["heading_text"] if sections else "",
                )
            )

        return chunks

    def _is_substantial_code(self, piece: str) -> bool:
        """判断含 code fence 的 piece 是否含足够实质内容。

        过滤目标：「样式收尾 + JSX 闭合」碎片，例如：
            ```css
            input { display: block; }
            ```

            </Sandpack>
            <Solution />

        默认保留。仅当**短 piece（< 200 字符）**且「无 prose + 仅含样式语言代码块
        + 含至少一个 JSX/HTML 闭合标签」时返回 False —— 这是 Sandpack 收尾的特征。

        Args:
            piece: 待检查的文本片段（可能包含 markdown + 代码块）

        Returns:
            True 如果有足够实质内容
        """
        # 长 piece 一定保留：避免误伤独立的长样式代码块
        if len(piece) >= 200:
            return True

        # 必须含 JSX/HTML 闭合标签才视为收尾碎片候选
        if not re.search(r"<[A-Za-z][^>]*/?>", piece):
            return True

        # 提取 fence 之外的内容
        without_codeblocks = re.sub(
            r"```[^\n]*\n.*?```", "", piece, flags=re.DOTALL
        )
        # 剥离 JSX/HTML 标签
        prose = re.sub(r"</?[A-Za-z][^>]*/?>", "", without_codeblocks)
        # 剥离 markdown 分隔符
        prose = re.sub(r"^-{3,}\s*$", "", prose, flags=re.MULTILINE)
        prose = re.sub(r"\s+", " ", prose).strip()
        if prose:
            return True

        # 检查所有代码块的语言
        code_block_re = re.compile(r"```([^\n]*)\n.*?```", re.DOTALL)
        languages = [
            (m.group(1).strip().lower().split()[0] if m.group(1).strip() else "")
            for m in code_block_re.finditer(piece)
        ]
        if not languages:
            return True
        style_langs = {"css", "scss", "sass", "less"}
        return not all(lang in style_langs for lang in languages)

    @staticmethod
    def _is_meaningful_content(text: str) -> bool:
        """判断文本是否有实际内容（非纯标记碎片）。

        过滤掉只包含 JSX 闭合标签、YAML 分隔符、单独的 markdown 链接/标题
        等无意义碎片。

        Args:
            text: 待检查的文本

        Returns:
            True 如果包含有意义的内容
        """
        # 去掉 HTML/JSX 标签后看是否还有实质内容
        stripped = re.sub(r"</?[A-Za-z][^>]*>", "", text).strip()
        # 去掉 YAML front matter 分隔符
        stripped = stripped.replace("---", "").strip()
        # 去掉 markdown anchor 注释（{/*xxx*/}）和单纯的标题井号
        stripped = re.sub(r"\{/\*[^*]*\*/\}", "", stripped).strip()
        stripped = re.sub(r"^#{1,6}\s+", "", stripped, flags=re.MULTILINE).strip()
        # 短于 40 字符且只含一句话的，认为是孤立的引用/标题碎片
        return len(stripped) >= 40

    def _merge_short_code_blocks(self, parts: list[str]) -> list[tuple[str, str]]:
        """将短代码块与相邻文本合并。

        遍历 split 后的 parts，短代码块（< min_code_chunk_size）不单独成块，
        而是追加到前一个文本段或作为下一个文本段的前缀。

        Args:
            parts: 由正则 split 产生的交替文本/代码块列表

        Returns:
            合并后的 (content, type) 列表，type 为 "text" 或 "code"
        """
        merged: list[tuple[str, str]] = []

        for part in parts:
            if not part.strip():
                # 空白部分追加到前一个文本段
                if merged and merged[-1][1] == "text":
                    merged[-1] = (merged[-1][0] + part, "text")
                else:
                    merged.append((part, "text"))
                continue

            if part.startswith("```"):
                if len(part) >= self.min_code_chunk_size:
                    # 长代码块：独立保留
                    merged.append((part, "code"))
                else:
                    # 短代码块：合并到前一个文本段
                    if merged and merged[-1][1] == "text":
                        merged[-1] = (merged[-1][0] + "\n" + part, "text")
                    else:
                        merged.append((part, "text"))
            else:
                # 文本段：尝试合并到前一个文本段（如果前一个也是文本）
                if merged and merged[-1][1] == "text":
                    merged[-1] = (merged[-1][0] + part, "text")
                else:
                    merged.append((part, "text"))

        return merged

    def _build_code_context(self, heading_text: str, preceding_text: str) -> str:
        """为代码块构建上下文文本。

        将标题和前一段文本拼接，从前文末尾截取最多 300 字符；
        优先在段落 / 句子 / 单词边界切，避免在单词或 markdown 链接中间硬切。

        Args:
            heading_text: 所属标题文本
            preceding_text: 代码块前的文本内容

        Returns:
            上下文字符串
        """
        max_context_len = 300

        trimmed = ""
        if preceding_text:
            if len(preceding_text) <= max_context_len:
                trimmed = preceding_text
            else:
                # 取末尾窗口
                window = preceding_text[-max_context_len:]
                # 从窗口起点向后找一个干净的边界
                # 优先级：段落（\n\n） > 句末标点（.!?。！？） > 单词边界（空白）
                candidates: list[int] = []
                for marker in ("\n\n", ". ", "! ", "? ", "。", "！", "？"):
                    idx = window.find(marker)
                    if idx != -1:
                        candidates.append(idx + len(marker))
                if candidates:
                    cut = min(candidates)
                else:
                    # 退回到第一个空白
                    import re
                    m = re.search(r"\s", window)
                    cut = m.end() if m else 0
                # 不要切掉超过一半的内容
                if cut > max_context_len // 2:
                    cut = 0
                trimmed = window[cut:].lstrip()

        parts = []
        if heading_text:
            parts.append(heading_text)
        if trimmed:
            parts.append(trimmed)
        return "\n".join(parts)

    def _split_by_headings(self, content: str) -> list[dict]:
        """按标题层级拆分文档。

        使用正则表达式识别 Markdown 标题（H1-H6），每个标题及其下属内容
        （直到下一个同级或更高级标题）构成一个 section。

        Args:
            content: Markdown 文档内容

        Returns:
            list of dicts with keys: content, heading_level, heading_text, start_line
        """
        lines = content.split("\n")
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

        sections: list[dict] = []
        current_section: dict | None = None

        for i, line in enumerate(lines):
            match = heading_pattern.match(line)
            if match:
                # Save previous section
                if current_section is not None:
                    sections.append(current_section)

                level = len(match.group(1))
                heading_text = line
                current_section = {
                    "content": line + "\n",
                    "heading_level": level,
                    "heading_text": heading_text,
                    "start_line": i + 1,  # 1-based line number
                }
            else:
                if current_section is None:
                    # Content before any heading
                    current_section = {
                        "content": line + "\n",
                        "heading_level": 0,
                        "heading_text": "",
                        "start_line": 1,
                    }
                else:
                    current_section["content"] += line + "\n"

        # Don't forget the last section
        if current_section is not None:
            sections.append(current_section)

        return sections

    def _merge_short_sections(self, sections: list[dict]) -> list[dict]:
        """合并过短的 section，避免产生碎片化 chunk。

        从前往后扫描，如果当前 section 内容长度 < min_section_size 且
        合并后不超过 max_chunk_size，则将其追加到前一个 section。
        如果第一个 section 自身过短，则把第二个 section 合并进它（这样
        frontmatter + intro + 第一个真正章节会成为一个 chunk，而不是
        让 frontmatter 单独成为孤立 chunk）。

        Args:
            sections: _split_by_headings 产生的 section 列表

        Returns:
            合并后的 section 列表
        """
        if len(sections) <= 1:
            return sections

        # 第一个 section 太短：把第二个 section 拼进来作为锚点
        first = sections[0]
        rest = sections[1:]
        if (
            len(first["content"]) < self.min_section_size
            and len(first["content"]) + len(rest[0]["content"]) <= self.max_chunk_size
        ):
            first = {
                **first,
                "content": first["content"] + rest[0]["content"],
            }
            rest = rest[1:]

        merged: list[dict] = [first]

        for section in rest:
            prev = merged[-1]
            section_len = len(section["content"])
            combined_len = len(prev["content"]) + section_len

            # 合并条件：当前 section 太短，且合并后不超过 max_chunk_size
            if section_len < self.min_section_size and combined_len <= self.max_chunk_size:
                prev["content"] += section["content"]
            else:
                merged.append(section)

        return merged

    def _split_long_text(self, text: str, start_line: int) -> list[str]:
        """将超长文本按句子边界拆分。

        在句子结束符（。？！. \\n）处拆分超长文本，确保每个子 Chunk
        不超过 max_chunk_size。

        Args:
            text: 待拆分的文本
            start_line: 文本在原文档中的起始行号

        Returns:
            拆分后的文本片段列表，每个片段 <= max_chunk_size
        """
        if len(text) <= self.max_chunk_size:
            return [text]

        pieces: list[str] = []
        remaining = text

        while len(remaining) > self.max_chunk_size:
            # Find the last sentence boundary within max_chunk_size
            search_region = remaining[: self.max_chunk_size]
            split_pos = self._find_sentence_boundary(search_region)

            if split_pos > 0:
                pieces.append(remaining[:split_pos])
                remaining = remaining[split_pos:]
            else:
                # No sentence boundary found; force split at max_chunk_size
                pieces.append(remaining[: self.max_chunk_size])
                remaining = remaining[self.max_chunk_size :]

        if remaining:
            pieces.append(remaining)

        return pieces

    def _find_sentence_boundary(self, text: str) -> int:
        """Find the last sentence boundary position in text.

        Sentence boundaries are: 。？！. \\n

        Args:
            text: Text to search for sentence boundaries

        Returns:
            Position after the last sentence boundary, or 0 if none found
        """
        # Search for the last occurrence of sentence-ending characters
        sentence_endings = "。？！.\n"
        last_pos = 0

        for i, char in enumerate(text):
            if char in sentence_endings:
                last_pos = i + 1  # Position after the boundary character

        return last_pos

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """为相邻 Chunk 添加重叠内容。

        重叠部分取自前一个 Chunk 的末尾，附加到后一个 Chunk 的开头。
        优先在段落 / 句子 / 单词边界处切，避免把 markdown 链接或单词拦腰切断。

        Args:
            chunks: 文本片段列表

        Returns:
            添加重叠后的文本片段列表
        """
        if len(chunks) <= 1 or self.overlap <= 0:
            return chunks

        result = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            overlap_text = self._clean_overlap_tail(prev_chunk)
            result.append(overlap_text + chunks[i])

        return result

    def _clean_overlap_tail(self, prev_chunk: str) -> str:
        """从 prev_chunk 末尾取最多 self.overlap 字符，并对齐到一个干净的边界。

        优先级：段落（\\n\\n） > 句末标点 > 单词边界。
        """
        if len(prev_chunk) <= self.overlap:
            return prev_chunk

        window = prev_chunk[-self.overlap:]
        candidates: list[int] = []
        for marker in ("\n\n", ". ", "! ", "? ", "。", "！", "？"):
            idx = window.find(marker)
            if idx != -1:
                candidates.append(idx + len(marker))
        if candidates:
            cut = min(candidates)
        else:
            import re
            m = re.search(r"\s", window)
            cut = m.end() if m else 0
        # 不要切掉超过一半的 overlap
        if cut > self.overlap // 2:
            cut = 0
        return window[cut:].lstrip()
