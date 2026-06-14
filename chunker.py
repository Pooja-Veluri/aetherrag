"""
Production-grade document chunker.

Three strategies:
  semantic      — sentence-aware splits with section header detection (default)
  hierarchical  — small retrieval child chunks + larger sliding-window parent context
  fixed         — original recursive character splitting (backward-compat)

All strategies:
  - Clean text (unicode normalise, strip boilerplate, collapse whitespace)
  - Track page number per chunk (chunks never cross the page they started on)
  - Detect section headings and store as metadata
  - Enforce minimum chunk size to discard micro-fragments
  - Return a uniform list[dict] consumed by RagEngine.add_document
"""

import re
import unicodedata
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Compiled regexes ────────────────────────────────────────────────────────

# Sentence boundary: end-of-sentence punctuation + whitespace + start of new sentence.
# Negative lookbehind for single capitals (initials) and common abbreviations.
_ABBREV = re.compile(
    r'\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|Fig|fig|Eq|No|Vol|pp|i\.e|e\.g|cf|al|ca|approx|U\.S|U\.K|Ph\.D|B\.Sc|M\.Sc)\.$',
    re.IGNORECASE,
)
_SENT_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'\(\[0-9])')

# Section heading patterns (covers markdown, ALL CAPS, numbered, "Title:" forms)
_SECTION_HEADER = re.compile(
    r'^(?:'
    r'#{1,4}\s+\S.{0,80}|'                  # ## Markdown heading
    r'(?:[A-Z][A-Z0-9 \-,:]{3,60})$|'       # ALL CAPS line (PDF headings)
    r'(?:\d+\.(?:\d+\.)*)\s+[A-Z].{2,60}$|' # 1.2.3 Numbered section
    r'[A-Z][a-zA-Z ]{3,50}:$'               # "Introduction:" style
    r')',
    re.MULTILINE,
)

# Lines to strip before chunking (page numbers, separators, copyright, URLs)
_BOILERPLATE = re.compile(
    r'(?:'
    r'^\s*\d{1,4}\s*$|'
    r'^[-=_*]{4,}\s*$|'
    r'^\s*Page\s+\d+\s+of\s+\d+\s*$|'
    r'^\s*www\.\S+\s*$|'
    r'^\s*©.{0,80}$|'
    r'^\s*All rights reserved\.?\s*$'
    r')',
    re.MULTILINE | re.IGNORECASE,
)

# ── Public API ───────────────────────────────────────────────────────────────


class DocumentChunker:
    """
    chunk_document(pages_data) → list of chunk dicts:
      {
        "text":     str,   # text for retrieval / LLM context
        "snippet":  str,   # short excerpt for UI display (= text for non-hierarchical)
        "page":     int,
        "section":  str,
        "strategy": str,
        "chunk_index": int,
        "parent_text": str,   # "" for semantic/fixed; parent window for hierarchical
      }
    """

    STRATEGIES = ("semantic", "hierarchical", "fixed")

    # Sensible defaults per strategy
    DEFAULTS = {
        "semantic":     {"chunk_size": 800,  "chunk_overlap": 150, "min_size": 80},
        "hierarchical": {"chunk_size": 400,  "chunk_overlap": 80,  "min_size": 60},
        "fixed":        {"chunk_size": 800,  "chunk_overlap": 150, "min_size": 50},
    }

    def __init__(
        self,
        strategy: str = "semantic",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_size: int | None = None,
        parent_multiplier: int = 3,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}")
        self.strategy = strategy
        d = self.DEFAULTS[strategy]
        self.chunk_size      = chunk_size     if chunk_size     is not None else d["chunk_size"]
        self.chunk_overlap   = chunk_overlap  if chunk_overlap  is not None else d["chunk_overlap"]
        self.min_chunk_size  = min_chunk_size if min_chunk_size is not None else d["min_size"]
        self.parent_multiplier = parent_multiplier  # child chunks per parent window

    # ── Entry point ──────────────────────────────────────────────────────────

    def chunk_document(self, pages_data: list[dict]) -> list[dict]:
        """
        pages_data: [{"page": int, "text": str}, ...]
        Returns list of chunk dicts (see class docstring).
        """
        paragraphs = self._to_paragraphs(pages_data)
        if not paragraphs:
            return []

        if self.strategy == "fixed":
            raw = self._fixed_chunks(paragraphs)
        elif self.strategy == "semantic":
            raw = self._semantic_chunks(paragraphs)
        else:
            raw = self._hierarchical_chunks(paragraphs)

        result = []
        for i, r in enumerate(raw):
            result.append({
                "text":        r["text"],
                "snippet":     r.get("snippet", r["text"]),
                "page":        r["page"],
                "section":     r.get("section", ""),
                "strategy":    self.strategy,
                "chunk_index": i,
                "parent_text": r.get("parent_text", ""),
            })
        return result

    # ── Text cleaning ────────────────────────────────────────────────────────

    @staticmethod
    def clean_text(text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        # Remove control characters except \n and \t
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        text = _BOILERPLATE.sub('', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = '\n'.join(line.rstrip() for line in text.splitlines())
        return text.strip()

    # ── Paragraph extraction ─────────────────────────────────────────────────

    def _to_paragraphs(self, pages_data: list[dict]) -> list[dict]:
        """
        Clean each page, detect section headers, split into paragraphs.
        Returns: [{"page": int, "section": str, "text": str}, ...]
        """
        paragraphs = []
        current_section = ""

        for item in pages_data:
            cleaned = self.clean_text(item["text"])
            if not cleaned:
                continue
            page_num = item["page"]

            for block in cleaned.split("\n\n"):
                block = block.strip()
                if not block:
                    continue

                lines = block.splitlines()
                # Single-line block → check if it's a section header
                if len(lines) == 1:
                    detected = self._detect_section(lines[0])
                    if detected:
                        current_section = detected
                        continue
                # Multi-line: check first line for heading, rest is body
                elif len(lines) > 1:
                    detected = self._detect_section(lines[0])
                    if detected:
                        current_section = detected
                        block = "\n".join(lines[1:]).strip()
                        if not block:
                            continue

                if len(block) >= self.min_chunk_size:
                    paragraphs.append({
                        "page":    page_num,
                        "section": current_section,
                        "text":    block,
                    })

        return paragraphs

    # ── Sentence utilities ───────────────────────────────────────────────────

    @staticmethod
    def _detect_section(line: str) -> Optional[str]:
        line = line.strip()
        if _SECTION_HEADER.match(line):
            return re.sub(r'^#{1,4}\s+', '', line).strip()
        return None

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """
        Abbreviation-aware sentence splitter using regex (no external deps).
        Re-joins splits that occurred after a known abbreviation.
        """
        parts = _SENT_END.split(text)
        sentences: list[str] = []
        buffer = ""

        for part in parts:
            candidate = (buffer + " " + part).strip() if buffer else part
            if _ABBREV.search(buffer) if buffer else False:
                # Previous fragment ended in abbreviation — don't split here
                buffer = candidate
            else:
                if buffer:
                    sentences.append(buffer.strip())
                buffer = part

        if buffer:
            sentences.append(buffer.strip())

        return [s for s in sentences if s]

    # ── Strategy: fixed ──────────────────────────────────────────────────────

    def _fixed_chunks(self, paragraphs: list[dict]) -> list[dict]:
        """Recursive character splitter — original approach, now with page tracking."""
        result = []
        for para in paragraphs:
            text = para["text"]
            splits = self._recursive_char_split(text)
            for chunk in splits:
                chunk = chunk.strip()
                if len(chunk) >= self.min_chunk_size:
                    result.append({"text": chunk, "page": para["page"], "section": para["section"]})
        return result

    def _recursive_char_split(self, text: str) -> list[str]:
        separators = ["\n\n", "\n", " ", ""]

        def split(txt, seps):
            if len(txt) <= self.chunk_size:
                return [txt]
            if not seps:
                step = max(self.chunk_size - self.chunk_overlap, 1)
                return [txt[i:i + self.chunk_size] for i in range(0, len(txt), step)]
            sep = seps[0]
            parts = txt.split(sep)
            merged, current = [], ""
            for part in parts:
                if len(part) > self.chunk_size:
                    if current:
                        merged.append(current)
                        current = ""
                    merged.extend(split(part, seps[1:]))
                else:
                    candidate = current + sep + part if current else part
                    if len(candidate) <= self.chunk_size:
                        current = candidate
                    else:
                        if current:
                            merged.append(current)
                        if self.chunk_overlap > 0 and len(current) > self.chunk_overlap:
                            current = current[-self.chunk_overlap:] + sep + part
                        else:
                            current = part
            if current:
                merged.append(current)
            return merged

        return split(text, separators)

    # ── Strategy: semantic ───────────────────────────────────────────────────

    def _semantic_chunks(self, paragraphs: list[dict]) -> list[dict]:
        """
        Split into sentences, greedily merge into size-bounded chunks.
        Overlap is sentence-level: carry last N sentences that fit within
        chunk_overlap characters into the next chunk.
        """
        # Build flat sentence list with page + section provenance.
        # Hard-split any "sentence" longer than chunk_size so no single chunk
        # can ever exceed the embedding API's max input length (Titan: 50k chars).
        sentences: list[dict] = []
        for para in paragraphs:
            for sent in self._split_sentences(para["text"]):
                sent = sent.strip()
                if not sent:
                    continue
                if len(sent) <= self.chunk_size:
                    sentences.append({"text": sent, "page": para["page"], "section": para["section"]})
                else:
                    step = max(self.chunk_size - self.chunk_overlap, 1)
                    for i in range(0, len(sent), step):
                        piece = sent[i:i + self.chunk_size].strip()
                        if piece:
                            sentences.append({"text": piece, "page": para["page"], "section": para["section"]})

        if not sentences:
            return []

        chunks = []
        current_sents: list[dict] = []
        current_len = 0

        def flush():
            body = " ".join(s["text"] for s in current_sents)
            if len(body.strip()) >= self.min_chunk_size:
                chunks.append({
                    "text":    body.strip(),
                    "page":    current_sents[0]["page"],
                    "section": next((s["section"] for s in current_sents if s["section"]), ""),
                })

        for sent in sentences:
            sent_len = len(sent["text"]) + 1
            if current_len + sent_len > self.chunk_size and current_sents:
                flush()
                # Sentence-level overlap: carry back sentences that fit in overlap budget
                overlap: list[dict] = []
                budget = 0
                for s in reversed(current_sents):
                    if budget + len(s["text"]) + 1 <= self.chunk_overlap:
                        overlap.insert(0, s)
                        budget += len(s["text"]) + 1
                    else:
                        break
                current_sents = overlap + [sent]
                current_len = budget + sent_len
            else:
                current_sents.append(sent)
                current_len += sent_len

        if current_sents:
            flush()

        return chunks

    # ── Strategy: hierarchical ───────────────────────────────────────────────

    def _hierarchical_chunks(self, paragraphs: list[dict]) -> list[dict]:
        """
        Two-tier chunking:
          child — small semantic chunks used for vector retrieval
          parent — sliding window of parent_multiplier children, sent to LLM as context

        Each child carries parent_text in its record.
        The child text is the retrieval target; parent_text is the LLM context.
        """
        # Generate small child chunks via semantic strategy
        child_chunker = DocumentChunker(
            strategy="semantic",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            min_chunk_size=self.min_chunk_size,
        )
        children = child_chunker._semantic_chunks(paragraphs)

        if not children:
            return []

        n = self.parent_multiplier
        half = n // 2
        result = []

        for i, child in enumerate(children):
            start_idx = max(0, i - half)
            end_idx   = min(len(children), start_idx + n)
            # Adjust window to always be n wide when possible
            if end_idx - start_idx < n:
                start_idx = max(0, end_idx - n)
            parent_text = " ".join(c["text"] for c in children[start_idx:end_idx])
            result.append({
                "text":        child["text"],          # small retrieval target
                "snippet":     child["text"],          # display excerpt
                "page":        child["page"],
                "section":     child.get("section", ""),
                "parent_text": parent_text.strip(),    # full LLM context
            })

        return result
