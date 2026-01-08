# uvicorn app_openai:app --reload --port 8000

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Optional

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document as LangChainDocument
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================
# Settings / Paths
# =========================

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Project layout
    app_dir: Path = Path(__file__).resolve().parent
    static_dir: Path = app_dir / "static"
    upload_dir: Path = app_dir / "uploads"
    chroma_base_dir: Path = app_dir / "chroma_dbs"

    # Embeddings / Retrieval
    embed_model_path: str = os.getenv("EMBED_MODEL_PATH", str(app_dir / "bge-large-zh-v1.5"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "600"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    collection_name: str = os.getenv("COLLECTION_NAME", "rag_collection")

    # Prompt shaping
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "12"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))

    # OCR (optional)
    ocr_lang: str = os.getenv("OCR_LANG", "chi_tra")
    ocr_dpi: int = int(os.getenv("OCR_DPI", "200"))
    poppler_path: Optional[str] = os.getenv("POPPLER_PATH") or None
    tesseract_cmd: str = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    tessdata_prefix: str = os.getenv("TESSDATA_PREFIX", r"C:\Program Files\Tesseract-OCR\tessdata")

    # OpenAI
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


SETTINGS = Settings()


def ensure_dirs() -> None:
    SETTINGS.static_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.upload_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.chroma_base_dir.mkdir(parents=True, exist_ok=True)


# =========================
# Optional OCR support
# =========================

def _autodetect_poppler() -> None:
    """Best-effort POPPLER_PATH auto-detect for Windows-friendly setups."""
    if SETTINGS.poppler_path:
        return

    base = SETTINGS.app_dir
    candidates = [
        base / "poppler-25.12.0" / "Library" / "bin",
        base / "poppler-25.12.0" / "bin",
    ]
    for c in candidates:
        if (c / "pdftoppm.exe").exists():
            os.environ["POPPLER_PATH"] = str(c)
            break


_autodetect_poppler()

try:
    import pytesseract  # type: ignore
    from pdf2image import convert_from_path  # type: ignore
    from pdf2image.pdf2image import pdfinfo_from_path  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None
    convert_from_path = None
    pdfinfo_from_path = None


def _ensure_ocr_ready() -> None:
    if pytesseract is None or convert_from_path is None or pdfinfo_from_path is None:
        raise RuntimeError("OCR dependencies are missing (pytesseract/pdf2image/poppler).")

    if SETTINGS.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = SETTINGS.tesseract_cmd
    if SETTINGS.tessdata_prefix and "TESSDATA_PREFIX" not in os.environ:
        os.environ["TESSDATA_PREFIX"] = SETTINGS.tessdata_prefix


def _pdf_page_count(pdf_path: str) -> int:
    _ensure_ocr_ready()
    info = pdfinfo_from_path(pdf_path, poppler_path=SETTINGS.poppler_path)
    return int(info.get("Pages", 0))


def ocr_pdf_to_page_docs(pdf_path: str) -> list[LangChainDocument]:
    """OCR a scanned PDF into one Document per page."""
    _ensure_ocr_ready()
    pages = _pdf_page_count(pdf_path)
    page_docs: list[LangChainDocument] = []

    for page_num in range(1, pages + 1):
        images = convert_from_path(
            pdf_path,
            dpi=SETTINGS.ocr_dpi,
            first_page=page_num,
            last_page=page_num,
            poppler_path=SETTINGS.poppler_path,
        )
        if not images:
            continue

        text = (pytesseract.image_to_string(images[0], lang=SETTINGS.ocr_lang) or "").strip()
        if not text:
            continue

        page_docs.append(
            LangChainDocument(
                page_content=text,
                metadata={"source": pdf_path, "page": page_num},
            )
        )

    return page_docs


# =========================
# PDF loading / chunking / vectorstore
# =========================

def load_pdf_to_page_docs(pdf_path: str) -> list[LangChainDocument]:
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    for d in docs:
        d.metadata.setdefault("source", pdf_path)
    return docs


def _normalize_pages_1_based(docs: list[LangChainDocument]) -> None:
    """
    Some loaders store pages as 0-based indices. Normalize to 1-based for human citations.
    """
    pages = [d.metadata.get("page") for d in docs if isinstance(d.metadata.get("page"), int)]
    if pages and min(pages) == 0:
        for d in docs:
            p = d.metadata.get("page")
            if isinstance(p, int):
                d.metadata["page"] = p + 1


def split_docs(page_docs: list[LangChainDocument]) -> list[LangChainDocument]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SETTINGS.chunk_size,
        chunk_overlap=SETTINGS.chunk_overlap,
    )
    return splitter.split_documents(page_docs)


def build_vectorstore(chunks: list[LangChainDocument], chroma_dir: str) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=SETTINGS.embed_model_path)

    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        client.delete_collection(SETTINGS.collection_name)
    except Exception:
        pass

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
        collection_name=SETTINGS.collection_name,
    )


# =========================
# Question intent heuristics + doc-card
# =========================

_CHAPTER_RE = re.compile(r"第[\d一二三四五六七八九十百千]+章")

def is_overview_q(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in [
        "介紹", "简介", "簡介", "概述", "總覽", "总览", "這本書", "这本书", "這份文件", "这份文件",
        "what is", "about this", "summary", "overview"
    ])

def is_structure_q(q: str) -> bool:
    q0 = q or ""
    ql = q0.lower()
    return (
        any(k in ql for k in ["幾章", "几章", "章節", "章节", "架構", "架构", "chapter", "chapters", "contents", "目錄", "目录"])
        or bool(_CHAPTER_RE.search(q0))
    )

OVERVIEW_MARKERS = [
    "摘要","Abstract","前言","序","導論","绪论","引言","introduction",
    "研究目的","目的","方法","method","研究方法","結論","结论","總結","总结",
    "本文","本書","本书","主要貢獻","主要贡献","contribution",
    "第一章","第二章","第三章","第四章","第五章","chapter 1","chapter"
]

def build_doc_card(page_docs: list[LangChainDocument], head_pages: int = 10, max_chars: int = 7000) -> Optional[LangChainDocument]:
    """
    Build a synthetic Document from:
    - first N pages (often preface/introduction)
    - pages containing overview markers (摘要/前言/第一章...)
    Helps retrieval for “介紹/幾章/架構” questions.
    """
    head = page_docs[:head_pages]
    hits = [d for d in page_docs if any(m in (d.page_content or "") for m in OVERVIEW_MARKERS)]

    seen_pages = set()
    picked: list[LangChainDocument] = []
    for d in head + hits:
        p = d.metadata.get("page")
        if p in seen_pages:
            continue
        seen_pages.add(p)
        picked.append(d)

    out: list[str] = []
    total = 0
    for d in picked:
        p = d.metadata.get("page", "?")
        txt = (d.page_content or "").strip()
        if not txt:
            continue
        chunk = f"[p.{p}]\n{txt}"
        if total + len(chunk) > max_chars:
            break
        out.append(chunk)
        total += len(chunk)

    if not out:
        return None

    return LangChainDocument(
        page_content="【DOC_CARD｜用于回答“介绍/概述/这份文件在讲什么”】【仅摘录原文】\n\n" + "\n\n---\n\n".join(out),
        metadata={"type": "doc_card", "page": 1, "source": page_docs[0].metadata.get("source") if page_docs else ""},
    )


# =========================
# Prompt shaping (improved)
# =========================

NOT_FOUND = "文档中未找到相关内容。"

PROMPT = PromptTemplate(
    template=(
        "你是一个严谨但表达自然的助手。请严格根据【上下文】回答问题。\n"
        "规则：\n"
        "1) 只写上下文出现的内容；不要补写、不要推测。\n"
        f"2) 若上下文完全没有相关信息：只回答“{NOT_FOUND}”。\n"
        "3) 当用户问“介绍/概述/summary/overview”：用 3–6 个要点概述；如出现章节线索再附结构要点。\n"
        "4) 当用户问“目录/幾章/章節/架構”：优先提取带“第一章/第二章/…/第X章”或类似结构线索的句子，并用要点列出。\n"
        "5) 用中文回答。\n\n"
        "【上下文】\n{context}\n\n"
        "【对话历史】\n{chat_history}\n\n"
        "【用户问题】\n{question}\n\n"
        "【回答】"
    ),
    input_variables=["context", "chat_history", "question"],
)


def messages_to_history_text(messages: list["ChatMessage"], max_turns: int) -> str:
    """Compact history lines (excluding the latest user question)."""
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user" and messages[i].content.strip():
            last_user_idx = i
            break

    upto = last_user_idx if last_user_idx is not None else len(messages)

    lines: list[str] = []
    for m in messages[:upto]:
        if m.role == "system":
            continue
        text = (m.content or "").strip()
        if not text:
            continue
        lines.append(f"{m.role}: {text}")

    return "\n".join(lines[-max_turns * 2 :])


def format_docs(docs: list[LangChainDocument], max_chars: int) -> str:
    """
    Build the context block:
    - prefixes with [p.X] for better grounding
    - truncates by a total character budget
    """
    parts: list[str] = []
    total = 0
    for d in docs or []:
        txt = (d.page_content or "").strip()
        if not txt:
            continue
        page = d.metadata.get("page", "?")
        chunk = f"[p.{page}]\n{txt}"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n\n".join(parts)


def _dedupe_docs(docs: list[LangChainDocument]) -> list[LangChainDocument]:
    seen = set()
    out: list[LangChainDocument] = []
    for d in docs or []:
        key = (d.metadata.get("type"), d.metadata.get("page"), (d.page_content or "")[:180])
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _pages_sources(docs: list[LangChainDocument], max_pages: int = 10) -> str:
    pages: list[int] = []
    for d in (docs or [])[:20]:
        if d.metadata.get("type") == "doc_card":
            continue
        p = d.metadata.get("page")
        if isinstance(p, int):
            pages.append(p)
    pages = sorted(set(pages))[:max_pages]
    return ", ".join(f"p.{p}" for p in pages)


# =========================
# RAG runtime state
# =========================

@dataclass
class RagComponents:
    vectorstore: Chroma
    retriever: Any
    retriever_overview: Any
    retriever_structure: Any
    prompt: PromptTemplate


class AppState:
    """Single-active-document state with a thread lock."""
    def __init__(self) -> None:
        self._lock = Lock()
        self.pdf_path: Optional[str] = None
        self.chroma_dir: Optional[str] = None
        self.rag: Optional[RagComponents] = None

    def get_rag(self) -> Optional[RagComponents]:
        with self._lock:
            return self.rag

    def set_active(self, pdf_path: str, chroma_dir: str, rag: RagComponents) -> None:
        with self._lock:
            self.pdf_path = pdf_path
            self.chroma_dir = chroma_dir
            self.rag = rag

    def clear(self) -> tuple[Optional[str], Optional[str]]:
        with self._lock:
            pdf_path = self.pdf_path
            chroma_dir = self.chroma_dir
            self.pdf_path = None
            self.chroma_dir = None
            self.rag = None
        return pdf_path, chroma_dir


STATE = AppState()


# =========================
# Index building
# =========================

def build_rag_components(vectorstore: Chroma) -> RagComponents:
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": SETTINGS.top_k, "fetch_k": 40, "lambda_mult": 0.5},
    )

    retriever_overview = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": max(SETTINGS.top_k, 12), "fetch_k": 60, "lambda_mult": 0.5},
    )

    retriever_structure = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": max(SETTINGS.top_k, 12), "fetch_k": 60, "lambda_mult": 0.5},
    )

    return RagComponents(
        vectorstore=vectorstore,
        retriever=retriever,
        retriever_overview=retriever_overview,
        retriever_structure=retriever_structure,
        prompt=PROMPT,
    )


def build_index_sync(pdf_path: str, chroma_dir: str) -> None:
    page_docs = load_pdf_to_page_docs(pdf_path)
    _normalize_pages_1_based(page_docs)

    # If PDF is scanned, native extraction may be almost empty -> OCR fallback
    total_chars = sum(len((d.page_content or "").strip()) for d in page_docs)
    if total_chars < 200:
        page_docs = ocr_pdf_to_page_docs(pdf_path)
        if not page_docs:
            raise RuntimeError("PDF looks scanned but OCR returned empty text.")

    # Build doc-card to help overview/structure questions
    doc_card = build_doc_card(page_docs, head_pages=10)
    if doc_card:
        doc_card.metadata.setdefault("source", pdf_path)
        page_docs.append(doc_card)

    chunks = split_docs(page_docs)
    vectorstore = build_vectorstore(chunks, chroma_dir)
    rag = build_rag_components(vectorstore)

    STATE.set_active(pdf_path=pdf_path, chroma_dir=chroma_dir, rag=rag)


# =========================
# FastAPI models + app
# =========================

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: Optional[str] = None
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    messages: list[ChatMessage]


ensure_dirs()

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(SETTINGS.static_dir)), name="static")

openai_client = OpenAI()


@app.get("/")
def home() -> FileResponse:
    index_path = SETTINGS.static_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(500, "Missing static/index.html")
    return FileResponse(str(index_path))


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Please upload a PDF (application/pdf).")

    raw = await file.read()
    file_id = hashlib.sha256(raw).hexdigest()[:12]

    pdf_path = SETTINGS.upload_dir / f"{file_id}.pdf"
    chroma_dir = SETTINGS.chroma_base_dir / file_id

    pdf_path.write_bytes(raw)

    try:
        await asyncio.to_thread(build_index_sync, str(pdf_path), str(chroma_dir))
    except Exception as e:
        # best-effort cleanup on failure
        try:
            if pdf_path.exists():
                pdf_path.unlink()
        except Exception:
            pass
        try:
            if chroma_dir.exists():
                shutil.rmtree(chroma_dir)
        except Exception:
            pass
        raise HTTPException(500, f"Indexing failed: {e}")

    return {"ok": True, "name": file.filename, "id": file_id}


@app.post("/chat")
async def chat(req: ChatRequest) -> dict[str, Any]:
    rag = STATE.get_rag()
    if rag is None:
        raise HTTPException(400, "No PDF indexed yet. Upload a PDF first.")

    user_msgs = [m for m in req.messages if m.role == "user" and (m.content or "").strip()]
    if not user_msgs:
        return {"reply": "Send a message first."}

    question = user_msgs[-1].content.strip()
    chat_history = messages_to_history_text(req.messages, max_turns=SETTINGS.max_history_turns)

    OVERVIEW_EXPANSIONS = [
        "摘要 前言 引言 结论 总结 目的 方法 主要贡献 本书 本文",
        "what is this document about abstract introduction conclusion contribution",
    ]
    STRUCTURE_EXPANSIONS = [
        "本書主要分為 本书主要分为 第一章 第二章 第三章 第四章 第五章 章節 章节 架構 架构 目录 目錄",
        "chapter 1 chapter 2 chapter 3 chapter 4 chapter 5 contents table of contents",
    ]

    def _retrieve_many(r, queries: list[str]) -> list[LangChainDocument]:
        got: list[LangChainDocument] = []
        for q in queries:
            try:
                got.extend(r.get_relevant_documents(q))
            except Exception:
                got.extend(r.invoke(q))
        return got

    def run_rag() -> str:
        # Mode selection
        if is_structure_q(question):
            mode = "structure"
        elif is_overview_q(question):
            mode = "overview"
        else:
            mode = "normal"

        # Try to fetch doc_card directly (if supported)
        doc_card_docs: list[LangChainDocument] = []
        try:
            doc_card_docs = rag.vectorstore.similarity_search("DOC_CARD", k=2, filter={"type": "doc_card"})
        except TypeError:
            doc_card_docs = rag.vectorstore.similarity_search("DOC_CARD", k=2)
        except Exception:
            doc_card_docs = []

        if mode == "overview":
            docs = doc_card_docs + _retrieve_many(rag.retriever_overview, [question] + OVERVIEW_EXPANSIONS)
        elif mode == "structure":
            docs = doc_card_docs + _retrieve_many(rag.retriever_structure, [question] + STRUCTURE_EXPANSIONS)
        else:
            docs = _retrieve_many(rag.retriever, [question])

        docs = _dedupe_docs(docs)

        def _answer_once(ctx_docs: list[LangChainDocument]) -> tuple[str, list[LangChainDocument]]:
            context_text = format_docs(ctx_docs, max_chars=SETTINGS.max_context_chars)
            prompt_text = rag.prompt.format(context=context_text, chat_history=chat_history, question=question)

            resp = openai_client.responses.create(
                model=SETTINGS.openai_model,
                input=prompt_text,
            )
            ans = (resp.output_text or "").strip()
            return ans, ctx_docs

        answer, used_docs = _answer_once(docs)

        # Second-pass fallback: broaden once if NOT_FOUND
        if answer.startswith(NOT_FOUND):
            broader = doc_card_docs + _retrieve_many(
                rag.retriever_overview, [question] + OVERVIEW_EXPANSIONS + STRUCTURE_EXPANSIONS
            )
            broader = _dedupe_docs(broader)
            answer2, used_docs2 = _answer_once(broader)
            if not answer2.startswith(NOT_FOUND):
                answer, used_docs = answer2, used_docs2

        sources = _pages_sources(used_docs, max_pages=10)
        if sources:
            answer = answer.rstrip() + "\n\nSources: " + sources

        return answer

    reply = await asyncio.to_thread(run_rag)
    return {"reply": reply}


@app.post("/clear")
def clear() -> dict[str, Any]:
    pdf_path, chroma_dir = STATE.clear()

    if pdf_path and os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
        except Exception:
            pass

    if chroma_dir and os.path.exists(chroma_dir):
        try:
            shutil.rmtree(chroma_dir)
        except Exception:
            pass

    return {"ok": True}
