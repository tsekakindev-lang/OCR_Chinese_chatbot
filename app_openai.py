# uvicorn app_openai:app --reload --port 8000

from __future__ import annotations

import asyncio
import hashlib
import os
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

print("RUNNING app.py FROM:", __file__)

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

    # Configure Windows defaults (only if OCR is invoked)
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


def split_docs(page_docs: list[LangChainDocument]) -> list[LangChainDocument]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SETTINGS.chunk_size,
        chunk_overlap=SETTINGS.chunk_overlap,
    )
    return splitter.split_documents(page_docs)


def build_vectorstore(chunks: list[LangChainDocument], chroma_dir: str) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=SETTINGS.embed_model_path)

    # Ensure a clean collection inside this persistent folder
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
# Prompt shaping
# =========================

PROMPT = PromptTemplate(
    template=(
        "你是一个严谨但表达自然的助手。请严格根据【上下文】回答问题。\n"
        "- 如果上下文中没有相关信息，请直接回答：“文档中未找到相关内容。”\n"
        "- 用中文回答。\n\n"
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
    parts: list[str] = []
    total = 0
    for d in docs or []:
        chunk = (d.page_content or "").strip()
        if not chunk:
            continue
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n\n".join(parts)


def extract_sources(docs: list[LangChainDocument], limit: int = 5) -> str:
    sources: list[str] = []
    for d in (docs or [])[:limit]:
        page = d.metadata.get("page", "?")
        sources.append(f"p.{page}")
    return ", ".join(sources)


# =========================
# RAG runtime state
# =========================

@dataclass
class RagComponents:
    retriever: Any
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
    return RagComponents(retriever=retriever, prompt=PROMPT)


def build_index_sync(pdf_path: str, chroma_dir: str) -> None:
    page_docs = load_pdf_to_page_docs(pdf_path)

    # If PDF is scanned, native extraction may be almost empty -> OCR fallback
    total_chars = sum(len((d.page_content or "").strip()) for d in page_docs)
    if total_chars < 200:
        page_docs = ocr_pdf_to_page_docs(pdf_path)
        if not page_docs:
            raise RuntimeError("PDF looks scanned but OCR returned empty text.")

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

    def run_rag() -> str:
        # Retrieval (prefer newer API, fallback for older)
        try:
            docs = rag.retriever.invoke(question)
        except Exception:
            docs = rag.retriever.get_relevant_documents(question)

        context_text = format_docs(docs, max_chars=SETTINGS.max_context_chars)
        prompt_text = rag.prompt.format(context=context_text, chat_history=chat_history, question=question)

        resp = openai_client.responses.create(
            model=SETTINGS.openai_model,
            input=prompt_text,
        )
        answer = (resp.output_text or "").strip()

        sources = extract_sources(docs)
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
