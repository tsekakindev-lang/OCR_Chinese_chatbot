# uvicorn app:app --reload --port 8000
# ^ Handy command to run the FastAPI server in dev mode.

# Debug print so you can confirm which file is being executed (useful if multiple copies exist).
print("RUNNING app.py FROM:", __file__)

import os
import shutil
import hashlib
import asyncio
from threading import Lock
from typing import List, Literal, Optional, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

# --- LangChain + RAG stack ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain_core.documents import Document as LangChainDocument
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader

# --- Chroma low-level client (used here mainly to delete/reset collections cleanly) ---
import chromadb

# ---------------------------
# Poppler / OCR support (Windows-friendly)
# ---------------------------
# pdf2image needs Poppler (pdftoppm.exe) to render PDF pages as images for OCR.
# This block tries to auto-detect a locally bundled Poppler folder and sets POPPLER_PATH.
from pathlib import Path

if not os.getenv("POPPLER_PATH"):
    base = Path(__file__).resolve().parent
    for c in [
        base / "poppler-25.12.0" / "Library" / "bin",
        base / "poppler-25.12.0" / "bin",
    ]:
        if (c / "pdftoppm.exe").exists():
            os.environ["POPPLER_PATH"] = str(c)
            break

# Optional OCR deps (for scanned PDFs).
# If these imports fail, OCR fallback will be disabled (native text extraction still works).
try:
    import pytesseract  # type: ignore
    from pdf2image import convert_from_path  # type: ignore
    from pdf2image.pdf2image import pdfinfo_from_path  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None
    convert_from_path = None
    pdfinfo_from_path = None

# Windows Tesseract defaults (only used if OCR is invoked).
# Note: This sets TESSDATA_PREFIX globally for the process.
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

# ---------------------------
# CONFIG
# ---------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Front-end files are served from /static (index.html, app.js, style.css).
STATIC_DIR = os.path.join(APP_DIR, "static")

# Uploaded PDFs are stored here (one file per upload, named by hash prefix).
UPLOAD_DIR = os.path.join(APP_DIR, "uploads")

# Each PDF gets its own persistent Chroma folder here (so indexes don't collide).
CHROMA_BASE_DIR = os.path.join(APP_DIR, "chroma_dbs")

# Ensure directories exist so the app can write files.
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_BASE_DIR, exist_ok=True)

# Local embedding model folder (HuggingFace). Defaults to ./bge-large-zh-v1.5 in the project dir.
EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH", os.path.join(APP_DIR, "bge-large-zh-v1.5"))

# LLM served by Ollama. Adjust to whichever model you have pulled locally.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:14b")

# Chunking parameters: affects retrieval granularity and vectorstore size.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Retriever top-k: how many chunks to fetch for answering a question.
TOP_K = int(os.getenv("TOP_K", "5"))

# OCR params: Traditional Chinese by default.
OCR_LANG = os.getenv("OCR_LANG", "chi_tra")
OCR_DPI = int(os.getenv("OCR_DPI", "200"))

# Poppler binary path (used by pdf2image). Auto-detected above if bundled.
POPPLER_PATH = os.getenv("POPPLER_PATH", "") or None

# Chroma collection name inside each per-PDF persistent store.
COLLECTION_NAME = "rag_collection"

# ---------------------------
# State (in-memory)
# ---------------------------
# This server keeps only ONE active indexed PDF at a time.
# STATE holds paths to disk artifacts + the currently active RAG components for /chat.
STATE_LOCK = Lock()

STATE: dict[str, Any] = {
    "pdf_path": None,     # Full path to the uploaded PDF
    "chroma_dir": None,   # Folder where Chroma vectors are persisted for that PDF
    "rag": None,          # Dict: vectorstore + retrievers + llm + prompt template
}

# ---------------------------
# Helpers: load PDF pages (native extraction)
# ---------------------------
def load_pdf_to_page_docs(pdf_path: str) -> list[LangChainDocument]:
    """
    Extract text from the PDF using PyMuPDFLoader.
    Typically returns one LangChain Document per page, with page metadata.
    """
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()  # usually per page
    for d in docs:
        # Ensure each page doc has a source marker for later tracing/debugging.
        d.metadata.setdefault("source", pdf_path)
    return docs


# ---------------------------
# Helpers: OCR fallback (for scanned PDFs)
# ---------------------------
def _ensure_ocr_ready() -> None:
    """Validate OCR dependencies and configure Tesseract binary path."""
    if pytesseract is None or convert_from_path is None or pdfinfo_from_path is None:
        raise RuntimeError(
            "OCR dependencies are missing. Install 'pytesseract' and 'pdf2image' "
            "and ensure Poppler is available on the system."
        )
    # Configure the tesseract binary path if present (Windows).
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def get_pdf_page_count(pdf_path: str) -> int:
    """Return page count using pdf2image + Poppler."""
    _ensure_ocr_ready()
    info = pdfinfo_from_path(pdf_path, poppler_path=POPPLER_PATH)
    return int(info.get("Pages", 0))


def ocr_pdf_to_page_docs(
    pdf_path: str,
    start_page: int = 1,
    end_page: int | None = None,
    lang: str = OCR_LANG,
    dpi: int = OCR_DPI,
) -> list[LangChainDocument]:
    """
    OCR a scanned PDF into one LangChain Document per page.
    - Pages are 1-indexed for start/end.
    - Each page is rendered to an image (pdf2image) then recognized (tesseract).
    """
    _ensure_ocr_ready()

    if end_page is None:
        end_page = get_pdf_page_count(pdf_path)
    if start_page < 1:
        start_page = 1
    if end_page < start_page:
        end_page = start_page

    page_docs: list[LangChainDocument] = []

    for page_num in range(start_page, end_page + 1):
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_num,
            last_page=page_num,
            poppler_path=POPPLER_PATH,
        )
        if not images:
            continue

        text = (pytesseract.image_to_string(images[0], lang=lang) or "").strip()
        if not text:
            continue

        page_docs.append(
            LangChainDocument(
                page_content=text,
                metadata={"source": pdf_path, "page": page_num},
            )
        )

    return page_docs


# ---------------------------
# Question intent heuristics (overview / structure)
# ---------------------------
# These helpers try to detect “介紹/概述/summary” style questions and “幾章/章節” structure questions.
# When detected, retrieval is broadened with pre-defined expansions and doc-card hints.
import re
_CHAPTER_RE = re.compile(r"第[\d一二三四五六七八九十百千]+章")

def is_overview_q(q: str) -> bool:
    """True if user asks 'what is this about' / summary / overview."""
    q = (q or "").lower()
    return any(k in q for k in [
        "介紹","简介","簡介","概述","總覽","总览","這本書","这本书","這份文件","这份文件",
        "what is","about this","summary","overview"
    ])

def is_structure_q(q: str) -> bool:
    """True if user asks about chapter count/structure/contents."""
    q0 = q or ""
    ql = q0.lower()
    return (
        any(k in ql for k in ["幾章","几章","章節","章节","架構","架构","chapter","chapters","contents","目錄","目录"])
        or bool(_CHAPTER_RE.search(q0))
    )

def _normalize_pages_1_based(docs: list[LangChainDocument]) -> None:
    """
    Some loaders store pages as 0-based indices. This normalizes to 1-based pages,
    so citations like p.1, p.2 match human expectation.
    """
    pages = [d.metadata.get("page") for d in docs if isinstance(d.metadata.get("page"), int)]
    if pages and min(pages) == 0:
        for d in docs:
            p = d.metadata.get("page")
            if isinstance(p, int):
                d.metadata["page"] = p + 1


# ---------------------------
# “Doc card” (optional) to help answer overview/structure questions
# ---------------------------
# The idea: pre-build a condensed “overview snippet” from likely-intro pages and markers,
# then append it as a special Document so the retriever can find it easily.
OVERVIEW_MARKERS = [
    "摘要","Abstract","前言","序","導論","绪论","引言","introduction",
    "研究目的","目的","方法","method","研究方法","結論","结论","總結","总结",
    "本文","本書","本书","主要貢獻","主要贡献","contribution",
    "第一章","第二章","第三章","第四章","第五章","chapter 1","chapter"
]

def build_doc_card(page_docs, head_pages=10, max_chars=7000):
    """
    Build a synthetic Document that contains:
    - First N pages (often have preface/introduction)
    - Any pages containing overview markers (摘要/前言/第一章...)
    This helps retrieval for “介紹/幾章” type questions.
    """
    head = page_docs[:head_pages]
    hits = [d for d in page_docs if any(m in (d.page_content or "") for m in OVERVIEW_MARKERS)]

    # De-duplicate by page number so we don't repeat the same page content.
    seen = set()
    picked = []
    for d in head + hits:
        p = d.metadata.get("page")
        if p in seen:
            continue
        seen.add(p)
        picked.append(d)

    out, total = [], 0
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
        metadata={"type": "doc_card", "page": 1}
    )


# ---------------------------
# Vectorstore build (chunk -> embed -> persist)
# ---------------------------
def build_vectorstore(page_docs: list[LangChainDocument], chroma_dir: str) -> Chroma:
    """
    Split docs into chunks, embed them, and persist into a per-PDF Chroma DB folder.
    - Each upload maps to a unique chroma_dir under CHROMA_BASE_DIR.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(page_docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_PATH)

    # Reset the collection inside this directory (avoid mixing old vectors with new ones).
    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
        collection_name=COLLECTION_NAME,
    )


# ---------------------------
# Chat history shaping (compact prompt history)
# ---------------------------
def _messages_to_history_text(messages: list["ChatMessage"], max_turns: int = 12) -> str:
    """
    Convert messages into compact text for the prompt.
    - Keeps the last ~max_turns user+assistant turns (system messages excluded)
    - Excludes the latest user message so it doesn't repeat inside history
    """
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user" and messages[i].content.strip():
            last_user_idx = i
            break

    # Keep everything BEFORE the last user question
    upto = last_user_idx if last_user_idx is not None else len(messages)

    lines = []
    for m in messages[:upto]:
        if m.role == "system":
            continue
        text = (m.content or "").strip()
        if not text:
            continue
        lines.append(f"{m.role}: {text}")

    return "\n".join(lines[-max_turns * 2 :])

def _format_docs(docs: list[LangChainDocument], max_chars: int = 12000) -> str:
    """
    Build the context block shown to the LLM.
    - Adds [p.X] prefixes for better grounded citations
    - Truncates by total character budget
    """
    parts, total = [], 0
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


# ---------------------------
# RAG components (retrievers + llm + prompt)
# ---------------------------
def build_rag_components(vectorstore: Chroma) -> dict[str, Any]:
    """
    Prepare:
    - multiple retrievers (normal/overview/structure) using MMR for diversity
    - llm (Ollama)
    - prompt that enforces "answer only from context" + special behavior for overview questions
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 40, "lambda_mult": 0.5},
    )

    # Wider retrievers used for broad questions (overview/structure) to increase recall.
    retriever_overview = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": max(TOP_K, 12), "fetch_k": 60, "lambda_mult": 0.5},
    )
    retriever_structure = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": max(TOP_K, 12), "fetch_k": 60, "lambda_mult": 0.5},
    )

    llm = OllamaLLM(model=OLLAMA_MODEL)

    # Prompt: Chinese answer, strictly grounded; adds special instruction for overview mode.
    prompt = PromptTemplate(
        template=(
            "你是一个严谨但表达自然的助手。请严格根据【上下文】回答问题。\n"
            "当用户问“介绍/概述/summary/overview”：用 3–6 个要点概述；如出现章节线索再附结构要点。\n"
            "只写上下文出现的内容；若只找到部分，标注“其余未在上下文出现”；不得补写。\n"
            "若上下文完全没有相关信息：回答“文档中未找到相关内容。”\n\n"
            "【上下文】\n{context}\n\n"
            "【对话历史】\n{chat_history}\n\n"
            "【用户问题】\n{question}\n\n"
            "【回答】"
        ),
        input_variables=["context", "chat_history", "question"],
    )

    return {
        "vectorstore": vectorstore,
        "retriever": retriever,
        "retriever_overview": retriever_overview,
        "retriever_structure": retriever_structure,
        "llm": llm,
        "prompt": prompt,
    }


# ---------------------------
# Index build (runs in a thread via asyncio.to_thread)
# ---------------------------
def _build_sync(pdf_path: str, chroma_dir: str) -> None:
    """
    Build the index for a single uploaded PDF:
    1) Try native text extraction
    2) If too little text, OCR fallback
    3) Build doc_card (optional) and append
    4) Create embeddings + persist to Chroma
    5) Build RAG components and store them as the active state
    """
    page_docs = load_pdf_to_page_docs(pdf_path)
    _normalize_pages_1_based(page_docs)

    # If the PDF is scanned, native extraction often returns almost nothing.
    total_chars = sum(len(((d.page_content or "").strip())) for d in page_docs)
    if total_chars < 200:
        try:
            page_docs = ocr_pdf_to_page_docs(pdf_path)
        except Exception as e:
            raise RuntimeError(
                "PyMuPDFLoader extracted almost no text and OCR fallback failed. "
                f"OCR error: {e}"
            )

        if not page_docs:
            raise RuntimeError(
                "PyMuPDFLoader extracted almost no text and OCR returned empty text. "
                "This PDF may be image-only or the OCR language/config is incorrect."
            )

    # Build a synthetic “doc card” to improve overview/structure Q&A.
    doc_card = build_doc_card(page_docs, head_pages=10)
    if doc_card:
        doc_card.metadata.setdefault("source", pdf_path)
        page_docs.append(doc_card)

    # Build + persist embeddings
    vectorstore = build_vectorstore(page_docs, chroma_dir)

    # Prepare retrievers + llm + prompt
    rag = build_rag_components(vectorstore)

    # Store as the single active document state.
    with STATE_LOCK:
        STATE["pdf_path"] = pdf_path
        STATE["chroma_dir"] = chroma_dir
        STATE["rag"] = rag


# ---------------------------
# API schemas (request bodies)
# ---------------------------
class ChatMessage(BaseModel):
    # Ignore unknown fields so frontend can send extra keys safely.
    model_config = ConfigDict(extra="ignore")

    id: Optional[str] = None
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    messages: List[ChatMessage]


# ---------------------------
# FastAPI app + static hosting
# ---------------------------
app = FastAPI()

# Serve /static/style.css, /static/app.js, /static/index.html, etc.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def home():
    """
    Serves the frontend HTML at GET /.
    Expects index.html at: static/index.html
    """
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(500, "Missing static/index.html (copy your HTML here).")
    return FileResponse(index_path)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF, save it to disk, build the index, and set global active RAG state.
    Returns a short stable ID (hash prefix) based on file bytes.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Please upload a PDF (application/pdf).")

    raw = await file.read()
    file_id = hashlib.sha256(raw).hexdigest()[:12]

    pdf_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    chroma_dir = os.path.join(CHROMA_BASE_DIR, file_id)

    # Persist PDF to disk so loaders/OCR can access it by path.
    with open(pdf_path, "wb") as f:
        f.write(raw)

    # Build index in a worker thread so the FastAPI event loop stays responsive.
    try:
        await asyncio.to_thread(_build_sync, pdf_path, chroma_dir)
    except Exception as e:
        raise HTTPException(500, f"Indexing failed: {e}")

    return {"ok": True, "name": file.filename, "id": file_id}


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Chat endpoint:
    - Uses active RAG state (must upload first)
    - Detects question type (structure/overview/normal)
    - Retrieves relevant chunks
    - Calls local LLM (Ollama) with a grounded prompt
    - Appends page “Sources” from the retrieved docs
    """
    with STATE_LOCK:
        rag = STATE.get("rag")

    if rag is None:
        raise HTTPException(400, "No PDF indexed yet. Upload a PDF first.")

    # Extract the latest user question from the message list.
    user_msgs = [m for m in req.messages if m.role == "user" and (m.content or "").strip()]
    if not user_msgs:
        return {"reply": "Send a message first."}
    question = user_msgs[-1].content.strip()

    # Compact history for prompt (history is NOT used for retrieval).
    chat_history = _messages_to_history_text(req.messages, max_turns=12)

    NOT_FOUND = "文档中未找到相关内容。"

    # Expansions are appended as additional retrieval queries to increase recall for broad questions.
    OVERVIEW_EXPANSIONS = [
        "摘要 前言 引言 结论 总结 目的 方法 主要贡献 本书 本文",
        "what is this document about abstract introduction conclusion contribution",
    ]
    STRUCTURE_EXPANSIONS = [
        "本書主要分為 本书主要分为 第一章 第二章 第三章 第四章 第五章",
        "chapter 1 chapter 2 chapter 3 chapter 4 chapter 5",
    ]

    def _dedupe_docs(docs: list[LangChainDocument]) -> list[LangChainDocument]:
        """Remove near-duplicate retrieved docs (page/type/content prefix as key)."""
        seen, out = set(), []
        for d in docs or []:
            key = (d.metadata.get("type"), d.metadata.get("page"), (d.page_content or "")[:180])
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
        return out

    def _retrieve_many(r, queries: list[str]) -> list[LangChainDocument]:
        """
        Run multiple retrieval queries against one retriever and merge the results.
        Supports both .get_relevant_documents() and .invoke() depending on version.
        """
        got = []
        for q in queries:
            try:
                got.extend(r.get_relevant_documents(q))
            except Exception:
                got.extend(r.invoke(q))
        return got

    def run_rag_sync() -> str:
        """
        Heavy work done off the event loop:
        - retrieval + prompt formatting + LLM call
        """
        vectorstore = rag["vectorstore"]
        r_normal = rag["retriever"]
        r_overview = rag["retriever_overview"]
        r_structure = rag["retriever_structure"]
        llm = rag["llm"]
        prompt = rag["prompt"]

        # Choose retrieval mode by question intent.
        if is_structure_q(question):
            mode = "structure"
        elif is_overview_q(question):
            mode = "overview"
        else:
            mode = "normal"

        # Try to fetch doc_card directly (if supported) so broad questions have a strong anchor.
        doc_card_docs = []
        try:
            doc_card_docs = vectorstore.similarity_search("DOC_CARD", k=1, filter={"type": "doc_card"})
        except TypeError:
            # Older API variant: no filter argument.
            doc_card_docs = vectorstore.similarity_search("DOC_CARD", k=1)
        except Exception:
            doc_card_docs = []

        # Retrieval strategy:
        # - overview/structure: doc_card + expanded queries using wider retrievers
        # - normal: just the main retriever with the raw question
        if mode == "overview":
            docs = doc_card_docs + _retrieve_many(r_overview, [question] + OVERVIEW_EXPANSIONS)
        elif mode == "structure":
            docs = doc_card_docs + _retrieve_many(r_structure, [question] + STRUCTURE_EXPANSIONS)
        else:
            docs = _retrieve_many(r_normal, [question])

        docs = _dedupe_docs(docs)

        def _answer_with(docs_for_ctx: list[LangChainDocument]) -> tuple[str, list[LangChainDocument]]:
            """Format context + run LLM once."""
            context_text = _format_docs(docs_for_ctx)
            prompt_text = prompt.format(context=context_text, chat_history=chat_history, question=question)
            ans = (llm.invoke(prompt_text) or "").strip()
            return ans, docs_for_ctx

        answer, used_docs = _answer_with(docs)

        # Second-pass fallback:
        # If the model responds NOT_FOUND, broaden retrieval once (overview + structure expansions).
        if answer.startswith(NOT_FOUND):
            broader = doc_card_docs + _retrieve_many(
                r_overview, [question] + OVERVIEW_EXPANSIONS + STRUCTURE_EXPANSIONS
            )
            broader = _dedupe_docs(broader)
            answer2, used_docs2 = _answer_with(broader)
            if not answer2.startswith(NOT_FOUND):
                answer, used_docs = answer2, used_docs2

        # Build "Sources: p.X, p.Y" from the docs actually used as context.
        pages = []
        for d in used_docs[:12]:
            if d.metadata.get("type") == "doc_card":
                continue
            p = d.metadata.get("page")
            if isinstance(p, int):
                pages.append(p)
        pages = sorted(set(pages))[:10]
        if pages:
            answer = answer.rstrip() + "\n\nSources: " + ", ".join(f"p.{p}" for p in pages)

        return answer

    # Run retrieval + LLM off-thread.
    reply = await asyncio.to_thread(run_rag_sync)
    return {"reply": reply}


@app.post("/clear")
def clear() -> dict[str, Any]:
    """
    Clear endpoint:
    - resets in-memory active document
    - deletes uploaded PDF file (best-effort)
    - deletes the per-PDF Chroma directory (best-effort)
    """
    with STATE_LOCK:
        pdf_path = STATE.get("pdf_path")
        chroma_dir = STATE.get("chroma_dir")
        STATE["pdf_path"] = None
        STATE["chroma_dir"] = None
        STATE["rag"] = None

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
