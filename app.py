# uvicorn app:app --reload --port 8000
# ^ Handy command to run the FastAPI server in dev mode.

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

# --- Chroma low-level client (for deleting collections) ---
import chromadb

# ---------------------------
# Poppler / OCR support (Windows-friendly)
# ---------------------------
# pdf2image needs Poppler (pdftoppm) to render PDF pages as images for OCR.
# This block tries to auto-detect a local bundled poppler folder and set POPPLER_PATH.
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

# Optional OCR deps (for scanned PDFs)
# If these imports fail, OCR fallback will be disabled.
try:
    import pytesseract  # type: ignore
    from pdf2image import convert_from_path  # type: ignore
    from pdf2image.pdf2image import pdfinfo_from_path  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None
    convert_from_path = None
    pdfinfo_from_path = None

# Windows Tesseract defaults (only used if OCR is invoked)
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

# ---------------------------
# CONFIG
# ---------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Where your front-end files live (index.html, app.js, style.css under /static)
STATIC_DIR = os.path.join(APP_DIR, "static")

# Where uploaded PDFs are saved
UPLOAD_DIR = os.path.join(APP_DIR, "uploads")

# Where per-PDF Chroma databases are created
CHROMA_BASE_DIR = os.path.join(APP_DIR, "chroma_dbs")

# Ensure directories exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_BASE_DIR, exist_ok=True)

# Local embedding model (HuggingFace) directory
# Default points to ./bge-large-zh-v1.5 inside this project folder
EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH", os.path.join(APP_DIR, "bge-large-zh-v1.5"))

# LLM served by Ollama
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:14b")

# Chunking parameters for building the vectorstore
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Retriever top-k (how many chunks to retrieve per question)
TOP_K = int(os.getenv("TOP_K", "5"))

# OCR params (Traditional Chinese by default)
OCR_LANG = os.getenv("OCR_LANG", "chi_tra")
OCR_DPI = int(os.getenv("OCR_DPI", "200"))

# Poppler binary path (used by pdf2image)
POPPLER_PATH = os.getenv("POPPLER_PATH", "") or None

# Chroma collection name inside each per-PDF persistent store
COLLECTION_NAME = "rag_collection"

# ---------------------------
# State (in-memory)
# ---------------------------
# This server keeps only ONE active indexed PDF at a time.
# The global dict holds where the PDF is and the “rag components” used by /chat.
STATE_LOCK = Lock()

STATE: dict[str, Any] = {
    "pdf_path": None,     # path to the uploaded PDF
    "chroma_dir": None,   # folder where Chroma persisted vectors for that PDF
    "rag": None,          # retriever + llm + prompt template
}

# ---------------------------
# Helpers: load PDF pages (native extraction)
# ---------------------------
def load_pdf_to_page_docs(pdf_path: str) -> list[LangChainDocument]:
    """
    Use PyMuPDFLoader to extract text from the PDF.
    Typically returns one LangChain Document per page.
    """
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()  # usually per page
    for d in docs:
        d.metadata.setdefault("source", pdf_path)
    return docs


# ---------------------------
# Helpers: OCR fallback (for scanned PDFs)
# ---------------------------
def _ensure_ocr_ready() -> None:
    """Validate OCR dependencies and configuration."""
    if pytesseract is None or convert_from_path is None or pdfinfo_from_path is None:
        raise RuntimeError(
            "OCR dependencies are missing. Install 'pytesseract' and 'pdf2image' "
            "and ensure Poppler is available on the system."
        )
    # Set the tesseract binary path if configured
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def get_pdf_page_count(pdf_path: str) -> int:
    """Return page count using pdf2image (Poppler)."""
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
    OCR a (scanned) PDF into one LangChain Document per page.
    Pages are 1-indexed for start/end inputs.
    """
    _ensure_ocr_ready()

    if end_page is None:
        end_page = get_pdf_page_count(pdf_path)
    if start_page < 1:
        start_page = 1
    if end_page < start_page:
        end_page = start_page

    page_docs: list[LangChainDocument] = []

    # Render each page to an image, then run Tesseract
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

import re
_CHAPTER_RE = re.compile(r"第[\d一二三四五六七八九十百千]+章")

def is_overview_q(q: str) -> bool:
    q = (q or "").lower()
    return any(k in q for k in [
        "介紹","简介","簡介","概述","總覽","总览","這本書","这本书","這份文件","这份文件",
        "what is","about this","summary","overview"
    ])

def is_structure_q(q: str) -> bool:
    q0 = q or ""
    ql = q0.lower()
    return (
        any(k in ql for k in ["幾章","几章","章節","章节","架構","架构","chapter","chapters","contents","目錄","目录"])
        or bool(_CHAPTER_RE.search(q0))
    )

def _normalize_pages_1_based(docs: list[LangChainDocument]) -> None:
    pages = [d.metadata.get("page") for d in docs if isinstance(d.metadata.get("page"), int)]
    if pages and min(pages) == 0:
        for d in docs:
            p = d.metadata.get("page")
            if isinstance(p, int):
                d.metadata["page"] = p + 1


OVERVIEW_MARKERS = [
    "摘要","Abstract","前言","序","導論","绪论","引言","introduction",
    "研究目的","目的","方法","method","研究方法","結論","结论","總結","总结",
    "本文","本書","本书","主要貢獻","主要贡献","contribution",
    "第一章","第二章","第三章","第四章","第五章","chapter 1","chapter"
]

def build_doc_card(page_docs, head_pages=10, max_chars=7000):
    head = page_docs[:head_pages]
    hits = [d for d in page_docs if any(m in (d.page_content or "") for m in OVERVIEW_MARKERS)]
    # de-dupe by page
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
# Vectorstore build
# ---------------------------
def build_vectorstore(page_docs: list[LangChainDocument], chroma_dir: str) -> Chroma:
    """
    Split docs into chunks, embed them, and persist into a Chroma database.
    Each upload gets its own persistent Chroma folder under CHROMA_BASE_DIR.
    """
    # Chunking so retrieval hits smaller, relevant spans
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(page_docs)

    # Local HF embeddings (great for Chinese if model is appropriate)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_PATH)

    # If a previous collection exists in this same chroma_dir, delete it first
    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # Build + persist the vector store
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
        collection_name=COLLECTION_NAME,
    )


# ---------------------------
# Chat history shaping
# ---------------------------
def _messages_to_history_text(messages: list["ChatMessage"], max_turns: int = 12) -> str:
    """
    Convert messages into compact history text.
    - Uses the last `max_turns` user+assistant turns (system messages excluded).
    - Excludes the latest user question from history (so it doesn't repeat).
    """
    # Find last user message index (the "current" question)
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user" and messages[i].content.strip():
            last_user_idx = i
            break

    # Keep everything BEFORE the last user question
    upto = last_user_idx if last_user_idx is not None else len(messages)

    lines = []
    for m in messages[:upto]:
        role = m.role
        if role == "system":
            continue
        text = (m.content or "").strip()
        if not text:
            continue
        lines.append(f"{role}: {text}")

    # Only keep the last N turns (approx: 2 lines per turn)
    return "\n".join(lines[-max_turns * 2 :])

def _format_docs(docs: list[LangChainDocument], max_chars: int = 12000) -> str:
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
# RAG components (retriever + llm + prompt)
# ---------------------------
def build_rag_components(vectorstore: Chroma) -> dict[str, Any]:
    """
    Prepare:
    - retriever (MMR for diversity)
    - llm (Ollama)
    - prompt template (Chinese, context-grounded)
    """
    # MMR: Maximal Marginal Relevance helps avoid returning redundant chunks
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 40, "lambda_mult": 0.5},
    )
    retriever_overview = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": max(TOP_K, 12), "fetch_k": 60, "lambda_mult": 0.5},
    )
    retriever_structure = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": max(TOP_K, 12), "fetch_k": 60, "lambda_mult": 0.5},
    )

    llm = OllamaLLM(model=OLLAMA_MODEL)

    # The prompt enforces "answer only from context" behavior.
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
    # 1) Try native text extraction
    page_docs = load_pdf_to_page_docs(pdf_path)
    _normalize_pages_1_based(page_docs)

    # If the PDF is scanned, PyMuPDFLoader will often extract almost nothing.
    total_chars = sum(len(((d.page_content or "").strip())) for d in page_docs)
    if total_chars < 200:
        # 2) OCR fallback (best effort)
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

    # ✅ Now page_docs is final (native or OCR): build doc_card
    doc_card = build_doc_card(page_docs, head_pages=10)
    if doc_card:
        doc_card.metadata.setdefault("source", pdf_path)
        page_docs.append(doc_card)

    # 3) Build + persist embeddings
    vectorstore = build_vectorstore(page_docs, chroma_dir)

    # 4) Prepare retriever + llm + prompt
    rag = build_rag_components(vectorstore)

    with STATE_LOCK:
        STATE["pdf_path"] = pdf_path
        STATE["chroma_dir"] = chroma_dir
        STATE["rag"] = rag



# ---------------------------
# API schemas (request bodies)
# ---------------------------
class ChatMessage(BaseModel):
    # Ignore unknown fields so the frontend can send extra keys safely
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
    Serves the frontend HTML.
    Your index.html is expected at: static/index.html
    """
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(500, "Missing static/index.html (copy your HTML here).")
    return FileResponse(index_path)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF, save it, build the index, and set the global active RAG state.
    Returns a short stable ID (hash prefix) to identify the upload.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Please upload a PDF (application/pdf).")

    # Read bytes, hash them (so same file gets same ID)
    raw = await file.read()
    file_id = hashlib.sha256(raw).hexdigest()[:12]

    pdf_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    chroma_dir = os.path.join(CHROMA_BASE_DIR, file_id)

    # Write PDF to disk
    with open(pdf_path, "wb") as f:
        f.write(raw)

    # Build index in a worker thread so the event loop stays responsive
    try:
        await asyncio.to_thread(_build_sync, pdf_path, chroma_dir)
    except Exception as e:
        raise HTTPException(500, f"Indexing failed: {e}")

    return {"ok": True, "name": file.filename, "id": file_id}


@app.post("/chat")
async def chat(req: ChatRequest):
    # Get current active RAG components
    with STATE_LOCK:
        rag = STATE.get("rag")

    if rag is None:
        raise HTTPException(400, "No PDF indexed yet. Upload a PDF first.")

    # Extract latest user question
    user_msgs = [m for m in req.messages if m.role == "user" and (m.content or "").strip()]
    if not user_msgs:
        return {"reply": "Send a message first."}
    question = user_msgs[-1].content.strip()

    # Compact history for the prompt (NOT used for retrieval)
    chat_history = _messages_to_history_text(req.messages, max_turns=12)

    NOT_FOUND = "文档中未找到相关内容。"

    OVERVIEW_EXPANSIONS = [
        "摘要 前言 引言 结论 总结 目的 方法 主要贡献 本书 本文",
        "what is this document about abstract introduction conclusion contribution",
    ]
    STRUCTURE_EXPANSIONS = [
        "本書主要分為 本书主要分为 第一章 第二章 第三章 第四章 第五章",
        "chapter 1 chapter 2 chapter 3 chapter 4 chapter 5",
    ]

    def _dedupe_docs(docs: list[LangChainDocument]) -> list[LangChainDocument]:
        seen, out = set(), []
        for d in docs or []:
            key = (d.metadata.get("type"), d.metadata.get("page"), (d.page_content or "")[:180])
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
        return out

    def _retrieve_many(r, queries: list[str]) -> list[LangChainDocument]:
        got = []
        for q in queries:
            try:
                got.extend(r.get_relevant_documents(q))
            except Exception:
                got.extend(r.invoke(q))
        return got

    def run_rag_sync() -> str:
        vectorstore = rag["vectorstore"]
        r_normal = rag["retriever"]
        r_overview = rag["retriever_overview"]
        r_structure = rag["retriever_structure"]
        llm = rag["llm"]
        prompt = rag["prompt"]

        # ✅ priority: structure > overview > normal
        if is_structure_q(question):
            mode = "structure"
        elif is_overview_q(question):
            mode = "overview"
        else:
            mode = "normal"

        # Try to fetch doc_card directly (filter)
        doc_card_docs = []
        try:
            doc_card_docs = vectorstore.similarity_search("DOC_CARD", k=1, filter={"type": "doc_card"})
        except TypeError:
            # older/variant API: no filter support
            doc_card_docs = vectorstore.similarity_search("DOC_CARD", k=1)
        except Exception:
            doc_card_docs = []


        if mode == "overview":
            docs = doc_card_docs + _retrieve_many(r_overview, [question] + OVERVIEW_EXPANSIONS)
        elif mode == "structure":
            docs = doc_card_docs + _retrieve_many(r_structure, [question] + STRUCTURE_EXPANSIONS)
        else:
            docs = _retrieve_many(r_normal, [question])

        docs = _dedupe_docs(docs)

        def _answer_with(docs_for_ctx: list[LangChainDocument]) -> tuple[str, list[LangChainDocument]]:
            context_text = _format_docs(docs_for_ctx)
            prompt_text = prompt.format(context=context_text, chat_history=chat_history, question=question)
            ans = (llm.invoke(prompt_text) or "").strip()
            return ans, docs_for_ctx

        answer, used_docs = _answer_with(docs)

        # 2-pass fallback: broaden once if NOT_FOUND
        if answer.startswith(NOT_FOUND):
            broader = doc_card_docs + _retrieve_many(
                r_overview, [question] + OVERVIEW_EXPANSIONS + STRUCTURE_EXPANSIONS
            )
            broader = _dedupe_docs(broader)
            answer2, used_docs2 = _answer_with(broader)
            if not answer2.startswith(NOT_FOUND):
                answer, used_docs = answer2, used_docs2


        # sources
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

    reply = await asyncio.to_thread(run_rag_sync)
    return {"reply": reply}

@app.post("/clear")
def clear() -> dict[str, Any]:
    # 1) snapshot paths + reset in-memory state (thread-safe)
    with STATE_LOCK:
        pdf_path = STATE.get("pdf_path")
        chroma_dir = STATE.get("chroma_dir")
        STATE["pdf_path"] = None
        STATE["chroma_dir"] = None
        STATE["rag"] = None

    # 2) best-effort delete local artifacts
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
