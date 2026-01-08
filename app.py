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

from dotenv import load_dotenv
load_dotenv()  # loads .env into environment variables

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
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:14b")

# Replace the local LLM with OpenAI
from openai import OpenAI
openai_client = OpenAI()  # reads OPENAI_API_KEY from env
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

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
    """
    Join retrieved documents into one context string, with a safety cap
    to avoid sending extremely large prompts to the LLM.
    """
    parts = []
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


# ---------------------------
# RAG components (retriever + llm + prompt)
# ---------------------------
# Local LLM
# def build_rag_components(vectorstore: Chroma) -> dict[str, Any]:
#     """
#     Prepare:
#     - retriever (MMR for diversity)
#     - llm (Ollama)
#     - prompt template (Chinese, context-grounded)
#     """
#     # MMR: Maximal Marginal Relevance helps avoid returning redundant chunks
#     retriever = vectorstore.as_retriever(
#         search_type="mmr",
#         search_kwargs={"k": TOP_K, "fetch_k": 40, "lambda_mult": 0.5},
#     )


#     # Load LLM
#     # llm = OllamaLLM(model=OLLAMA_MODEL)

#     # The prompt enforces "answer only from context" behavior.
#     prompt = PromptTemplate(
#         template=(
#             "你是一个严谨但表达自然的助手。请严格根据【上下文】回答问题。\n"
#             "- 如果上下文中没有相关信息，请直接回答：“文档中未找到相关内容。”\n"
#             "- 用中文回答。\n\n"
#             "【上下文】\n{context}\n\n"
#             "【对话历史】\n{chat_history}\n\n"
#             "【用户问题】\n{question}\n\n"
#             "【回答】"
#         ),
#         input_variables=["context", "chat_history", "question"],
#     )

#     return {"retriever": retriever, "llm": llm, "prompt": prompt}

# Use OpenAI API
def build_rag_components(vectorstore: Chroma) -> dict[str, Any]:
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 40, "lambda_mult": 0.5},
    )

    prompt = PromptTemplate(
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

    return {"retriever": retriever, "prompt": prompt}


# ---------------------------
# Index build (runs in a thread via asyncio.to_thread)
# ---------------------------
def _build_sync(pdf_path: str, chroma_dir: str) -> None:
    """
    Build the entire RAG state synchronously:
    1) Extract text via PyMuPDFLoader
    2) If almost empty -> OCR fallback
    3) Build vectorstore (Chroma)
    4) Build retriever/llm/prompt and store in global STATE
    """
    # 1) Try native text extraction
    page_docs = load_pdf_to_page_docs(pdf_path)

    # If the PDF is scanned, PyMuPDFLoader will often extract almost nothing.
    total_chars = sum(len(d.page_content.strip()) for d in page_docs)
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

    # 3) Build + persist embeddings
    vectorstore = build_vectorstore(page_docs, chroma_dir)

    # 4) Prepare retriever + llm + prompt
    rag = build_rag_components(vectorstore)

    # Save active state (single active PDF at a time)
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
    """
    Main chat endpoint:
    - reads latest user question
    - retrieves relevant chunks from Chroma
    - formats prompt (context + history + question)
    - calls Ollama LLM
    - returns answer + page source hints
    """
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

    # Use OpenAI API
    retriever = rag["retriever"]
    prompt = rag["prompt"]

    # Local LLM
    # retriever = rag["retriever"]
    # llm = rag["llm"]
    # prompt = rag["prompt"]

    # Run retrieval+generation in a background thread (sync calls)
    def run_rag():
        # Retrieve using ONLY the current question (better than mixing history into retrieval)
        try:
            docs = retriever.get_relevant_documents(question)
        except Exception:
            # Some retrievers use .invoke() in newer LangChain versions
            docs = retriever.invoke(question)

        context_text = _format_docs(docs)
        prompt_text = prompt.format(
            context=context_text,
            chat_history=chat_history,
            question=question
        )
        
        # Local LLM
        # answer = (llm.invoke(prompt_text) or "").strip()

        # Use OpenAI API
        resp = openai_client.responses.create(
            model=OPENAI_MODEL,
            input=prompt_text,
        )

        answer = (resp.output_text or "").strip()

        # Attach quick sources (page numbers) from metadata, best-effort
        sources = []
        for d in (docs or [])[:5]:
            page = d.metadata.get("page", "?")
            sources.append(f"p.{page}")
        if sources:
            answer = answer.rstrip() + "\n\nSources: " + ", ".join(sources)

        return answer

    reply = await asyncio.to_thread(run_rag)
    return {"reply": reply}


@app.post("/clear")
def clear():
    """
    Clears the active PDF + vector DB for a “fresh start”.
    Also deletes the saved PDF file and its chroma directory on disk (best effort).
    """
    with STATE_LOCK:
        pdf_path = STATE.get("pdf_path")
        chroma_dir = STATE.get("chroma_dir")
        STATE["pdf_path"] = None
        STATE["chroma_dir"] = None
        STATE["rag"] = None   # ✅ correct key name (you store rag dict)

    # best-effort cleanup (ignore failures)
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
