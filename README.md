PDF Chat (RAG) — FastAPI + Chroma + Ollama

A simple local web app that lets you upload a PDF and chat with it using Retrieval-Augmented Generation (RAG).

Backend: FastAPI

Vector DB: Chroma (persisted per PDF)

Embeddings: local HuggingFace model (default: ./bge-large-zh-v1.5)

LLM: Ollama (default model: deepseek-r1:14b)

OCR fallback (optional): Tesseract + pdf2image + Poppler (helpful for scanned PDFs)

Features

Upload a PDF from the browser

Automatically indexes the PDF into a per-file Chroma DB

Ask questions and get answers with page references (best-effort)

“Clear” button resets the active PDF and deletes its stored vectors/files

Note: This server keeps only one active indexed PDF at a time.

Project Structure
.
├── app.py
├── static/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── uploads/        # created at runtime
└── chroma_dbs/     # created at runtime (per-PDF folders)

Requirements
Must-have

Python 3.10+ (3.11+ recommended)

Ollama installed and running

Optional (only if you need OCR for scanned PDFs)

Tesseract OCR

Poppler (needed by pdf2image)

Setup
1) Create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

2) Install Python dependencies

If you have requirements.txt, use it:

pip install -r requirements.txt


If you don’t, install the common deps:

pip install fastapi uvicorn pydantic chromadb PyMuPDF \
  langchain-core langchain-community langchain-text-splitters \
  langchain-huggingface langchain-chroma langchain-ollama


Optional OCR deps:

pip install pdf2image pytesseract pillow

3) Make sure Ollama has the model

Default model used by the app: deepseek-r1:14b

ollama pull deepseek-r1:14b


(Or choose another model and set OLLAMA_MODEL—see config below.)

4) Embedding model folder

By default the app looks for:

./bge-large-zh-v1.5

If your embedding model lives somewhere else, set:

# macOS/Linux
export EMBED_MODEL_PATH="/path/to/bge-large-zh-v1.5"

# Windows PowerShell
setx EMBED_MODEL_PATH "C:\path\to\bge-large-zh-v1.5"

Run the App

From the project root:

uvicorn app:app --reload --port 8000


Open:

http://127.0.0.1:8000

Configuration (Environment Variables)

These are read in app.py:

Variable	Default	What it does
EMBED_MODEL_PATH	./bge-large-zh-v1.5	Local HF embedding model directory
OLLAMA_MODEL	deepseek-r1:14b	Ollama model name
CHUNK_SIZE	600	Text chunk size for indexing
CHUNK_OVERLAP	100	Overlap between chunks
TOP_K	5	Retrieved chunks per question
OCR_LANG	chi_tra	Tesseract OCR language
OCR_DPI	200	Render DPI for OCR images
POPPLER_PATH	(auto-detect on Windows if bundled)	Poppler bin path for pdf2image
OCR notes (optional)

OCR activates only when normal PDF text extraction returns almost nothing.

On Windows, the code currently uses a default Tesseract path in app.py. If your Tesseract is installed elsewhere (or you’re on macOS/Linux), update that constant or set your environment accordingly.

API Endpoints
POST /upload

Upload a PDF (multipart/form-data with field name file).

Example:

curl -F "file=@your.pdf" http://127.0.0.1:8000/upload


Returns:

{ "ok": true, "name": "your.pdf", "id": "abcdef123456" }

POST /chat

Send messages:

curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is this PDF about?"}]}'


Returns:

{ "reply": "..." }

POST /clear

Clears current PDF + deletes its stored files/vectors:

curl -X POST http://127.0.0.1:8000/clear

Troubleshooting

“No PDF indexed yet. Upload a PDF first.”
Upload a PDF via the UI or POST /upload before chatting.

OCR errors / “OCR dependencies are missing…”
Install pytesseract and pdf2image, and ensure Tesseract + Poppler are installed/available.

Large PDF indexing is slow
That’s normal. Try smaller PDFs first or reduce CHUNK_SIZE/TOP_K.

Security Notes

This app accepts file uploads and runs a local LLM workflow. If you deploy it beyond localhost, add authentication and restrict file handling.
