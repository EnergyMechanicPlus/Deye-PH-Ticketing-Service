
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
from datetime import datetime
import uuid
import json
from typing import Optional, List, Dict, Any
import os

# Vector DB / embeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import numpy as np

# ----------------------
# Config
# ----------------------
ROOT = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
INDEX_DIR = DATA_DIR / "chroma"
META_FILE = DATA_DIR / "index.json"

for p in [DATA_DIR, PDF_DIR, INDEX_DIR]:
    p.mkdir(parents=True, exist_ok=True)

if not META_FILE.exists():
    META_FILE.write_text(json.dumps({"files": {}}, indent=2))

API_KEY = os.getenv("VAULT_API_KEY", "")

# ----------------------
# App
# ----------------------
app = FastAPI(title="PDF Memory Vault", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# Security dependency
# ----------------------
def require_api_key(x_api_key: Optional[str] = Header(None)):
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# ----------------------
# Embeddings + Chroma
# ----------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path=str(INDEX_DIR))
collection = client.get_or_create_collection(name="pdf_chunks")

# ----------------------
# Helpers
# ----------------------
def load_meta() -> Dict[str, Any]:
    return json.loads(META_FILE.read_text())

def save_meta(meta: Dict[str, Any]):
    META_FILE.write_text(json.dumps(meta, indent=2))

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def extract_pdf_text(pdf_path: Path) -> List[Dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append({"page": i + 1, "text": page.extract_text() or ""})
        except Exception:
            pages.append({"page": i + 1, "text": ""})
    return pages

# ----------------------
# Routes
# ----------------------
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/upload", dependencies=[Depends(require_api_key)] if API_KEY else None)
async def upload_pdf(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_id = str(uuid.uuid4())
    safe_name = f"{file_id}_{Path(file.filename).name}"
    dest = PDF_DIR / safe_name

    with dest.open("wb") as f:
        f.write(await file.read())

    # Read & index
    pages = extract_pdf_text(dest)
    all_chunks, all_ids, all_metas = [], [], []
    for p in pages:
        if not p["text"]:
            continue
        for idx, chunk in enumerate(chunk_text(p["text"])):
            chunk_id = f"{file_id}_p{p['page']}_c{idx}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metas.append({
                "file_id": file_id,
                "filename": dest.name,
                "page": p["page"],
                "title": title or Path(file.filename).stem,
                "tags": [t.strip() for t in (tags.split(",") if tags else []) if t.strip()],
                "notes": notes or "",
                "uploaded_at": datetime.utcnow().isoformat() + "Z",
            })

    if all_chunks:
        embeddings = model.encode(all_chunks, convert_to_numpy=True).tolist()
        collection.add(
            documents=all_chunks, 
            metadatas=all_metas, 
            ids=all_ids, 
            embeddings=embeddings
        )

    meta = load_meta()
    meta["files"][file_id] = {
        "file_id": file_id,
        "original_filename": file.filename,
        "stored_filename": dest.name,
        "path": str(dest),
        "title": title or Path(file.filename).stem,
        "tags": [t.strip() for t in (tags.split(",") if tags else []) if t.strip()],
        "notes": notes or "",
        "pages": len(pages),
        "uploaded_at": datetime.utcnow().isoformat() + "Z",
    }
    save_meta(meta)

    return {"ok": True, "file_id": file_id, "stored": dest.name, "pages": len(pages)}

from fastapi import Depends

@app.get("/list", dependencies=[Depends(require_api_key)] if API_KEY else None)
async def list_files():
    return load_meta()["files"]

@app.get("/file/{file_id}", dependencies=[Depends(require_api_key)] if API_KEY else None)
async def get_file(file_id: str):
    rec = load_meta()["files"].get(file_id)
    if not rec:
        raise HTTPException(status_code=404, detail="File not found")
    path = Path(rec["path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Stored file missing")
    return FileResponse(path)

@app.delete("/file/{file_id}", dependencies=[Depends(require_api_key)] if API_KEY else None)
async def delete_file(file_id: str):
    meta = load_meta()
    rec = meta["files"].get(file_id)
    if not rec:
        raise HTTPException(status_code=404, detail="File not found")

    # Remove from vector store
    res = collection.get(where={"file_id": file_id})
    ids = res.get("ids") if res else None
    if ids:
        collection.delete(ids=ids)

    # Remove file
    try:
        Path(rec["path"]).unlink(missing_ok=True)
    except Exception:
        pass

    del meta["files"][file_id]
    save_meta(meta)
    return {"ok": True}

@app.get("/search", dependencies=[Depends(require_api_key)] if API_KEY else None)
async def search(q: str, n: int = 5):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query 'q' is required")
    q_emb = model.encode([q], convert_to_numpy=True).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=n)
    hits = []
    for doc, meta, _id in zip(res.get("documents", [[]])[0], res.get("metadatas", [[]])[0], res.get("ids", [[]])[0]):
        hits.append({
            "id": _id,
            "file_id": meta.get("file_id"),
            "title": meta.get("title"),
            "page": meta.get("page"),
            "snippet": doc[:400],
            "filename": meta.get("filename"),
            "tags": meta.get("tags"),
            "notes": meta.get("notes"),
        })
    return {"query": q, "results": hits}
