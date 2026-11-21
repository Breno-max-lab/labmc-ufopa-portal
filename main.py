import json
import os
import re
from pathlib import Path
from typing import List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader

load_dotenv()

app = FastAPI(
    title="Portal Institucional LabMC - UFOPA",
    description="Backend, CMS em JSON, RAG de Documentos e servidor estático do LabMC UFOPA",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DOCS_DIR = STATIC_DIR / "docs"
DATA_FILE = BASE_DIR / "data" / "portal_content.json"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

BASE_SYSTEM_INSTRUCTION = """
Você é o Assistente Institucional e Técnico do Laboratório de Modelagem Computacional (LabMC) da Universidade Federal do Oeste do Pará (UFOPA).

DIRETRIZES DE COMPORTAMENTO E ESTILO:
- Responda sempre em português do Brasil, com tom acadêmico, técnico, claro e objetivo.
- Nunca utilize travessões longos (— ou –) no texto; prefira vírgulas, parênteses ou dois-pontos.
- Utilize prioritariamente os trechos extraídos dos documentos oficiais e do catálogo do portal fornecidos no contexto abaixo.
- Se uma informação específica de edital (como data limite ou número de vagas) não constar nos documentos carregados, informe claramente que o detalhe não está nos documentos disponíveis e oriente o discente a consultar a secretaria/coordenação no Bloco 26.
- Quando a pergunta tratar de conceitos técnicos (Dinâmica dos Fluidos Computacional / CFD, equações de Navier-Stokes, OpenFOAM, programação paralela em GPU com NVIDIA CUDA, Python científico, NumPy, SciPy ou métodos numéricos), forneça explicações didáticas e exemplos de código claros.

DADOS INSTITUCIONAIS FIXOS DO LABMC (UFOPA):
- Nome completo: Laboratório de Modelagem Computacional (LabMC).
- Vinculação institucional: Programa de Ciência e Tecnologia (PC&T) e Instituto de Engenharia e Geociências (IEG) da Universidade Federal do Oeste do Pará (UFOPA).
- Localização física: Bloco 26, Unidade Tapajós, Santarém, Pará. Área física dedicada: 36 metros quadrados.
- Coordenação: Prof. Dr. Josecley Fialho Góes.
- Funcionamento regular: Segunda a sexta-feira, das 08h00 às 18h00.
- Canais oficiais: E-mail labmc@ufopa.edu.br e Instagram @labmc.ufopa.
"""


class HistoryItem(BaseModel):
    role: str
    text: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[HistoryItem]] = []


class ChatResponse(BaseModel):
    reply: str
    mode: str
    sources_used: List[str] = []


def extract_text_from_file(file_path: Path) -> str:
    """Extrai texto limpo de arquivos .pdf, .md ou .txt."""
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".pdf":
            reader = PdfReader(str(file_path))
            pages_text = []
            for page in reader.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(text.strip())
            return "\n".join(pages_text)
        elif suffix in {".md", ".txt"}:
            return file_path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""
    return ""


def load_knowledge_chunks() -> List[dict]:
    """Carrega blocos de conhecimento de static/docs/ e data/portal_content.json."""
    chunks = []

    # 1. Lê documentos oficiais em static/docs/
    if DOCS_DIR.exists():
        for file_path in sorted(DOCS_DIR.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in {".pdf", ".md", ".txt"}:
                raw_text = extract_text_from_file(file_path)
                if not raw_text:
                    continue
                # Divide em blocos de ~1200 caracteres com sobreposição
                step = 1000
                size = 1200
                for i in range(0, len(raw_text), step):
                    snippet = raw_text[i : i + size].strip()
                    if snippet:
                        chunks.append({"source": file_path.name, "text": snippet})

    # 2. Lê dados estruturados do CMS (data/portal_content.json)
    if DATA_FILE.exists():
        try:
            data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
            for ev in data.get("eventos", []):
                chunks.append({
                    "source": "Catálogo de Eventos e Editais",
                    "text": f"Evento/Edital ({ev.get('badge', '')} - {ev.get('local', '')}): {ev.get('titulo', '')}. {ev.get('descricao', '')} {ev.get('detalhes', '')}"
                })
            for proj in data.get("projetos", []):
                chunks.append({
                    "source": "Projetos de Pesquisa",
                    "text": f"Projeto ({proj.get('tag', '')}): {proj.get('title', '')}. {proj.get('summary', '')} Ferramentas: {', '.join(proj.get('tools', []))}."
                })
            for doc in data.get("documentos", []):
                chunks.append({
                    "source": "Normas e Manuais",
                    "text": f"Documento ({doc.get('titulo', '')}): {doc.get('descricao', '')} {doc.get('resumo', '')}"
                })
            for pub in data.get("producoes", []):
                chunks.append({
                    "source": "Produção Científica",
                    "text": f"Publicação ({pub.get('year', '')} - {pub.get('type', '')}): {pub.get('title', '')}. Autores: {pub.get('authors', '')}. Veículo: {pub.get('venue', '')}."
                })
        except Exception:
            pass

    return chunks


def retrieve_relevant_context(query: str, top_k: int = 4) -> tuple[str, List[str]]:
    """Seleciona os trechos mais relevantes para a pergunta do usuário."""
    chunks = load_knowledge_chunks()
    if not chunks:
        return "", []

    tokens = set(re.findall(r"\w{3,}", query.lower()))
    if not tokens:
        selected = chunks[:top_k]
    else:
        scored = []
        for chunk in chunks:
            text_lower = chunk["text"].lower()
            score = sum(1 for tok in tokens if tok in text_lower)
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [item[1] for item in scored[:top_k] if item[0] > 0]
        if not selected:
            selected = chunks[:2]

    context_blocks = []
    sources_used = []
    for item in selected:
        src = item["source"]
        if src not in sources_used:
            sources_used.append(src)
        context_blocks.append(f"[Fonte: {src}]\n{item['text']}")

    return "\n\n---\n\n".join(context_blocks), sources_used


@app.get("/api/health")
async def health_check():
    docs_count = 0
    if DOCS_DIR.exists():
        docs_count = len([
            f for f in DOCS_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in {".pdf", ".md", ".txt"}
        ])
    return {
        "status": "ok",
        "service": "labmc-ufopa-portal",
        "ai_configured": bool(GEMINI_API_KEY),
        "indexed_docs": docs_count,
    }


@app.get("/api/content")
async def get_portal_content():
    if not DATA_FILE.exists():
        raise HTTPException(status_code=404, detail="Arquivo portal_content.json não encontrado.")
    try:
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Erro ao ler dados estruturados do portal.") from exc


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    user_message = payload.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="A mensagem não pode estar vazia.")

    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY não configurada no ambiente do servidor.",
        )

    retrieved_context, sources_used = retrieve_relevant_context(user_message)

    full_system_instruction = BASE_SYSTEM_INSTRUCTION
    if retrieved_context:
        full_system_instruction += (
            "\n\nCONTEXTO EXTRAÍDO DOS DOCUMENTOS E REGISTROS DO LABMC:\n"
            f"{retrieved_context}"
        )

    contents = []
    for item in (payload.history or [])[-10:]:
        role = "user" if item.role == "user" else "model"
        contents.append({"role": role, "parts": [{"text": item.text}]})

    contents.append({"role": "user", "parts": [{"text": user_message}]})

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

    request_body = {
        "systemInstruction": {"parts": [{"text": full_system_instruction}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 900,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, json=request_body)
            response.raise_for_status()
            data = response.json()

        reply_text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )

        if not reply_text:
            raise ValueError("Resposta vazia retornada pelo modelo.")

        reply_text = reply_text.replace(" — ", ", ").replace("—", ", ").replace(" – ", ", ")

        return ChatResponse(reply=reply_text, mode="api", sources_used=sources_used)

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Erro na comunicação com a API externa: {exc.response.status_code}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Falha interna ao processar a resposta do assistente.",
        ) from exc


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Arquivo static/index.html não encontrado.")
    return FileResponse(str(index_file))