import base64
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader

load_dotenv()

app = FastAPI(
    title="Portal Institucional LabMC - UFOPA",
    description="Backend, CMS Completo em JSON, Agendamento de Máquinas/Espaço e RAG do LabMC UFOPA",
    version="2.1.0",
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

DOCS_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "labmc2026").strip()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
GITHUB_REPO = os.getenv("GITHUB_REPO", "Breno-max-lab/labmc-ufopa-portal").strip()
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main").strip()

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


class BookingCreateRequest(BaseModel):
    tipo_solicitacao: str  # "maquina" ou "espaco"
    solicitante_nome: str
    matricula_siape: str
    email: str
    vinculo_curso: str
    recurso_ou_finalidade: str
    data_desejada: str
    horario_inicio: str
    horario_fim: str
    justificativa: str


class BookingDecisionRequest(BaseModel):
    status: str  # "APROVADO" ou "RECUSADO"
    observacao_admin: str = ""


def verify_admin(x_admin_token: Optional[str]):
    if not x_admin_token or x_admin_token.strip() != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Senha administrativa inválida ou não informada.")


async def github_upsert_file(repo_path: str, content_bytes: bytes, commit_message: str) -> bool:
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return False

    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{repo_path}"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        sha = None
        get_res = await client.get(api_url, headers=headers, params={"ref": GITHUB_BRANCH})
        if get_res.status_code == 200:
            sha = get_res.json().get("sha")

        payload = {
            "message": commit_message,
            "content": base64.b64encode(content_bytes).decode("utf-8"),
            "branch": GITHUB_BRANCH,
        }
        if sha:
            payload["sha"] = sha

        put_res = await client.put(api_url, headers=headers, json=payload)
        return put_res.status_code in {200, 201}


async def github_delete_file(repo_path: str, commit_message: str) -> bool:
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return False

    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{repo_path}"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        get_res = await client.get(api_url, headers=headers, params={"ref": GITHUB_BRANCH})
        if get_res.status_code != 200:
            return False

        sha = get_res.json().get("sha")
        if not sha:
            return False

        del_res = await client.request(
            "DELETE",
            api_url,
            headers=headers,
            json={
                "message": commit_message,
                "sha": sha,
                "branch": GITHUB_BRANCH,
            },
        )
        return del_res.status_code == 200


def extract_text_from_file(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".pdf":
            reader = PdfReader(str(file_path))
            pages_text = [page.extract_text().strip() for page in reader.pages if (page.extract_text() or "").strip()]
            return "\n".join(pages_text)
        elif suffix in {".md", ".txt"}:
            return file_path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""
    return ""


def load_knowledge_chunks() -> List[dict]:
    chunks = []
    if DOCS_DIR.exists():
        for file_path in sorted(DOCS_DIR.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in {".pdf", ".md", ".txt"}:
                raw_text = extract_text_from_file(file_path)
                if not raw_text:
                    continue
                for i in range(0, len(raw_text), 1000):
                    snippet = raw_text[i : i + 1200].strip()
                    if snippet:
                        chunks.append({"source": file_path.name, "text": snippet})

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
            for eq in data.get("infraestrutura", []):
                chunks.append({
                    "source": "Infraestrutura e Equipamentos",
                    "text": f"Recurso ({eq.get('recurso', '')}): {eq.get('especificacao', '')}. Aplicação: {eq.get('aplicacao', '')}. Status: {eq.get('status', '')}."
                })
            for mem in data.get("equipe", []):
                chunks.append({
                    "source": "Equipe do LabMC",
                    "text": f"Membro ({mem.get('role', '')}): {mem.get('name', '')}. Área: {mem.get('area', '')}."
                })
        except Exception:
            pass

    return chunks


def retrieve_relevant_context(query: str, top_k: int = 4) -> tuple[str, List[str]]:
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
    docs = [f.name for f in DOCS_DIR.iterdir() if f.is_file() and f.suffix.lower() in {".pdf", ".md", ".txt"}] if DOCS_DIR.exists() else []
    return {
        "status": "ok",
        "service": "labmc-ufopa-portal",
        "ai_configured": bool(GEMINI_API_KEY),
        "github_sync": bool(GITHUB_TOKEN),
        "indexed_docs": len(docs),
        "docs_list": docs,
    }


@app.get("/api/content")
async def get_portal_content():
    if not DATA_FILE.exists():
        raise HTTPException(status_code=404, detail="Arquivo portal_content.json não encontrado.")
    try:
        data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        # Oculta dados pessoais sensíveis na rota pública, mantendo apenas agenda aprovada
        public_bookings = [
            {
                "protocolo": b.get("protocolo"),
                "tipo_solicitacao": b.get("tipo_solicitacao"),
                "recurso_ou_finalidade": b.get("recurso_ou_finalidade"),
                "data_desejada": b.get("data_desejada"),
                "horario_inicio": b.get("horario_inicio"),
                "horario_fim": b.get("horario_fim"),
                "status": b.get("status"),
                "solicitante_resumo": b.get("solicitante_nome", "").split(" ")[0],
            }
            for b in data.get("agendamentos", [])
            if b.get("status") == "APROVADO"
        ]
        safe_data = {k: v for k, v in data.items() if k != "agendamentos"}
        safe_data["agendamentos_aprovados"] = public_bookings
        return safe_data
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Erro ao ler dados estruturados do portal.") from exc


@app.post("/api/bookings")
async def create_booking_request(payload: BookingCreateRequest):
    """Recebe uma solicitação pública de Máquina (PC) ou Espaço Físico (Evento/Reunião/Aula) com status PENDENTE."""
    if not DATA_FILE.exists():
        raise HTTPException(status_code=404, detail="Arquivo portal_content.json não encontrado.")

    if payload.tipo_solicitacao not in {"maquina", "espaco"}:
        raise HTTPException(status_code=400, detail="Tipo de solicitação inválido.")

    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    agendamentos = data.setdefault("agendamentos", [])

    protocolo = f"LABMC-{uuid.uuid4().hex[:6].upper()}"
    novo_registro = {
        "protocolo": protocolo,
        "tipo_solicitacao": payload.tipo_solicitacao,
        "solicitante_nome": payload.solicitante_nome.strip(),
        "matricula_siape": payload.matricula_siape.strip(),
        "email": payload.email.strip(),
        "vinculo_curso": payload.vinculo_curso.strip(),
        "recurso_ou_finalidade": payload.recurso_ou_finalidade.strip(),
        "data_desejada": payload.data_desejada.strip(),
        "horario_inicio": payload.horario_inicio.strip(),
        "horario_fim": payload.horario_fim.strip(),
        "justificativa": payload.justificativa.strip(),
        "status": "PENDENTE",
        "observacao_admin": "",
        "criado_em": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }

    agendamentos.insert(0, novo_registro)
    updated_json = json.dumps(data, ensure_ascii=False, indent=2)
    DATA_FILE.write_text(updated_json, encoding="utf-8")

    await github_upsert_file(
        repo_path="data/portal_content.json",
        content_bytes=updated_json.encode("utf-8"),
        commit_message=f"feat(booking): nova solicitacao {protocolo} ({payload.tipo_solicitacao})",
    )

    return {"status": "ok", "protocolo": protocolo, "situacao": "PENDENTE"}


@app.get("/api/bookings/status/{query}")
async def check_booking_status(query: str):
    """Permite ao solicitante consultar o status pelo Protocolo ou Matrícula/SIAPE."""
    if not DATA_FILE.exists():
        raise HTTPException(status_code=404, detail="Base não encontrada.")
    q = query.strip().upper()
    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    matches = [
        {
            "protocolo": b.get("protocolo"),
            "tipo_solicitacao": b.get("tipo_solicitacao"),
            "solicitante_nome": b.get("solicitante_nome"),
            "recurso_ou_finalidade": b.get("recurso_ou_finalidade"),
            "data_desejada": b.get("data_desejada"),
            "horario_inicio": b.get("horario_inicio"),
            "horario_fim": b.get("horario_fim"),
            "status": b.get("status"),
            "observacao_admin": b.get("observacao_admin", ""),
        }
        for b in data.get("agendamentos", [])
        if b.get("protocolo", "").upper() == q or b.get("matricula_siape", "").upper() == q
    ]
    return {"results": matches}


@app.post("/api/admin/login")
async def admin_login(x_admin_token: Optional[str] = Header(None)):
    verify_admin(x_admin_token)
    return {"authenticated": True, "github_sync": bool(GITHUB_TOKEN)}


@app.get("/api/admin/full-content")
async def admin_get_full_content(x_admin_token: Optional[str] = Header(None)):
    """Retorna todos os dados completos, incluindo solicitações pendentes e contatos."""
    verify_admin(x_admin_token)
    if not DATA_FILE.exists():
        raise HTTPException(status_code=404, detail="Arquivo portal_content.json não encontrado.")
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


@app.put("/api/admin/content")
async def admin_update_full_content(
    payload: Dict[str, Any] = Body(...),
    commit_msg: str = Header("chore(admin): atualiza conteudo do portal LabMC", alias="X-Commit-Message"),
    x_admin_token: Optional[str] = Header(None),
):
    verify_admin(x_admin_token)
    try:
        updated_json = json.dumps(payload, ensure_ascii=False, indent=2)
        DATA_FILE.write_text(updated_json, encoding="utf-8")
        synced = await github_upsert_file(
            repo_path="data/portal_content.json",
            content_bytes=updated_json.encode("utf-8"),
            commit_message=commit_msg,
        )
        return {"status": "ok", "github_synced": synced}
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Erro ao salvar conteúdo estruturado do portal.") from exc


@app.put("/api/admin/bookings/{protocolo}")
async def admin_decide_booking(
    protocolo: str,
    decision: BookingDecisionRequest,
    x_admin_token: Optional[str] = Header(None),
):
    """Aprova ou recusa uma solicitação de Máquina (PC) ou Espaço Físico."""
    verify_admin(x_admin_token)
    if decision.status not in {"APROVADO", "RECUSADO", "PENDENTE"}:
        raise HTTPException(status_code=400, detail="Status inválido.")

    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    agendamentos = data.get("agendamentos", [])
    found = None
    for item in agendamentos:
        if item.get("protocolo") == protocolo:
            item["status"] = decision.status
            item["observacao_admin"] = decision.observacao_admin.strip()
            found = item
            break

    if not found:
        raise HTTPException(status_code=404, detail="Solicitação não encontrada.")

    updated_json = json.dumps(data, ensure_ascii=False, indent=2)
    DATA_FILE.write_text(updated_json, encoding="utf-8")

    synced = await github_upsert_file(
        repo_path="data/portal_content.json",
        content_bytes=updated_json.encode("utf-8"),
        commit_message=f"chore(admin): {decision.status.lower()} solicitacao {protocolo}",
    )
    return {"status": "ok", "github_synced": synced}


@app.delete("/api/admin/bookings/{protocolo}")
async def admin_delete_booking(protocolo: str, x_admin_token: Optional[str] = Header(None)):
    verify_admin(x_admin_token)
    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    agendamentos = data.get("agendamentos", [])
    new_list = [b for b in agendamentos if b.get("protocolo") != protocolo]
    if len(new_list) == len(agendamentos):
        raise HTTPException(status_code=404, detail="Solicitação não encontrada.")

    data["agendamentos"] = new_list
    updated_json = json.dumps(data, ensure_ascii=False, indent=2)
    DATA_FILE.write_text(updated_json, encoding="utf-8")

    synced = await github_upsert_file(
        repo_path="data/portal_content.json",
        content_bytes=updated_json.encode("utf-8"),
        commit_message=f"chore(admin): remove agendamento {protocolo}",
    )
    return {"status": "deleted", "github_synced": synced}


@app.post("/api/admin/upload-doc")
async def admin_upload_doc(
    file: UploadFile = File(...),
    titulo: str = Form(""),
    tag: str = Form("DOCUMENTO OFICIAL"),
    descricao: str = Form(""),
    add_to_portal: bool = Form(True),
    x_admin_token: Optional[str] = Header(None),
):
    verify_admin(x_admin_token)

    filename = Path(file.filename or "documento.pdf").name
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
    if Path(safe_name).suffix.lower() not in {".pdf", ".md", ".txt"}:
        raise HTTPException(status_code=400, detail="Formato não suportado. Envie arquivos .pdf, .md ou .txt.")

    dest_path = DOCS_DIR / safe_name
    content_bytes = await file.read()
    dest_path.write_bytes(content_bytes)

    synced_doc = await github_upsert_file(
        repo_path=f"static/docs/{safe_name}",
        content_bytes=content_bytes,
        commit_message=f"docs(admin): adiciona documento {safe_name}",
    )

    if add_to_portal and DATA_FILE.exists():
        try:
            data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
            docs_list = data.setdefault("documentos", [])
            doc_title = titulo.strip() or safe_name
            doc_desc = descricao.strip() or f"Documento oficial disponibilizado pela coordenação do LabMC ({safe_name})."
            docs_list.insert(0, {
                "tag": tag.strip().upper() or "DOCUMENTO",
                "titulo": doc_title,
                "descricao": doc_desc,
                "resumo": doc_desc,
                "arquivo_pdf": f"/static/docs/{safe_name}"
            })
            updated_json = json.dumps(data, ensure_ascii=False, indent=2)
            DATA_FILE.write_text(updated_json, encoding="utf-8")
            await github_upsert_file(
                repo_path="data/portal_content.json",
                content_bytes=updated_json.encode("utf-8"),
                commit_message=f"chore(admin): registra {safe_name} no catalogo do portal",
            )
        except Exception:
            pass

    return {"status": "ok", "filename": safe_name, "github_synced": synced_doc}


@app.delete("/api/admin/docs/{filename}")
async def admin_delete_doc(filename: str, x_admin_token: Optional[str] = Header(None)):
    verify_admin(x_admin_token)
    safe_name = Path(filename).name
    target = DOCS_DIR / safe_name
    if target.exists():
        target.unlink()

    synced_del = await github_delete_file(
        repo_path=f"static/docs/{safe_name}",
        commit_message=f"docs(admin): remove documento {safe_name}",
    )

    if DATA_FILE.exists():
        try:
            data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
            docs_list = data.get("documentos", [])
            new_docs = [d for d in docs_list if not str(d.get("arquivo_pdf", "")).endswith(f"/{safe_name}")]
            if len(new_docs) != len(docs_list):
                data["documentos"] = new_docs
                updated_json = json.dumps(data, ensure_ascii=False, indent=2)
                DATA_FILE.write_text(updated_json, encoding="utf-8")
                await github_upsert_file(
                    repo_path="data/portal_content.json",
                    content_bytes=updated_json.encode("utf-8"),
                    commit_message=f"chore(admin): remove referencia de {safe_name} do catalogo",
                )
        except Exception:
            pass

    return {"status": "deleted", "filename": safe_name, "github_synced": synced_del}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    user_message = payload.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="A mensagem não pode estar vazia.")

    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY não configurada no ambiente do servidor.")

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
        raise HTTPException(status_code=502, detail=f"Erro na comunicação com a API externa: {exc.response.status_code}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Falha interna ao processar a resposta do assistente.") from exc


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Arquivo static/index.html não encontrado.")
    return FileResponse(str(index_file))


@app.get("/admin")
@app.get("/admin/")
@app.get("/Admin")
@app.get("/Admin/")
async def serve_admin():
    admin_file = STATIC_DIR / "admin.html"
    if not admin_file.exists():
        raise HTTPException(status_code=404, detail="Arquivo static/admin.html não encontrado.")
    return FileResponse(str(admin_file))