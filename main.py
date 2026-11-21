import os
from pathlib import Path
from typing import List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

app = FastAPI(
    title="Portal Institucional LabMC - UFOPA",
    description="Backend e servidor estático do Laboratório de Modelagem Computacional da UFOPA",
    version="1.0.0",
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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

SYSTEM_INSTRUCTION = """
Você é o Assistente Institucional e Técnico do Laboratório de Modelagem Computacional (LabMC) da Universidade Federal do Oeste do Pará (UFOPA).

DIRETRIZES DE COMPORTAMENTO E ESTILO:
- Responda sempre em português do Brasil, com tom acadêmico, técnico, claro e objetivo.
- Nunca utilize travessões longos (— ou –) no texto; prefira vírgulas, parênteses ou dois-pontos.
- Quando a pergunta tratar de regras, localização, estrutura ou atividades do laboratório, utilize rigorosamente os dados oficiais listados abaixo.
- Quando a pergunta tratar de conceitos técnicos (Dinâmica dos Fluidos Computacional / CFD, equações de Navier-Stokes, OpenFOAM, programação paralela em GPU com NVIDIA CUDA, Python científico, NumPy, SciPy ou métodos numéricos), forneça explicações didáticas e exemplos de código claros.

DADOS OFICIAIS DO LABMC (UFOPA):
- Nome completo: Laboratório de Modelagem Computacional (LabMC).
- Vinculação institucional: Programa de Ciência e Tecnologia (PC&T) e Instituto de Engenharia e Geociências (IEG) da Universidade Federal do Oeste do Pará (UFOPA).
- Localização física: Bloco 26, Unidade Tapajós, Santarém, Pará. Área física dedicada: 36 metros quadrados.
- Coordenação: Prof. Dr. Josecley Fialho Góes.
- Áreas principais de atuação: Dinâmica dos Fluidos Computacional (DFC/CFD), processamento paralelo por meio da API CUDA da NVIDIA, métodos numéricos aplicados, modelagem matemática, ciência de dados e suporte computacional às disciplinas do PC&T e demais cursos do IEG.
- Infraestrutura computacional: Estações de trabalho equipadas com GPUs NVIDIA com suporte a arquitetura CUDA, sistema operacional Ubuntu Linux, CUDA Toolkit, compiladores GCC/G++, OpenFOAM, ParaView e ecossistema Python científico (NumPy, SciPy, Matplotlib, Jupyter).
- Funcionamento e horário de atendimento: Segunda a sexta-feira, das 08h00 às 18h00, com escalas presenciais de monitores e pesquisadores conforme o calendário acadêmico.
- Acesso e reserva de bancadas: Discentes de graduação (TCC, iniciação científica PIBIC/PIBITI, estágio obrigatório) e pós-graduação vinculados ao IEG/PC&T podem solicitar bancada mediante preenchimento do formulário institucional, assinatura do Termo de Responsabilidade e ciência do orientador ou da coordenação.
- Oportunidades e atividades: Edital de Monitoria do LabMC, Curso online em Dinâmica dos Fluidos (CFD) e Curso de Machine Learning.
- Canais oficiais: E-mail labmc@ufopa.edu.br e perfil oficial no Instagram @labmc.ufopa.
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


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "service": "labmc-ufopa-portal",
        "ai_configured": bool(GEMINI_API_KEY),
    }


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
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.35,
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

        # Garante remoção de travessões longos caso ocorram na geração
        reply_text = reply_text.replace(" — ", ", ").replace("—", ", ").replace(" – ", ", ")

        return ChatResponse(reply=reply_text, mode="api")

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


# Serve arquivos estáticos e o portal principal
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Arquivo static/index.html não encontrado.")
    return FileResponse(str(index_file))