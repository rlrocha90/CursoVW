from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import List
import time
from prometheus_client import Counter, Summary, generate_latest, CONTENT_TYPE_LATEST

# Certifique-se de que os recursos do NLTK estão baixados
nltk.download("punkt")
nltk.download("stopwords")

# Inicializa a aplicação FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite requisições de qualquer origem
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos HTTP
    allow_headers=["*"],  # Permite todos os cabeçalhos
)

# Dicionário de fluxos de trabalho com múltiplas palavras-chave
fluxos = {
    "reunião": ["reunião", "agenda", "marcar encontro", "reuniões"],
    "relatório": ["relatório", "gerar relatório", "criar relatório", "relatórios"],
    "projeto": ["projeto", "gestão de projetos", "gerenciamento de projetos"],
    "cliente": ["suporte ao cliente", "atendimento ao cliente", "cliente"],
    "financeiro": ["financeiro", "contas", "relatório financeiro", "finanças"],
}

OLLAMA_URL = "http://localhost:11434/api/generate"  # Substitua com a URL do Ollama
PREDICTION_TIME = Summary("inference_duration_seconds", "Duração da inferência")
PREDICTION_COUNT = Counter("inference_count", "Contagem da inferência")

class Pergunta(BaseModel):
    pergunta: str

def extrair_palavras_chave(texto: str) -> List[str]:
    """Extrai palavras-chave da pergunta"""
    stop_words = set(stopwords.words("portuguese"))
    tokens = word_tokenize(texto.lower())  # Tokeniza e converte em minúsculas
    palavras_chave = [t for t in tokens if t.isalnum() and t not in stop_words]
    return palavras_chave

def determinar_fluxo(palavras_chave: List[str]) -> str:
    """Identifica o fluxo de trabalho baseado nas palavras-chave"""
    for fluxo, palavras in fluxos.items():
        # Verifica se alguma palavra-chave do fluxo aparece nas palavras extraídas
        if any(palavra in palavras_chave for palavra in palavras):
            return f"{fluxo.capitalize()}"
    return "Nenhum fluxo específico identificado."

def obter_resposta_llama(pergunta: str) -> str:
    """Obtém a resposta do modelo LLaMA"""
    payload = {
        "model": "llama3.2",
        "prompt": pergunta,
        "stream": False
    }
    resposta = requests.post(OLLAMA_URL, json=payload)
    return resposta.json().get("response", "Erro ao obter resposta")

@app.get("/")
def home():
    return {"message": "API llama funcionando!"}

@app.get("/metrics")
def get_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/pergunta/")
async def fazer_pergunta(pergunta: Pergunta):
    """Processa a pergunta, envia ao LLaMA e retorna a resposta"""
    PREDICTION_COUNT.inc()
    start_time = time.time()
    palavras_chave = extrair_palavras_chave(pergunta.pergunta)
    fluxo = determinar_fluxo(palavras_chave)
    
    # Obter resposta do LLaMA
    resposta_llama = obter_resposta_llama(pergunta.pergunta)
    duration = time.time() - start_time
    PREDICTION_TIME.observe(duration)
    
    return {"fluxo": fluxo, "duração": duration, "resposta": resposta_llama}

