"""
API FastAPI para o Gêmeo Digital - Avaliador Lei do Bem
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import sys
from pathlib import Path

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from avaliador import AvaliadorLeiBem, Projeto, ResultadoAvaliacao

# =============================================================================
# Modelos Pydantic para API
# =============================================================================

class ProjetoInput(BaseModel):
    """Projeto a ser avaliado"""
    titulo: str = Field(..., description="Título do projeto")
    descricao: str = Field(..., description="Descrição do projeto")
    elemento_novo: str = Field(..., description="Elemento tecnologicamente novo ou inovador")
    barreira_tecnologica: str = Field(..., description="Barreira ou desafio tecnológico")
    metodologia: str = Field(..., description="Metodologia e métodos utilizados")
    area: str = Field(default="eletroeletronica", description="Área do projeto")
    trl_inicial: Optional[int] = Field(None, ge=1, le=9, description="TRL inicial")
    trl_final: Optional[int] = Field(None, ge=1, le=9, description="TRL final")

    model_config = {
        "json_schema_extra": {
            "example": {
                "titulo": "Sistema de Monitoramento IoT para Redes Elétricas",
                "descricao": "Desenvolvimento de sistema embarcado para monitoramento...",
                "elemento_novo": "Algoritmo proprietário de detecção de anomalias...",
                "barreira_tecnologica": "Desafio de executar modelos de ML em hardware limitado...",
                "metodologia": "1. Revisão do estado da arte 2. Prototipagem...",
                "area": "eletroeletronica",
                "trl_inicial": 3,
                "trl_final": 6
            }
        }
    }


class AvaliacaoResponse(BaseModel):
    """Resposta da avaliação"""
    titulo: str
    area: str
    score_geral: float
    decisao: str
    probabilidade_aprovacao: float
    parecer_resumido: str
    pontos_fortes: list[str]
    pontos_fracos: list[str]
    sugestoes_melhoria: list[str]
    scores_pilares: dict
    frascati: dict


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    area: str


# =============================================================================
# Aplicação FastAPI
# =============================================================================

app = FastAPI(
    title="Gêmeo Digital - Avaliador Lei do Bem",
    description="""
    API para avaliação de projetos de P&D&I submetidos à Lei do Bem.

    ## Funcionalidades

    - **Avaliação de Projetos**: Avalia projetos e retorna score, decisão e parecer
    - **Análise por Pilar**: Score detalhado para cada pilar (descrição, elemento novo, barreira, metodologia)
    - **Análise Frascati**: Verifica aderência aos 5 critérios do Manual de Frascati
    - **Sugestões de Melhoria**: Recomendações para melhorar projetos com problemas

    ## Área Suportada

    Atualmente: **Eletroeletrônica**

    Outras áreas em desenvolvimento.
    """,
    version="0.1.0",
    contact={
        "name": "Suporte",
        "email": "suporte@exemplo.com"
    }
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar origens
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instância global do avaliador
avaliador = AvaliadorLeiBem(area="eletroeletronica")


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check e informações básicas"""
    return HealthResponse(
        status="online",
        version="0.1.0",
        area="eletroeletronica"
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check"""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        area="eletroeletronica"
    )


@app.post("/avaliar", response_model=AvaliacaoResponse)
async def avaliar_projeto(projeto: ProjetoInput):
    """
    Avalia um projeto de P&D&I

    Retorna score geral, decisão (APROVADO/DILIGÊNCIA/REJEITADO),
    parecer detalhado e sugestões de melhoria.
    """
    try:
        # Converte para objeto Projeto
        projeto_obj = Projeto(
            titulo=projeto.titulo,
            descricao=projeto.descricao,
            elemento_novo=projeto.elemento_novo,
            barreira_tecnologica=projeto.barreira_tecnologica,
            metodologia=projeto.metodologia,
            area=projeto.area,
            trl_inicial=projeto.trl_inicial,
            trl_final=projeto.trl_final
        )

        # Avalia
        resultado = avaliador.avaliar(projeto_obj)

        # Converte para response
        return AvaliacaoResponse(
            titulo=resultado.titulo,
            area=resultado.area,
            score_geral=round(resultado.score_geral, 2),
            decisao=resultado.decisao.value,
            probabilidade_aprovacao=round(resultado.probabilidade_aprovacao, 1),
            parecer_resumido=resultado.parecer_resumido,
            pontos_fortes=resultado.pontos_fortes,
            pontos_fracos=resultado.pontos_fracos,
            sugestoes_melhoria=resultado.sugestoes_melhoria,
            scores_pilares={
                nome: {
                    "score": round(p.score, 2),
                    "peso": p.peso,
                    "justificativa": p.justificativa
                }
                for nome, p in resultado.scores_pilares.items()
            },
            frascati={
                "inovadora": resultado.frascati.inovadora,
                "criativa": resultado.frascati.criativa,
                "incerta": resultado.frascati.incerta,
                "sistematica": resultado.frascati.sistematica,
                "transferivel": resultado.frascati.transferivel,
                "total_atendidos": resultado.frascati.total_atendidos
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na avaliação: {str(e)}")


@app.post("/avaliar/lote")
async def avaliar_lote(projetos: list[ProjetoInput]):
    """
    Avalia múltiplos projetos de uma vez

    Retorna lista de avaliações.
    """
    resultados = []
    for projeto in projetos:
        try:
            resultado = await avaliar_projeto(projeto)
            resultados.append({"status": "success", "resultado": resultado})
        except HTTPException as e:
            resultados.append({"status": "error", "erro": e.detail})

    return {"total": len(projetos), "resultados": resultados}


@app.get("/criterios")
async def listar_criterios():
    """
    Retorna os critérios de avaliação utilizados

    Útil para entender como os projetos são avaliados.
    """
    return avaliador.criterios


@app.get("/areas")
async def listar_areas():
    """
    Lista áreas suportadas pelo avaliador

    Atualmente apenas eletroeletrônica está implementada.
    """
    return {
        "implementadas": ["eletroeletronica"],
        "em_desenvolvimento": ["quimica", "software", "mecanica", "biotecnologia"],
        "area_atual": avaliador.area
    }


# =============================================================================
# Execução
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
