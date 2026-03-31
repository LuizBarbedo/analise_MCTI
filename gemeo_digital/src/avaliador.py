"""
Gêmeo Digital - Avaliador Lei do Bem
Módulo principal de avaliação de projetos P&D&I
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import yaml
from pathlib import Path


class Decisao(Enum):
    APROVADO = "APROVADO"
    DILIGENCIA = "DILIGÊNCIA"
    REJEITADO = "REJEITADO"


@dataclass
class ScorePilar:
    """Score de um pilar individual"""
    nome: str
    score: float
    peso: float
    justificativa: str
    pontos_fortes: list[str]
    pontos_fracos: list[str]

    @property
    def score_ponderado(self) -> float:
        return self.score * self.peso


@dataclass
class AnaliseRascati:
    """Análise dos 5 critérios de Frascati"""
    inovadora: bool = False
    criativa: bool = False
    incerta: bool = False
    sistematica: bool = False
    transferivel: bool = False
    justificativas: dict[str, str] = field(default_factory=dict)

    @property
    def total_atendidos(self) -> int:
        return sum([
            self.inovadora,
            self.criativa,
            self.incerta,
            self.sistematica,
            self.transferivel
        ])

    @property
    def criterios_essenciais_ok(self) -> bool:
        """Verifica se inovadora e incerta (obrigatórios) estão atendidos"""
        return self.inovadora and self.incerta


@dataclass
class ResultadoAvaliacao:
    """Resultado completo da avaliação de um projeto"""
    # Identificação
    titulo: str
    area: str

    # Scores
    score_geral: float
    scores_pilares: dict[str, ScorePilar]

    # Frascati
    frascati: AnaliseRascati

    # Decisão
    decisao: Decisao
    probabilidade_aprovacao: float

    # Parecer
    parecer_resumido: str
    parecer_detalhado: str
    pontos_fortes: list[str]
    pontos_fracos: list[str]
    sugestoes_melhoria: list[str]

    # Similaridade com base de treinamento
    similaridade_aprovados: float
    similaridade_rejeitados: float

    def to_dict(self) -> dict:
        """Converte resultado para dicionário"""
        return {
            "titulo": self.titulo,
            "area": self.area,
            "score_geral": round(self.score_geral, 2),
            "scores_pilares": {
                nome: {
                    "score": round(p.score, 2),
                    "peso": p.peso,
                    "score_ponderado": round(p.score_ponderado, 2),
                    "justificativa": p.justificativa,
                    "pontos_fortes": p.pontos_fortes,
                    "pontos_fracos": p.pontos_fracos
                }
                for nome, p in self.scores_pilares.items()
            },
            "frascati": {
                "inovadora": self.frascati.inovadora,
                "criativa": self.frascati.criativa,
                "incerta": self.frascati.incerta,
                "sistematica": self.frascati.sistematica,
                "transferivel": self.frascati.transferivel,
                "total_atendidos": self.frascati.total_atendidos,
                "justificativas": self.frascati.justificativas
            },
            "decisao": self.decisao.value,
            "probabilidade_aprovacao": round(self.probabilidade_aprovacao, 1),
            "parecer_resumido": self.parecer_resumido,
            "parecer_detalhado": self.parecer_detalhado,
            "pontos_fortes": self.pontos_fortes,
            "pontos_fracos": self.pontos_fracos,
            "sugestoes_melhoria": self.sugestoes_melhoria,
            "similaridade_aprovados": round(self.similaridade_aprovados, 3),
            "similaridade_rejeitados": round(self.similaridade_rejeitados, 3)
        }


@dataclass
class Projeto:
    """Representa um projeto a ser avaliado"""
    titulo: str
    descricao: str
    elemento_novo: str
    barreira_tecnologica: str
    metodologia: str
    area: str = "eletroeletronica"
    trl_inicial: Optional[int] = None
    trl_final: Optional[int] = None
    classificacao: Optional[str] = None  # Pesquisa Básica/Aplicada/Desenvolvimento Experimental

    @classmethod
    def from_dict(cls, data: dict) -> "Projeto":
        """Cria projeto a partir de dicionário"""
        return cls(
            titulo=data.get("titulo", data.get("daproj_noprojeto", "")),
            descricao=data.get("descricao", data.get("daproj_dsprojeto", "")),
            elemento_novo=data.get("elemento_novo", data.get("daproj_dselementotecnologico", "")),
            barreira_tecnologica=data.get("barreira_tecnologica", data.get("daproj_dsdesafiotecnologico", "")),
            metodologia=data.get("metodologia", data.get("daproj_dsmetodologiautilizada", "")),
            area=data.get("area", data.get("daproj_dsareaprojeto", "eletroeletronica")),
            trl_inicial=data.get("trl_inicial"),
            trl_final=data.get("trl_final"),
            classificacao=data.get("classificacao", data.get("daproj_tpnatureza"))
        )

    def to_dict(self) -> dict:
        return {
            "titulo": self.titulo,
            "descricao": self.descricao,
            "elemento_novo": self.elemento_novo,
            "barreira_tecnologica": self.barreira_tecnologica,
            "metodologia": self.metodologia,
            "area": self.area,
            "trl_inicial": self.trl_inicial,
            "trl_final": self.trl_final,
            "classificacao": self.classificacao
        }


class AvaliadorLeiBem:
    """
    Gêmeo Digital - Avaliador de Projetos Lei do Bem

    Este avaliador simula o comportamento de um avaliador técnico do MCTI,
    utilizando:
    - RAG (Retrieval-Augmented Generation) para contexto legal
    - Few-Shot Learning com exemplos de projetos aprovados/rejeitados
    - Prompt Engineering estruturado para avaliação consistente
    """

    def __init__(
        self,
        area: str = "eletroeletronica",
        modelo: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        config_path: Optional[Path] = None
    ):
        """
        Inicializa o avaliador

        Args:
            area: Área de avaliação (ex: eletroeletronica, quimica, software)
            modelo: Modelo LLM a usar
            api_key: Chave da API Anthropic (ou via env ANTHROPIC_API_KEY)
            config_path: Caminho para arquivos de configuração
        """
        self.area = area
        self.modelo = modelo
        self.api_key = api_key

        # Caminhos
        self.base_path = Path(__file__).parent.parent
        self.config_path = config_path or self.base_path / "config"
        self.data_path = self.base_path / "data"

        # Carrega configurações
        self.criterios = self._carregar_criterios()
        self.prompts = self._carregar_prompts()

        # Componentes (inicializados sob demanda)
        self._llm_client = None
        self._rag_engine = None
        self._exemplos_aprovados = None
        self._exemplos_rejeitados = None
        self._exemplos_limitrofes = None

    def _carregar_criterios(self) -> dict:
        """Carrega critérios de avaliação do YAML"""
        criterios_file = self.config_path / "criterios_avaliacao.yaml"
        if criterios_file.exists():
            with open(criterios_file, encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def _carregar_prompts(self) -> dict:
        """Carrega templates de prompts do YAML"""
        prompts_file = self.config_path / "prompts.yaml"
        if prompts_file.exists():
            with open(prompts_file, encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    @property
    def llm_client(self):
        """Inicializa cliente LLM sob demanda"""
        if self._llm_client is None:
            # TODO: Implementar inicialização do cliente Anthropic
            # from anthropic import Anthropic
            # self._llm_client = Anthropic(api_key=self.api_key)
            pass
        return self._llm_client

    @property
    def rag_engine(self):
        """Inicializa motor RAG sob demanda"""
        if self._rag_engine is None:
            # TODO: Implementar RAGEngine
            # from .rag_engine import RAGEngine
            # self._rag_engine = RAGEngine(self.data_path / "base_conhecimento")
            pass
        return self._rag_engine

    def carregar_exemplos(self, tipo: str = "todos") -> dict:
        """
        Carrega exemplos de projetos para few-shot learning

        Args:
            tipo: "aprovados", "rejeitados", "limitrofes" ou "todos"
        """
        exemplos = {}

        if tipo in ["aprovados", "todos"]:
            path = self.data_path / "exemplos" / "aprovados"
            exemplos["aprovados"] = self._carregar_exemplos_pasta(path)

        if tipo in ["rejeitados", "todos"]:
            path = self.data_path / "exemplos" / "rejeitados"
            exemplos["rejeitados"] = self._carregar_exemplos_pasta(path)

        if tipo in ["limitrofes", "todos"]:
            path = self.data_path / "exemplos" / "limitrofes"
            exemplos["limitrofes"] = self._carregar_exemplos_pasta(path)

        return exemplos

    def _carregar_exemplos_pasta(self, pasta: Path) -> list[dict]:
        """Carrega exemplos de uma pasta"""
        exemplos = []
        if pasta.exists():
            for arquivo in pasta.glob("*.yaml"):
                with open(arquivo, encoding="utf-8") as f:
                    exemplos.append(yaml.safe_load(f))
            for arquivo in pasta.glob("*.json"):
                import json
                with open(arquivo, encoding="utf-8") as f:
                    exemplos.append(json.load(f))
        return exemplos

    def avaliar(self, projeto: Projeto | dict) -> ResultadoAvaliacao:
        """
        Avalia um projeto e retorna o resultado completo

        Args:
            projeto: Projeto a avaliar (objeto Projeto ou dict)

        Returns:
            ResultadoAvaliacao com scores, decisão e parecer
        """
        # Converte dict para Projeto se necessário
        if isinstance(projeto, dict):
            projeto = Projeto.from_dict(projeto)

        # TODO: Implementar avaliação real com LLM
        # Por enquanto, retorna resultado placeholder
        return self._avaliar_placeholder(projeto)

    def _avaliar_placeholder(self, projeto: Projeto) -> ResultadoAvaliacao:
        """Avaliação placeholder para testes (sem LLM)"""
        # Scores placeholder
        scores_pilares = {
            "descricao": ScorePilar(
                nome="Descrição do Projeto",
                score=7.0,
                peso=0.15,
                justificativa="[Aguardando implementação com LLM]",
                pontos_fortes=["[Placeholder]"],
                pontos_fracos=["[Placeholder]"]
            ),
            "elemento_novo": ScorePilar(
                nome="Elemento Novo/Inovador",
                score=6.5,
                peso=0.30,
                justificativa="[Aguardando implementação com LLM]",
                pontos_fortes=["[Placeholder]"],
                pontos_fracos=["[Placeholder]"]
            ),
            "barreira": ScorePilar(
                nome="Barreira Tecnológica",
                score=6.0,
                peso=0.30,
                justificativa="[Aguardando implementação com LLM]",
                pontos_fortes=["[Placeholder]"],
                pontos_fracos=["[Placeholder]"]
            ),
            "metodologia": ScorePilar(
                nome="Metodologia",
                score=7.0,
                peso=0.25,
                justificativa="[Aguardando implementação com LLM]",
                pontos_fortes=["[Placeholder]"],
                pontos_fracos=["[Placeholder]"]
            )
        }

        # Calcula score geral
        score_geral = sum(p.score_ponderado for p in scores_pilares.values())

        # Frascati placeholder
        frascati = AnaliseRascati(
            inovadora=True,
            criativa=True,
            incerta=True,
            sistematica=True,
            transferivel=False,
            justificativas={
                "inovadora": "[Aguardando implementação com LLM]",
                "criativa": "[Aguardando implementação com LLM]",
                "incerta": "[Aguardando implementação com LLM]",
                "sistematica": "[Aguardando implementação com LLM]",
                "transferivel": "[Aguardando implementação com LLM]"
            }
        )

        # Determina decisão
        thresholds = self.criterios.get("thresholds", {})
        if score_geral >= thresholds.get("aprovado", {}).get("score_minimo", 7.0):
            decisao = Decisao.APROVADO
            prob = min(95, 70 + (score_geral - 7) * 10)
        elif score_geral >= thresholds.get("diligencia", {}).get("score_minimo", 5.0):
            decisao = Decisao.DILIGENCIA
            prob = 40 + (score_geral - 5) * 15
        else:
            decisao = Decisao.REJEITADO
            prob = max(5, score_geral * 8)

        return ResultadoAvaliacao(
            titulo=projeto.titulo,
            area=projeto.area,
            score_geral=score_geral,
            scores_pilares=scores_pilares,
            frascati=frascati,
            decisao=decisao,
            probabilidade_aprovacao=prob,
            parecer_resumido=(
                f"[PLACEHOLDER] O projeto '{projeto.titulo}' foi avaliado com score {score_geral:.2f}/10. "
                f"Esta é uma avaliação placeholder - implemente a integração com LLM para avaliação real."
            ),
            parecer_detalhado="[Aguardando implementação com LLM]",
            pontos_fortes=["[Placeholder - aguardando LLM]"],
            pontos_fracos=["[Placeholder - aguardando LLM]"],
            sugestoes_melhoria=["[Placeholder - aguardando LLM]"],
            similaridade_aprovados=0.0,
            similaridade_rejeitados=0.0
        )

    def avaliar_lote(self, projetos: list[Projeto | dict]) -> list[ResultadoAvaliacao]:
        """Avalia múltiplos projetos"""
        return [self.avaliar(p) for p in projetos]

    def gerar_relatorio(self, resultado: ResultadoAvaliacao, formato: str = "markdown") -> str:
        """
        Gera relatório formatado da avaliação

        Args:
            resultado: ResultadoAvaliacao
            formato: "markdown", "html" ou "txt"
        """
        if formato == "markdown":
            return self._relatorio_markdown(resultado)
        elif formato == "html":
            return self._relatorio_html(resultado)
        else:
            return self._relatorio_txt(resultado)

    def _relatorio_markdown(self, r: ResultadoAvaliacao) -> str:
        """Gera relatório em Markdown"""
        md = f"""# Parecer Técnico - Lei do Bem

## Projeto: {r.titulo}
**Área:** {r.area}
**Data:** [Data da avaliação]

---

## Resultado

| Métrica | Valor |
|---------|-------|
| **Score Geral** | {r.score_geral:.2f}/10 |
| **Decisão** | **{r.decisao.value}** |
| **Probabilidade de Aprovação** | {r.probabilidade_aprovacao:.1f}% |
| **Critérios Frascati Atendidos** | {r.frascati.total_atendidos}/5 |

---

## Avaliação por Pilar

"""
        for nome, pilar in r.scores_pilares.items():
            md += f"""### {pilar.nome}
- **Score:** {pilar.score:.1f}/10 (Peso: {pilar.peso*100:.0f}%)
- **Justificativa:** {pilar.justificativa}
- **Pontos Fortes:** {', '.join(pilar.pontos_fortes)}
- **Pontos Fracos:** {', '.join(pilar.pontos_fracos)}

"""

        md += f"""---

## Análise Frascati

| Critério | Atendido | Justificativa |
|----------|----------|---------------|
| Inovadora | {'Sim' if r.frascati.inovadora else 'Nao'} | {r.frascati.justificativas.get('inovadora', '-')} |
| Criativa | {'Sim' if r.frascati.criativa else 'Nao'} | {r.frascati.justificativas.get('criativa', '-')} |
| Incerta | {'Sim' if r.frascati.incerta else 'Nao'} | {r.frascati.justificativas.get('incerta', '-')} |
| Sistemática | {'Sim' if r.frascati.sistematica else 'Nao'} | {r.frascati.justificativas.get('sistematica', '-')} |
| Transferível | {'Sim' if r.frascati.transferivel else 'Nao'} | {r.frascati.justificativas.get('transferivel', '-')} |

---

## Parecer

{r.parecer_resumido}

### Pontos Fortes
"""
        for ponto in r.pontos_fortes:
            md += f"- {ponto}\n"

        md += "\n### Pontos a Melhorar\n"
        for ponto in r.pontos_fracos:
            md += f"- {ponto}\n"

        if r.sugestoes_melhoria:
            md += "\n### Sugestões de Melhoria\n"
            for sug in r.sugestoes_melhoria:
                md += f"- {sug}\n"

        md += """
---

*Avaliação gerada pelo Gêmeo Digital - Avaliador Lei do Bem*
*Este parecer é uma simulação e não substitui a avaliação oficial do MCTI*
"""
        return md

    def _relatorio_html(self, r: ResultadoAvaliacao) -> str:
        """Gera relatório em HTML"""
        # TODO: Implementar
        return "<html><body>TODO: Implementar relatório HTML</body></html>"

    def _relatorio_txt(self, r: ResultadoAvaliacao) -> str:
        """Gera relatório em texto plano"""
        # TODO: Implementar
        return "TODO: Implementar relatório TXT"


# =============================================================================
# Exemplo de uso
# =============================================================================
if __name__ == "__main__":
    # Exemplo de uso básico
    avaliador = AvaliadorLeiBem(area="eletroeletronica")

    projeto_exemplo = {
        "titulo": "Sistema de Monitoramento IoT para Redes Elétricas Inteligentes",
        "descricao": """
            Desenvolvimento de um sistema embarcado para monitoramento em tempo real
            de redes elétricas de distribuição, utilizando sensores IoT e algoritmos
            de machine learning para detecção de anomalias e predição de falhas.
        """,
        "elemento_novo": """
            Algoritmo proprietário de detecção de anomalias baseado em redes neurais
            recorrentes (LSTM) otimizado para execução em microcontroladores de baixo
            consumo, permitindo análise edge sem necessidade de conexão constante.
        """,
        "barreira_tecnologica": """
            Desafio de executar modelos de ML em hardware com recursos limitados
            (32KB RAM) mantendo precisão aceitável. Soluções existentes requerem
            hardware mais robusto ou conexão com nuvem.
        """,
        "metodologia": """
            1. Revisão do estado da arte em TinyML e edge computing
            2. Prototipagem de arquiteturas de rede neural compactadas
            3. Quantização e otimização de modelos
            4. Testes em bancada com dados reais de subestações
            5. Validação em campo com piloto em 3 subestações
        """
    }

    resultado = avaliador.avaliar(projeto_exemplo)
    print(avaliador.gerar_relatorio(resultado))
